"""Training loop for IR Signature Recognition.

Provides the Trainer class that orchestrates:
- Combined loss (CrossEntropy + TripletMarginLoss)
- Epoch-level metrics logging (EpochMetrics)
- VRAM OOM recovery (halve batch size, retry)
- NaN loss recovery (reduce LR by 0.5, max 3 retries)
- Checkpoint saving (LoRA weights, head weights, config)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ir_recognition.models import (
    EpochMetrics,
    SUPPORTED_VEHICLE_CLASSES,
    TrainingConfig,
)
from ir_recognition.signature_db.database import SignatureDatabase
from ir_recognition.training.dataloader import (
    GradientAccumulationConfig,
    IRSignatureDataset,
    TripletBatchSampler,
)
from ir_recognition.training.model import (
    IRRecognitionModel,
    get_peak_vram_usage_gb,
    get_vram_usage_gb,
)

logger = logging.getLogger(__name__)


def _compute_embedding_quality(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> float:
    """Compute embedding quality: mean intra-class similarity - mean inter-class similarity.

    Args:
        embeddings: L2-normalized embeddings of shape (N, D).
        labels: Integer class labels of shape (N,).

    Returns:
        Embedding quality score (higher is better).
    """
    if embeddings.shape[0] < 2:
        return 0.0

    # Compute pairwise cosine similarity matrix
    # Since embeddings are L2-normalized, dot product = cosine similarity
    sim_matrix = torch.mm(embeddings, embeddings.t())

    # Create masks for intra-class and inter-class pairs
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    # Exclude self-similarity (diagonal)
    mask_diag = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)

    intra_mask = label_matrix & mask_diag
    inter_mask = ~label_matrix & mask_diag

    intra_count = intra_mask.sum().item()
    inter_count = inter_mask.sum().item()

    if intra_count == 0 or inter_count == 0:
        return 0.0

    mean_intra = sim_matrix[intra_mask].mean().item()
    mean_inter = sim_matrix[inter_mask].mean().item()

    return mean_intra - mean_inter


def _mine_hard_triplets(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online hard triplet mining from batch embeddings.

    For each anchor, find the hardest positive (farthest same-class)
    and hardest negative (closest different-class).

    Args:
        embeddings: L2-normalized embeddings of shape (N, D).
        labels: Integer class labels of shape (N,).

    Returns:
        Tuple of (anchors, positives, negatives) tensors, each shape (num_triplets, D).
    """
    device = embeddings.device
    n = embeddings.shape[0]

    # Pairwise distance matrix (squared Euclidean)
    # For L2-normalized vectors: ||a - b||^2 = 2 - 2*cos(a,b)
    dist_matrix = 2.0 - 2.0 * torch.mm(embeddings, embeddings.t())

    anchors = []
    positives = []
    negatives = []

    for i in range(n):
        # Find same-class indices (excluding self)
        same_class = (labels == labels[i]) & (
            torch.arange(n, device=device) != i
        )
        # Find different-class indices
        diff_class = labels != labels[i]

        if not same_class.any() or not diff_class.any():
            continue

        # Hardest positive: max distance among same class
        pos_dists = dist_matrix[i][same_class]
        hardest_pos_idx = torch.where(same_class)[0][pos_dists.argmax()]

        # Hardest negative: min distance among different class
        neg_dists = dist_matrix[i][diff_class]
        hardest_neg_idx = torch.where(diff_class)[0][neg_dists.argmin()]

        anchors.append(embeddings[i])
        positives.append(embeddings[hardest_pos_idx])
        negatives.append(embeddings[hardest_neg_idx])

    if not anchors:
        # Return empty tensors if no valid triplets found
        return (
            torch.zeros(0, embeddings.shape[1], device=device),
            torch.zeros(0, embeddings.shape[1], device=device),
            torch.zeros(0, embeddings.shape[1], device=device),
        )

    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
    )


class Trainer:
    """Training orchestrator for IR Signature Recognition.

    Manages the full training loop including:
    - Combined loss (classification + metric learning)
    - VRAM OOM recovery (halve batch size, retry)
    - NaN loss recovery (reduce LR by 0.5, max 3 retries)
    - Epoch metrics logging
    - Checkpoint saving

    Args:
        config: TrainingConfig with all training hyperparameters.
        database: SignatureDatabase with train/val splits populated.
        model: Optional pre-built IRRecognitionModel. If None, will be loaded
            from config using load_quantized_model().
    """

    def __init__(
        self,
        config: TrainingConfig,
        database: SignatureDatabase,
        model: Optional[IRRecognitionModel] = None,
    ) -> None:
        self.config = config
        self.database = database

        # Model setup
        if model is not None:
            self.model = model
        else:
            from ir_recognition.training.model import load_quantized_model

            self.model = load_quantized_model(config)

        # Determine device
        self.device = self._get_device()

        # Loss functions
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=config.triplet_margin)

        # Current batch size (may be reduced on OOM)
        self._current_batch_size = config.batch_size

        # NaN recovery state
        self._nan_retries = 0
        self._max_nan_retries = 3

    def _get_device(self) -> torch.device:
        """Determine the device the model is on."""
        try:
            param = next(self.model.parameters())
            return param.device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self) -> list[EpochMetrics]:
        """Run the full training loop.

        Returns:
            List of EpochMetrics, one per completed epoch.

        Raises:
            RuntimeError: If NaN loss persists after max retries.
        """
        # Build datasets
        train_dataset = IRSignatureDataset(self.database, split="train")
        val_dataset = IRSignatureDataset(self.database, split="val")

        # Store the training class order for checkpoint saving
        self._class_names = [
            train_dataset._idx_to_class[i]
            for i in range(len(train_dataset._idx_to_class))
        ]

        # Set up optimizer (only trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.config.learning_rate)
        current_lr = self.config.learning_rate

        # Gradient accumulation config
        grad_accum = GradientAccumulationConfig(
            batch_size=self._current_batch_size,
            accumulation_steps=max(1, 16 // self._current_batch_size),
        )

        metrics_history: list[EpochMetrics] = []

        # Reset peak VRAM tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        epoch = 0
        while epoch < self.config.num_epochs:
            try:
                epoch_metrics = self._train_epoch(
                    epoch=epoch,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    optimizer=optimizer,
                    current_lr=current_lr,
                    grad_accum=grad_accum,
                )
                metrics_history.append(epoch_metrics)
                self._nan_retries = 0  # Reset NaN counter on success

                logger.info(
                    f"Epoch {epoch}: train_loss={epoch_metrics.train_loss:.4f}, "
                    f"val_loss={epoch_metrics.val_loss:.4f}, "
                    f"accuracy={epoch_metrics.classification_accuracy:.4f}, "
                    f"embedding_quality={epoch_metrics.embedding_quality:.4f}, "
                    f"vram_peak={epoch_metrics.vram_peak_gb:.2f}GB, "
                    f"lr={epoch_metrics.learning_rate:.2e}"
                )
                epoch += 1

            except torch.cuda.OutOfMemoryError:
                # VRAM OOM recovery: halve batch size and retry
                self._current_batch_size = max(1, self._current_batch_size // 2)
                grad_accum = GradientAccumulationConfig(
                    batch_size=self._current_batch_size,
                    accumulation_steps=max(1, 16 // self._current_batch_size),
                )
                logger.warning(
                    f"CUDA OOM at epoch {epoch}. "
                    f"Halving batch size to {self._current_batch_size}. Retrying."
                )
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Retry same epoch (don't increment)

            except _NaNLossError:
                # NaN loss recovery: reduce LR by 0.5, max 3 retries
                self._nan_retries += 1
                if self._nan_retries > self._max_nan_retries:
                    raise RuntimeError(
                        f"NaN loss detected {self._nan_retries} times. "
                        f"Max retries ({self._max_nan_retries}) exceeded."
                    )
                current_lr *= 0.5
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                logger.warning(
                    f"NaN loss at epoch {epoch}. "
                    f"Reducing LR to {current_lr:.2e}. "
                    f"Retry {self._nan_retries}/{self._max_nan_retries}."
                )
                # Retry same epoch (don't increment)

        return metrics_history

    def _train_epoch(
        self,
        epoch: int,
        train_dataset: IRSignatureDataset,
        val_dataset: IRSignatureDataset,
        optimizer: AdamW,
        current_lr: float,
        grad_accum: GradientAccumulationConfig,
    ) -> EpochMetrics:
        """Train for a single epoch and compute metrics.

        Args:
            epoch: Current epoch number.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            optimizer: The optimizer.
            current_lr: Current learning rate.
            grad_accum: Gradient accumulation configuration.

        Returns:
            EpochMetrics for this epoch.

        Raises:
            _NaNLossError: If NaN loss is detected.
            torch.cuda.OutOfMemoryError: If VRAM is exhausted.
        """
        # Build data loaders with triplet-aware sampling
        train_sampler = TripletBatchSampler(
            labels=train_dataset.labels,
            batch_size=self._current_batch_size,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self._current_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        # Training phase
        self.model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits, embeddings = self.model(images)

            # Classification loss
            cls_loss = self.classification_loss_fn(logits, labels)

            # Triplet loss with online hard mining
            anchors, positives, negatives = _mine_hard_triplets(embeddings, labels)
            if anchors.shape[0] > 0:
                metric_loss = self.triplet_loss_fn(anchors, positives, negatives)
            else:
                metric_loss = torch.tensor(0.0, device=self.device)

            # Combined loss
            combined_loss = (
                self.config.loss_weights["classification"] * cls_loss
                + self.config.loss_weights["metric"] * metric_loss
            )

            # Check for NaN
            if torch.isnan(combined_loss):
                raise _NaNLossError("NaN loss detected during training")

            # Scale for gradient accumulation
            scaled_loss = grad_accum.scale_loss(combined_loss)
            scaled_loss.backward()

            # Optimizer step (respecting gradient accumulation)
            if grad_accum.should_step(batch_idx):
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += combined_loss.item()
            num_train_batches += 1

        # Final optimizer step if there are remaining accumulated gradients
        if num_train_batches % grad_accum.accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / max(num_train_batches, 1)

        # Validation phase
        val_loss, val_accuracy, embedding_quality = self._validate(val_loader)

        # Get peak VRAM
        vram_peak = get_peak_vram_usage_gb()

        return EpochMetrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            classification_accuracy=val_accuracy,
            embedding_quality=embedding_quality,
            vram_peak_gb=vram_peak,
            learning_rate=current_lr,
        )

    @torch.no_grad()
    def _validate(
        self, val_loader: DataLoader
    ) -> tuple[float, float, float]:
        """Run validation and compute metrics.

        Args:
            val_loader: Validation data loader.

        Returns:
            Tuple of (val_loss, classification_accuracy, embedding_quality).
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_embeddings = []
        all_labels = []
        num_batches = 0

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits, embeddings = self.model(images)

            # Classification loss
            cls_loss = self.classification_loss_fn(logits, labels)

            # Triplet loss
            anchors, positives, negatives = _mine_hard_triplets(embeddings, labels)
            if anchors.shape[0] > 0:
                metric_loss = self.triplet_loss_fn(anchors, positives, negatives)
            else:
                metric_loss = torch.tensor(0.0, device=self.device)

            combined_loss = (
                self.config.loss_weights["classification"] * cls_loss
                + self.config.loss_weights["metric"] * metric_loss
            )

            total_loss += combined_loss.item()
            num_batches += 1

            # Accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

            # Collect embeddings for quality metric
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)

        # Compute embedding quality
        if all_embeddings:
            all_emb = torch.cat(all_embeddings, dim=0)
            all_lbl = torch.cat(all_labels, dim=0)
            emb_quality = _compute_embedding_quality(all_emb, all_lbl)
        else:
            emb_quality = 0.0

        return avg_loss, accuracy, emb_quality

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint with LoRA weights, head weights, and config.

        Checkpoint format:
        - lora_weights.pt: LoRA adapter state dict (full model trainable params)
        - classification_head.pt: Classification head state dict
        - embedding_head.pt: Embedding head state dict
        - config.json: TrainingConfig as JSON (includes encoder_dim, class_names)

        Args:
            path: Directory to save the checkpoint to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter weights (all trainable parameters from the full model)
        lora_state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lora_state_dict[name] = param.cpu().detach()
        torch.save(lora_state_dict, path / "lora_weights.pt")

        # Save classification head
        torch.save(
            self.model.classification_head.state_dict(),
            path / "classification_head.pt",
        )

        # Save embedding head
        torch.save(
            self.model.embedding_head.state_dict(),
            path / "embedding_head.pt",
        )

        # Save config as JSON (include encoder_dim and class_names for inference)
        config_dict = asdict(self.config)
        # Add inference-relevant metadata
        config_dict["encoder_dim"] = self.model.classification_head.linear.in_features
        # Save class names in the TRAINING order (sorted alphabetically by dataloader)
        # This must match the class_to_idx mapping used during training
        config_dict["class_names"] = self._class_names
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Checkpoint saved to {path}")

    def get_vram_usage(self) -> float:
        """Return current VRAM usage in GB.

        Returns:
            Current VRAM usage in GB, or 0.0 if CUDA is not available.
        """
        return get_vram_usage_gb()


class _NaNLossError(Exception):
    """Internal exception for NaN loss detection."""

    pass
