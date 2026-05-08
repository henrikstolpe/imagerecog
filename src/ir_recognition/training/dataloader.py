"""Data loading and batch construction for IR Signature Recognition training.

Provides:
- IRSignatureDataset: PyTorch Dataset that loads images from SignatureDatabase
- TripletBatchSampler: Ensures each batch has at least 2 images per class
  for valid online triplet mining
- GradientAccumulationWrapper: Helper for effective larger batch sizes

Preprocessing pipeline:
    Load PNG → resize to 224×224 → normalize to [0,1] → replicate grayscale to 3 channels
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from ir_recognition.models import ImageMetadata
from ir_recognition.signature_db.database import SignatureDatabase

# ImageNet normalization constants (used for SigLIP/ViT models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Target image size for model input
TARGET_SIZE = (224, 224)


class IRSignatureDataset(Dataset):
    """PyTorch Dataset that loads IR images from a SignatureDatabase.

    Each item is a tuple of (image_tensor, class_label_index) where:
    - image_tensor: float32 tensor of shape (3, 224, 224), normalized
    - class_label_index: integer index of the vehicle class

    Preprocessing pipeline:
    1. Load PNG from disk using PIL
    2. Resize to 224×224 using bilinear interpolation
    3. Convert to float tensor in [0, 1]
    4. Replicate grayscale to 3 channels
    5. Apply ImageNet-style normalization

    Args:
        database: SignatureDatabase instance to load images from.
        split: Which split to use ('train', 'val', or 'test').
        normalize: Whether to apply ImageNet normalization. Default True.
        target_size: Target image dimensions (H, W). Default (224, 224).
    """

    def __init__(
        self,
        database: SignatureDatabase,
        split: str,
        normalize: bool = True,
        target_size: tuple[int, int] = TARGET_SIZE,
    ) -> None:
        self.database = database
        self.split = split
        self.normalize = normalize
        self.target_size = target_size

        # Load metadata for the requested split
        self._metadata: list[ImageMetadata] = database.get_split(split)

        # Build class-to-index mapping (sorted for determinism)
        classes = sorted(set(m.vehicle_class for m in self._metadata))
        self._class_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(classes)
        }
        self._idx_to_class: dict[int, str] = {
            idx: cls for cls, idx in self._class_to_idx.items()
        }

        # Build index-to-class mapping for the sampler
        self._labels: list[int] = [
            self._class_to_idx[m.vehicle_class] for m in self._metadata
        ]

    @property
    def num_classes(self) -> int:
        """Number of distinct classes in this dataset split."""
        return len(self._class_to_idx)

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Mapping from class name to integer index."""
        return self._class_to_idx

    @property
    def labels(self) -> list[int]:
        """List of integer class labels, one per sample."""
        return self._labels

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load and preprocess a single image.

        Args:
            index: Index into the dataset.

        Returns:
            Tuple of (image_tensor, class_label_index).
        """
        metadata = self._metadata[index]
        label = self._labels[index]

        # Load image from disk
        image_path = self.database.db_path / metadata.file_path
        image = self._load_and_preprocess(image_path)

        return image, label

    def _load_and_preprocess(self, image_path: Path) -> torch.Tensor:
        """Load a PNG image and apply the full preprocessing pipeline.

        Args:
            image_path: Path to the PNG image file.

        Returns:
            Float32 tensor of shape (3, 224, 224).
        """
        # Step 1: Load image using PIL
        pil_image = Image.open(image_path).convert("L")  # Ensure grayscale

        # Step 2: Resize to target size using bilinear interpolation
        pil_image = pil_image.resize(
            (self.target_size[1], self.target_size[0]),  # PIL uses (width, height)
            Image.BILINEAR,
        )

        # Step 3: Convert to float tensor in [0, 1]
        np_image = np.array(pil_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np_image).unsqueeze(0)  # Shape: (1, H, W)

        # Step 4: Replicate grayscale to 3 channels
        tensor = tensor.repeat(3, 1, 1)  # Shape: (3, H, W)

        # Step 5: Apply ImageNet-style normalization
        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor


class TripletBatchSampler(Sampler[list[int]]):
    """Batch sampler that ensures valid triplet mining within each batch.

    For online triplet mining to work, each batch must contain at least
    2 images from at least 2 different classes. This sampler groups images
    by class and constructs batches that satisfy this constraint.

    Strategy:
    - Select classes for each batch (at least 2)
    - For each selected class, include at least 2 images
    - Fill remaining batch slots with additional images from selected classes

    Args:
        labels: List of integer class labels for all samples.
        batch_size: Number of samples per batch.
        classes_per_batch: Number of distinct classes per batch.
            Default None means use min(num_classes, batch_size // 2).
        samples_per_class: Minimum number of samples per class in each batch.
            Default 2 (minimum for triplet mining).
        drop_last: Whether to drop the last incomplete batch. Default True.
        seed: Random seed for reproducibility. Default None.
    """

    def __init__(
        self,
        labels: list[int],
        batch_size: int,
        classes_per_batch: int | None = None,
        samples_per_class: int = 2,
        drop_last: bool = True,
        seed: int | None = None,
    ) -> None:
        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last

        # Group sample indices by class
        self._class_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self._class_indices[label].append(idx)

        # Filter classes that have enough samples
        self._valid_classes = [
            cls for cls, indices in self._class_indices.items()
            if len(indices) >= samples_per_class
        ]

        if len(self._valid_classes) < 2:
            raise ValueError(
                f"Need at least 2 classes with >= {samples_per_class} samples each "
                f"for triplet mining. Found {len(self._valid_classes)} valid classes."
            )

        # Determine classes per batch
        if classes_per_batch is None:
            # Use as many classes as fit given batch_size and samples_per_class
            max_classes = batch_size // samples_per_class
            self.classes_per_batch = min(len(self._valid_classes), max(2, max_classes))
        else:
            self.classes_per_batch = min(classes_per_batch, len(self._valid_classes))

        # Ensure constraint: classes_per_batch * samples_per_class <= batch_size
        if self.classes_per_batch * self.samples_per_class > self.batch_size:
            self.classes_per_batch = self.batch_size // self.samples_per_class

        # Ensure at least 2 classes
        if self.classes_per_batch < 2:
            raise ValueError(
                f"batch_size ({batch_size}) is too small for "
                f"samples_per_class ({samples_per_class}) with at least 2 classes. "
                f"Need batch_size >= {2 * samples_per_class}."
            )

        self._rng = random.Random(seed)
        self._num_batches = self._compute_num_batches()

    def _compute_num_batches(self) -> int:
        """Compute the number of batches per epoch."""
        total_samples = len(self.labels)
        if self.drop_last:
            return total_samples // self.batch_size
        return (total_samples + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Generate batches with triplet-mining-friendly composition.

        Each batch contains at least `classes_per_batch` classes with
        at least `samples_per_class` samples each.
        """
        # Create shuffled copies of indices per class
        class_pools: dict[int, list[int]] = {}
        for cls in self._valid_classes:
            pool = self._class_indices[cls].copy()
            self._rng.shuffle(pool)
            class_pools[cls] = pool

        # Track position in each class pool
        class_positions: dict[int, int] = {cls: 0 for cls in self._valid_classes}

        for _ in range(self._num_batches):
            batch: list[int] = []

            # Select classes for this batch
            selected_classes = self._rng.sample(
                self._valid_classes, self.classes_per_batch
            )

            # For each selected class, pick samples_per_class samples
            for cls in selected_classes:
                pool = class_pools[cls]
                pos = class_positions[cls]

                for _ in range(self.samples_per_class):
                    if pos >= len(pool):
                        # Reshuffle and reset
                        self._rng.shuffle(pool)
                        pos = 0
                    batch.append(pool[pos])
                    pos += 1

                class_positions[cls] = pos

            # Fill remaining slots with additional samples from selected classes
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                # Round-robin from selected classes
                fill_idx = 0
                while len(batch) < self.batch_size:
                    cls = selected_classes[fill_idx % len(selected_classes)]
                    pool = class_pools[cls]
                    pos = class_positions[cls]

                    if pos >= len(pool):
                        self._rng.shuffle(pool)
                        pos = 0

                    batch.append(pool[pos])
                    class_positions[cls] = pos + 1
                    fill_idx += 1

            yield batch


class GradientAccumulationConfig:
    """Configuration for gradient accumulation to achieve effective larger batch sizes.

    With gradient accumulation, the effective batch size is:
        effective_batch_size = batch_size * accumulation_steps

    For example, with batch_size=4 and accumulation_steps=8,
    the effective batch size is 32.

    Args:
        batch_size: Actual batch size per forward pass (constrained by VRAM).
        accumulation_steps: Number of forward passes before a weight update.
    """

    def __init__(self, batch_size: int, accumulation_steps: int = 1) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if accumulation_steps <= 0:
            raise ValueError(
                f"accumulation_steps must be > 0, got {accumulation_steps}"
            )
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps

    @property
    def effective_batch_size(self) -> int:
        """The effective batch size after accumulation."""
        return self.batch_size * self.accumulation_steps

    @classmethod
    def from_target_batch_size(
        cls, target_batch_size: int, max_batch_size: int
    ) -> "GradientAccumulationConfig":
        """Create config to achieve a target effective batch size.

        Args:
            target_batch_size: Desired effective batch size.
            max_batch_size: Maximum batch size that fits in VRAM.

        Returns:
            GradientAccumulationConfig with appropriate accumulation_steps.
        """
        if target_batch_size <= 0:
            raise ValueError(
                f"target_batch_size must be > 0, got {target_batch_size}"
            )
        if max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be > 0, got {max_batch_size}"
            )

        actual_batch_size = min(target_batch_size, max_batch_size)
        accumulation_steps = max(1, target_batch_size // actual_batch_size)

        return cls(
            batch_size=actual_batch_size,
            accumulation_steps=accumulation_steps,
        )

    def should_step(self, batch_idx: int) -> bool:
        """Check if optimizer should step after this batch.

        Args:
            batch_idx: Zero-based index of the current batch within the epoch.

        Returns:
            True if the optimizer should step (i.e., accumulation is complete).
        """
        return (batch_idx + 1) % self.accumulation_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation.

        When accumulating gradients, the loss should be divided by
        accumulation_steps so that the total gradient magnitude is
        equivalent to a single large batch.

        Args:
            loss: The raw loss tensor.

        Returns:
            Scaled loss tensor.
        """
        if self.accumulation_steps == 1:
            return loss
        return loss / self.accumulation_steps
