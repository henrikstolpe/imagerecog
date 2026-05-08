"""Inference pipeline for IR Signature Recognition.

Provides the InferencePipeline class that handles classification and
similarity-based recognition of IR images. Supports both real GPU-based
models and injectable mock models for testing.

Pipeline flow:
    1. Preprocess input image (resize, normalize)
    2. Forward pass through vision encoder + LoRA weights
    3. Classification head → softmax → top-3 predictions
    4. Embedding head → L2-normalized vector
    5. Cosine similarity against pre-computed database embeddings
    6. Return combined results with confidence/threshold flags
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, Union

import numpy as np
import torch
import torch.nn.functional as F

from ir_recognition.inference.preprocessing import preprocess_image
from ir_recognition.models import (
    SUPPORTED_VEHICLE_CLASSES,
    ClassificationResult,
    RecognitionResult,
    SimilarityMatch,
)

logger = logging.getLogger(__name__)

# Threshold constants
LOW_CONFIDENCE_THRESHOLD = 0.5
NO_MATCH_THRESHOLD = 0.3


class ModelBackend(Protocol):
    """Protocol for model backends, enabling mock injection for testing.

    Any object implementing forward() with the correct signature can be
    used as a model backend.
    """

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Args:
            pixel_values: Input tensor of shape (1, 3, 224, 224).

        Returns:
            Tuple of (logits, embeddings) where:
                - logits: shape (1, num_classes) raw classification logits
                - embeddings: shape (1, embedding_dim) L2-normalized embeddings
        """
        ...


class InferencePipeline:
    """Unified inference pipeline for classification and similarity recognition.

    Loads a trained model checkpoint (LoRA weights + classification/embedding heads)
    and pre-computed database embeddings to perform:
    - Classification: top-3 vehicle class predictions with confidence scores
    - Similarity search: top-5 most similar images from the database
    - Combined recognition: both classification and similarity in one call

    The pipeline can be initialized with either:
    - A model_path pointing to a checkpoint directory (for production use)
    - A model_backend object implementing the ModelBackend protocol (for testing)

    Args:
        model_path: Path to the checkpoint directory containing model weights.
        database: SignatureDatabase instance for loading pre-computed embeddings.
        model_backend: Optional injectable model backend for testing.
        device: Device to run inference on. Default auto-detects.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        database=None,
        model_backend: ModelBackend | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the inference pipeline.

        Either model_path or model_backend must be provided.

        Args:
            model_path: Path to checkpoint directory with model weights and config.
            database: SignatureDatabase instance for embedding lookup.
            model_backend: Optional model backend (overrides model_path loading).
            device: Torch device string. Auto-detected if None.
        """
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._database = database
        self._class_names: list[str] = list(SUPPORTED_VEHICLE_CLASSES)

        # Load or set model backend
        if model_backend is not None:
            self._model = model_backend
        elif model_path is not None:
            self._model = self._load_model_from_checkpoint(model_path)
        else:
            raise ValueError("Either model_path or model_backend must be provided")

        # Load pre-computed embeddings from database
        self._db_embeddings: np.ndarray | None = None
        self._db_image_ids: list[str] = []
        self._db_labels: list[str] = []
        self._load_database_embeddings(database)

    def recognize(self, image: Union[Path, np.ndarray]) -> RecognitionResult:
        """Run full recognition pipeline on a single image.

        Performs both classification and similarity search, returning
        combined results with threshold flags.

        Args:
            image: Either a file path to an image or a numpy array.

        Returns:
            RecognitionResult with top-3 classifications, top-5 similarity
            matches, and threshold flags.

        Raises:
            ValueError: If the image cannot be preprocessed.
        """
        classifications = self.classify(image)
        similarity_matches = self.find_similar(image)

        # Compute threshold flags
        low_confidence = classifications[0].confidence < LOW_CONFIDENCE_THRESHOLD if classifications else True
        no_match = all(
            m.similarity_score < NO_MATCH_THRESHOLD for m in similarity_matches
        ) if similarity_matches else True

        return RecognitionResult(
            classifications=classifications,
            similarity_matches=similarity_matches,
            low_confidence=low_confidence,
            no_match=no_match,
        )

    def classify(self, image: Union[Path, np.ndarray]) -> list[ClassificationResult]:
        """Classify an image, returning top-3 predictions sorted by confidence.

        Args:
            image: Either a file path to an image or a numpy array.

        Returns:
            List of top-3 ClassificationResult sorted by confidence descending.

        Raises:
            ValueError: If the image cannot be preprocessed.
        """
        tensor = self._preprocess(image)
        logits, _ = self._forward(tensor)

        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=-1).squeeze(0)  # Shape: (num_classes,)

        # Get top-3 predictions
        top_k = min(3, len(self._class_names))
        top_values, top_indices = torch.topk(probabilities, top_k)

        results = []
        for confidence, idx in zip(top_values.tolist(), top_indices.tolist()):
            results.append(
                ClassificationResult(
                    vehicle_class=self._class_names[idx],
                    confidence=float(confidence),
                )
            )

        return results

    def find_similar(self, image: Union[Path, np.ndarray]) -> list[SimilarityMatch]:
        """Find the most similar images in the database.

        Computes the embedding for the input image and performs cosine
        similarity against all pre-computed database embeddings.

        Args:
            image: Either a file path to an image or a numpy array.

        Returns:
            List of top-5 SimilarityMatch sorted by similarity_score descending.
            Returns empty list if no database embeddings are available.

        Raises:
            ValueError: If the image cannot be preprocessed.
        """
        if self._db_embeddings is None or len(self._db_image_ids) == 0:
            logger.warning("No database embeddings available for similarity search")
            return []

        tensor = self._preprocess(image)
        _, embedding = self._forward(tensor)

        # Convert embedding to numpy for similarity computation
        query_embedding = embedding.squeeze(0).detach().cpu().numpy()  # Shape: (embedding_dim,)

        # Compute cosine similarity against all database embeddings
        # Since embeddings are L2-normalized, cosine similarity = dot product
        similarities = self._compute_cosine_similarities(query_embedding, self._db_embeddings)

        # Get top-5 (or fewer if database is smaller)
        top_k = min(5, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                SimilarityMatch(
                    vehicle_class=self._db_labels[idx],
                    similarity_score=float(similarities[idx]),
                    image_id=self._db_image_ids[idx],
                )
            )

        return results

    def _preprocess(self, image: Union[Path, np.ndarray]) -> torch.Tensor:
        """Preprocess an image for model input.

        Args:
            image: Either a file path or numpy array.

        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224).

        Raises:
            ValueError: If preprocessing fails.
        """
        tensor, error = preprocess_image(image)
        if tensor is None:
            raise ValueError(f"Image preprocessing failed: {error}")
        return tensor.to(self._device)

    def _forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Args:
            tensor: Preprocessed input tensor of shape (1, 3, 224, 224).

        Returns:
            Tuple of (logits, embeddings).
        """
        with torch.no_grad():
            logits, embeddings = self._model.forward(tensor)
        return logits, embeddings

    def _load_model_from_checkpoint(self, model_path: Path) -> ModelBackend:
        """Load model from a checkpoint directory.

        Expected checkpoint structure:
            model_path/
            ├── config.json
            ├── lora_weights.pt
            ├── classification_head.pt
            └── embedding_head.pt

        Args:
            model_path: Path to the checkpoint directory.

        Returns:
            A loaded model backend ready for inference.

        Raises:
            FileNotFoundError: If checkpoint files are missing.
            RuntimeError: If model loading fails.
        """
        model_path = Path(model_path)

        # Verify checkpoint files exist
        config_path = model_path / "config.json"
        lora_path = model_path / "lora_weights.pt"
        cls_head_path = model_path / "classification_head.pt"
        emb_head_path = model_path / "embedding_head.pt"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not cls_head_path.exists():
            raise FileNotFoundError(
                f"Classification head not found: {cls_head_path}"
            )
        if not emb_head_path.exists():
            raise FileNotFoundError(f"Embedding head not found: {emb_head_path}")

        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        encoder_dim = config.get("encoder_dim", 1152)
        num_classes = config.get("num_classes", 5)
        embedding_dim = config.get("embedding_dim", 256)
        self._class_names = config.get("class_names", list(SUPPORTED_VEHICLE_CLASSES))

        # Import model components
        from ir_recognition.training.model import (
            ClassificationHead,
            EmbeddingHead,
            IRRecognitionModel,
        )

        # Create heads and load weights
        classification_head = ClassificationHead(encoder_dim, num_classes)
        classification_head.load_state_dict(
            torch.load(cls_head_path, map_location=self._device, weights_only=True)
        )

        embedding_head = EmbeddingHead(encoder_dim, embedding_dim)
        embedding_head.load_state_dict(
            torch.load(emb_head_path, map_location=self._device, weights_only=True)
        )

        # Load vision encoder with LoRA weights if available
        if lora_path.exists():
            vision_encoder = self._load_vision_encoder_with_lora(
                config, lora_path
            )
        else:
            logger.warning(
                "LoRA weights not found, using base vision encoder"
            )
            vision_encoder = self._load_base_vision_encoder(config)

        # Create combined model
        # Cast heads to float16 to match vision encoder output dtype
        # and move to the correct device
        device = self._device
        classification_head = classification_head.half().to(device)
        embedding_head = embedding_head.half().to(device)

        model = IRRecognitionModel(
            vision_encoder=vision_encoder,
            classification_head=classification_head,
            embedding_head=embedding_head,
            pool_strategy=config.get("pool_strategy", "mean"),
        )
        model.eval()

        return model

    def _load_vision_encoder_with_lora(self, config: dict, lora_path: Path):
        """Load the vision encoder with LoRA weights for inference.

        Strategy: Load the base model, apply LoRA to the encoder sub-module
        only (avoiding the SiglipVisionModel forward-pass conflict), load
        trained weights, and return a wrapper that manually handles the
        embedding step + encoder forward.

        Args:
            config: Model configuration dictionary.
            lora_path: Path to LoRA weights file (lora_weights.pt).

        Returns:
            A module that accepts pixel_values and returns encoder output.
        """
        from transformers import PaliGemmaForConditionalGeneration
        from peft import get_peft_model, LoraConfig, TaskType

        from ir_recognition.training.model import get_quantization_config

        base_model_name = config.get("base_model", "google/paligemma2-3b-pt-224")
        quant_bits = config.get("quantization_bits", 4)
        lora_rank = config.get("lora_rank", 16)
        lora_alpha = config.get("lora_alpha", 32)

        quant_config = get_quantization_config(quant_bits)
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Extract vision tower components
        vision_tower = base_model.model.vision_tower
        embeddings = vision_tower.embeddings
        encoder = vision_tower.encoder
        post_layernorm = vision_tower.post_layernorm

        # Apply LoRA to the encoder sub-module only
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        encoder_with_lora = get_peft_model(encoder, lora_config)

        # Load trained LoRA weights
        saved_state = torch.load(
            lora_path, map_location=self._device, weights_only=True
        )

        # Remap keys from training format to PEFT encoder format
        remapped_state = {}
        prefix_to_strip = "vision_encoder.encoder."
        prefix_to_add = "base_model.model."
        for key, value in saved_state.items():
            if key.startswith(prefix_to_strip) and "lora_" in key:
                new_key = prefix_to_add + key[len(prefix_to_strip):]
                remapped_state[new_key] = value

        if remapped_state:
            encoder_with_lora.load_state_dict(remapped_state, strict=False)
            logger.info(f"Loaded {len(remapped_state)} LoRA weight tensors")
        else:
            logger.warning("No LoRA weights found in checkpoint")

        encoder_with_lora.eval()

        # Create a wrapper module that does embedding + encoder + layernorm
        class _VisionEncoderWrapper(torch.nn.Module):
            def __init__(self, embeddings, encoder, post_layernorm):
                super().__init__()
                self.embeddings = embeddings
                self.encoder = encoder
                self.post_layernorm = post_layernorm

            def forward(self, pixel_values=None, **kwargs):
                hidden_states = self.embeddings(pixel_values)
                encoder_output = self.encoder(inputs_embeds=hidden_states)
                # encoder_output is a BaseModelOutput
                last_hidden = encoder_output.last_hidden_state if hasattr(encoder_output, 'last_hidden_state') else encoder_output[0]
                last_hidden = self.post_layernorm(last_hidden)
                # Return in same format as SiglipVisionModel
                from transformers.modeling_outputs import BaseModelOutputWithPooling
                return BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden,
                    pooler_output=None,
                )

        wrapper = _VisionEncoderWrapper(embeddings, encoder_with_lora, post_layernorm)
        wrapper.eval()
        return wrapper

    def _load_base_vision_encoder(self, config: dict):
        """Load the base vision encoder without LoRA (fallback).

        Args:
            config: Model configuration dictionary.

        Returns:
            Base vision encoder module.
        """
        from transformers import PaliGemmaForConditionalGeneration
        from ir_recognition.training.model import get_quantization_config

        base_model_name = config.get("base_model", "google/paligemma2-3b-pt-224")
        quant_bits = config.get("quantization_bits", 4)

        quant_config = get_quantization_config(quant_bits)
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        if hasattr(base_model, "model") and hasattr(base_model.model, "vision_tower"):
            return base_model.model.vision_tower
        elif hasattr(base_model, "vision_tower"):
            return base_model.vision_tower
        elif hasattr(base_model, "vision_model"):
            return base_model.vision_model
        else:
            raise RuntimeError("Could not find vision encoder in model")

    def _load_database_embeddings(self, database) -> None:
        """Load pre-computed embeddings from the database's embeddings directory.

        Expected file: {db_path}/embeddings/embeddings.npz
        Keys: "embeddings" (N x embedding_dim), "image_ids" (N,), "labels" (N,)

        Handles the case where embeddings.npz doesn't exist (empty database)
        gracefully by leaving embeddings as None.

        Args:
            database: SignatureDatabase instance, or None.
        """
        if database is None:
            logger.info("No database provided, similarity search will be unavailable")
            return

        embeddings_path = Path(database.db_path) / "embeddings" / "embeddings.npz"

        if not embeddings_path.exists():
            logger.info(
                f"No pre-computed embeddings found at {embeddings_path}. "
                "Similarity search will return empty results."
            )
            return

        try:
            data = np.load(embeddings_path, allow_pickle=True)
            self._db_embeddings = data["embeddings"].astype(np.float32)
            self._db_image_ids = data["image_ids"].tolist()
            self._db_labels = data["labels"].tolist()

            logger.info(
                f"Loaded {len(self._db_image_ids)} database embeddings "
                f"from {embeddings_path}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load database embeddings from {embeddings_path}: {e}. "
                "Similarity search will return empty results."
            )
            self._db_embeddings = None
            self._db_image_ids = []
            self._db_labels = []

    @staticmethod
    def _compute_cosine_similarities(
        query: np.ndarray, database: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between a query vector and all database vectors.

        For L2-normalized vectors, cosine similarity equals the dot product.
        This method handles both normalized and unnormalized vectors.

        Args:
            query: Query embedding vector of shape (embedding_dim,).
            database: Database embeddings of shape (N, embedding_dim).

        Returns:
            Array of similarity scores of shape (N,), each in [-1, 1].
        """
        # Normalize query vector
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return np.zeros(len(database), dtype=np.float32)
        query_normalized = query / query_norm

        # Normalize database vectors
        db_norms = np.linalg.norm(database, axis=1, keepdims=True)
        # Avoid division by zero
        db_norms = np.maximum(db_norms, 1e-10)
        db_normalized = database / db_norms

        # Cosine similarity = dot product of normalized vectors
        similarities = db_normalized @ query_normalized

        return similarities.astype(np.float32)
