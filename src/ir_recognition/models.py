"""Core data models and type definitions for the IR Signature Recognition system.

Defines dataclasses for configuration, metadata, and results used throughout
the pipeline: IR generation, database management, training, and inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np

# Type alias for IR images: HxW float32 array with values in [0, 1]
IRImage = np.ndarray

# Supported vehicle classes
SUPPORTED_VEHICLE_CLASSES = (
    "T-72", "T-80", "BMP-3", "CV90", "civilian_car",
    "Challenger-2", "Leopard-2",
)

# Valid source types for image metadata
VALID_SOURCE_TYPES = ("synthetic", "real")


@dataclass
class IRGeneratorConfig:
    """Configuration for generating a single synthetic IR image.

    Attributes:
        vehicle_type: Vehicle class to generate (must be in SUPPORTED_VEHICLE_CLASSES).
        azimuth: Viewing azimuth angle in degrees, range [0, 360).
        elevation: Viewing elevation angle in degrees, range [0, 90].
        ambient_temp: Ambient temperature in Kelvin (must be > 0). Default 293.0 (20°C).
        image_size: Output image dimensions (height, width). Default (224, 224).
    """

    vehicle_type: str
    azimuth: float
    elevation: float
    ambient_temp: float = 293.0
    image_size: tuple[int, int] = (224, 224)

    def __post_init__(self) -> None:
        if self.vehicle_type not in SUPPORTED_VEHICLE_CLASSES:
            raise ValueError(
                f"Unknown vehicle class: '{self.vehicle_type}'. "
                f"Supported: {SUPPORTED_VEHICLE_CLASSES}"
            )
        if not (0.0 <= self.azimuth < 360.0):
            raise ValueError(
                f"Azimuth must be in [0, 360), got {self.azimuth}"
            )
        if not (0.0 <= self.elevation <= 90.0):
            raise ValueError(
                f"Elevation must be in [0, 90], got {self.elevation}"
            )
        if self.ambient_temp <= 0:
            raise ValueError(
                f"Ambient temperature must be > 0 Kelvin, got {self.ambient_temp}"
            )
        if (
            not isinstance(self.image_size, tuple)
            or len(self.image_size) != 2
            or self.image_size[0] <= 0
            or self.image_size[1] <= 0
        ):
            raise ValueError(
                f"image_size must be a tuple of two positive integers, got {self.image_size}"
            )


@dataclass
class ImageMetadata:
    """Metadata for an image stored in the signature database.

    Attributes:
        image_id: Unique identifier for the image.
        vehicle_class: Vehicle class label (must be in SUPPORTED_VEHICLE_CLASSES).
        azimuth: Viewing azimuth angle in degrees, range [0, 360).
        elevation: Viewing elevation angle in degrees, range [0, 90].
        source_type: Origin of the image, either "synthetic" or "real".
        file_path: Path to the stored image file.
    """

    image_id: str
    vehicle_class: str
    azimuth: float
    elevation: float
    source_type: str
    file_path: Path

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("image_id must be a non-empty string")
        if self.vehicle_class not in SUPPORTED_VEHICLE_CLASSES:
            raise ValueError(
                f"Unknown vehicle class: '{self.vehicle_class}'. "
                f"Supported: {SUPPORTED_VEHICLE_CLASSES}"
            )
        if not (0.0 <= self.azimuth < 360.0):
            raise ValueError(
                f"Azimuth must be in [0, 360), got {self.azimuth}"
            )
        if not (0.0 <= self.elevation <= 90.0):
            raise ValueError(
                f"Elevation must be in [0, 90], got {self.elevation}"
            )
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {VALID_SOURCE_TYPES}, got '{self.source_type}'"
            )
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)


@dataclass
class ClassificationResult:
    """A single classification prediction.

    Attributes:
        vehicle_class: Predicted vehicle class.
        confidence: Confidence score in [0, 1].
    """

    vehicle_class: str
    confidence: float

    def __post_init__(self) -> None:
        if self.vehicle_class not in SUPPORTED_VEHICLE_CLASSES:
            raise ValueError(
                f"Unknown vehicle class: '{self.vehicle_class}'. "
                f"Supported: {SUPPORTED_VEHICLE_CLASSES}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be in [0, 1], got {self.confidence}"
            )


@dataclass
class SimilarityMatch:
    """A single similarity match result.

    Attributes:
        vehicle_class: Vehicle class of the matched image.
        similarity_score: Cosine similarity score in [-1, 1].
        image_id: ID of the matched image in the database.
    """

    vehicle_class: str
    similarity_score: float
    image_id: str

    def __post_init__(self) -> None:
        if self.vehicle_class not in SUPPORTED_VEHICLE_CLASSES:
            raise ValueError(
                f"Unknown vehicle class: '{self.vehicle_class}'. "
                f"Supported: {SUPPORTED_VEHICLE_CLASSES}"
            )
        if not (-1.0 <= self.similarity_score <= 1.0):
            raise ValueError(
                f"Similarity score must be in [-1, 1], got {self.similarity_score}"
            )
        if not self.image_id:
            raise ValueError("image_id must be a non-empty string")


@dataclass
class RecognitionResult:
    """Combined recognition result from the inference pipeline.

    Attributes:
        classifications: Top-3 classification predictions sorted by confidence descending.
        similarity_matches: Top-5 similarity matches sorted by score descending.
        low_confidence: True if top-1 classification confidence < 0.5.
        no_match: True if all similarity scores < 0.3.
    """

    classifications: list[ClassificationResult]
    similarity_matches: list[SimilarityMatch]
    low_confidence: bool
    no_match: bool

    def __post_init__(self) -> None:
        if not isinstance(self.classifications, list):
            raise ValueError("classifications must be a list")
        if not isinstance(self.similarity_matches, list):
            raise ValueError("similarity_matches must be a list")

        # Validate low_confidence flag
        if self.classifications:
            top_confidence = self.classifications[0].confidence
            expected_low_confidence = top_confidence < 0.5
            if self.low_confidence != expected_low_confidence:
                raise ValueError(
                    f"low_confidence should be {expected_low_confidence} "
                    f"(top-1 confidence is {top_confidence}), got {self.low_confidence}"
                )

        # Validate no_match flag
        if self.similarity_matches:
            all_below_threshold = all(
                m.similarity_score < 0.3 for m in self.similarity_matches
            )
            if self.no_match != all_below_threshold:
                raise ValueError(
                    f"no_match should be {all_below_threshold} based on similarity scores, "
                    f"got {self.no_match}"
                )


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Attributes:
        base_model: Hugging Face model identifier for the base model.
        lora_rank: Rank of LoRA adapters.
        lora_alpha: Alpha scaling factor for LoRA.
        quantization_bits: Number of bits for quantization (4 for QLoRA).
        embedding_dim: Dimensionality of the embedding vector.
        num_classes: Number of vehicle classes to classify.
        batch_size: Training batch size (constrained by VRAM).
        learning_rate: Initial learning rate.
        num_epochs: Number of training epochs.
        triplet_margin: Margin for triplet loss.
        loss_weights: Dictionary with 'classification' and 'metric' loss weights.
        max_vram_gb: Maximum allowed VRAM usage in GB.
    """

    base_model: str = "google/paligemma2-3b-pt-224"
    lora_rank: int = 16
    lora_alpha: int = 32
    quantization_bits: int = 4
    embedding_dim: int = 256
    num_classes: int = 5
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 20
    triplet_margin: float = 0.3
    loss_weights: dict = field(
        default_factory=lambda: {"classification": 0.5, "metric": 0.5}
    )
    max_vram_gb: float = 8.0

    def __post_init__(self) -> None:
        if not self.base_model:
            raise ValueError("base_model must be a non-empty string")
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be > 0, got {self.lora_rank}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be > 0, got {self.lora_alpha}")
        if self.quantization_bits not in (4, 8):
            raise ValueError(
                f"quantization_bits must be 4 or 8, got {self.quantization_bits}"
            )
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be > 0, got {self.embedding_dim}"
            )
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {self.num_classes}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be > 0, got {self.num_epochs}")
        if self.triplet_margin <= 0:
            raise ValueError(
                f"triplet_margin must be > 0, got {self.triplet_margin}"
            )
        if not isinstance(self.loss_weights, dict):
            raise ValueError("loss_weights must be a dictionary")
        if "classification" not in self.loss_weights or "metric" not in self.loss_weights:
            raise ValueError(
                "loss_weights must contain 'classification' and 'metric' keys"
            )
        if self.max_vram_gb <= 0:
            raise ValueError(
                f"max_vram_gb must be > 0, got {self.max_vram_gb}"
            )


@dataclass
class EpochMetrics:
    """Metrics recorded after each training epoch.

    Attributes:
        epoch: Epoch number (0-indexed).
        train_loss: Average training loss for the epoch.
        val_loss: Average validation loss for the epoch.
        classification_accuracy: Classification accuracy on validation set.
        embedding_quality: Mean intra-class similarity minus mean inter-class similarity.
        vram_peak_gb: Peak VRAM usage during the epoch in GB.
        learning_rate: Learning rate used during the epoch.
    """

    epoch: int
    train_loss: float
    val_loss: float
    classification_accuracy: float
    embedding_quality: float
    vram_peak_gb: float
    learning_rate: float

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {self.epoch}")
        if self.train_loss < 0:
            raise ValueError(
                f"train_loss must be >= 0, got {self.train_loss}"
            )
        if self.val_loss < 0:
            raise ValueError(f"val_loss must be >= 0, got {self.val_loss}")
        if not (0.0 <= self.classification_accuracy <= 1.0):
            raise ValueError(
                f"classification_accuracy must be in [0, 1], got {self.classification_accuracy}"
            )
        if self.vram_peak_gb < 0:
            raise ValueError(
                f"vram_peak_gb must be >= 0, got {self.vram_peak_gb}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )
