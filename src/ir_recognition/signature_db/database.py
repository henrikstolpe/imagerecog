"""Signature Database for IR image storage and manifest management.

Provides file-based storage of IR images organized by vehicle class,
with a JSON manifest tracking all images and their metadata.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ir_recognition.models import SUPPORTED_VEHICLE_CLASSES, ImageMetadata


class SignatureDatabase:
    """File-based signature database for IR images.

    Storage layout:
        {db_path}/
        ├── manifest.json
        ├── images/
        │   ├── T-72/
        │   ├── T-80/
        │   ├── BMP-3/
        │   ├── CV90/
        │   └── civilian_car/
        └── embeddings/
            └── embeddings.npz

    Attributes:
        db_path: Root directory of the database.
    """

    MANIFEST_VERSION = "1.0"

    def __init__(self, db_path: Path | str) -> None:
        """Initialize the signature database.

        Args:
            db_path: Root directory for the database. Created if it doesn't exist.
        """
        self.db_path = Path(db_path)
        self._manifest_path = self.db_path / "manifest.json"
        self._images_dir = self.db_path / "images"
        self._embeddings_dir = self.db_path / "embeddings"

        # Create directory structure
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        for vehicle_class in SUPPORTED_VEHICLE_CLASSES:
            (self._images_dir / vehicle_class).mkdir(exist_ok=True)

        # Load or initialize manifest
        self._manifest = self._load_manifest()

    def add_image(self, image: np.ndarray, metadata: ImageMetadata) -> str:
        """Add an image to the database.

        Validates metadata, saves the image as an 8-bit grayscale PNG,
        and updates the manifest.

        Args:
            image: HxW float32 numpy array with values in [0, 1].
            metadata: Image metadata with required fields.

        Returns:
            The image_id of the stored image.

        Raises:
            ValueError: If metadata is missing required fields or has invalid values.
            ValueError: If image is not a valid 2D numpy array.
        """
        # Validate image
        self._validate_image(image)

        # Validate metadata (ImageMetadata __post_init__ handles field validation)
        self._validate_metadata(metadata)

        # Determine file path
        file_path = Path("images") / metadata.vehicle_class / f"{metadata.image_id}.png"
        full_path = self.db_path / file_path

        # Convert float32 [0, 1] to uint8 [0, 255] and save as PNG
        image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8, mode="L")
        pil_image.save(full_path)

        # Update metadata file_path to the relative path within the database
        metadata.file_path = file_path

        # Add to manifest
        image_entry = {
            "image_id": metadata.image_id,
            "vehicle_class": metadata.vehicle_class,
            "azimuth": metadata.azimuth,
            "elevation": metadata.elevation,
            "source_type": metadata.source_type,
            "file_path": str(file_path),
            "split": None,
        }
        self._manifest["images"].append(image_entry)

        # Save manifest
        self._save_manifest()

        return metadata.image_id

    def get_manifest(self) -> pd.DataFrame:
        """Return the full manifest as a pandas DataFrame.

        Returns:
            DataFrame with columns: image_id, vehicle_class, azimuth,
            elevation, source_type, file_path, split.
        """
        if not self._manifest["images"]:
            return pd.DataFrame(
                columns=[
                    "image_id",
                    "vehicle_class",
                    "azimuth",
                    "elevation",
                    "source_type",
                    "file_path",
                    "split",
                ]
            )
        return pd.DataFrame(self._manifest["images"])

    def generate_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int | None = None,
    ) -> None:
        """Assign images to train/val/test splits, stratified by vehicle class.

        Each vehicle class is split independently so that every class has
        proportional representation in each split. For N >= 20 images,
        the resulting proportions will be within ±5% of the target ratios.

        Every image is assigned to exactly one split after this method completes.

        Args:
            train_ratio: Proportion of images for the training split.
            val_ratio: Proportion of images for the validation split.
            test_ratio: Proportion of images for the test split.
            seed: Optional random seed for reproducibility.

        Raises:
            ValueError: If ratios don't sum to approximately 1.0 or are negative.
            ValueError: If the database contains no images.
        """
        # Validate ratios
        if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError("Split ratios must be non-negative")
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {ratio_sum:.6f}"
            )
        if not self._manifest["images"]:
            raise ValueError("Cannot generate splits on an empty database")

        # Group images by vehicle class for stratified splitting
        class_indices: dict[str, list[int]] = defaultdict(list)
        for idx, entry in enumerate(self._manifest["images"]):
            class_indices[entry["vehicle_class"]].append(idx)

        # Set up random number generator
        rng = random.Random(seed)

        # Assign splits per class (stratified)
        for vehicle_class, indices in class_indices.items():
            rng.shuffle(indices)
            n = len(indices)

            # Calculate split sizes ensuring every image is assigned
            n_train = round(n * train_ratio)
            n_val = round(n * val_ratio)
            n_test = n - n_train - n_val  # Remainder goes to test to ensure sum = n

            # Ensure no split gets negative count
            if n_test < 0:
                # Adjust: take from train
                n_train += n_test
                n_test = 0

            # Assign splits
            for i, idx in enumerate(indices):
                if i < n_train:
                    self._manifest["images"][idx]["split"] = "train"
                elif i < n_train + n_val:
                    self._manifest["images"][idx]["split"] = "val"
                else:
                    self._manifest["images"][idx]["split"] = "test"

        # Update split ratios in manifest
        self._manifest["splits"] = {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        }

        # Save manifest
        self._save_manifest()

    def get_split(self, split: str) -> list[ImageMetadata]:
        """Get all images assigned to a given split.

        Args:
            split: One of 'train', 'val', or 'test'.

        Returns:
            List of ImageMetadata for all images in the specified split.

        Raises:
            ValueError: If split is not one of 'train', 'val', 'test'.
        """
        valid_splits = ("train", "val", "test")
        if split not in valid_splits:
            raise ValueError(
                f"split must be one of {valid_splits}, got '{split}'"
            )

        results = []
        for entry in self._manifest["images"]:
            if entry["split"] == split:
                results.append(
                    ImageMetadata(
                        image_id=entry["image_id"],
                        vehicle_class=entry["vehicle_class"],
                        azimuth=entry["azimuth"],
                        elevation=entry["elevation"],
                        source_type=entry["source_type"],
                        file_path=Path(entry["file_path"]),
                    )
                )
        return results

    def get_stats(self) -> dict:
        """Return statistics about the database contents.

        Returns:
            Dictionary with:
                - 'per_class': dict mapping vehicle_class to image count
                - 'per_angle': dict mapping azimuth bucket (str) to image count
                - 'per_source_type': dict mapping source_type to image count
                - 'total': total number of images
        """
        per_class: dict[str, int] = defaultdict(int)
        per_angle: dict[str, int] = defaultdict(int)
        per_source_type: dict[str, int] = defaultdict(int)

        for entry in self._manifest["images"]:
            per_class[entry["vehicle_class"]] += 1
            # Bucket azimuth into 45-degree bins
            azimuth = entry["azimuth"]
            bucket = int(azimuth // 45) * 45
            angle_label = f"{bucket}-{bucket + 45}"
            per_angle[angle_label] += 1
            per_source_type[entry["source_type"]] += 1

        return {
            "per_class": dict(per_class),
            "per_angle": dict(per_angle),
            "per_source_type": dict(per_source_type),
            "total": len(self._manifest["images"]),
        }

    def _validate_image(self, image: np.ndarray) -> None:
        """Validate that the image is a proper 2D numpy array.

        Args:
            image: The image array to validate.

        Raises:
            ValueError: If image is not a valid 2D numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy ndarray")
        if image.ndim != 2:
            raise ValueError(
                f"Image must be 2D (grayscale), got {image.ndim}D array"
            )
        if image.size == 0:
            raise ValueError("Image must not be empty")

    def _validate_metadata(self, metadata: ImageMetadata) -> None:
        """Validate metadata has all required fields with valid values.

        The ImageMetadata dataclass already validates fields in __post_init__,
        but we perform additional checks here for database-level constraints.

        Args:
            metadata: The metadata to validate.

        Raises:
            ValueError: If metadata is invalid.
        """
        if not metadata.image_id:
            raise ValueError("image_id must be a non-empty string")
        if not metadata.vehicle_class:
            raise ValueError("Missing required field: vehicle_class")
        if metadata.vehicle_class not in SUPPORTED_VEHICLE_CLASSES:
            raise ValueError(
                f"Unknown vehicle class: '{metadata.vehicle_class}'. "
                f"Supported: {SUPPORTED_VEHICLE_CLASSES}"
            )
        if metadata.source_type not in ("synthetic", "real"):
            raise ValueError(
                f"source_type must be 'synthetic' or 'real', got '{metadata.source_type}'"
            )

        # Check for duplicate image_id
        existing_ids = {img["image_id"] for img in self._manifest["images"]}
        if metadata.image_id in existing_ids:
            raise ValueError(
                f"Image with id '{metadata.image_id}' already exists in database"
            )

    def _load_manifest(self) -> dict[str, Any]:
        """Load manifest from disk or create a new one.

        Returns:
            The manifest dictionary.
        """
        if self._manifest_path.exists():
            with open(self._manifest_path, "r") as f:
                return json.load(f)
        return self._create_empty_manifest()

    def _save_manifest(self) -> None:
        """Save the current manifest to disk as JSON."""
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def _create_empty_manifest(self) -> dict[str, Any]:
        """Create a new empty manifest structure.

        Returns:
            Empty manifest dictionary with proper schema.
        """
        return {
            "version": self.MANIFEST_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vehicle_classes": list(SUPPORTED_VEHICLE_CLASSES),
            "images": [],
            "splits": {
                "train": 0.70,
                "val": 0.15,
                "test": 0.15,
            },
        }
