"""Unit tests for the training dataloader module.

Tests:
- IRSignatureDataset: loading, preprocessing, output shape/range
- TripletBatchSampler: batch composition guarantees
- GradientAccumulationConfig: accumulation logic
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ir_recognition.models import ImageMetadata
from ir_recognition.signature_db.database import SignatureDatabase
from ir_recognition.training.dataloader import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    GradientAccumulationConfig,
    IRSignatureDataset,
    TripletBatchSampler,
)


@pytest.fixture
def populated_database(tmp_path: Path) -> SignatureDatabase:
    """Create a SignatureDatabase with enough images for testing.

    Creates 4 images per class for 3 classes (12 total), then generates splits.
    """
    db = SignatureDatabase(tmp_path / "test_db")

    classes = ["T-72", "T-80", "BMP-3"]
    img_count = 0
    for cls in classes:
        for i in range(4):
            # Create a simple test image with distinct patterns per class
            rng = np.random.default_rng(seed=img_count)
            image = rng.random((64, 64), dtype=np.float32)
            metadata = ImageMetadata(
                image_id=f"{cls.lower().replace('-', '')}_{i:03d}",
                vehicle_class=cls,
                azimuth=float(i * 90),
                elevation=30.0,
                source_type="synthetic",
                file_path=Path("placeholder"),
            )
            db.add_image(image, metadata)
            img_count += 1

    db.generate_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    return db


@pytest.fixture
def large_database(tmp_path: Path) -> SignatureDatabase:
    """Create a larger database with 10 images per class for 5 classes."""
    db = SignatureDatabase(tmp_path / "large_db")

    classes = ["T-72", "T-80", "BMP-3", "CV90", "civilian_car"]
    img_count = 0
    for cls in classes:
        for i in range(10):
            rng = np.random.default_rng(seed=img_count)
            image = rng.random((100, 100), dtype=np.float32)
            metadata = ImageMetadata(
                image_id=f"{cls.lower().replace('-', '')}_{i:03d}",
                vehicle_class=cls,
                azimuth=float(i * 36),
                elevation=30.0,
                source_type="synthetic",
                file_path=Path("placeholder"),
            )
            db.add_image(image, metadata)
            img_count += 1

    db.generate_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    return db


class TestIRSignatureDataset:
    """Tests for IRSignatureDataset."""

    def test_dataset_length(self, populated_database: SignatureDatabase):
        """Dataset length matches number of images in the split."""
        dataset = IRSignatureDataset(populated_database, split="train")
        train_metadata = populated_database.get_split("train")
        assert len(dataset) == len(train_metadata)

    def test_output_shape(self, populated_database: SignatureDatabase):
        """Each item returns a (3, 224, 224) tensor."""
        dataset = IRSignatureDataset(populated_database, split="train")
        if len(dataset) == 0:
            pytest.skip("No training samples available")
        image, label = dataset[0]
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32

    def test_output_label_is_int(self, populated_database: SignatureDatabase):
        """Labels are non-negative integers."""
        dataset = IRSignatureDataset(populated_database, split="train")
        if len(dataset) == 0:
            pytest.skip("No training samples available")
        _, label = dataset[0]
        assert isinstance(label, int)
        assert label >= 0

    def test_grayscale_replicated_to_3_channels(self, populated_database: SignatureDatabase):
        """All 3 channels have the same base content before normalization."""
        dataset = IRSignatureDataset(populated_database, split="train", normalize=False)
        if len(dataset) == 0:
            pytest.skip("No training samples available")
        image, _ = dataset[0]
        # Without normalization, all channels should be identical
        assert torch.allclose(image[0], image[1])
        assert torch.allclose(image[1], image[2])

    def test_unnormalized_range(self, populated_database: SignatureDatabase):
        """Without normalization, pixel values are in [0, 1]."""
        dataset = IRSignatureDataset(populated_database, split="train", normalize=False)
        if len(dataset) == 0:
            pytest.skip("No training samples available")
        image, _ = dataset[0]
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_normalized_values(self, populated_database: SignatureDatabase):
        """With normalization, values are shifted by ImageNet mean/std."""
        dataset = IRSignatureDataset(populated_database, split="train", normalize=True)
        if len(dataset) == 0:
            pytest.skip("No training samples available")
        image, _ = dataset[0]
        # After ImageNet normalization, values can be negative
        # The range should be roughly [-2.5, 2.5] for typical images
        assert image.min() < 0.0  # Normalization shifts values below 0

    def test_class_to_idx_mapping(self, populated_database: SignatureDatabase):
        """class_to_idx provides a valid mapping."""
        dataset = IRSignatureDataset(populated_database, split="train")
        assert len(dataset.class_to_idx) > 0
        # All indices should be unique and sequential
        indices = sorted(dataset.class_to_idx.values())
        assert indices == list(range(len(indices)))

    def test_num_classes(self, populated_database: SignatureDatabase):
        """num_classes matches the number of distinct classes in the split."""
        dataset = IRSignatureDataset(populated_database, split="train")
        assert dataset.num_classes == len(dataset.class_to_idx)

    def test_labels_list(self, populated_database: SignatureDatabase):
        """labels list has correct length and valid values."""
        dataset = IRSignatureDataset(populated_database, split="train")
        assert len(dataset.labels) == len(dataset)
        for label in dataset.labels:
            assert 0 <= label < dataset.num_classes

    def test_resize_from_different_sizes(self, tmp_path: Path):
        """Images of different original sizes are all resized to 224×224."""
        db = SignatureDatabase(tmp_path / "resize_db")

        # Add images of different sizes
        for i, size in enumerate([(50, 50), (300, 400), (224, 224)]):
            rng = np.random.default_rng(seed=i)
            image = rng.random(size, dtype=np.float32)
            metadata = ImageMetadata(
                image_id=f"test_{i:03d}",
                vehicle_class="T-72",
                azimuth=float(i * 120),
                elevation=30.0,
                source_type="synthetic",
                file_path=Path("placeholder"),
            )
            db.add_image(image, metadata)

        db.generate_splits(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0, seed=42)
        dataset = IRSignatureDataset(db, split="train")

        for i in range(len(dataset)):
            image, _ = dataset[i]
            assert image.shape == (3, 224, 224)


class TestTripletBatchSampler:
    """Tests for TripletBatchSampler."""

    def test_batch_size(self, large_database: SignatureDatabase):
        """Each batch has exactly batch_size samples."""
        dataset = IRSignatureDataset(large_database, split="train")
        batch_size = 8
        sampler = TripletBatchSampler(
            labels=dataset.labels,
            batch_size=batch_size,
            seed=42,
        )
        for batch in sampler:
            assert len(batch) == batch_size

    def test_at_least_two_classes_per_batch(self, large_database: SignatureDatabase):
        """Each batch contains images from at least 2 different classes."""
        dataset = IRSignatureDataset(large_database, split="train")
        sampler = TripletBatchSampler(
            labels=dataset.labels,
            batch_size=8,
            seed=42,
        )
        for batch in sampler:
            batch_labels = {dataset.labels[idx] for idx in batch}
            assert len(batch_labels) >= 2

    def test_at_least_two_samples_per_class(self, large_database: SignatureDatabase):
        """Each class in a batch has at least 2 samples."""
        dataset = IRSignatureDataset(large_database, split="train")
        sampler = TripletBatchSampler(
            labels=dataset.labels,
            batch_size=8,
            samples_per_class=2,
            seed=42,
        )
        for batch in sampler:
            # Count samples per class in this batch
            class_counts: dict[int, int] = {}
            for idx in batch:
                label = dataset.labels[idx]
                class_counts[label] = class_counts.get(label, 0) + 1

            # Each class that appears should have at least 2 samples
            # (the guaranteed classes from selection)
            # Note: due to fill logic, some classes may have more
            for count in class_counts.values():
                assert count >= 2

    def test_valid_indices(self, large_database: SignatureDatabase):
        """All indices in batches are valid dataset indices."""
        dataset = IRSignatureDataset(large_database, split="train")
        sampler = TripletBatchSampler(
            labels=dataset.labels,
            batch_size=8,
            seed=42,
        )
        for batch in sampler:
            for idx in batch:
                assert 0 <= idx < len(dataset)

    def test_sampler_length(self, large_database: SignatureDatabase):
        """Sampler length matches expected number of batches."""
        dataset = IRSignatureDataset(large_database, split="train")
        batch_size = 8
        sampler = TripletBatchSampler(
            labels=dataset.labels,
            batch_size=batch_size,
            drop_last=True,
            seed=42,
        )
        expected_batches = len(dataset) // batch_size
        assert len(sampler) == expected_batches

    def test_raises_on_insufficient_classes(self):
        """Raises ValueError if fewer than 2 classes have enough samples."""
        # Only 1 class
        labels = [0] * 10
        with pytest.raises(ValueError, match="Need at least 2 classes"):
            TripletBatchSampler(labels=labels, batch_size=4)

    def test_raises_on_batch_too_small(self):
        """Raises ValueError if batch_size can't fit 2 classes × samples_per_class."""
        labels = [0, 0, 1, 1, 2, 2]
        with pytest.raises(ValueError):
            TripletBatchSampler(
                labels=labels,
                batch_size=3,  # Too small for 2 classes × 2 samples
                samples_per_class=2,
            )

    def test_reproducibility_with_seed(self, large_database: SignatureDatabase):
        """Same seed produces same batch sequence."""
        dataset = IRSignatureDataset(large_database, split="train")
        sampler1 = TripletBatchSampler(
            labels=dataset.labels, batch_size=8, seed=123
        )
        sampler2 = TripletBatchSampler(
            labels=dataset.labels, batch_size=8, seed=123
        )
        batches1 = list(sampler1)
        batches2 = list(sampler2)
        assert batches1 == batches2


class TestGradientAccumulationConfig:
    """Tests for GradientAccumulationConfig."""

    def test_effective_batch_size(self):
        """effective_batch_size = batch_size * accumulation_steps."""
        config = GradientAccumulationConfig(batch_size=4, accumulation_steps=8)
        assert config.effective_batch_size == 32

    def test_no_accumulation(self):
        """With accumulation_steps=1, effective equals actual."""
        config = GradientAccumulationConfig(batch_size=4, accumulation_steps=1)
        assert config.effective_batch_size == 4

    def test_should_step(self):
        """should_step returns True every accumulation_steps batches."""
        config = GradientAccumulationConfig(batch_size=4, accumulation_steps=4)
        # batch_idx is 0-based
        assert not config.should_step(0)
        assert not config.should_step(1)
        assert not config.should_step(2)
        assert config.should_step(3)  # 4th batch (idx=3) → step
        assert not config.should_step(4)
        assert config.should_step(7)  # 8th batch → step

    def test_scale_loss(self):
        """Loss is scaled by 1/accumulation_steps."""
        config = GradientAccumulationConfig(batch_size=4, accumulation_steps=4)
        loss = torch.tensor(4.0)
        scaled = config.scale_loss(loss)
        assert torch.isclose(scaled, torch.tensor(1.0))

    def test_scale_loss_no_accumulation(self):
        """With accumulation_steps=1, loss is unchanged."""
        config = GradientAccumulationConfig(batch_size=4, accumulation_steps=1)
        loss = torch.tensor(2.5)
        scaled = config.scale_loss(loss)
        assert torch.isclose(scaled, loss)

    def test_from_target_batch_size(self):
        """from_target_batch_size computes correct accumulation steps."""
        config = GradientAccumulationConfig.from_target_batch_size(
            target_batch_size=32, max_batch_size=4
        )
        assert config.batch_size == 4
        assert config.accumulation_steps == 8
        assert config.effective_batch_size == 32

    def test_from_target_batch_size_no_accumulation_needed(self):
        """When target fits in VRAM, no accumulation needed."""
        config = GradientAccumulationConfig.from_target_batch_size(
            target_batch_size=4, max_batch_size=8
        )
        assert config.batch_size == 4
        assert config.accumulation_steps == 1

    def test_invalid_batch_size(self):
        """Raises ValueError for invalid batch_size."""
        with pytest.raises(ValueError):
            GradientAccumulationConfig(batch_size=0, accumulation_steps=1)

    def test_invalid_accumulation_steps(self):
        """Raises ValueError for invalid accumulation_steps."""
        with pytest.raises(ValueError):
            GradientAccumulationConfig(batch_size=4, accumulation_steps=0)
