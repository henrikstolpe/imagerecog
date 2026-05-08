"""Unit tests for embedding pre-computation module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ir_recognition.inference.embeddings import (
    compute_all_embeddings,
    compute_incremental_embeddings,
    _get_embeddings_path,
    _load_existing_embeddings,
    _save_embeddings,
)
from ir_recognition.models import ImageMetadata
from ir_recognition.signature_db.database import SignatureDatabase


class MockModelBackend:
    """Mock model backend that returns deterministic embeddings based on input."""

    def __init__(self, embedding_dim: int = 256, num_classes: int = 5):
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.call_count = 0

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic logits and L2-normalized embeddings."""
        self.call_count += 1
        batch_size = pixel_values.shape[0]

        # Generate deterministic logits
        logits = torch.randn(batch_size, self.num_classes)

        # Generate deterministic embedding and L2-normalize
        embedding = torch.randn(batch_size, self.embedding_dim)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

        return logits, embedding


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SignatureDatabase with some images."""
    db = SignatureDatabase(tmp_path / "test_db")

    # Add a few test images
    for i in range(5):
        image = np.random.rand(224, 224).astype(np.float32)
        metadata = ImageMetadata(
            image_id=f"test_img_{i:03d}",
            vehicle_class="T-72",
            azimuth=float(i * 45) % 360,
            elevation=30.0,
            source_type="synthetic",
            file_path=Path(f"images/T-72/test_img_{i:03d}.png"),
        )
        db.add_image(image, metadata)

    return db


@pytest.fixture
def mock_model():
    """Create a mock model backend."""
    return MockModelBackend()


class TestGetEmbeddingsPath:
    """Tests for _get_embeddings_path."""

    def test_returns_correct_path(self, tmp_db):
        path = _get_embeddings_path(tmp_db)
        expected = Path(tmp_db.db_path) / "embeddings" / "embeddings.npz"
        assert path == expected


class TestLoadExistingEmbeddings:
    """Tests for _load_existing_embeddings."""

    def test_returns_none_when_file_missing(self, tmp_path):
        path = tmp_path / "nonexistent.npz"
        embeddings, ids, labels = _load_existing_embeddings(path)
        assert embeddings is None
        assert ids == []
        assert labels == []

    def test_loads_valid_npz(self, tmp_path):
        path = tmp_path / "embeddings.npz"
        test_embeddings = np.random.randn(3, 256).astype(np.float32)
        test_ids = ["img_0", "img_1", "img_2"]
        test_labels = ["T-72", "T-80", "BMP-3"]

        np.savez(
            path,
            embeddings=test_embeddings,
            image_ids=np.array(test_ids, dtype=object),
            labels=np.array(test_labels, dtype=object),
        )

        embeddings, ids, labels = _load_existing_embeddings(path)
        assert embeddings is not None
        assert embeddings.shape == (3, 256)
        assert ids == test_ids
        assert labels == test_labels


class TestSaveEmbeddings:
    """Tests for _save_embeddings."""

    def test_saves_and_loads_correctly(self, tmp_path):
        path = tmp_path / "embeddings" / "embeddings.npz"
        test_embeddings = np.random.randn(4, 256).astype(np.float32)
        test_ids = ["a", "b", "c", "d"]
        test_labels = ["T-72", "T-80", "BMP-3", "CV90"]

        _save_embeddings(path, test_embeddings, test_ids, test_labels)

        assert path.exists()
        data = np.load(path, allow_pickle=True)
        np.testing.assert_array_almost_equal(data["embeddings"], test_embeddings)
        assert data["image_ids"].tolist() == test_ids
        assert data["labels"].tolist() == test_labels


class TestComputeAllEmbeddings:
    """Tests for compute_all_embeddings."""

    def test_computes_embeddings_for_all_images(self, tmp_db, mock_model):
        compute_all_embeddings(mock_model, tmp_db, device="cpu")

        embeddings_path = _get_embeddings_path(tmp_db)
        assert embeddings_path.exists()

        data = np.load(embeddings_path, allow_pickle=True)
        assert data["embeddings"].shape == (5, 256)
        assert len(data["image_ids"]) == 5
        assert len(data["labels"]) == 5

        # All labels should be T-72 since that's what we added
        assert all(label == "T-72" for label in data["labels"].tolist())

    def test_embeddings_are_float32(self, tmp_db, mock_model):
        compute_all_embeddings(mock_model, tmp_db, device="cpu")

        embeddings_path = _get_embeddings_path(tmp_db)
        data = np.load(embeddings_path, allow_pickle=True)
        assert data["embeddings"].dtype == np.float32

    def test_empty_database_does_nothing(self, tmp_path, mock_model):
        db = SignatureDatabase(tmp_path / "empty_db")
        compute_all_embeddings(mock_model, db, device="cpu")

        embeddings_path = _get_embeddings_path(db)
        assert not embeddings_path.exists()

    def test_model_called_for_each_image(self, tmp_db, mock_model):
        compute_all_embeddings(mock_model, tmp_db, device="cpu")
        assert mock_model.call_count == 5


class TestComputeIncrementalEmbeddings:
    """Tests for compute_incremental_embeddings."""

    def test_computes_all_when_no_existing(self, tmp_db, mock_model):
        count = compute_incremental_embeddings(mock_model, tmp_db, device="cpu")
        assert count == 5

        embeddings_path = _get_embeddings_path(tmp_db)
        data = np.load(embeddings_path, allow_pickle=True)
        assert data["embeddings"].shape == (5, 256)

    def test_only_computes_new_images(self, tmp_db, mock_model):
        # First, compute embeddings for existing images
        compute_all_embeddings(mock_model, tmp_db, device="cpu")
        initial_count = mock_model.call_count

        # Add a new image
        image = np.random.rand(224, 224).astype(np.float32)
        metadata = ImageMetadata(
            image_id="new_img_001",
            vehicle_class="T-80",
            azimuth=90.0,
            elevation=30.0,
            source_type="synthetic",
            file_path=Path("images/T-80/new_img_001.png"),
        )
        tmp_db.add_image(image, metadata)

        # Incremental update should only process the new image
        mock_model.call_count = 0
        count = compute_incremental_embeddings(mock_model, tmp_db, device="cpu")
        assert count == 1
        assert mock_model.call_count == 1

        # Verify total embeddings
        embeddings_path = _get_embeddings_path(tmp_db)
        data = np.load(embeddings_path, allow_pickle=True)
        assert data["embeddings"].shape == (6, 256)
        assert "new_img_001" in data["image_ids"].tolist()

    def test_returns_zero_when_all_computed(self, tmp_db, mock_model):
        compute_all_embeddings(mock_model, tmp_db, device="cpu")
        mock_model.call_count = 0

        count = compute_incremental_embeddings(mock_model, tmp_db, device="cpu")
        assert count == 0
        assert mock_model.call_count == 0

    def test_returns_zero_for_empty_database(self, tmp_path, mock_model):
        db = SignatureDatabase(tmp_path / "empty_db")
        count = compute_incremental_embeddings(mock_model, db, device="cpu")
        assert count == 0

    def test_preserves_existing_embeddings(self, tmp_db, mock_model):
        # Compute initial embeddings
        compute_all_embeddings(mock_model, tmp_db, device="cpu")

        embeddings_path = _get_embeddings_path(tmp_db)
        data_before = np.load(embeddings_path, allow_pickle=True)
        ids_before = data_before["image_ids"].tolist()

        # Add new image and run incremental
        image = np.random.rand(224, 224).astype(np.float32)
        metadata = ImageMetadata(
            image_id="extra_img",
            vehicle_class="BMP-3",
            azimuth=180.0,
            elevation=45.0,
            source_type="synthetic",
            file_path=Path("images/BMP-3/extra_img.png"),
        )
        tmp_db.add_image(image, metadata)

        compute_incremental_embeddings(mock_model, tmp_db, device="cpu")

        data_after = np.load(embeddings_path, allow_pickle=True)
        ids_after = data_after["image_ids"].tolist()

        # All original IDs should still be present
        for img_id in ids_before:
            assert img_id in ids_after
        # New ID should be added
        assert "extra_img" in ids_after
