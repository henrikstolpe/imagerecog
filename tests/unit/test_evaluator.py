"""Unit tests for the Evaluator class."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ir_recognition.evaluation.evaluator import Evaluator
from ir_recognition.models import (
    ClassificationResult,
    ImageMetadata,
    RecognitionResult,
    SimilarityMatch,
)


@pytest.fixture
def mock_pipeline():
    """Create a mock inference pipeline."""
    pipeline = MagicMock()
    return pipeline


@pytest.fixture
def mock_database(tmp_path):
    """Create a mock database with test images."""
    db = MagicMock()
    db.db_path = tmp_path

    # Create some fake image files
    images_dir = tmp_path / "images" / "T-72"
    images_dir.mkdir(parents=True)

    test_metadata = []
    for i in range(5):
        img_path = Path("images") / "T-72" / f"img_{i}.png"
        full_path = tmp_path / img_path
        # Write a minimal valid PNG-like file (just needs to exist for path resolution)
        full_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        test_metadata.append(
            ImageMetadata(
                image_id=f"img_{i}",
                vehicle_class="T-72",
                azimuth=float(i * 45),
                elevation=30.0,
                source_type="synthetic",
                file_path=img_path,
            )
        )

    # Add some images from another class
    images_dir2 = tmp_path / "images" / "T-80"
    images_dir2.mkdir(parents=True)
    for i in range(5):
        img_path = Path("images") / "T-80" / f"img_{i}.png"
        full_path = tmp_path / img_path
        full_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        test_metadata.append(
            ImageMetadata(
                image_id=f"t80_img_{i}",
                vehicle_class="T-80",
                azimuth=float(i * 45),
                elevation=30.0,
                source_type="synthetic",
                file_path=img_path,
            )
        )

    db.get_split.return_value = test_metadata
    return db


class TestEvaluateClassification:
    """Tests for evaluate_classification()."""

    def test_perfect_classification(self, mock_pipeline, mock_database):
        """Test accuracy computation with perfect predictions."""
        # Mock classify to always return correct class
        def classify_side_effect(image_path):
            # Determine class from path
            if "T-72" in str(image_path):
                return [ClassificationResult(vehicle_class="T-72", confidence=0.9)]
            else:
                return [ClassificationResult(vehicle_class="T-80", confidence=0.9)]

        mock_pipeline.classify.side_effect = classify_side_effect

        evaluator = Evaluator(mock_pipeline, mock_database)
        result = evaluator.evaluate_classification()

        assert result["accuracy"] == 1.0
        assert result["num_samples"] == 10
        assert "T-72" in result["per_class"]
        assert "T-80" in result["per_class"]
        assert result["per_class"]["T-72"]["precision"] == 1.0
        assert result["per_class"]["T-72"]["recall"] == 1.0
        assert result["per_class"]["T-72"]["f1"] == 1.0

    def test_all_wrong_classification(self, mock_pipeline, mock_database):
        """Test accuracy computation when all predictions are wrong."""
        # Mock classify to always return wrong class
        def classify_side_effect(image_path):
            if "T-72" in str(image_path):
                return [ClassificationResult(vehicle_class="T-80", confidence=0.9)]
            else:
                return [ClassificationResult(vehicle_class="T-72", confidence=0.9)]

        mock_pipeline.classify.side_effect = classify_side_effect

        evaluator = Evaluator(mock_pipeline, mock_database)
        result = evaluator.evaluate_classification()

        assert result["accuracy"] == 0.0
        assert result["num_samples"] == 10
        # Precision for T-72: predicted T-72 5 times, all were actually T-80
        assert result["per_class"]["T-72"]["precision"] == 0.0
        assert result["per_class"]["T-72"]["recall"] == 0.0

    def test_empty_test_split(self, mock_pipeline, mock_database):
        """Test with no test images."""
        mock_database.get_split.return_value = []

        evaluator = Evaluator(mock_pipeline, mock_database)
        result = evaluator.evaluate_classification()

        assert result["accuracy"] == 0.0
        assert result["per_class"] == {}
        assert result["num_samples"] == 0

    def test_classification_error_handling(self, mock_pipeline, mock_database):
        """Test that classification errors are handled gracefully."""
        mock_pipeline.classify.side_effect = RuntimeError("Model error")

        evaluator = Evaluator(mock_pipeline, mock_database)
        result = evaluator.evaluate_classification()

        # All predictions should be empty string (wrong), so accuracy = 0
        assert result["accuracy"] == 0.0
        assert result["num_samples"] == 10


class TestEvaluateSimilarity:
    """Tests for evaluate_similarity()."""

    def test_perfect_similarity(self, mock_pipeline, mock_database):
        """Test recall@5 when correct class always appears in top-5."""
        def find_similar_side_effect(image_path):
            if "T-72" in str(image_path):
                return [
                    SimilarityMatch(vehicle_class="T-72", similarity_score=0.95, image_id="match1"),
                    SimilarityMatch(vehicle_class="T-72", similarity_score=0.90, image_id="match2"),
                    SimilarityMatch(vehicle_class="T-80", similarity_score=0.80, image_id="match3"),
                ]
            else:
                return [
                    SimilarityMatch(vehicle_class="T-80", similarity_score=0.92, image_id="match4"),
                    SimilarityMatch(vehicle_class="T-72", similarity_score=0.85, image_id="match5"),
                ]

        mock_pipeline.find_similar.side_effect = find_similar_side_effect

        evaluator = Evaluator(mock_pipeline, mock_database)
        result = evaluator.evaluate_similarity()

        assert result["recall_at_5"] == 1.0
        assert result["mean_similarity_correct"] > 0.0
        assert result["num_samples"] == 10

    def test_no_correct_matches(self, mock_pipeline, mock_database):
        """Test recall@5 when correct class never appears."""
        def find_similar_side_effect(image_path):
            if "T-72" in str(image_path):
                return [
                    SimilarityMatch(vehicle_class="T-80", similarity_score=0.9, image_id="m1"),
                ]
            else:
                return [
                    SimilarityMatch(vehicle_class="T-72", similarity_score=0.9, image_id="m2"),
                ]

        mock_pipeline.find_similar.side_effect = find_similar_side_effect

        evaluator = Evaluator(mock_pipeline, mock_database)
        result = evaluator.evaluate_similarity()

        assert result["recall_at_5"] == 0.0
        assert result["mean_similarity_correct"] == 0.0

    def test_empty_test_split_similarity(self, mock_pipeline, mock_database):
        """Test with no test images."""
        mock_database.get_split.return_value = []

        evaluator = Evaluator(mock_pipeline, mock_database)
        result = evaluator.evaluate_similarity()

        assert result["recall_at_5"] == 0.0
        assert result["mean_similarity_correct"] == 0.0
        assert result["num_samples"] == 0


class TestMeasureLatency:
    """Tests for measure_latency()."""

    def test_latency_returns_valid_values(self, mock_pipeline, mock_database):
        """Test that latency measurement returns positive timing values."""
        # Mock recognize to take a small amount of time
        def recognize_side_effect(image_path):
            time.sleep(0.001)  # 1ms
            return RecognitionResult(
                classifications=[ClassificationResult(vehicle_class="T-72", confidence=0.9)],
                similarity_matches=[],
                low_confidence=False,
                no_match=True,
            )

        mock_pipeline.recognize.side_effect = recognize_side_effect

        evaluator = Evaluator(mock_pipeline, mock_database)
        result = evaluator.measure_latency(num_samples=5)

        assert result["mean_seconds"] > 0.0
        assert result["p95_seconds"] > 0.0
        assert result["max_seconds"] > 0.0
        assert result["p95_seconds"] >= result["mean_seconds"]
        assert result["max_seconds"] >= result["p95_seconds"]
        assert result["num_samples"] == 5

    def test_latency_limits_samples(self, mock_pipeline, mock_database):
        """Test that latency measurement respects num_samples limit."""
        mock_pipeline.recognize.return_value = RecognitionResult(
            classifications=[ClassificationResult(vehicle_class="T-72", confidence=0.9)],
            similarity_matches=[],
            low_confidence=False,
            no_match=True,
        )

        evaluator = Evaluator(mock_pipeline, mock_database)
        result = evaluator.measure_latency(num_samples=3)

        assert result["num_samples"] == 3
        assert mock_pipeline.recognize.call_count == 3

    def test_latency_empty_test_split(self, mock_pipeline, mock_database):
        """Test latency with no test images."""
        mock_database.get_split.return_value = []

        evaluator = Evaluator(mock_pipeline, mock_database)
        result = evaluator.measure_latency()

        assert result["mean_seconds"] == 0.0
        assert result["p95_seconds"] == 0.0
        assert result["max_seconds"] == 0.0
        assert result["num_samples"] == 0


class TestMeasureVram:
    """Tests for measure_vram()."""

    def test_vram_no_cuda(self, mock_pipeline, mock_database):
        """Test VRAM measurement when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            evaluator = Evaluator(mock_pipeline, mock_database)
            result = evaluator.measure_vram()

        assert result == 0.0

    def test_vram_with_cuda(self, mock_pipeline, mock_database):
        """Test VRAM measurement when CUDA is available."""
        mock_pipeline.recognize.return_value = RecognitionResult(
            classifications=[ClassificationResult(vehicle_class="T-72", confidence=0.9)],
            similarity_matches=[],
            low_confidence=False,
            no_match=True,
        )

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.reset_peak_memory_stats") as mock_reset, \
             patch("torch.cuda.max_memory_allocated", return_value=4 * 1024**3):  # 4GB
            evaluator = Evaluator(mock_pipeline, mock_database)
            result = evaluator.measure_vram()

        mock_reset.assert_called_once()
        assert result == pytest.approx(4.0, abs=0.01)


class TestPerClassMetrics:
    """Tests for the _compute_per_class_metrics static method."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = ["A", "A", "B", "B"]
        y_pred = ["A", "A", "B", "B"]

        result = Evaluator._compute_per_class_metrics(y_true, y_pred)

        assert result["A"]["precision"] == 1.0
        assert result["A"]["recall"] == 1.0
        assert result["A"]["f1"] == 1.0
        assert result["B"]["precision"] == 1.0
        assert result["B"]["recall"] == 1.0
        assert result["B"]["f1"] == 1.0

    def test_mixed_predictions(self):
        """Test metrics with some correct and some wrong predictions."""
        y_true = ["A", "A", "B", "B"]
        y_pred = ["A", "B", "B", "A"]

        result = Evaluator._compute_per_class_metrics(y_true, y_pred)

        # A: TP=1, FP=1, FN=1 -> precision=0.5, recall=0.5, f1=0.5
        assert result["A"]["precision"] == 0.5
        assert result["A"]["recall"] == 0.5
        assert result["A"]["f1"] == 0.5
        # B: TP=1, FP=1, FN=1 -> precision=0.5, recall=0.5, f1=0.5
        assert result["B"]["precision"] == 0.5
        assert result["B"]["recall"] == 0.5
        assert result["B"]["f1"] == 0.5

    def test_no_predictions_for_class(self):
        """Test metrics when a class is never predicted."""
        y_true = ["A", "A", "B"]
        y_pred = ["B", "B", "B"]

        result = Evaluator._compute_per_class_metrics(y_true, y_pred)

        # A: TP=0, FP=0, FN=2 -> precision=0, recall=0, f1=0
        assert result["A"]["precision"] == 0.0
        assert result["A"]["recall"] == 0.0
        assert result["A"]["f1"] == 0.0
        # B: TP=1, FP=2, FN=0 -> precision=1/3, recall=1.0, f1=0.5
        assert result["B"]["precision"] == pytest.approx(1 / 3)
        assert result["B"]["recall"] == 1.0
        assert result["B"]["f1"] == pytest.approx(0.5)
