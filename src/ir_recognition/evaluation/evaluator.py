"""Evaluation module for IR Signature Recognition.

Provides the Evaluator class that computes classification metrics,
similarity metrics, latency measurements, and VRAM usage on the test split.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ir_recognition.inference.pipeline import InferencePipeline
    from ir_recognition.signature_db.database import SignatureDatabase

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates the inference pipeline on the test split of the signature database.

    Computes classification metrics (accuracy, per-class precision/recall/F1),
    similarity metrics (recall@5, mean similarity for correct matches),
    latency measurements (mean, p95, max), and peak VRAM usage.

    Args:
        pipeline: InferencePipeline instance for running inference.
        database: SignatureDatabase instance containing test split images.
    """

    def __init__(self, pipeline: "InferencePipeline", database: "SignatureDatabase") -> None:
        """Initialize the evaluator.

        Args:
            pipeline: Trained inference pipeline for classification and similarity.
            database: Signature database with test split assigned.
        """
        self._pipeline = pipeline
        self._database = database

    def evaluate_classification(self) -> dict:
        """Run classification on the test split and compute metrics.

        Evaluates the pipeline's classify() method on all test images,
        computing overall accuracy and per-class precision, recall, and F1.

        Returns:
            Dictionary with:
                - 'accuracy': float, overall classification accuracy
                - 'per_class': dict mapping vehicle_class to {precision, recall, f1}
                - 'num_samples': int, number of test images evaluated
        """
        test_images = self._database.get_split("test")

        if not test_images:
            logger.warning("No test images found in database")
            return {"accuracy": 0.0, "per_class": {}, "num_samples": 0}

        # Collect predictions and ground truth
        y_true: list[str] = []
        y_pred: list[str] = []

        for metadata in test_images:
            image_path = self._database.db_path / metadata.file_path
            try:
                classifications = self._pipeline.classify(image_path)
                if classifications:
                    predicted_class = classifications[0].vehicle_class
                else:
                    predicted_class = ""
            except Exception as e:
                logger.warning(f"Classification failed for {metadata.image_id}: {e}")
                predicted_class = ""

            y_true.append(metadata.vehicle_class)
            y_pred.append(predicted_class)

        # Compute overall accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true) if y_true else 0.0

        # Compute per-class precision, recall, F1
        per_class = self._compute_per_class_metrics(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "per_class": per_class,
            "num_samples": len(y_true),
        }

    def evaluate_similarity(self) -> dict:
        """Run similarity queries on the test split and compute metrics.

        For each test image, runs find_similar() and checks whether the
        correct vehicle class appears in the top-5 results (recall@5).
        Also computes mean similarity score for correct matches.

        Returns:
            Dictionary with:
                - 'recall_at_5': float, fraction of test images where correct
                  class appears in top-5 similar results
                - 'mean_similarity_correct': float, mean similarity score for
                  matches that have the correct class
                - 'num_samples': int, number of test images evaluated
        """
        test_images = self._database.get_split("test")

        if not test_images:
            logger.warning("No test images found in database")
            return {
                "recall_at_5": 0.0,
                "mean_similarity_correct": 0.0,
                "num_samples": 0,
            }

        hits = 0
        correct_similarities: list[float] = []

        for metadata in test_images:
            image_path = self._database.db_path / metadata.file_path
            try:
                matches = self._pipeline.find_similar(image_path)
            except Exception as e:
                logger.warning(f"Similarity search failed for {metadata.image_id}: {e}")
                continue

            # Check if correct class appears in top-5
            correct_class = metadata.vehicle_class
            found_correct = False
            for match in matches:
                if match.vehicle_class == correct_class:
                    if not found_correct:
                        hits += 1
                        found_correct = True
                    correct_similarities.append(match.similarity_score)

        num_samples = len(test_images)
        recall_at_5 = hits / num_samples if num_samples > 0 else 0.0
        mean_sim = (
            float(np.mean(correct_similarities))
            if correct_similarities
            else 0.0
        )

        return {
            "recall_at_5": recall_at_5,
            "mean_similarity_correct": mean_sim,
            "num_samples": num_samples,
        }

    def measure_latency(self, num_samples: int = 50) -> dict:
        """Measure inference latency over N samples from the test split.

        Runs the full recognize() pipeline on up to num_samples test images
        and reports timing statistics.

        Args:
            num_samples: Number of samples to measure. Uses min(num_samples,
                available test images).

        Returns:
            Dictionary with:
                - 'mean_seconds': float, mean inference time per image
                - 'p95_seconds': float, 95th percentile inference time
                - 'max_seconds': float, maximum inference time
                - 'num_samples': int, actual number of samples measured
        """
        test_images = self._database.get_split("test")

        if not test_images:
            logger.warning("No test images found for latency measurement")
            return {
                "mean_seconds": 0.0,
                "p95_seconds": 0.0,
                "max_seconds": 0.0,
                "num_samples": 0,
            }

        # Limit to num_samples
        samples = test_images[:num_samples]
        latencies: list[float] = []

        for metadata in samples:
            image_path = self._database.db_path / metadata.file_path
            start = time.perf_counter()
            try:
                self._pipeline.recognize(image_path)
            except Exception as e:
                logger.warning(f"Inference failed for {metadata.image_id}: {e}")
                continue
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        if not latencies:
            return {
                "mean_seconds": 0.0,
                "p95_seconds": 0.0,
                "max_seconds": 0.0,
                "num_samples": 0,
            }

        latencies_arr = np.array(latencies)
        return {
            "mean_seconds": float(np.mean(latencies_arr)),
            "p95_seconds": float(np.percentile(latencies_arr, 95)),
            "max_seconds": float(np.max(latencies_arr)),
            "num_samples": len(latencies),
        }

    def measure_vram(self) -> float:
        """Measure peak VRAM usage during inference.

        Runs a single inference pass and reports the peak GPU memory
        allocated. Returns 0.0 if CUDA is not available.

        Returns:
            Peak VRAM usage in GB during inference.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                logger.info("CUDA not available, returning 0.0 for VRAM measurement")
                return 0.0

            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()

            # Run a single inference to measure peak VRAM
            test_images = self._database.get_split("test")
            if test_images:
                image_path = self._database.db_path / test_images[0].file_path
                try:
                    self._pipeline.recognize(image_path)
                except Exception as e:
                    logger.warning(f"Inference failed during VRAM measurement: {e}")

            # Get peak memory in bytes, convert to GB
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_gb = peak_bytes / (1024**3)

            return peak_gb

        except ImportError:
            logger.warning("PyTorch not available for VRAM measurement")
            return 0.0

    @staticmethod
    def _compute_per_class_metrics(
        y_true: list[str], y_pred: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute per-class precision, recall, and F1 score.

        Args:
            y_true: Ground truth class labels.
            y_pred: Predicted class labels.

        Returns:
            Dictionary mapping each class to {precision, recall, f1}.
        """
        # Get all unique classes from ground truth
        classes = sorted(set(y_true))

        per_class: dict[str, dict[str, float]] = {}

        for cls in classes:
            # True positives: predicted cls and actually cls
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            # False positives: predicted cls but actually not cls
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            # False negatives: actually cls but predicted something else
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        return per_class
