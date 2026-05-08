"""Unit tests for core data models and type definitions."""

from pathlib import Path

import numpy as np
import pytest

from ir_recognition.models import (
    SUPPORTED_VEHICLE_CLASSES,
    VALID_SOURCE_TYPES,
    ClassificationResult,
    EpochMetrics,
    ImageMetadata,
    IRGeneratorConfig,
    IRImage,
    RecognitionResult,
    SimilarityMatch,
    TrainingConfig,
)


class TestIRImage:
    """Tests for the IRImage type alias."""

    def test_ir_image_is_ndarray(self):
        """IRImage is a type alias for np.ndarray."""
        assert IRImage is np.ndarray


class TestIRGeneratorConfig:
    """Tests for IRGeneratorConfig dataclass."""

    def test_valid_config(self):
        config = IRGeneratorConfig(
            vehicle_type="T-72",
            azimuth=45.0,
            elevation=30.0,
            ambient_temp=293.0,
            image_size=(224, 224),
        )
        assert config.vehicle_type == "T-72"
        assert config.azimuth == 45.0
        assert config.elevation == 30.0
        assert config.ambient_temp == 293.0
        assert config.image_size == (224, 224)

    def test_default_values(self):
        config = IRGeneratorConfig(
            vehicle_type="T-80", azimuth=0.0, elevation=0.0
        )
        assert config.ambient_temp == 293.0
        assert config.image_size == (224, 224)

    def test_all_supported_vehicle_classes(self):
        for vc in SUPPORTED_VEHICLE_CLASSES:
            config = IRGeneratorConfig(
                vehicle_type=vc, azimuth=0.0, elevation=0.0
            )
            assert config.vehicle_type == vc

    def test_invalid_vehicle_type(self):
        with pytest.raises(ValueError, match="Unknown vehicle class"):
            IRGeneratorConfig(
                vehicle_type="unknown", azimuth=0.0, elevation=0.0
            )

    def test_azimuth_lower_bound(self):
        config = IRGeneratorConfig(
            vehicle_type="T-72", azimuth=0.0, elevation=0.0
        )
        assert config.azimuth == 0.0

    def test_azimuth_upper_bound_exclusive(self):
        with pytest.raises(ValueError, match="Azimuth must be in"):
            IRGeneratorConfig(
                vehicle_type="T-72", azimuth=360.0, elevation=0.0
            )

    def test_azimuth_negative(self):
        with pytest.raises(ValueError, match="Azimuth must be in"):
            IRGeneratorConfig(
                vehicle_type="T-72", azimuth=-1.0, elevation=0.0
            )

    def test_elevation_lower_bound(self):
        config = IRGeneratorConfig(
            vehicle_type="T-72", azimuth=0.0, elevation=0.0
        )
        assert config.elevation == 0.0

    def test_elevation_upper_bound(self):
        config = IRGeneratorConfig(
            vehicle_type="T-72", azimuth=0.0, elevation=90.0
        )
        assert config.elevation == 90.0

    def test_elevation_out_of_range(self):
        with pytest.raises(ValueError, match="Elevation must be in"):
            IRGeneratorConfig(
                vehicle_type="T-72", azimuth=0.0, elevation=91.0
            )

    def test_elevation_negative(self):
        with pytest.raises(ValueError, match="Elevation must be in"):
            IRGeneratorConfig(
                vehicle_type="T-72", azimuth=0.0, elevation=-1.0
            )

    def test_ambient_temp_zero(self):
        with pytest.raises(ValueError, match="Ambient temperature must be > 0"):
            IRGeneratorConfig(
                vehicle_type="T-72", azimuth=0.0, elevation=0.0, ambient_temp=0.0
            )

    def test_ambient_temp_negative(self):
        with pytest.raises(ValueError, match="Ambient temperature must be > 0"):
            IRGeneratorConfig(
                vehicle_type="T-72", azimuth=0.0, elevation=0.0, ambient_temp=-10.0
            )

    def test_invalid_image_size(self):
        with pytest.raises(ValueError, match="image_size must be a tuple"):
            IRGeneratorConfig(
                vehicle_type="T-72",
                azimuth=0.0,
                elevation=0.0,
                image_size=(0, 224),
            )


class TestImageMetadata:
    """Tests for ImageMetadata dataclass."""

    def test_valid_metadata(self):
        meta = ImageMetadata(
            image_id="t72_001",
            vehicle_class="T-72",
            azimuth=45.0,
            elevation=30.0,
            source_type="synthetic",
            file_path=Path("images/T-72/img001.png"),
        )
        assert meta.image_id == "t72_001"
        assert meta.vehicle_class == "T-72"
        assert meta.source_type == "synthetic"

    def test_string_file_path_converted(self):
        meta = ImageMetadata(
            image_id="t72_001",
            vehicle_class="T-72",
            azimuth=0.0,
            elevation=0.0,
            source_type="real",
            file_path="images/T-72/img001.png",
        )
        assert isinstance(meta.file_path, Path)

    def test_empty_image_id(self):
        with pytest.raises(ValueError, match="image_id must be a non-empty string"):
            ImageMetadata(
                image_id="",
                vehicle_class="T-72",
                azimuth=0.0,
                elevation=0.0,
                source_type="synthetic",
                file_path=Path("img.png"),
            )

    def test_invalid_vehicle_class(self):
        with pytest.raises(ValueError, match="Unknown vehicle class"):
            ImageMetadata(
                image_id="id1",
                vehicle_class="invalid",
                azimuth=0.0,
                elevation=0.0,
                source_type="synthetic",
                file_path=Path("img.png"),
            )

    def test_invalid_source_type(self):
        with pytest.raises(ValueError, match="source_type must be one of"):
            ImageMetadata(
                image_id="id1",
                vehicle_class="T-72",
                azimuth=0.0,
                elevation=0.0,
                source_type="generated",
                file_path=Path("img.png"),
            )

    def test_azimuth_out_of_range(self):
        with pytest.raises(ValueError, match="Azimuth must be in"):
            ImageMetadata(
                image_id="id1",
                vehicle_class="T-72",
                azimuth=360.0,
                elevation=0.0,
                source_type="synthetic",
                file_path=Path("img.png"),
            )

    def test_elevation_out_of_range(self):
        with pytest.raises(ValueError, match="Elevation must be in"):
            ImageMetadata(
                image_id="id1",
                vehicle_class="T-72",
                azimuth=0.0,
                elevation=95.0,
                source_type="synthetic",
                file_path=Path("img.png"),
            )


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_valid_result(self):
        result = ClassificationResult(vehicle_class="T-72", confidence=0.87)
        assert result.vehicle_class == "T-72"
        assert result.confidence == 0.87

    def test_confidence_zero(self):
        result = ClassificationResult(vehicle_class="T-72", confidence=0.0)
        assert result.confidence == 0.0

    def test_confidence_one(self):
        result = ClassificationResult(vehicle_class="T-72", confidence=1.0)
        assert result.confidence == 1.0

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="Confidence must be in"):
            ClassificationResult(vehicle_class="T-72", confidence=1.1)

    def test_confidence_negative(self):
        with pytest.raises(ValueError, match="Confidence must be in"):
            ClassificationResult(vehicle_class="T-72", confidence=-0.1)

    def test_invalid_vehicle_class(self):
        with pytest.raises(ValueError, match="Unknown vehicle class"):
            ClassificationResult(vehicle_class="unknown", confidence=0.5)


class TestSimilarityMatch:
    """Tests for SimilarityMatch dataclass."""

    def test_valid_match(self):
        match = SimilarityMatch(
            vehicle_class="T-72", similarity_score=0.92, image_id="t72_001"
        )
        assert match.vehicle_class == "T-72"
        assert match.similarity_score == 0.92
        assert match.image_id == "t72_001"

    def test_similarity_score_bounds(self):
        # -1 is valid (opposite embeddings)
        match = SimilarityMatch(
            vehicle_class="T-72", similarity_score=-1.0, image_id="id1"
        )
        assert match.similarity_score == -1.0

        match = SimilarityMatch(
            vehicle_class="T-72", similarity_score=1.0, image_id="id1"
        )
        assert match.similarity_score == 1.0

    def test_similarity_score_out_of_range(self):
        with pytest.raises(ValueError, match="Similarity score must be in"):
            SimilarityMatch(
                vehicle_class="T-72", similarity_score=1.1, image_id="id1"
            )

    def test_empty_image_id(self):
        with pytest.raises(ValueError, match="image_id must be a non-empty string"):
            SimilarityMatch(
                vehicle_class="T-72", similarity_score=0.5, image_id=""
            )

    def test_invalid_vehicle_class(self):
        with pytest.raises(ValueError, match="Unknown vehicle class"):
            SimilarityMatch(
                vehicle_class="invalid", similarity_score=0.5, image_id="id1"
            )


class TestRecognitionResult:
    """Tests for RecognitionResult dataclass."""

    def test_valid_result(self):
        classifications = [
            ClassificationResult(vehicle_class="T-72", confidence=0.87),
            ClassificationResult(vehicle_class="T-80", confidence=0.09),
            ClassificationResult(vehicle_class="BMP-3", confidence=0.03),
        ]
        similarity_matches = [
            SimilarityMatch(vehicle_class="T-72", similarity_score=0.92, image_id="id1"),
            SimilarityMatch(vehicle_class="T-72", similarity_score=0.88, image_id="id2"),
        ]
        result = RecognitionResult(
            classifications=classifications,
            similarity_matches=similarity_matches,
            low_confidence=False,
            no_match=False,
        )
        assert len(result.classifications) == 3
        assert len(result.similarity_matches) == 2
        assert result.low_confidence is False
        assert result.no_match is False

    def test_low_confidence_flag_true(self):
        classifications = [
            ClassificationResult(vehicle_class="T-72", confidence=0.4),
        ]
        result = RecognitionResult(
            classifications=classifications,
            similarity_matches=[],
            low_confidence=True,
            no_match=True,
        )
        assert result.low_confidence is True

    def test_low_confidence_flag_mismatch(self):
        classifications = [
            ClassificationResult(vehicle_class="T-72", confidence=0.8),
        ]
        with pytest.raises(ValueError, match="low_confidence should be"):
            RecognitionResult(
                classifications=classifications,
                similarity_matches=[],
                low_confidence=True,
                no_match=True,
            )

    def test_no_match_flag_true(self):
        similarity_matches = [
            SimilarityMatch(vehicle_class="T-72", similarity_score=0.2, image_id="id1"),
            SimilarityMatch(vehicle_class="T-80", similarity_score=0.1, image_id="id2"),
        ]
        result = RecognitionResult(
            classifications=[
                ClassificationResult(vehicle_class="T-72", confidence=0.3),
            ],
            similarity_matches=similarity_matches,
            low_confidence=True,
            no_match=True,
        )
        assert result.no_match is True

    def test_no_match_flag_mismatch(self):
        similarity_matches = [
            SimilarityMatch(vehicle_class="T-72", similarity_score=0.5, image_id="id1"),
        ]
        with pytest.raises(ValueError, match="no_match should be"):
            RecognitionResult(
                classifications=[
                    ClassificationResult(vehicle_class="T-72", confidence=0.8),
                ],
                similarity_matches=similarity_matches,
                low_confidence=False,
                no_match=True,
            )


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.base_model == "google/paligemma2-3b-pt-224"
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.quantization_bits == 4
        assert config.embedding_dim == 256
        assert config.num_classes == 5
        assert config.batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.num_epochs == 20
        assert config.triplet_margin == 0.3
        assert config.loss_weights == {"classification": 0.5, "metric": 0.5}
        assert config.max_vram_gb == 8.0

    def test_custom_values(self):
        config = TrainingConfig(
            lora_rank=8,
            batch_size=2,
            num_epochs=10,
        )
        assert config.lora_rank == 8
        assert config.batch_size == 2
        assert config.num_epochs == 10

    def test_invalid_lora_rank(self):
        with pytest.raises(ValueError, match="lora_rank must be > 0"):
            TrainingConfig(lora_rank=0)

    def test_invalid_quantization_bits(self):
        with pytest.raises(ValueError, match="quantization_bits must be 4 or 8"):
            TrainingConfig(quantization_bits=16)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            TrainingConfig(batch_size=0)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            TrainingConfig(learning_rate=-0.001)

    def test_invalid_loss_weights_missing_key(self):
        with pytest.raises(ValueError, match="loss_weights must contain"):
            TrainingConfig(loss_weights={"classification": 1.0})

    def test_empty_base_model(self):
        with pytest.raises(ValueError, match="base_model must be a non-empty string"):
            TrainingConfig(base_model="")

    def test_invalid_max_vram(self):
        with pytest.raises(ValueError, match="max_vram_gb must be > 0"):
            TrainingConfig(max_vram_gb=0.0)


class TestEpochMetrics:
    """Tests for EpochMetrics dataclass."""

    def test_valid_metrics(self):
        metrics = EpochMetrics(
            epoch=0,
            train_loss=1.5,
            val_loss=1.8,
            classification_accuracy=0.65,
            embedding_quality=0.2,
            vram_peak_gb=6.5,
            learning_rate=2e-4,
        )
        assert metrics.epoch == 0
        assert metrics.train_loss == 1.5
        assert metrics.classification_accuracy == 0.65

    def test_negative_epoch(self):
        with pytest.raises(ValueError, match="epoch must be >= 0"):
            EpochMetrics(
                epoch=-1,
                train_loss=1.0,
                val_loss=1.0,
                classification_accuracy=0.5,
                embedding_quality=0.1,
                vram_peak_gb=5.0,
                learning_rate=1e-4,
            )

    def test_negative_train_loss(self):
        with pytest.raises(ValueError, match="train_loss must be >= 0"):
            EpochMetrics(
                epoch=0,
                train_loss=-0.1,
                val_loss=1.0,
                classification_accuracy=0.5,
                embedding_quality=0.1,
                vram_peak_gb=5.0,
                learning_rate=1e-4,
            )

    def test_accuracy_out_of_range(self):
        with pytest.raises(ValueError, match="classification_accuracy must be in"):
            EpochMetrics(
                epoch=0,
                train_loss=1.0,
                val_loss=1.0,
                classification_accuracy=1.5,
                embedding_quality=0.1,
                vram_peak_gb=5.0,
                learning_rate=1e-4,
            )

    def test_negative_vram(self):
        with pytest.raises(ValueError, match="vram_peak_gb must be >= 0"):
            EpochMetrics(
                epoch=0,
                train_loss=1.0,
                val_loss=1.0,
                classification_accuracy=0.5,
                embedding_quality=0.1,
                vram_peak_gb=-1.0,
                learning_rate=1e-4,
            )

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            EpochMetrics(
                epoch=0,
                train_loss=1.0,
                val_loss=1.0,
                classification_accuracy=0.5,
                embedding_quality=0.1,
                vram_peak_gb=5.0,
                learning_rate=0.0,
            )
