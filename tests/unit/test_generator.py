"""Unit tests for the IRGenerator class.

Tests single image generation, batch generation, and input validation.
Requirements: 1.2, 1.4, 1.5
"""

import numpy as np
import pytest

from ir_recognition.ir_generator import IRGenerator
from ir_recognition.models import IRGeneratorConfig, SUPPORTED_VEHICLE_CLASSES


class TestIRGeneratorGenerate:
    """Tests for IRGenerator.generate() method."""

    def test_generate_returns_correct_shape(self):
        gen = IRGenerator(seed=42)
        config = IRGeneratorConfig(vehicle_type="T-72", azimuth=45.0, elevation=30.0)
        img = gen.generate(config)
        assert img.shape == (224, 224)

    def test_generate_returns_float32(self):
        gen = IRGenerator(seed=42)
        config = IRGeneratorConfig(vehicle_type="T-72", azimuth=0.0, elevation=0.0)
        img = gen.generate(config)
        assert img.dtype == np.float32

    def test_generate_values_in_zero_one(self):
        gen = IRGenerator(seed=42)
        config = IRGeneratorConfig(vehicle_type="BMP-3", azimuth=90.0, elevation=45.0)
        img = gen.generate(config)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_generate_non_zero_variance(self):
        gen = IRGenerator(seed=42)
        config = IRGeneratorConfig(vehicle_type="CV90", azimuth=180.0, elevation=60.0)
        img = gen.generate(config)
        assert img.var() > 0

    def test_generate_custom_image_size(self):
        gen = IRGenerator(seed=42)
        config = IRGeneratorConfig(
            vehicle_type="T-80", azimuth=0.0, elevation=0.0, image_size=(256, 256)
        )
        img = gen.generate(config)
        assert img.shape == (256, 256)

    @pytest.mark.parametrize("vehicle_type", SUPPORTED_VEHICLE_CLASSES)
    def test_generate_all_vehicle_classes(self, vehicle_type):
        gen = IRGenerator(seed=42)
        config = IRGeneratorConfig(vehicle_type=vehicle_type, azimuth=90.0, elevation=30.0)
        img = gen.generate(config)
        assert img.shape == (224, 224)
        assert img.dtype == np.float32
        assert 0.0 <= img.min() and img.max() <= 1.0

    def test_generate_boundary_azimuth_zero(self):
        gen = IRGenerator(seed=42)
        config = IRGeneratorConfig(vehicle_type="T-72", azimuth=0.0, elevation=30.0)
        img = gen.generate(config)
        assert img.shape == (224, 224)

    def test_generate_boundary_azimuth_near_360(self):
        gen = IRGenerator(seed=42)
        config = IRGeneratorConfig(vehicle_type="T-72", azimuth=359.0, elevation=30.0)
        img = gen.generate(config)
        assert img.shape == (224, 224)

    def test_generate_boundary_elevation_zero(self):
        gen = IRGenerator(seed=42)
        config = IRGeneratorConfig(vehicle_type="T-72", azimuth=45.0, elevation=0.0)
        img = gen.generate(config)
        assert img.shape == (224, 224)

    def test_generate_boundary_elevation_90(self):
        gen = IRGenerator(seed=42)
        config = IRGeneratorConfig(vehicle_type="T-72", azimuth=45.0, elevation=90.0)
        img = gen.generate(config)
        assert img.shape == (224, 224)

    def test_different_angles_produce_different_images(self):
        gen = IRGenerator(seed=42)
        config1 = IRGeneratorConfig(vehicle_type="T-72", azimuth=0.0, elevation=30.0)
        config2 = IRGeneratorConfig(vehicle_type="T-72", azimuth=180.0, elevation=30.0)
        img1 = gen.generate(config1)
        img2 = gen.generate(config2)
        # Images from front and rear should differ
        assert not np.allclose(img1, img2)


class TestIRGeneratorGenerateBatch:
    """Tests for IRGenerator.generate_batch() method."""

    def test_batch_returns_correct_count(self):
        gen = IRGenerator(seed=42)
        batch = gen.generate_batch("T-72", num_angles=8)
        assert len(batch) == 8

    def test_batch_default_num_angles(self):
        gen = IRGenerator(seed=42)
        batch = gen.generate_batch("T-72")
        assert len(batch) == 8

    def test_batch_evenly_spaced_azimuths(self):
        gen = IRGenerator(seed=42)
        batch = gen.generate_batch("T-72", num_angles=8)
        azimuths = [meta["azimuth"] for _, meta in batch]
        expected = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        assert azimuths == expected

    def test_batch_4_angles(self):
        gen = IRGenerator(seed=42)
        batch = gen.generate_batch("T-72", num_angles=4)
        azimuths = [meta["azimuth"] for _, meta in batch]
        expected = [0.0, 90.0, 180.0, 270.0]
        assert azimuths == expected

    def test_batch_metadata_contains_required_keys(self):
        gen = IRGenerator(seed=42)
        batch = gen.generate_batch("BMP-3", num_angles=4)
        for _, meta in batch:
            assert "vehicle_type" in meta
            assert "azimuth" in meta
            assert "elevation" in meta
            assert "ambient_temp" in meta

    def test_batch_metadata_vehicle_type(self):
        gen = IRGenerator(seed=42)
        batch = gen.generate_batch("CV90", num_angles=4)
        for _, meta in batch:
            assert meta["vehicle_type"] == "CV90"

    def test_batch_metadata_elevation(self):
        gen = IRGenerator(seed=42)
        batch = gen.generate_batch("T-72", num_angles=4, elevation=45.0)
        for _, meta in batch:
            assert meta["elevation"] == 45.0

    def test_batch_metadata_ambient_temp(self):
        gen = IRGenerator(seed=42)
        batch = gen.generate_batch("T-72", num_angles=4, ambient_temp=300.0)
        for _, meta in batch:
            assert meta["ambient_temp"] == 300.0

    def test_batch_images_are_valid(self):
        gen = IRGenerator(seed=42)
        batch = gen.generate_batch("T-80", num_angles=4)
        for img, _ in batch:
            assert img.shape == (224, 224)
            assert img.dtype == np.float32
            assert 0.0 <= img.min() and img.max() <= 1.0
            assert img.var() > 0

    def test_batch_distinct_azimuths(self):
        gen = IRGenerator(seed=42)
        batch = gen.generate_batch("T-72", num_angles=8)
        azimuths = [meta["azimuth"] for _, meta in batch]
        # All azimuths should be distinct
        assert len(set(azimuths)) == 8


class TestIRGeneratorValidation:
    """Tests for input validation in IRGenerator."""

    def test_generate_invalid_vehicle_type(self):
        gen = IRGenerator(seed=42)
        with pytest.raises(ValueError, match="Unknown vehicle class"):
            IRGeneratorConfig(vehicle_type="unknown", azimuth=0.0, elevation=0.0)

    def test_generate_invalid_azimuth_negative(self):
        gen = IRGenerator(seed=42)
        with pytest.raises(ValueError, match="Azimuth"):
            IRGeneratorConfig(vehicle_type="T-72", azimuth=-1.0, elevation=0.0)

    def test_generate_invalid_azimuth_360(self):
        gen = IRGenerator(seed=42)
        with pytest.raises(ValueError, match="Azimuth"):
            IRGeneratorConfig(vehicle_type="T-72", azimuth=360.0, elevation=0.0)

    def test_generate_invalid_elevation_negative(self):
        gen = IRGenerator(seed=42)
        with pytest.raises(ValueError, match="Elevation"):
            IRGeneratorConfig(vehicle_type="T-72", azimuth=0.0, elevation=-1.0)

    def test_generate_invalid_elevation_over_90(self):
        gen = IRGenerator(seed=42)
        with pytest.raises(ValueError, match="Elevation"):
            IRGeneratorConfig(vehicle_type="T-72", azimuth=0.0, elevation=91.0)

    def test_batch_invalid_vehicle_type(self):
        gen = IRGenerator(seed=42)
        with pytest.raises(ValueError, match="Unknown vehicle class"):
            gen.generate_batch("invalid_vehicle")

    def test_batch_invalid_num_angles_zero(self):
        gen = IRGenerator(seed=42)
        with pytest.raises(ValueError, match="num_angles must be >= 1"):
            gen.generate_batch("T-72", num_angles=0)

    def test_batch_invalid_num_angles_negative(self):
        gen = IRGenerator(seed=42)
        with pytest.raises(ValueError, match="num_angles must be >= 1"):
            gen.generate_batch("T-72", num_angles=-1)

    def test_batch_invalid_elevation(self):
        gen = IRGenerator(seed=42)
        with pytest.raises(ValueError, match="Elevation"):
            gen.generate_batch("T-72", elevation=100.0)
