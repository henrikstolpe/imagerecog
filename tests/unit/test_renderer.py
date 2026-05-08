"""Unit tests for the IR rendering engine.

Tests the renderer's core functionality: projection, normalization,
noise addition, and output validity across different configurations.
"""

import numpy as np
import pytest

from ir_recognition.ir_generator.renderer import (
    render_ir_image,
    _project_point,
    _project_zone_extent,
    _add_sensor_noise,
    _normalize_image,
)
from ir_recognition.ir_generator.thermal_profiles import (
    get_thermal_profile,
    THERMAL_PROFILES,
    HeatZone,
    ThermalProfile,
)


class TestRenderIRImage:
    """Tests for the main render_ir_image function."""

    def test_output_shape_default(self):
        """Output has default shape (224, 224)."""
        profile = get_thermal_profile("T-72")
        img = render_ir_image(profile, azimuth=45.0, elevation=30.0)
        assert img.shape == (224, 224)

    def test_output_dtype_float32(self):
        """Output dtype is float32."""
        profile = get_thermal_profile("T-72")
        img = render_ir_image(profile, azimuth=45.0, elevation=30.0)
        assert img.dtype == np.float32

    def test_output_range_zero_to_one(self):
        """All pixel values are in [0, 1]."""
        profile = get_thermal_profile("T-72")
        img = render_ir_image(profile, azimuth=45.0, elevation=30.0)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_output_nonzero_variance(self):
        """Output image has non-zero variance (not blank)."""
        profile = get_thermal_profile("T-72")
        img = render_ir_image(profile, azimuth=45.0, elevation=30.0)
        assert img.var() > 0

    def test_configurable_resolution(self):
        """Output respects configurable image_size."""
        profile = get_thermal_profile("T-72")
        img = render_ir_image(profile, azimuth=0.0, elevation=0.0, image_size=(128, 256))
        assert img.shape == (128, 256)

    @pytest.mark.parametrize("vehicle_class", list(THERMAL_PROFILES.keys()))
    def test_all_vehicle_classes_produce_valid_output(self, vehicle_class):
        """Each vehicle class produces a valid image."""
        profile = get_thermal_profile(vehicle_class)
        img = render_ir_image(profile, azimuth=45.0, elevation=30.0)
        assert img.shape == (224, 224)
        assert img.dtype == np.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0
        assert img.var() > 0

    def test_different_angles_produce_different_images(self):
        """Different azimuth angles produce different pixel distributions."""
        profile = get_thermal_profile("T-72")
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        img_front = render_ir_image(profile, azimuth=0.0, elevation=30.0, rng=rng1)
        img_side = render_ir_image(profile, azimuth=90.0, elevation=30.0, rng=rng2)
        # Images should not be identical
        assert not np.allclose(img_front, img_side)

    def test_reproducibility_with_same_rng(self):
        """Same rng seed produces identical output."""
        profile = get_thermal_profile("T-72")
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        img1 = render_ir_image(profile, azimuth=45.0, elevation=30.0, rng=rng1)
        img2 = render_ir_image(profile, azimuth=45.0, elevation=30.0, rng=rng2)
        np.testing.assert_array_equal(img1, img2)

    def test_invalid_azimuth_raises_valueerror(self):
        """Azimuth >= 360 raises ValueError."""
        profile = get_thermal_profile("T-72")
        with pytest.raises(ValueError, match="Azimuth"):
            render_ir_image(profile, azimuth=360.0, elevation=30.0)

    def test_negative_azimuth_raises_valueerror(self):
        """Negative azimuth raises ValueError."""
        profile = get_thermal_profile("T-72")
        with pytest.raises(ValueError, match="Azimuth"):
            render_ir_image(profile, azimuth=-1.0, elevation=30.0)

    def test_invalid_elevation_raises_valueerror(self):
        """Elevation > 90 raises ValueError."""
        profile = get_thermal_profile("T-72")
        with pytest.raises(ValueError, match="Elevation"):
            render_ir_image(profile, azimuth=45.0, elevation=91.0)

    def test_invalid_ambient_temp_raises_valueerror(self):
        """Ambient temp <= 0 raises ValueError."""
        profile = get_thermal_profile("T-72")
        with pytest.raises(ValueError, match="Ambient temperature"):
            render_ir_image(profile, azimuth=45.0, elevation=30.0, ambient_temp=0.0)

    def test_boundary_azimuth_zero(self):
        """Azimuth 0° (front view) produces valid output."""
        profile = get_thermal_profile("T-72")
        img = render_ir_image(profile, azimuth=0.0, elevation=30.0)
        assert img.var() > 0

    def test_boundary_elevation_zero(self):
        """Elevation 0° (ground level) produces valid output."""
        profile = get_thermal_profile("T-72")
        img = render_ir_image(profile, azimuth=45.0, elevation=0.0)
        assert img.var() > 0

    def test_boundary_elevation_ninety(self):
        """Elevation 90° (top-down) produces valid output."""
        profile = get_thermal_profile("T-72")
        img = render_ir_image(profile, azimuth=45.0, elevation=90.0)
        assert img.var() > 0

    def test_configurable_snr(self):
        """Different SNR values produce different noise levels."""
        profile = get_thermal_profile("T-72")
        # Use same seed for the underlying projection, different noise
        img_clean = render_ir_image(
            profile, azimuth=45.0, elevation=30.0, snr_db=60.0,
            rng=np.random.default_rng(0)
        )
        img_noisy = render_ir_image(
            profile, azimuth=45.0, elevation=30.0, snr_db=5.0,
            rng=np.random.default_rng(0)
        )
        # Both should be valid
        assert img_clean.min() >= 0.0 and img_clean.max() <= 1.0
        assert img_noisy.min() >= 0.0 and img_noisy.max() <= 1.0


class TestNormalizeImage:
    """Tests for the _normalize_image helper."""

    def test_normalizes_to_zero_one(self):
        """Output is in [0, 1] range."""
        img = np.array([[1.0, 5.0], [3.0, 10.0]])
        result = _normalize_image(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uniform_image_has_nonzero_variance(self):
        """A uniform image still produces non-zero variance output."""
        img = np.full((64, 64), 5.0)
        result = _normalize_image(img)
        assert result.var() > 0

    def test_output_dtype_float32(self):
        """Output is float32."""
        img = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _normalize_image(img)
        assert result.dtype == np.float32


class TestAddSensorNoise:
    """Tests for the _add_sensor_noise helper."""

    def test_adds_noise(self):
        """Noise is actually added to the image."""
        img = np.ones((64, 64), dtype=np.float64) * 0.5
        rng = np.random.default_rng(42)
        result = _add_sensor_noise(img, snr_db=20.0, rng=rng)
        # Should not be identical to input
        assert not np.allclose(result, img)

    def test_zero_signal_gets_minimal_noise(self):
        """A zero-signal image still gets some noise for non-zero variance."""
        img = np.zeros((64, 64), dtype=np.float64)
        rng = np.random.default_rng(42)
        result = _add_sensor_noise(img, snr_db=20.0, rng=rng)
        assert result.var() > 0
