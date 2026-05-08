"""IRGenerator class for producing synthetic IR images.

Provides a high-level interface for generating single IR images from a
configuration object, or batches of images across evenly-spaced azimuth angles.

Requirements: 1.2, 1.4, 1.5
"""

from __future__ import annotations

import numpy as np

from ir_recognition.models import IRGeneratorConfig, SUPPORTED_VEHICLE_CLASSES
from ir_recognition.ir_generator.renderer import render_ir_image
from ir_recognition.ir_generator.thermal_profiles import get_thermal_profile


class IRGenerator:
    """High-level generator for synthetic IR vehicle images.

    Uses thermal profiles and the IR rendering engine to produce synthetic
    infrared images of vehicles at specified viewing angles and conditions.

    Example usage::

        generator = IRGenerator()
        config = IRGeneratorConfig(vehicle_type="T-72", azimuth=45.0, elevation=30.0)
        image = generator.generate(config)

        # Generate a batch across 8 evenly-spaced azimuths
        batch = generator.generate_batch("T-72", num_angles=8)
    """

    def __init__(
        self,
        blur_sigma: float = 3.0,
        snr_db: float = 25.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the IR generator.

        Args:
            blur_sigma: Standard deviation for Gaussian blur (thermal diffusion).
                Default 3.0.
            snr_db: Signal-to-noise ratio in decibels for sensor noise. Default 25.0.
            seed: Optional random seed for reproducible image generation.
        """
        self.blur_sigma = blur_sigma
        self.snr_db = snr_db
        self._rng = np.random.default_rng(seed)

    def generate(self, config: IRGeneratorConfig) -> np.ndarray:
        """Generate a single synthetic IR image from a configuration.

        Looks up the thermal profile for the specified vehicle type and renders
        an IR image at the given viewing angle and ambient conditions.

        Args:
            config: An IRGeneratorConfig specifying vehicle type, viewing angle,
                ambient temperature, and image size.

        Returns:
            A numpy array of shape (H, W) with dtype float32 and values in [0, 1].

        Raises:
            ValueError: If the vehicle type is not supported, azimuth is not in
                [0, 360), or elevation is not in [0, 90].
        """
        # Validate parameters (IRGeneratorConfig.__post_init__ already validates,
        # but we re-check here for clarity and in case a raw config is passed)
        self._validate_vehicle_type(config.vehicle_type)
        self._validate_azimuth(config.azimuth)
        self._validate_elevation(config.elevation)

        # Look up the thermal profile for this vehicle class
        profile = get_thermal_profile(config.vehicle_type)

        # Render the IR image
        image = render_ir_image(
            profile=profile,
            azimuth=config.azimuth,
            elevation=config.elevation,
            ambient_temp=config.ambient_temp,
            image_size=config.image_size,
            blur_sigma=self.blur_sigma,
            snr_db=self.snr_db,
            rng=self._rng,
        )

        return image

    def generate_batch(
        self,
        vehicle_type: str,
        num_angles: int = 8,
        elevation: float = 30.0,
        ambient_temp: float = 293.0,
        image_size: tuple[int, int] = (224, 224),
    ) -> list[tuple[np.ndarray, dict]]:
        """Generate images for a vehicle across evenly-spaced azimuth angles.

        Produces ``num_angles`` images at evenly-spaced azimuth values spanning
        the full 360° range. For example, with ``num_angles=8``, the azimuths
        will be 0, 45, 90, 135, 180, 225, 270, 315 degrees.

        Args:
            vehicle_type: Vehicle class to generate (must be in SUPPORTED_VEHICLE_CLASSES).
            num_angles: Number of evenly-spaced azimuth angles to generate.
                Must be >= 1. Default 8.
            elevation: Viewing elevation angle in degrees [0, 90]. Default 30.0.
            ambient_temp: Ambient temperature in Kelvin (must be > 0). Default 293.0.
            image_size: Output image dimensions (height, width). Default (224, 224).

        Returns:
            A list of tuples ``(image, metadata)`` where:
            - ``image`` is a numpy array of shape (H, W), dtype float32, values in [0, 1]
            - ``metadata`` is a dict with keys: vehicle_type, azimuth, elevation, ambient_temp

        Raises:
            ValueError: If vehicle_type is not supported, elevation is out of range,
                or num_angles < 1.
        """
        self._validate_vehicle_type(vehicle_type)
        self._validate_elevation(elevation)

        if num_angles < 1:
            raise ValueError(f"num_angles must be >= 1, got {num_angles}")

        # Compute evenly-spaced azimuth angles
        azimuths = [360.0 * i / num_angles for i in range(num_angles)]

        results: list[tuple[np.ndarray, dict]] = []
        for azimuth in azimuths:
            config = IRGeneratorConfig(
                vehicle_type=vehicle_type,
                azimuth=azimuth,
                elevation=elevation,
                ambient_temp=ambient_temp,
                image_size=image_size,
            )
            image = self.generate(config)
            metadata = {
                "vehicle_type": vehicle_type,
                "azimuth": azimuth,
                "elevation": elevation,
                "ambient_temp": ambient_temp,
            }
            results.append((image, metadata))

        return results

    @staticmethod
    def _validate_vehicle_type(vehicle_type: str) -> None:
        """Validate that the vehicle type is supported."""
        if vehicle_type not in SUPPORTED_VEHICLE_CLASSES:
            raise ValueError(
                f"Unknown vehicle class: '{vehicle_type}'. "
                f"Supported: {SUPPORTED_VEHICLE_CLASSES}"
            )

    @staticmethod
    def _validate_azimuth(azimuth: float) -> None:
        """Validate that azimuth is in [0, 360)."""
        if not (0.0 <= azimuth < 360.0):
            raise ValueError(f"Azimuth must be in [0, 360), got {azimuth}")

    @staticmethod
    def _validate_elevation(elevation: float) -> None:
        """Validate that elevation is in [0, 90]."""
        if not (0.0 <= elevation <= 90.0):
            raise ValueError(f"Elevation must be in [0, 90], got {elevation}")
