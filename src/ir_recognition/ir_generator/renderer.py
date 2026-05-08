"""IR rendering engine for projecting thermal profiles into 2D images.

Takes a ThermalProfile and viewing parameters (azimuth, elevation) and produces
a 2D grayscale IR image simulating what an IRST sensor would observe.

The rendering pipeline:
1. Project 3D heat zones onto a 2D image plane based on viewing angle
2. Apply Gaussian blur for thermal diffusion simulation
3. Add sensor noise (Gaussian noise with configurable SNR)
4. Normalize output to [0, 1] float32 range

Requirements: 1.1, 1.2, 1.5
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from ir_recognition.ir_generator.thermal_profiles import HeatZone, ThermalProfile


def render_ir_image(
    profile: ThermalProfile,
    azimuth: float,
    elevation: float,
    ambient_temp: float = 293.0,
    image_size: tuple[int, int] = (224, 224),
    blur_sigma: float = 3.0,
    snr_db: float = 25.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render a synthetic IR image from a thermal profile at a given viewing angle.

    Projects the 3D heat zones of the thermal profile onto a 2D image plane
    based on the azimuth and elevation angles, applies Gaussian blur for
    thermal diffusion, adds sensor noise, and normalizes to [0, 1].

    Args:
        profile: ThermalProfile defining the vehicle's heat zones.
        azimuth: Viewing azimuth angle in degrees [0, 360).
                 0° = front, 90° = right side, 180° = rear, 270° = left side.
        elevation: Viewing elevation angle in degrees [0, 90].
                   0° = ground level, 90° = directly above.
        ambient_temp: Ambient temperature in Kelvin (must be > 0). Default 293.0.
        image_size: Output image dimensions (height, width). Default (224, 224).
        blur_sigma: Standard deviation for Gaussian blur (thermal diffusion).
                    Default 3.0.
        snr_db: Signal-to-noise ratio in decibels for sensor noise. Default 25.0.
        rng: Optional numpy random Generator for reproducible noise.

    Returns:
        A numpy array of shape (H, W) with dtype float32 and values in [0, 1].
        The image will have non-zero variance (not blank).

    Raises:
        ValueError: If azimuth, elevation, or ambient_temp are out of range.
    """
    if not (0.0 <= azimuth < 360.0):
        raise ValueError(f"Azimuth must be in [0, 360), got {azimuth}")
    if not (0.0 <= elevation <= 90.0):
        raise ValueError(f"Elevation must be in [0, 90], got {elevation}")
    if ambient_temp <= 0:
        raise ValueError(f"Ambient temperature must be > 0 Kelvin, got {ambient_temp}")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError(f"image_size must have positive dimensions, got {image_size}")

    if rng is None:
        rng = np.random.default_rng()

    height, width = image_size

    # Step 1: Create the raw thermal image by projecting heat zones
    image = _project_heat_zones(profile, azimuth, elevation, height, width)

    # Step 2: Apply Gaussian blur for thermal diffusion simulation
    if blur_sigma > 0:
        image = gaussian_filter(image, sigma=blur_sigma)

    # Step 3: Add sensor noise
    image = _add_sensor_noise(image, snr_db, rng)

    # Step 4: Normalize to [0, 1] float32
    image = _normalize_image(image)

    return image


def _project_heat_zones(
    profile: ThermalProfile,
    azimuth: float,
    elevation: float,
    height: int,
    width: int,
) -> np.ndarray:
    """Project 3D heat zones onto a 2D image plane.

    The coordinate system for heat zones:
      - x: 0.0 (front) to 1.0 (rear) along vehicle longitudinal axis
      - y: 0.0 (top) to 1.0 (bottom) in vertical plane
      - z: 0.0 (left) to 1.0 (right) across vehicle width

    The viewing angle determines how these 3D zones map to 2D:
      - Azimuth rotates around the vertical axis (y)
      - Elevation tilts the view from ground level (0°) to top-down (90°)

    Args:
        profile: ThermalProfile with heat zones.
        azimuth: Viewing azimuth in degrees.
        elevation: Viewing elevation in degrees.
        height: Output image height in pixels.
        width: Output image width in pixels.

    Returns:
        Raw thermal image as float64 array (not yet normalized).
    """
    image = np.zeros((height, width), dtype=np.float64)

    # Convert angles to radians
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)

    # Precompute trigonometric values
    cos_az = math.cos(az_rad)
    sin_az = math.sin(az_rad)
    cos_el = math.cos(el_rad)
    sin_el = math.sin(el_rad)

    for zone in profile.zones:
        # Project the 3D zone center to 2D image coordinates
        proj_u, proj_v = _project_point(
            zone.center_x, zone.center_y, zone.center_z,
            cos_az, sin_az, cos_el, sin_el,
        )

        # Project the zone extents to determine 2D size
        proj_width, proj_height = _project_zone_extent(
            zone.length, zone.height, zone.width,
            cos_az, sin_az, cos_el, sin_el,
        )

        # Compute the effective temperature contribution
        effective_intensity = zone.temp_delta * zone.intensity

        # Draw the zone as an elliptical Gaussian blob on the image
        _draw_zone(
            image, proj_u, proj_v, proj_width, proj_height,
            effective_intensity, height, width,
        )

    return image


def _project_point(
    x: float, y: float, z: float,
    cos_az: float, sin_az: float,
    cos_el: float, sin_el: float,
) -> tuple[float, float]:
    """Project a 3D normalized point to 2D image coordinates [0, 1].

    The projection applies:
    1. Azimuth rotation around the vertical (y) axis
    2. Elevation tilt to mix vertical and depth information

    Returns (u, v) in [0, 1] range representing horizontal and vertical
    position on the image plane.
    """
    # Center coordinates around origin for rotation
    cx = x - 0.5
    cz = z - 0.5
    cy = y - 0.5

    # Rotate around vertical axis (azimuth)
    # Azimuth 0° = viewing from front (looking along +x axis)
    # Azimuth 90° = viewing from right side (looking along -z axis)
    rotated_depth = cx * cos_az + cz * sin_az
    rotated_horiz = -cx * sin_az + cz * cos_az

    # Apply elevation: mix vertical and depth
    # At elevation 0° (ground level), we see the side view
    # At elevation 90° (top-down), we see the top view
    projected_v = cy * cos_el - rotated_depth * sin_el
    # projected_depth = cy * sin_el + rotated_depth * cos_el  # not used for 2D

    # Map back to [0, 1] range
    u = rotated_horiz + 0.5
    v = projected_v + 0.5

    return u, v


def _project_zone_extent(
    length: float, zone_height: float, zone_width: float,
    cos_az: float, sin_az: float,
    cos_el: float, sin_el: float,
) -> tuple[float, float]:
    """Compute the projected 2D extent of a 3D zone.

    Returns (projected_width, projected_height) as fractions of image size.
    """
    # The visible width depends on the azimuth angle:
    # - From front/rear (az=0/180): we see the width (z-extent)
    # - From the side (az=90/270): we see the length (x-extent)
    proj_w = abs(zone_width * cos_az) + abs(length * abs(sin_az))

    # The visible height depends on elevation:
    # - At ground level (el=0): we see the full height
    # - At top-down (el=90): we see the length/width as "height"
    proj_h = abs(zone_height * cos_el) + abs(length * sin_el * abs(cos_az)) + abs(zone_width * sin_el * abs(sin_az))

    # Ensure minimum visible size
    proj_w = max(proj_w, 0.03)
    proj_h = max(proj_h, 0.03)

    return proj_w, proj_h


def _draw_zone(
    image: np.ndarray,
    center_u: float,
    center_v: float,
    zone_width: float,
    zone_height: float,
    intensity: float,
    img_height: int,
    img_width: int,
) -> None:
    """Draw a heat zone as an elliptical Gaussian blob on the image.

    Args:
        image: The image array to draw on (modified in-place).
        center_u: Horizontal center position [0, 1].
        center_v: Vertical center position [0, 1].
        zone_width: Width of the zone as fraction of image width.
        zone_height: Height of the zone as fraction of image height.
        intensity: Peak intensity value for this zone.
        img_height: Image height in pixels.
        img_width: Image width in pixels.
    """
    # Convert normalized coordinates to pixel coordinates
    cx_px = center_u * img_width
    cy_px = center_v * img_height

    # Convert zone size to pixel sigma (zone extent ~ 2 sigma)
    sigma_x = max(zone_width * img_width / 4.0, 1.0)
    sigma_y = max(zone_height * img_height / 4.0, 1.0)

    # Determine the bounding box for efficient computation
    # Only compute within 3 sigma of center
    x_min = max(0, int(cx_px - 3 * sigma_x))
    x_max = min(img_width, int(cx_px + 3 * sigma_x) + 1)
    y_min = max(0, int(cy_px - 3 * sigma_y))
    y_max = min(img_height, int(cy_px + 3 * sigma_y) + 1)

    if x_min >= x_max or y_min >= y_max:
        return

    # Create coordinate grids for the bounding box
    y_coords = np.arange(y_min, y_max, dtype=np.float64)
    x_coords = np.arange(x_min, x_max, dtype=np.float64)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    # Compute 2D Gaussian
    dx = (xx - cx_px) / sigma_x
    dy = (yy - cy_px) / sigma_y
    gaussian = np.exp(-0.5 * (dx * dx + dy * dy))

    # Add the zone contribution (additive blending)
    image[y_min:y_max, x_min:x_max] += intensity * gaussian


def _add_sensor_noise(
    image: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian sensor noise to the image.

    The noise level is determined by the signal-to-noise ratio (SNR) in dB.
    SNR_dB = 10 * log10(signal_power / noise_power)

    Args:
        image: Input image array.
        snr_db: Signal-to-noise ratio in decibels.
        rng: Random number generator for reproducibility.

    Returns:
        Image with added noise.
    """
    signal_power = np.mean(image ** 2)

    if signal_power < 1e-10:
        # If signal is essentially zero, add minimal noise to ensure non-zero variance
        noise_std = 0.01
    else:
        # Convert SNR from dB to linear scale
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = math.sqrt(noise_power)

    noise = rng.normal(0.0, noise_std, size=image.shape)
    return image + noise


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] float32 range.

    Ensures the output has non-zero variance by adding minimal noise
    if the image is completely uniform.

    Args:
        image: Input image array.

    Returns:
        Normalized float32 array with values in [0, 1] and non-zero variance.
    """
    img_min = image.min()
    img_max = image.max()

    if img_max - img_min < 1e-10:
        # Image is essentially uniform; create a minimal gradient to ensure
        # non-zero variance while keeping values in [0, 1]
        result = np.full(image.shape, 0.5, dtype=np.float32)
        # Add a tiny gradient
        h, w = image.shape
        gradient = np.linspace(0.0, 0.01, w, dtype=np.float32)
        result += gradient[np.newaxis, :]
        return result

    # Min-max normalization to [0, 1]
    result = (image - img_min) / (img_max - img_min)
    return result.astype(np.float32)
