"""Image preprocessing for the inference pipeline.

Handles loading, validation, resizing, and normalization of input images
for the IR Signature Recognition model. Matches the training preprocessing
pipeline to ensure consistent model behavior.

Preprocessing pipeline:
    1. Load image from file (PNG, JPEG, TIFF) or accept numpy array
    2. Validate input (file exists, readable, valid format)
    3. Resize to 224×224 using bilinear interpolation
    4. Convert to float tensor in [0, 1]
    5. Replicate grayscale to 3 channels
    6. Apply ImageNet-style normalization

Error handling:
    Returns (None, error_message) for invalid inputs instead of raising exceptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

# ImageNet normalization constants (matches training preprocessing)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Target image size for model input
TARGET_SIZE = (224, 224)

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def preprocess_image(
    image_input: Union[str, Path, np.ndarray],
    target_size: tuple[int, int] = TARGET_SIZE,
    normalize: bool = True,
) -> tuple[torch.Tensor | None, str | None]:
    """Preprocess an image for model inference.

    Accepts either a file path (str or Path) to a PNG, JPEG, or TIFF image,
    or a numpy array. Returns a preprocessed tensor ready for model input,
    or an error message if the input is invalid.

    Args:
        image_input: Either a file path (str/Path) to an image file,
            or a numpy array (HxW or HxWxC).
        target_size: Target dimensions (height, width). Default (224, 224).
        normalize: Whether to apply ImageNet normalization. Default True.

    Returns:
        A tuple of (tensor, error) where:
        - On success: (tensor of shape (1, 3, 224, 224), None)
        - On failure: (None, descriptive error message string)
    """
    if isinstance(image_input, (str, Path)):
        return _preprocess_from_file(Path(image_input), target_size, normalize)
    elif isinstance(image_input, np.ndarray):
        return _preprocess_from_array(image_input, target_size, normalize)
    else:
        return None, (
            f"Unsupported input type: {type(image_input).__name__}. "
            "Expected file path (str/Path) or numpy array."
        )


def _preprocess_from_file(
    file_path: Path,
    target_size: tuple[int, int],
    normalize: bool,
) -> tuple[torch.Tensor | None, str | None]:
    """Load and preprocess an image from a file path.

    Validates that the file exists, has a supported extension, and is a
    valid image before preprocessing.

    Args:
        file_path: Path to the image file.
        target_size: Target dimensions (height, width).
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Tuple of (tensor, None) on success or (None, error_message) on failure.
    """
    # Check file exists
    if not file_path.exists():
        return None, f"File not found: {file_path}"

    # Check file is not a directory
    if file_path.is_dir():
        return None, f"Path is a directory, not a file: {file_path}"

    # Check file extension
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None, (
            f"Unsupported image format '{ext}'. "
            f"Accepted: PNG, JPEG, TIFF"
        )

    # Check file is readable and non-empty
    try:
        file_size = file_path.stat().st_size
        if file_size == 0:
            return None, f"Unable to read image file: file is empty ({file_path})"
    except OSError as e:
        return None, f"Unable to read image file: {e}"

    # Attempt to open and decode the image
    try:
        pil_image = Image.open(file_path)
        # Force full decode to catch truncated/corrupted files
        pil_image.load()
    except UnidentifiedImageError:
        return None, (
            f"Unable to read image file: not a valid image ({file_path})"
        )
    except (OSError, SyntaxError) as e:
        return None, f"Unable to read image file: {e}"
    except Exception as e:
        return None, f"Unable to read image file: unexpected error - {e}"

    # Convert PIL image to tensor
    return _pil_to_tensor(pil_image, target_size, normalize)


def _preprocess_from_array(
    array: np.ndarray,
    target_size: tuple[int, int],
    normalize: bool,
) -> tuple[torch.Tensor | None, str | None]:
    """Preprocess a numpy array image.

    Accepts arrays of shape (H, W) for grayscale or (H, W, C) for color.
    Values can be in [0, 1] (float) or [0, 255] (uint8).

    Args:
        array: Numpy array representing the image.
        target_size: Target dimensions (height, width).
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Tuple of (tensor, None) on success or (None, error_message) on failure.
    """
    # Validate array dimensions
    if array.ndim < 2 or array.ndim > 3:
        return None, (
            f"Invalid image dimensions: expected 2D (HxW) or 3D (HxWxC) array, "
            f"got {array.ndim}D array with shape {array.shape}"
        )

    if array.size == 0:
        return None, "Invalid image dimensions: array is empty"

    # Validate minimum size
    h, w = array.shape[:2]
    if h < 1 or w < 1:
        return None, f"Invalid image dimensions: {h}x{w}"

    # Convert to PIL Image for consistent resize behavior
    try:
        if array.ndim == 2:
            # Grayscale: (H, W)
            pil_image = _array_to_pil_grayscale(array)
        else:
            # Color: (H, W, C)
            channels = array.shape[2]
            if channels == 1:
                # Single channel treated as grayscale
                pil_image = _array_to_pil_grayscale(array[:, :, 0])
            elif channels == 3:
                pil_image = _array_to_pil_rgb(array)
            elif channels == 4:
                # RGBA - drop alpha channel
                pil_image = _array_to_pil_rgb(array[:, :, :3])
            else:
                return None, (
                    f"Unsupported number of channels: {channels}. "
                    "Expected 1, 3, or 4."
                )
    except Exception as e:
        return None, f"Unable to process image array: {e}"

    return _pil_to_tensor(pil_image, target_size, normalize)


def _array_to_pil_grayscale(array: np.ndarray) -> Image.Image:
    """Convert a 2D numpy array to a grayscale PIL Image.

    Handles both float [0, 1] and uint8 [0, 255] inputs.
    """
    if array.dtype in (np.float32, np.float64, float):
        # Clip and convert float [0, 1] to uint8 [0, 255]
        array_clipped = np.clip(array, 0.0, 1.0)
        array_uint8 = (array_clipped * 255).astype(np.uint8)
    elif array.dtype == np.uint8:
        array_uint8 = array
    else:
        # Convert other integer types
        array_uint8 = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array_uint8, mode="L")


def _array_to_pil_rgb(array: np.ndarray) -> Image.Image:
    """Convert a 3D numpy array (H, W, 3) to an RGB PIL Image.

    Handles both float [0, 1] and uint8 [0, 255] inputs.
    """
    if array.dtype in (np.float32, np.float64, float):
        array_clipped = np.clip(array, 0.0, 1.0)
        array_uint8 = (array_clipped * 255).astype(np.uint8)
    elif array.dtype == np.uint8:
        array_uint8 = array
    else:
        array_uint8 = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array_uint8, mode="RGB")


def _pil_to_tensor(
    pil_image: Image.Image,
    target_size: tuple[int, int],
    normalize: bool,
) -> tuple[torch.Tensor | None, str | None]:
    """Convert a PIL Image to a preprocessed model-input tensor.

    Pipeline:
    1. Convert to grayscale (if not already)
    2. Resize to target_size using bilinear interpolation
    3. Convert to float32 tensor in [0, 1]
    4. Replicate to 3 channels
    5. Apply ImageNet normalization (if enabled)
    6. Add batch dimension

    Args:
        pil_image: PIL Image to process.
        target_size: Target dimensions (height, width).
        normalize: Whether to apply ImageNet normalization.

    Returns:
        Tuple of (tensor of shape (1, 3, H, W), None) on success,
        or (None, error_message) on failure.
    """
    try:
        # Step 1: Convert to grayscale
        grayscale = pil_image.convert("L")

        # Step 2: Resize to target size (PIL uses width, height)
        resized = grayscale.resize(
            (target_size[1], target_size[0]),
            Image.BILINEAR,
        )

        # Step 3: Convert to float32 tensor in [0, 1]
        np_image = np.array(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np_image).unsqueeze(0)  # Shape: (1, H, W)

        # Step 4: Replicate grayscale to 3 channels
        tensor = tensor.repeat(3, 1, 1)  # Shape: (3, H, W)

        # Step 5: Apply ImageNet-style normalization
        if normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std

        # Step 6: Add batch dimension
        tensor = tensor.unsqueeze(0)  # Shape: (1, 3, H, W)

        return tensor, None

    except Exception as e:
        return None, f"Unable to process image: {e}"
