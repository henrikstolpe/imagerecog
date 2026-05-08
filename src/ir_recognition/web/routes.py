"""API route definitions for the IR Recognition Web UI.

Defines endpoints for recognition, generation, health check, and
database image serving.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from ir_recognition.models import IRGeneratorConfig
from ir_recognition.web.schemas import (
    ClassificationItem,
    GenerateRequest,
    GenerateResponse,
    RecognitionResponse,
    SimilarityItem,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Allowed image extensions for upload
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Maximum upload file size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize(request: Request, file: UploadFile = File(...)) -> RecognitionResponse:
    """Run recognition on an uploaded IR image.

    Accepts a multipart file upload, validates the image format,
    runs the inference pipeline, and returns structured results.

    Args:
        request: The incoming HTTP request (used to access app state).
        file: The uploaded image file.

    Returns:
        RecognitionResponse with classifications, similarity matches,
        threshold flags, and processing time.
    """
    # Check if pipeline is available
    pipeline = request.app.state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=503, content={"detail": "Model not ready"}
        )

    # Validate file extension
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. "
            f"Accepted: PNG, JPEG, TIFF"
        )

    # Read file bytes and validate size
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=413, content={"detail": "File size exceeds 10MB limit"}
        )

    # Decode image to numpy array
    try:
        pil_image = Image.open(io.BytesIO(file_bytes))
        # Convert to RGB if needed (handles grayscale, RGBA, etc.)
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        image_array = np.array(pil_image)
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

    # Run recognition pipeline with timing
    start_time = time.perf_counter()
    result = pipeline.recognize(image_array)
    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000.0

    # Map domain model to response schema
    classifications = [
        ClassificationItem(
            vehicle_class=c.vehicle_class,
            confidence=c.confidence,
        )
        for c in result.classifications
    ]

    similarity_matches = [
        SimilarityItem(
            vehicle_class=m.vehicle_class,
            similarity_score=m.similarity_score,
            image_id=m.image_id,
            thumbnail_url=f"/api/database/images/{m.image_id}",
        )
        for m in result.similarity_matches
    ]

    return RecognitionResponse(
        classifications=classifications,
        similarity_matches=similarity_matches,
        low_confidence=result.low_confidence,
        no_match=result.no_match,
        processing_time_ms=processing_time_ms,
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    """Generate a synthetic IR image from the given parameters.

    Accepts generation parameters as JSON, invokes the IR generator,
    and returns the result as a base64-encoded PNG.

    Args:
        request: The incoming HTTP request (used to access app state).
        body: Validated generation parameters.

    Returns:
        GenerateResponse with base64 PNG and echoed parameters.
    """
    generator = request.app.state.generator

    # Create config from request
    config = IRGeneratorConfig(
        vehicle_type=body.vehicle_type,
        azimuth=body.azimuth,
        elevation=body.elevation,
        ambient_temp=body.ambient_temp,
    )

    # Generate the image (returns numpy array float32 [0, 1])
    image_array = generator.generate(config)

    # Encode as PNG then base64
    image_uint8 = (np.clip(image_array, 0.0, 1.0) * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8, mode="L")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return GenerateResponse(
        image_base64=image_base64,
        vehicle_type=body.vehicle_type,
        azimuth=body.azimuth,
        elevation=body.elevation,
        ambient_temp=body.ambient_temp,
    )


@router.get("/health")
async def health(request: Request) -> dict:
    """Health check endpoint.

    Returns:
        Dictionary with status and model_loaded flag.
    """
    return {
        "status": "ok",
        "model_loaded": request.app.state.model_loaded,
    }


@router.get("/database/images/{image_id}")
async def get_database_image(request: Request, image_id: str) -> StreamingResponse:
    """Serve a thumbnail PNG from the signature database.

    Looks up the image file in the database directory structure
    and returns it as a streaming PNG response.

    Args:
        request: The incoming HTTP request (used to access app state).
        image_id: The unique identifier of the image in the database.

    Returns:
        StreamingResponse with the PNG image data.
    """
    db_path = request.app.state.db_path

    if db_path is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Image not found: {image_id}"},
        )

    # Search for the image file in the database images directory
    images_dir = Path(db_path) / "images"
    image_file = _find_image_in_database(images_dir, image_id)

    if image_file is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Image not found: {image_id}"},
        )

    # Read and stream the image file
    image_bytes = image_file.read_bytes()
    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type="image/png",
    )


def _find_image_in_database(images_dir: Path, image_id: str) -> Path | None:
    """Find an image file in the database directory by image_id.

    Searches all vehicle class subdirectories for a file named
    {image_id}.png.

    Args:
        images_dir: The images directory of the signature database.
        image_id: The image identifier to look up.

    Returns:
        Path to the image file if found, None otherwise.
    """
    if not images_dir.exists():
        return None

    # Look for {image_id}.png in any vehicle class subdirectory
    for vehicle_dir in images_dir.iterdir():
        if vehicle_dir.is_dir():
            candidate = vehicle_dir / f"{image_id}.png"
            if candidate.exists():
                return candidate

    return None
