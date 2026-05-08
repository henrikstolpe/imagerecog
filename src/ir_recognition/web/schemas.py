"""Pydantic request/response schemas for the IR Recognition Web API.

Defines validated models for all API endpoints including generation requests,
classification results, similarity matches, and recognition responses.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

# Supported vehicle classes (mirrors models.SUPPORTED_VEHICLE_CLASSES)
SUPPORTED_VEHICLE_CLASSES = ("T-72", "T-80", "BMP-3", "CV90", "civilian_car")


class GenerateRequest(BaseModel):
    """Request body for the /api/generate endpoint.

    Attributes:
        vehicle_type: Vehicle class to generate (must be in SUPPORTED_VEHICLE_CLASSES).
        azimuth: Viewing azimuth angle in degrees, range [0, 360).
        elevation: Viewing elevation angle in degrees, range [0, 90].
        ambient_temp: Ambient temperature in Kelvin (must be > 0). Default 293.0.
    """

    vehicle_type: str
    azimuth: float
    elevation: float
    ambient_temp: float = 293.0

    @field_validator("vehicle_type")
    @classmethod
    def validate_vehicle_type(cls, v: str) -> str:
        if v not in SUPPORTED_VEHICLE_CLASSES:
            raise ValueError(
                f"Unknown vehicle class: '{v}'. "
                f"Supported: {SUPPORTED_VEHICLE_CLASSES}"
            )
        return v

    @field_validator("azimuth")
    @classmethod
    def validate_azimuth(cls, v: float) -> float:
        if not (0.0 <= v < 360.0):
            raise ValueError(f"Azimuth must be in [0, 360), got {v}")
        return v

    @field_validator("elevation")
    @classmethod
    def validate_elevation(cls, v: float) -> float:
        if not (0.0 <= v <= 90.0):
            raise ValueError(f"Elevation must be in [0, 90], got {v}")
        return v

    @field_validator("ambient_temp")
    @classmethod
    def validate_ambient_temp(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                f"Ambient temperature must be > 0 Kelvin, got {v}"
            )
        return v


class ClassificationItem(BaseModel):
    """A single classification prediction in the recognition response.

    Attributes:
        vehicle_class: Predicted vehicle class label.
        confidence: Confidence score in [0, 1].
    """

    vehicle_class: str
    confidence: float

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {v}")
        return v


class SimilarityItem(BaseModel):
    """A single similarity match in the recognition response.

    Attributes:
        vehicle_class: Vehicle class of the matched reference image.
        similarity_score: Cosine similarity score in [-1, 1].
        image_id: Unique identifier of the matched image in the database.
        thumbnail_url: Relative URL to fetch the thumbnail image.
    """

    vehicle_class: str
    similarity_score: float
    image_id: str
    thumbnail_url: str

    @field_validator("similarity_score")
    @classmethod
    def validate_similarity_score(cls, v: float) -> float:
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"Similarity score must be in [-1, 1], got {v}")
        return v


class RecognitionResponse(BaseModel):
    """Response body for the /api/recognize endpoint.

    Attributes:
        classifications: Top-3 classification predictions sorted by confidence.
        similarity_matches: Top-5 similarity matches sorted by score descending.
        low_confidence: True if top-1 classification confidence < 0.5.
        no_match: True if all similarity scores < 0.3.
        processing_time_ms: Time taken for recognition in milliseconds.
    """

    classifications: list[ClassificationItem]
    similarity_matches: list[SimilarityItem]
    low_confidence: bool
    no_match: bool
    processing_time_ms: float


class GenerateResponse(BaseModel):
    """Response body for the /api/generate endpoint.

    Attributes:
        image_base64: Base64-encoded PNG image data.
        vehicle_type: Vehicle class that was generated.
        azimuth: Azimuth angle used for generation.
        elevation: Elevation angle used for generation.
        ambient_temp: Ambient temperature used for generation.
    """

    image_base64: str
    vehicle_type: str
    azimuth: float
    elevation: float
    ambient_temp: float
