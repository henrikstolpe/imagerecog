"""Integration tests for the IR Recognition Web UI end-to-end flow.

Verifies:
- Static file serving (index.html, JS, CSS)
- Upload → recognize → display results flow with mock pipeline
- Generate → display → recognize flow
- Error states (invalid file, server error, connection error)

Requirements: 1.1, 1.3, 2.1, 3.1, 4.1, 5.4, 7.1, 7.2
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from httpx import ASGITransport, AsyncClient
from PIL import Image

from ir_recognition.inference.pipeline import InferencePipeline
from ir_recognition.models import (
    ClassificationResult,
    RecognitionResult,
    SimilarityMatch,
)
from ir_recognition.web.app import create_app


# ---------------------------------------------------------------------------
# Mock model backend
# ---------------------------------------------------------------------------


class MockModelBackend:
    """Mock model backend that returns deterministic classification logits and embeddings.

    Implements the ModelBackend protocol: forward(pixel_values) -> (logits, embeddings).
    Returns logits that produce a clear top-1 prediction for T-72 (index 0).
    """

    def __init__(self, num_classes: int = 5, embedding_dim: int = 256):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dummy logits and embeddings."""
        batch_size = pixel_values.shape[0]
        # Logits: strong prediction for class 0 (T-72)
        logits = torch.zeros(batch_size, self.num_classes)
        logits[:, 0] = 5.0  # T-72
        logits[:, 1] = 2.0  # T-80
        logits[:, 2] = 1.0  # BMP-3

        # Embeddings: random unit vector
        embeddings = torch.randn(batch_size, self.embedding_dim)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return logits, embeddings


class FailingModelBackend:
    """Mock model backend that raises an exception on forward pass."""

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Model inference failed: GPU out of memory")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_test_png(width: int = 224, height: int = 224) -> bytes:
    """Create a minimal valid PNG image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), mode="RGB"
    )
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def mock_backend():
    """Create a mock model backend."""
    return MockModelBackend()


@pytest.fixture
def app(mock_backend):
    """Create a FastAPI app with mock pipeline (no database)."""
    return create_app(model_backend=mock_backend)


@pytest.fixture
def failing_app():
    """Create a FastAPI app with a failing model backend."""
    return create_app(model_backend=FailingModelBackend())


@pytest.fixture
def valid_png_bytes():
    """A valid 224x224 RGB PNG image as bytes."""
    return _create_test_png()


# ---------------------------------------------------------------------------
# Tests: Static file serving
# ---------------------------------------------------------------------------


class TestStaticFileServing:
    """Verify static files are served correctly at root path."""

    @pytest.mark.anyio
    async def test_root_returns_index_html(self, app):
        """GET / should return the index.html content."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "IR Signature Recognition System" in response.text

    @pytest.mark.anyio
    async def test_js_app_served(self, app):
        """GET /js/app.js should return JavaScript content."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/js/app.js")

        assert response.status_code == 200
        content_type = response.headers["content-type"]
        assert "javascript" in content_type or "text/" in content_type
        assert "import" in response.text  # ES module imports

    @pytest.mark.anyio
    async def test_css_served(self, app):
        """GET /css/styles.css should return CSS content."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/css/styles.css")

        assert response.status_code == 200
        content_type = response.headers["content-type"]
        assert "css" in content_type or "text/" in content_type


# ---------------------------------------------------------------------------
# Tests: Recognition endpoint (upload → recognize → results)
# ---------------------------------------------------------------------------


class TestRecognizeEndpoint:
    """Verify upload → recognize → display results flow with mock pipeline."""

    @pytest.mark.anyio
    async def test_recognize_valid_png(self, app, valid_png_bytes):
        """POST /api/recognize with a valid PNG returns RecognitionResponse JSON."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/recognize",
                files={"file": ("test.png", valid_png_bytes, "image/png")},
            )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "classifications" in data
        assert "similarity_matches" in data
        assert "low_confidence" in data
        assert "no_match" in data
        assert "processing_time_ms" in data

        # Verify classifications
        classifications = data["classifications"]
        assert 1 <= len(classifications) <= 3
        for item in classifications:
            assert "vehicle_class" in item
            assert "confidence" in item
            assert 0.0 <= item["confidence"] <= 1.0

        # Top-1 should be T-72 (from mock backend logits)
        assert classifications[0]["vehicle_class"] == "T-72"

        # Processing time should be positive
        assert data["processing_time_ms"] > 0

    @pytest.mark.anyio
    async def test_recognize_valid_jpeg(self, app):
        """POST /api/recognize with a valid JPEG returns 200."""
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8), mode="RGB"
        )
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        jpeg_bytes = buffer.getvalue()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/recognize",
                files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["classifications"]) >= 1

    @pytest.mark.anyio
    async def test_recognize_invalid_file_returns_400(self, app):
        """POST /api/recognize with a text file returns HTTP 400."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/recognize",
                files={"file": ("test.txt", b"not an image", "text/plain")},
            )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert len(data["detail"]) > 0

    @pytest.mark.anyio
    async def test_recognize_unsupported_extension_returns_400(self, app):
        """POST /api/recognize with .bmp extension returns HTTP 400."""
        # Create a valid image but with unsupported extension
        img_bytes = _create_test_png()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/recognize",
                files={"file": ("test.bmp", img_bytes, "image/bmp")},
            )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Unsupported" in data["detail"] or "format" in data["detail"].lower()


# ---------------------------------------------------------------------------
# Tests: Generate endpoint (generate → display → recognize)
# ---------------------------------------------------------------------------


class TestGenerateEndpoint:
    """Verify generate → display → recognize flow."""

    @pytest.mark.anyio
    async def test_generate_valid_params(self, app):
        """POST /api/generate with valid params returns GenerateResponse JSON."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/generate",
                json={
                    "vehicle_type": "T-72",
                    "azimuth": 45.0,
                    "elevation": 30.0,
                    "ambient_temp": 293.0,
                },
            )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "image_base64" in data
        assert "vehicle_type" in data
        assert "azimuth" in data
        assert "elevation" in data
        assert "ambient_temp" in data

        # Verify echoed parameters
        assert data["vehicle_type"] == "T-72"
        assert data["azimuth"] == 45.0
        assert data["elevation"] == 30.0
        assert data["ambient_temp"] == 293.0

        # Verify base64 image is non-empty and decodable
        import base64

        image_bytes = base64.b64decode(data["image_base64"])
        assert len(image_bytes) > 0

        # Verify it's a valid PNG
        img = Image.open(io.BytesIO(image_bytes))
        assert img.format == "PNG"

    @pytest.mark.anyio
    async def test_generate_then_recognize(self, app):
        """Generate an image, then submit it for recognition (end-to-end flow)."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Step 1: Generate
            gen_response = await client.post(
                "/api/generate",
                json={
                    "vehicle_type": "BMP-3",
                    "azimuth": 90.0,
                    "elevation": 15.0,
                    "ambient_temp": 300.0,
                },
            )
            assert gen_response.status_code == 200
            gen_data = gen_response.json()

            # Step 2: Decode generated image and submit for recognition
            import base64

            image_bytes = base64.b64decode(gen_data["image_base64"])

            rec_response = await client.post(
                "/api/recognize",
                files={"file": ("generated.png", image_bytes, "image/png")},
            )

        assert rec_response.status_code == 200
        rec_data = rec_response.json()
        assert len(rec_data["classifications"]) >= 1
        assert rec_data["processing_time_ms"] > 0

    @pytest.mark.anyio
    async def test_generate_invalid_vehicle_type(self, app):
        """POST /api/generate with invalid vehicle_type returns 422."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/generate",
                json={
                    "vehicle_type": "INVALID",
                    "azimuth": 0.0,
                    "elevation": 0.0,
                    "ambient_temp": 293.0,
                },
            )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests: Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Verify health check endpoint."""

    @pytest.mark.anyio
    async def test_health_returns_ok(self, app):
        """GET /api/health returns status ok and model_loaded flag."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    @pytest.mark.anyio
    async def test_health_no_model(self):
        """GET /api/health with no model returns model_loaded=False."""
        app = create_app()  # No model_backend, no model_path
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify error states display correctly."""

    @pytest.mark.anyio
    async def test_pipeline_exception_returns_500(self, failing_app, valid_png_bytes):
        """When pipeline raises exception, server returns 500 with detail message."""
        async with AsyncClient(
            transport=ASGITransport(app=failing_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/recognize",
                files={"file": ("test.png", valid_png_bytes, "image/png")},
            )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert len(data["detail"]) > 0

    @pytest.mark.anyio
    async def test_server_remains_operational_after_error(
        self, failing_app, valid_png_bytes
    ):
        """After a 500 error, the server still responds to subsequent requests."""
        async with AsyncClient(
            transport=ASGITransport(app=failing_app), base_url="http://test"
        ) as client:
            # First request: triggers error
            response1 = await client.post(
                "/api/recognize",
                files={"file": ("test.png", valid_png_bytes, "image/png")},
            )
            assert response1.status_code == 500

            # Second request: server should still be operational
            response2 = await client.get("/api/health")
            assert response2.status_code == 200
            assert response2.json()["status"] == "ok"

    @pytest.mark.anyio
    async def test_no_model_returns_503(self, valid_png_bytes):
        """When no model is loaded, recognize returns 503."""
        app = create_app()  # No model
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/recognize",
                files={"file": ("test.png", valid_png_bytes, "image/png")},
            )

        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "not ready" in data["detail"].lower() or "Model" in data["detail"]
