"""Shared test fixtures for the IR Recognition Web UI tests.

Provides:
- mock_backend: A configurable MockModelBackend returning deterministic logits/embeddings
- app: FastAPI application wired with the mock backend
- test_client: httpx AsyncClient configured with the app
- valid_image_bytes: Minimal valid 224x224 PNG as bytes
- valid_jpeg_bytes: Minimal valid 224x224 JPEG as bytes

The mock backend is configurable — tests can override num_classes, embedding_dim,
or inject custom logits via the mock_backend_factory fixture.

Requirements: 1.1, 1.5
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pytest
import torch
from httpx import ASGITransport, AsyncClient
from PIL import Image

from ir_recognition.web.app import create_app


# ---------------------------------------------------------------------------
# Configurable Mock Model Backend
# ---------------------------------------------------------------------------


class MockModelBackend:
    """Mock model backend that returns deterministic classification logits and embeddings.

    Implements the ModelBackend protocol: forward(pixel_values) -> (logits, embeddings).

    Args:
        num_classes: Number of output classes for logits tensor.
        embedding_dim: Dimensionality of the embedding vector.
        custom_logits: Optional custom logits tensor to return. If provided,
            num_classes is ignored and the shape of custom_logits is used.
    """

    def __init__(
        self,
        num_classes: int = 5,
        embedding_dim: int = 256,
        custom_logits: Optional[torch.Tensor] = None,
    ):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.custom_logits = custom_logits

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic logits and embeddings.

        Logits produce a clear top-1 prediction for T-72 (index 0) by default.
        If custom_logits were provided, those are returned instead (broadcast to batch).
        """
        batch_size = pixel_values.shape[0]

        if self.custom_logits is not None:
            # Broadcast custom logits to batch size
            if self.custom_logits.dim() == 1:
                logits = self.custom_logits.unsqueeze(0).expand(batch_size, -1)
            else:
                logits = self.custom_logits[:batch_size]
        else:
            # Default: strong prediction for class 0 (T-72)
            logits = torch.zeros(batch_size, self.num_classes)
            logits[:, 0] = 5.0  # T-72
            logits[:, 1] = 2.0  # T-80
            logits[:, 2] = 1.0  # BMP-3

        # Embeddings: deterministic unit vector (seeded for reproducibility)
        rng = torch.Generator().manual_seed(42)
        embeddings = torch.randn(batch_size, self.embedding_dim, generator=rng)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return logits, embeddings


class FailingModelBackend:
    """Mock model backend that raises an exception on forward pass.

    Useful for testing error isolation (Requirement 1.5).
    """

    def __init__(self, error_message: str = "Model inference failed: GPU out of memory"):
        self.error_message = error_message

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(self.error_message)


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------


def _create_png_bytes(width: int = 224, height: int = 224) -> bytes:
    """Create a minimal valid PNG image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), mode="RGB"
    )
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _create_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    """Create a minimal valid JPEG image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), mode="RGB"
    )
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend_factory():
    """Factory fixture for creating configurable MockModelBackend instances.

    Usage in tests:
        def test_custom(mock_backend_factory):
            backend = mock_backend_factory(num_classes=3, embedding_dim=128)
            # or with custom logits:
            backend = mock_backend_factory(custom_logits=torch.tensor([1.0, 0.5, 0.1]))
    """

    def _factory(
        num_classes: int = 5,
        embedding_dim: int = 256,
        custom_logits: Optional[torch.Tensor] = None,
    ) -> MockModelBackend:
        return MockModelBackend(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            custom_logits=custom_logits,
        )

    return _factory


@pytest.fixture
def mock_backend():
    """Create a default MockModelBackend (5 classes, 256-dim embeddings).

    Returns deterministic logits with T-72 as the top-1 prediction.
    """
    return MockModelBackend()


@pytest.fixture
def app(mock_backend):
    """Create a FastAPI app with mock pipeline (no database)."""
    return create_app(model_backend=mock_backend)


@pytest.fixture
def failing_app():
    """Create a FastAPI app with a failing model backend (for error isolation tests)."""
    return create_app(model_backend=FailingModelBackend())


@pytest.fixture
async def test_client(app):
    """Async httpx client configured with the FastAPI app.

    Usage:
        async def test_something(test_client):
            response = await test_client.get("/api/health")
            assert response.status_code == 200
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def valid_image_bytes():
    """A minimal valid 224x224 RGB PNG image as bytes."""
    return _create_png_bytes()


@pytest.fixture
def valid_jpeg_bytes():
    """A minimal valid 224x224 RGB JPEG image as bytes."""
    return _create_jpeg_bytes()
