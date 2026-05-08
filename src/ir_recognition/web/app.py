"""FastAPI application factory for the IR Recognition Web UI.

Creates and configures the FastAPI application with:
- Static file serving for the frontend SPA
- Exception handlers for structured error responses
- InferencePipeline and IRGenerator wired as application state
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from ir_recognition.inference.pipeline import InferencePipeline
from ir_recognition.ir_generator.generator import IRGenerator
from ir_recognition.signature_db.database import SignatureDatabase
from ir_recognition.web.routes import router

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class _ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware that catches unhandled exceptions and returns JSON 500."""

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            logger.error(f"Unhandled error: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500, content={"detail": f"Internal error: {exc}"}
            )


def create_app(
    model_path: Path | None = None,
    db_path: Path | None = None,
    model_backend=None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_path: Path to the model checkpoint directory. If None, the
            pipeline is created without a loaded model (model_loaded=False).
        db_path: Path to the signature database directory. If None, no
            database is used.
        model_backend: Optional injectable model backend for testing.
            If provided, overrides model_path for the InferencePipeline.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(title="IR Signature Recognition")

    # --- Exception handlers ---

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"detail": f"Internal error: {exc}"}
        )

    # Add middleware to catch exceptions not handled by FastAPI's exception handlers
    app.add_middleware(_ErrorHandlingMiddleware)

    # --- Wire application state ---

    database = None
    if db_path is not None:
        database = SignatureDatabase(db_path)

    # Create InferencePipeline: needs either model_backend or model_path
    if model_backend is not None:
        pipeline = InferencePipeline(
            model_backend=model_backend,
            database=database,
        )
        app.state.model_loaded = True
    elif model_path is not None:
        pipeline = InferencePipeline(
            model_path=model_path,
            database=database,
        )
        app.state.model_loaded = True
    else:
        # No model available - pipeline cannot be created
        pipeline = None
        app.state.model_loaded = False

    app.state.pipeline = pipeline
    app.state.db_path = db_path

    # Create IRGenerator
    app.state.generator = IRGenerator()

    # --- Register routes ---

    app.include_router(router)

    # --- Mount static files ---
    # Mounted last so API routes take precedence over static file serving
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app
