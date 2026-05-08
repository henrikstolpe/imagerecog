"""CLI entry point for the IR Recognition Web UI.

Usage:
    python -m ir_recognition.web --model-path ./checkpoints --db-path ./data/signature_db

Environment variable fallbacks:
    IR_MODEL_PATH: Path to model checkpoint directory
    IR_DB_PATH: Path to signature database directory
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from ir_recognition.web.app import create_app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the web server."""
    parser = argparse.ArgumentParser(
        prog="python -m ir_recognition.web",
        description="Launch the IR Signature Recognition Web UI server.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint directory (env: IR_MODEL_PATH)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to signature database directory (env: IR_DB_PATH)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    return parser.parse_args(argv)


def resolve_path(cli_value: str | None, env_var: str) -> Path | None:
    """Resolve a path from CLI argument or environment variable fallback.

    Priority: CLI argument > environment variable > None.
    """
    value = cli_value if cli_value is not None else os.environ.get(env_var)
    if value is not None:
        return Path(value)
    return None


def main(argv: list[str] | None = None) -> None:
    """Main entry point: parse args, create app, and launch uvicorn."""
    args = parse_args(argv)

    model_path = resolve_path(args.model_path, "IR_MODEL_PATH")
    db_path = resolve_path(args.db_path, "IR_DB_PATH")

    app = create_app(model_path=model_path, db_path=db_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
