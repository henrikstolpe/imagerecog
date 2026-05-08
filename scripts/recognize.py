#!/usr/bin/env python
"""Recognize vehicles in IR images using the trained model.

Loads a trained InferencePipeline with a model checkpoint and signature
database, runs recognition on the input image, and prints JSON results.

Usage:
    python scripts/recognize.py path/to/image.png --checkpoint checkpoints/best --database data/signature_db
    python scripts/recognize.py image.jpg --checkpoint checkpoints/best --database data/signature_db

Requirements: 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Recognize vehicles in IR images using the trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input IR image (PNG, JPEG, or TIFF).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best"),
        help="Path to the trained model checkpoint directory.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/signature_db"),
        help="Path to the signature database directory.",
    )
    return parser.parse_args(argv)


def format_result_as_json(result) -> str:
    """Format a RecognitionResult as a JSON string.

    Args:
        result: RecognitionResult from the inference pipeline.

    Returns:
        Formatted JSON string matching the RecognitionResult schema.
    """
    output = {
        "classifications": [
            {
                "vehicle_class": c.vehicle_class,
                "confidence": round(c.confidence, 4),
            }
            for c in result.classifications
        ],
        "similarity_matches": [
            {
                "vehicle_class": m.vehicle_class,
                "similarity_score": round(m.similarity_score, 4),
                "image_id": m.image_id,
            }
            for m in result.similarity_matches
        ],
        "low_confidence": result.low_confidence,
        "no_match": result.no_match,
    }
    return json.dumps(output, indent=2)


def main() -> None:
    """Entry point for the recognition script."""
    args = parse_args()

    # Validate image path
    if not args.image.exists():
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Validate checkpoint path
    if not args.checkpoint.exists():
        print(
            f"Error: Checkpoint directory not found: {args.checkpoint}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate database path
    if not args.database.exists():
        print(
            f"Error: Database directory not found: {args.database}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import here to avoid slow imports when just checking args
    try:
        from ir_recognition.inference.pipeline import InferencePipeline
        from ir_recognition.signature_db.database import SignatureDatabase
    except ImportError as e:
        print(
            f"Error: Failed to import required modules: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load database
    try:
        database = SignatureDatabase(args.database)
    except Exception as e:
        print(
            f"Error: Failed to load signature database: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load inference pipeline
    try:
        pipeline = InferencePipeline(
            model_path=args.checkpoint,
            database=database,
        )
    except FileNotFoundError as e:
        print(f"Error: Model loading failed: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: Model loading failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to initialize inference pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    # Run recognition
    try:
        result = pipeline.recognize(args.image)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Recognition failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print JSON results to stdout
    print(format_result_as_json(result))


if __name__ == "__main__":
    main()
