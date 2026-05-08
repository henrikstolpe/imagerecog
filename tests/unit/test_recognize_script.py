"""Unit tests for the recognize.py CLI script.

Tests argument parsing, JSON formatting, and error handling behavior.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from recognize import format_result_as_json, parse_args


class TestParseArgs:
    """Tests for command-line argument parsing."""

    def test_minimal_args(self):
        """Test parsing with only the required image argument."""
        args = parse_args(["test_image.png"])
        assert args.image == Path("test_image.png")
        assert args.checkpoint == Path("checkpoints/best")
        assert args.database == Path("data/signature_db")

    def test_all_args(self):
        """Test parsing with all arguments specified."""
        args = parse_args([
            "my_image.jpg",
            "--checkpoint", "models/v2",
            "--database", "db/signatures",
        ])
        assert args.image == Path("my_image.jpg")
        assert args.checkpoint == Path("models/v2")
        assert args.database == Path("db/signatures")

    def test_missing_image_arg_raises(self):
        """Test that missing image argument causes SystemExit."""
        with pytest.raises(SystemExit):
            parse_args([])


class TestFormatResultAsJson:
    """Tests for JSON output formatting."""

    def test_basic_result_format(self):
        """Test that a RecognitionResult is formatted correctly as JSON."""
        from ir_recognition.models import (
            ClassificationResult,
            RecognitionResult,
            SimilarityMatch,
        )

        result = RecognitionResult(
            classifications=[
                ClassificationResult(vehicle_class="T-72", confidence=0.87),
                ClassificationResult(vehicle_class="T-80", confidence=0.09),
                ClassificationResult(vehicle_class="BMP-3", confidence=0.03),
            ],
            similarity_matches=[
                SimilarityMatch(
                    vehicle_class="T-72",
                    similarity_score=0.92,
                    image_id="t72_synth_az045_el030_001",
                ),
                SimilarityMatch(
                    vehicle_class="T-72",
                    similarity_score=0.88,
                    image_id="t72_synth_az090_el030_002",
                ),
            ],
            low_confidence=False,
            no_match=False,
        )

        json_str = format_result_as_json(result)
        parsed = json.loads(json_str)

        assert len(parsed["classifications"]) == 3
        assert parsed["classifications"][0]["vehicle_class"] == "T-72"
        assert parsed["classifications"][0]["confidence"] == 0.87
        assert len(parsed["similarity_matches"]) == 2
        assert parsed["similarity_matches"][0]["image_id"] == "t72_synth_az045_el030_001"
        assert parsed["low_confidence"] is False
        assert parsed["no_match"] is False

    def test_low_confidence_flag(self):
        """Test JSON output with low_confidence flag set."""
        from ir_recognition.models import (
            ClassificationResult,
            RecognitionResult,
            SimilarityMatch,
        )

        result = RecognitionResult(
            classifications=[
                ClassificationResult(vehicle_class="T-72", confidence=0.3),
                ClassificationResult(vehicle_class="T-80", confidence=0.2),
                ClassificationResult(vehicle_class="BMP-3", confidence=0.1),
            ],
            similarity_matches=[
                SimilarityMatch(
                    vehicle_class="T-72",
                    similarity_score=0.5,
                    image_id="t72_001",
                ),
            ],
            low_confidence=True,
            no_match=False,
        )

        json_str = format_result_as_json(result)
        parsed = json.loads(json_str)

        assert parsed["low_confidence"] is True
        assert parsed["no_match"] is False

    def test_no_match_flag(self):
        """Test JSON output with no_match flag set."""
        from ir_recognition.models import (
            ClassificationResult,
            RecognitionResult,
            SimilarityMatch,
        )

        result = RecognitionResult(
            classifications=[
                ClassificationResult(vehicle_class="T-72", confidence=0.8),
                ClassificationResult(vehicle_class="T-80", confidence=0.1),
                ClassificationResult(vehicle_class="BMP-3", confidence=0.05),
            ],
            similarity_matches=[
                SimilarityMatch(
                    vehicle_class="CV90",
                    similarity_score=0.2,
                    image_id="cv90_001",
                ),
            ],
            low_confidence=False,
            no_match=True,
        )

        json_str = format_result_as_json(result)
        parsed = json.loads(json_str)

        assert parsed["low_confidence"] is False
        assert parsed["no_match"] is True


class TestCLIErrorHandling:
    """Tests for CLI error handling using subprocess."""

    def test_nonexistent_image_exits_with_error(self):
        """Test that a non-existent image file produces an error and exit code 1."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/recognize.py",
                "nonexistent_image.png",
                "--checkpoint", "checkpoints/best",
                "--database", "data/signature_db",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr
        assert "Image file not found" in result.stderr

    def test_nonexistent_checkpoint_exits_with_error(self, tmp_path):
        """Test that a non-existent checkpoint produces an error and exit code 1."""
        # Create a dummy image file so we pass the image check
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"fake image data")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/recognize.py",
                str(image_file),
                "--checkpoint", "nonexistent_checkpoint_dir",
                "--database", "data/signature_db",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr
        assert "Checkpoint directory not found" in result.stderr

    def test_nonexistent_database_exits_with_error(self, tmp_path):
        """Test that a non-existent database produces an error and exit code 1."""
        # Create a dummy image and checkpoint dir
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"fake image data")
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "scripts/recognize.py",
                str(image_file),
                "--checkpoint", str(checkpoint_dir),
                "--database", "nonexistent_database_dir",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr
        assert "Database directory not found" in result.stderr
