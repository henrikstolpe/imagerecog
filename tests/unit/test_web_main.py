"""Unit tests for the ir_recognition.web.__main__ CLI entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ir_recognition.web.__main__ import main, parse_args, resolve_path


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_defaults(self):
        args = parse_args([])
        assert args.model_path is None
        assert args.db_path is None
        assert args.host == "0.0.0.0"
        assert args.port == 8000

    def test_model_path(self):
        args = parse_args(["--model-path", "/tmp/checkpoints"])
        assert args.model_path == "/tmp/checkpoints"

    def test_db_path(self):
        args = parse_args(["--db-path", "/tmp/signature_db"])
        assert args.db_path == "/tmp/signature_db"

    def test_host(self):
        args = parse_args(["--host", "127.0.0.1"])
        assert args.host == "127.0.0.1"

    def test_port(self):
        args = parse_args(["--port", "9000"])
        assert args.port == 9000

    def test_all_arguments(self):
        args = parse_args([
            "--model-path", "./models",
            "--db-path", "./data/db",
            "--host", "localhost",
            "--port", "3000",
        ])
        assert args.model_path == "./models"
        assert args.db_path == "./data/db"
        assert args.host == "localhost"
        assert args.port == 3000


class TestResolvePath:
    """Tests for path resolution with env var fallback."""

    def test_cli_value_takes_priority(self, monkeypatch):
        monkeypatch.setenv("IR_MODEL_PATH", "/env/path")
        result = resolve_path("/cli/path", "IR_MODEL_PATH")
        assert result == Path("/cli/path")

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("IR_MODEL_PATH", "/env/path")
        result = resolve_path(None, "IR_MODEL_PATH")
        assert result == Path("/env/path")

    def test_returns_none_when_neither_set(self, monkeypatch):
        monkeypatch.delenv("IR_MODEL_PATH", raising=False)
        result = resolve_path(None, "IR_MODEL_PATH")
        assert result is None

    def test_returns_path_object(self):
        result = resolve_path("./relative/path", "UNUSED_VAR")
        assert isinstance(result, Path)
        assert result == Path("./relative/path")


class TestMain:
    """Tests for the main entry point function."""

    @patch("ir_recognition.web.__main__.uvicorn.run")
    @patch("ir_recognition.web.__main__.create_app")
    def test_launches_uvicorn_with_defaults(self, mock_create_app, mock_uvicorn_run):
        mock_app = mock_create_app.return_value

        main(["--host", "127.0.0.1", "--port", "5000"])

        mock_create_app.assert_called_once_with(model_path=None, db_path=None)
        mock_uvicorn_run.assert_called_once_with(
            mock_app, host="127.0.0.1", port=5000
        )

    @patch("ir_recognition.web.__main__.uvicorn.run")
    @patch("ir_recognition.web.__main__.create_app")
    def test_passes_paths_to_create_app(self, mock_create_app, mock_uvicorn_run):
        main(["--model-path", "/models", "--db-path", "/data"])

        mock_create_app.assert_called_once_with(
            model_path=Path("/models"),
            db_path=Path("/data"),
        )

    @patch("ir_recognition.web.__main__.uvicorn.run")
    @patch("ir_recognition.web.__main__.create_app")
    def test_env_var_fallback_for_model_path(
        self, mock_create_app, mock_uvicorn_run, monkeypatch
    ):
        monkeypatch.setenv("IR_MODEL_PATH", "/env/models")
        monkeypatch.delenv("IR_DB_PATH", raising=False)

        main([])

        mock_create_app.assert_called_once_with(
            model_path=Path("/env/models"),
            db_path=None,
        )

    @patch("ir_recognition.web.__main__.uvicorn.run")
    @patch("ir_recognition.web.__main__.create_app")
    def test_env_var_fallback_for_db_path(
        self, mock_create_app, mock_uvicorn_run, monkeypatch
    ):
        monkeypatch.delenv("IR_MODEL_PATH", raising=False)
        monkeypatch.setenv("IR_DB_PATH", "/env/db")

        main([])

        mock_create_app.assert_called_once_with(
            model_path=None,
            db_path=Path("/env/db"),
        )

    @patch("ir_recognition.web.__main__.uvicorn.run")
    @patch("ir_recognition.web.__main__.create_app")
    def test_cli_overrides_env_vars(
        self, mock_create_app, mock_uvicorn_run, monkeypatch
    ):
        monkeypatch.setenv("IR_MODEL_PATH", "/env/models")
        monkeypatch.setenv("IR_DB_PATH", "/env/db")

        main(["--model-path", "/cli/models", "--db-path", "/cli/db"])

        mock_create_app.assert_called_once_with(
            model_path=Path("/cli/models"),
            db_path=Path("/cli/db"),
        )
