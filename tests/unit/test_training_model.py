"""Unit tests for training model components.

Tests the architecture classes (heads, combined model) without requiring GPU.
"""

import pytest
import torch
import torch.nn as nn

from ir_recognition.training.model import (
    ClassificationHead,
    EmbeddingHead,
    IRRecognitionModel,
    get_quantization_config,
    get_lora_config,
    get_vram_usage_gb,
    get_peak_vram_usage_gb,
)
from ir_recognition.models import TrainingConfig


class TestClassificationHead:
    """Tests for the ClassificationHead module."""

    def test_output_shape(self):
        """Classification head produces correct output shape."""
        head = ClassificationHead(encoder_dim=1152, num_classes=5)
        x = torch.randn(4, 1152)
        output = head(x)
        assert output.shape == (4, 5)

    def test_single_sample(self):
        """Works with single sample batch."""
        head = ClassificationHead(encoder_dim=1152, num_classes=5)
        x = torch.randn(1, 1152)
        output = head(x)
        assert output.shape == (1, 5)

    def test_output_is_logits(self):
        """Output is raw logits (not softmax), can be negative."""
        head = ClassificationHead(encoder_dim=1152, num_classes=5)
        torch.manual_seed(42)
        x = torch.randn(8, 1152)
        output = head(x)
        # Raw logits can be negative
        assert output.min().item() < 0 or output.max().item() > 1

    def test_softmax_sums_to_one(self):
        """Applying softmax to output sums to 1 per sample."""
        head = ClassificationHead(encoder_dim=1152, num_classes=5)
        x = torch.randn(4, 1152)
        output = head(x)
        probs = torch.softmax(output, dim=-1)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-6)

    def test_different_encoder_dims(self):
        """Works with various encoder dimensions."""
        for dim in [256, 512, 768, 1152]:
            head = ClassificationHead(encoder_dim=dim, num_classes=5)
            x = torch.randn(2, dim)
            output = head(x)
            assert output.shape == (2, 5)

    def test_different_num_classes(self):
        """Works with various numbers of classes."""
        for n_classes in [2, 5, 10, 100]:
            head = ClassificationHead(encoder_dim=1152, num_classes=n_classes)
            x = torch.randn(2, 1152)
            output = head(x)
            assert output.shape == (2, n_classes)


class TestEmbeddingHead:
    """Tests for the EmbeddingHead module."""

    def test_output_shape(self):
        """Embedding head produces correct output shape."""
        head = EmbeddingHead(encoder_dim=1152, embedding_dim=256)
        x = torch.randn(4, 1152)
        output = head(x)
        assert output.shape == (4, 256)

    def test_l2_normalized(self):
        """Output vectors are L2-normalized (unit norm)."""
        head = EmbeddingHead(encoder_dim=1152, embedding_dim=256)
        x = torch.randn(8, 1152)
        output = head(x)
        norms = torch.norm(output, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-6)

    def test_single_sample_normalized(self):
        """Single sample is also L2-normalized."""
        head = EmbeddingHead(encoder_dim=1152, embedding_dim=256)
        x = torch.randn(1, 1152)
        output = head(x)
        norm = torch.norm(output, p=2, dim=-1)
        assert torch.allclose(norm, torch.ones(1), atol=1e-6)

    def test_output_finite(self):
        """All output values are finite (no NaN or Inf)."""
        head = EmbeddingHead(encoder_dim=1152, embedding_dim=256)
        x = torch.randn(4, 1152)
        output = head(x)
        assert torch.isfinite(output).all()

    def test_different_embedding_dims(self):
        """Works with various embedding dimensions."""
        for emb_dim in [64, 128, 256, 512]:
            head = EmbeddingHead(encoder_dim=1152, embedding_dim=emb_dim)
            x = torch.randn(2, 1152)
            output = head(x)
            assert output.shape == (2, emb_dim)
            norms = torch.norm(output, p=2, dim=-1)
            assert torch.allclose(norms, torch.ones(2), atol=1e-6)


class TestIRRecognitionModel:
    """Tests for the combined IRRecognitionModel."""

    def _make_mock_encoder(self, hidden_dim: int = 64, seq_len: int = 16):
        """Create a simple mock vision encoder for testing.

        Uses small dimensions to avoid memory issues in tests.
        """

        class MockEncoder(nn.Module):
            def __init__(self, hidden_dim, seq_len):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.seq_len = seq_len
                # Use a small projection to avoid memory issues
                self.proj = nn.Linear(3 * 16 * 16, seq_len * hidden_dim)

            def forward(self, pixel_values):
                batch_size = pixel_values.shape[0]
                # Downsample input to small size before projection
                downsampled = nn.functional.adaptive_avg_pool2d(pixel_values, (16, 16))
                flat = downsampled.view(batch_size, -1)
                out = self.proj(flat)
                return out.view(batch_size, self.seq_len, self.hidden_dim)

        return MockEncoder(hidden_dim, seq_len)

    def test_forward_output_shapes(self):
        """Combined model produces correct output shapes."""
        encoder = self._make_mock_encoder(hidden_dim=64, seq_len=16)
        cls_head = ClassificationHead(encoder_dim=64, num_classes=5)
        emb_head = EmbeddingHead(encoder_dim=64, embedding_dim=256)

        model = IRRecognitionModel(encoder, cls_head, emb_head)
        x = torch.randn(2, 3, 224, 224)
        logits, embeddings = model(x)

        assert logits.shape == (2, 5)
        assert embeddings.shape == (2, 256)

    def test_embeddings_normalized(self):
        """Embeddings from combined model are L2-normalized."""
        encoder = self._make_mock_encoder(hidden_dim=64, seq_len=16)
        cls_head = ClassificationHead(encoder_dim=64, num_classes=5)
        emb_head = EmbeddingHead(encoder_dim=64, embedding_dim=256)

        model = IRRecognitionModel(encoder, cls_head, emb_head)
        x = torch.randn(4, 3, 224, 224)
        _, embeddings = model(x)

        norms = torch.norm(embeddings, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-6)

    def test_mean_pooling(self):
        """Mean pooling strategy averages over sequence dimension."""
        encoder = self._make_mock_encoder(hidden_dim=64, seq_len=16)
        cls_head = ClassificationHead(encoder_dim=64, num_classes=5)
        emb_head = EmbeddingHead(encoder_dim=64, embedding_dim=32)

        model = IRRecognitionModel(encoder, cls_head, emb_head, pool_strategy="mean")
        x = torch.randn(2, 3, 224, 224)
        logits, embeddings = model(x)

        assert logits.shape == (2, 5)
        assert embeddings.shape == (2, 32)

    def test_cls_pooling(self):
        """CLS pooling strategy takes first token."""
        encoder = self._make_mock_encoder(hidden_dim=64, seq_len=16)
        cls_head = ClassificationHead(encoder_dim=64, num_classes=5)
        emb_head = EmbeddingHead(encoder_dim=64, embedding_dim=32)

        model = IRRecognitionModel(encoder, cls_head, emb_head, pool_strategy="cls")
        x = torch.randn(2, 3, 224, 224)
        logits, embeddings = model(x)

        assert logits.shape == (2, 5)
        assert embeddings.shape == (2, 32)

    def test_handles_encoder_with_last_hidden_state(self):
        """Handles encoder output with last_hidden_state attribute."""

        class EncoderWithAttr(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3 * 224 * 224, 16 * 64)

            def forward(self, pixel_values):
                batch_size = pixel_values.shape[0]
                flat = pixel_values.view(batch_size, -1)
                out = self.linear(flat).view(batch_size, 16, 64)

                class Output:
                    last_hidden_state = out

                return Output()

        encoder = EncoderWithAttr()
        cls_head = ClassificationHead(encoder_dim=64, num_classes=5)
        emb_head = EmbeddingHead(encoder_dim=64, embedding_dim=32)

        model = IRRecognitionModel(encoder, cls_head, emb_head)
        x = torch.randn(2, 3, 224, 224)
        logits, embeddings = model(x)

        assert logits.shape == (2, 5)
        assert embeddings.shape == (2, 32)

    def test_handles_encoder_tuple_output(self):
        """Handles encoder output as tuple (first element is hidden states)."""

        class TupleEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3 * 224 * 224, 16 * 64)

            def forward(self, pixel_values):
                batch_size = pixel_values.shape[0]
                flat = pixel_values.view(batch_size, -1)
                out = self.linear(flat).view(batch_size, 16, 64)
                return (out, None)  # Tuple format

        encoder = TupleEncoder()
        cls_head = ClassificationHead(encoder_dim=64, num_classes=5)
        emb_head = EmbeddingHead(encoder_dim=64, embedding_dim=32)

        model = IRRecognitionModel(encoder, cls_head, emb_head)
        x = torch.randn(2, 3, 224, 224)
        logits, embeddings = model(x)

        assert logits.shape == (2, 5)
        assert embeddings.shape == (2, 32)


class TestQuantizationConfig:
    """Tests for quantization configuration."""

    def test_4bit_config(self):
        """4-bit config uses NF4 quantization."""
        config = get_quantization_config(4)
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True

    def test_8bit_config(self):
        """8-bit config uses load_in_8bit."""
        config = get_quantization_config(8)
        assert config.load_in_8bit is True

    def test_invalid_bits_raises(self):
        """Invalid quantization bits raises ValueError."""
        with pytest.raises(ValueError, match="quantization_bits must be 4 or 8"):
            get_quantization_config(16)


class TestLoraConfig:
    """Tests for LoRA configuration."""

    def test_default_config(self):
        """Default LoRA config has correct rank and alpha."""
        config = get_lora_config(lora_rank=16, lora_alpha=32)
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.bias == "none"

    def test_custom_target_modules(self):
        """Custom target modules are respected."""
        modules = ["q_proj", "v_proj"]
        config = get_lora_config(lora_rank=8, lora_alpha=16, target_modules=modules)
        assert config.r == 8
        assert config.lora_alpha == 16
        # PEFT stores target_modules as a set
        assert set(config.target_modules) == set(modules)

    def test_default_targets_vision_encoder(self):
        """Default target modules target vision encoder attention layers."""
        config = get_lora_config()
        for module in config.target_modules:
            assert "vision" in module or "self_attn" in module


class TestVRAMUtilities:
    """Tests for VRAM utility functions."""

    def test_vram_usage_returns_float(self):
        """VRAM usage returns a float (0.0 if no CUDA)."""
        usage = get_vram_usage_gb()
        assert isinstance(usage, float)
        assert usage >= 0.0

    def test_peak_vram_usage_returns_float(self):
        """Peak VRAM usage returns a float (0.0 if no CUDA)."""
        usage = get_peak_vram_usage_gb()
        assert isinstance(usage, float)
        assert usage >= 0.0


class TestTrainingConfigIntegration:
    """Tests that TrainingConfig works with model loading functions."""

    def test_default_config_valid(self):
        """Default TrainingConfig has valid parameters for model loading."""
        config = TrainingConfig()
        assert config.base_model == "google/paligemma2-3b-pt-224"
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.quantization_bits == 4
        assert config.embedding_dim == 256
        assert config.num_classes == 5
        assert config.max_vram_gb == 8.0

    def test_config_produces_valid_quant_config(self):
        """TrainingConfig parameters produce valid quantization config."""
        config = TrainingConfig()
        quant_config = get_quantization_config(config.quantization_bits)
        assert quant_config.load_in_4bit is True

    def test_config_produces_valid_lora_config(self):
        """TrainingConfig parameters produce valid LoRA config."""
        config = TrainingConfig()
        lora_config = get_lora_config(
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
        )
        assert lora_config.r == config.lora_rank
        assert lora_config.lora_alpha == config.lora_alpha
