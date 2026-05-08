"""Model loading and architecture for IR Signature Recognition.

Provides:
- QLoRA quantization configuration for PaliGemma 2 3B
- LoRA adapter application to vision encoder attention layers
- Classification and embedding heads
- Combined model wrapping everything together

The architecture classes (heads, combined model) can be instantiated
and tested without GPU. The actual model loading from HuggingFace
requires GPU and is handled by `load_quantized_model()`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ir_recognition.models import TrainingConfig


class ClassificationHead(nn.Module):
    """Linear classification head mapping encoder features to class logits.

    Args:
        encoder_dim: Dimensionality of the vision encoder output.
        num_classes: Number of vehicle classes to classify.
    """

    def __init__(self, encoder_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(encoder_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: Encoder output tensor of shape (batch_size, encoder_dim).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.linear(x)


class EmbeddingHead(nn.Module):
    """Embedding head mapping encoder features to L2-normalized vectors.

    Args:
        encoder_dim: Dimensionality of the vision encoder output.
        embedding_dim: Dimensionality of the output embedding (default 256).
    """

    def __init__(self, encoder_dim: int, embedding_dim: int = 256) -> None:
        super().__init__()
        self.linear = nn.Linear(encoder_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning L2-normalized embeddings.

        Args:
            x: Encoder output tensor of shape (batch_size, encoder_dim).

        Returns:
            L2-normalized embedding tensor of shape (batch_size, embedding_dim).
        """
        projected = self.linear(x)
        return F.normalize(projected, p=2, dim=-1)


class IRRecognitionModel(nn.Module):
    """Combined model wrapping vision encoder with classification and embedding heads.

    This model takes image tensors and produces both classification logits
    and L2-normalized embeddings in a single forward pass.

    Args:
        vision_encoder: The vision encoder backbone (e.g., SigLIP from PaliGemma).
        classification_head: Head for vehicle class prediction.
        embedding_head: Head for metric learning embeddings.
        pool_strategy: How to pool vision encoder outputs. One of "mean", "cls".
            Default "mean" averages all patch tokens.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        classification_head: ClassificationHead,
        embedding_head: EmbeddingHead,
        pool_strategy: str = "mean",
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.classification_head = classification_head
        self.embedding_head = embedding_head
        self.pool_strategy = pool_strategy

    def pool_features(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Pool sequence of patch tokens into a single feature vector.

        Args:
            encoder_output: Tensor of shape (batch_size, seq_len, hidden_dim).

        Returns:
            Pooled tensor of shape (batch_size, hidden_dim).
        """
        if self.pool_strategy == "cls":
            return encoder_output[:, 0, :]
        # Default: mean pooling over all tokens
        return encoder_output.mean(dim=1)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and both heads.

        Args:
            pixel_values: Input image tensor of shape (batch_size, 3, 224, 224).

        Returns:
            Tuple of (logits, embeddings) where:
                - logits: shape (batch_size, num_classes)
                - embeddings: shape (batch_size, embedding_dim), L2-normalized
        """
        # Get vision encoder output
        encoder_output = self.vision_encoder(pixel_values=pixel_values)

        # Handle different encoder output formats
        if hasattr(encoder_output, "last_hidden_state"):
            hidden_states = encoder_output.last_hidden_state
        elif isinstance(encoder_output, tuple):
            hidden_states = encoder_output[0]
        else:
            hidden_states = encoder_output

        # Pool to single vector per image
        pooled = self.pool_features(hidden_states)

        # Pass through both heads
        logits = self.classification_head(pooled)
        embeddings = self.embedding_head(pooled)

        return logits, embeddings


def get_quantization_config(quantization_bits: int = 4):
    """Create BitsAndBytes quantization config for QLoRA.

    Args:
        quantization_bits: Number of bits for quantization (4 or 8).

    Returns:
        BitsAndBytesConfig for model loading.

    Raises:
        ValueError: If quantization_bits is not 4 or 8.
    """
    from transformers import BitsAndBytesConfig

    if quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"quantization_bits must be 4 or 8, got {quantization_bits}"
        )


def get_lora_config(
    lora_rank: int = 16,
    lora_alpha: int = 32,
    target_modules: list[str] | None = None,
):
    """Create LoRA configuration for vision encoder attention layers.

    Args:
        lora_rank: Rank of LoRA adapters.
        lora_alpha: Alpha scaling factor for LoRA.
        target_modules: List of module name patterns to apply LoRA to.
            Defaults to vision encoder attention projection layers.

    Returns:
        LoraConfig for PEFT model wrapping.
    """
    from peft import LoraConfig, TaskType

    if target_modules is None:
        # Target attention layers in the SigLIP vision encoder
        # PaliGemma 2 uses a SigLIP-So400m vision encoder with standard
        # attention projections: q_proj, k_proj, v_proj, out_proj
        # Use simple module name matching (peft matches by suffix)
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

    return LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )


def get_vision_encoder_dim(model) -> int:
    """Extract the hidden dimension of the vision encoder from a PaliGemma model.

    Args:
        model: A loaded PaliGemma model instance.

    Returns:
        The hidden dimension (int) of the vision encoder.
    """
    if hasattr(model, "config") and hasattr(model.config, "vision_config"):
        return model.config.vision_config.hidden_size
    # Fallback: SigLIP-So400m/14 has hidden_size=1152
    return 1152


def load_quantized_model(config: TrainingConfig) -> IRRecognitionModel:
    """Load PaliGemma 2 with QLoRA quantization and create the full model.

    This function requires GPU access and downloads the model from HuggingFace.

    Steps:
    1. Load base model with NF4 quantization via bitsandbytes
    2. Extract the vision encoder (SigLIP-So400m)
    3. Apply LoRA adapters to vision encoder attention layers
    4. Create classification and embedding heads
    5. Wrap everything in IRRecognitionModel

    Args:
        config: TrainingConfig with model parameters.

    Returns:
        IRRecognitionModel ready for training.

    Raises:
        RuntimeError: If CUDA is not available or VRAM budget is exceeded.
    """
    from transformers import PaliGemmaForConditionalGeneration
    from peft import get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Model loading requires GPU access."
        )

    # Step 1: Load base model with quantization
    quant_config = get_quantization_config(config.quantization_bits)

    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        config.base_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Step 2: Prepare for k-bit training (enables gradient computation on quantized model)
    base_model = prepare_model_for_kbit_training(base_model)

    # Step 3: Apply LoRA adapters to the full model (targeting vision encoder attention)
    lora_config = get_lora_config(
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
    )
    # Apply LoRA to the full model — targets will match vision encoder attention layers
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Step 4: Extract vision encoder for forward passes (not PEFT-wrapped)
    if hasattr(base_model.model, "model") and hasattr(base_model.model.model, "vision_tower"):
        vision_encoder = base_model.model.model.vision_tower
    elif hasattr(base_model, "model") and hasattr(base_model.model, "vision_tower"):
        vision_encoder = base_model.model.vision_tower
    elif hasattr(base_model, "vision_tower"):
        vision_encoder = base_model.vision_tower
    elif hasattr(base_model, "vision_model"):
        vision_encoder = base_model.vision_model
    elif hasattr(base_model, "model") and hasattr(base_model.model, "vision_model"):
        vision_encoder = base_model.model.vision_model
    else:
        raise RuntimeError(
            "Could not find vision encoder in model. "
            "Expected 'vision_tower' or 'vision_model' attribute."
        )

    # Step 5: Determine encoder dimension and create heads
    encoder_dim = get_vision_encoder_dim(base_model)

    classification_head = ClassificationHead(
        encoder_dim=encoder_dim,
        num_classes=config.num_classes,
    )
    embedding_head = EmbeddingHead(
        encoder_dim=encoder_dim,
        embedding_dim=config.embedding_dim,
    )

    # Move heads to GPU
    device = next(vision_encoder.parameters()).device
    classification_head = classification_head.to(device)
    embedding_head = embedding_head.to(device)

    # Step 6: Create combined model
    model = IRRecognitionModel(
        vision_encoder=vision_encoder,
        classification_head=classification_head,
        embedding_head=embedding_head,
        pool_strategy="mean",
    )

    # Step 7: Verify VRAM usage
    vram_gb = get_vram_usage_gb()
    if vram_gb > config.max_vram_gb:
        raise RuntimeError(
            f"VRAM usage ({vram_gb:.2f} GB) exceeds budget "
            f"({config.max_vram_gb} GB). Consider reducing model size "
            f"or increasing quantization."
        )

    return model


def get_vram_usage_gb() -> float:
    """Get current GPU VRAM usage in GB.

    Returns:
        VRAM usage in GB, or 0.0 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**3)


def get_peak_vram_usage_gb() -> float:
    """Get peak GPU VRAM usage in GB.

    Returns:
        Peak VRAM usage in GB, or 0.0 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)
