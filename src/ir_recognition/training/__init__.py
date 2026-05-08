"""Model Training module.

Fine-tunes PaliGemma 2 vision encoder with QLoRA,
trains classification and embedding heads.
"""

from ir_recognition.training.model import (
    ClassificationHead,
    EmbeddingHead,
    IRRecognitionModel,
    get_lora_config,
    get_quantization_config,
    get_vram_usage_gb,
    get_peak_vram_usage_gb,
    get_vision_encoder_dim,
    load_quantized_model,
)
from ir_recognition.training.dataloader import (
    GradientAccumulationConfig,
    IRSignatureDataset,
    TripletBatchSampler,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_SIZE,
)
from ir_recognition.training.trainer import Trainer

__all__ = [
    "ClassificationHead",
    "EmbeddingHead",
    "IRRecognitionModel",
    "Trainer",
    "get_lora_config",
    "get_quantization_config",
    "get_vram_usage_gb",
    "get_peak_vram_usage_gb",
    "get_vision_encoder_dim",
    "load_quantized_model",
    "GradientAccumulationConfig",
    "IRSignatureDataset",
    "TripletBatchSampler",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "TARGET_SIZE",
]
