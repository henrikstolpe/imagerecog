"""Inference Pipeline module.

Unified entry point for classification and similarity-based recognition.
"""

from ir_recognition.inference.embeddings import (
    compute_all_embeddings,
    compute_incremental_embeddings,
)
from ir_recognition.inference.pipeline import InferencePipeline
from ir_recognition.inference.preprocessing import preprocess_image

__all__ = [
    "InferencePipeline",
    "compute_all_embeddings",
    "compute_incremental_embeddings",
    "preprocess_image",
]
