#!/usr/bin/env python
"""Train the IR Signature Recognition model.

Loads a signature database, initializes the Trainer with QLoRA-quantized
PaliGemma 2 3B, runs the training loop with combined classification and
metric learning loss, saves the best checkpoint, and computes embeddings
for all database images.

Usage:
    python scripts/train.py --database data/signature_db --checkpoint checkpoints/best
    python scripts/train.py --database data/signature_db --epochs 10 --batch-size 2
    python scripts/train.py --database data/signature_db --lora-rank 8 --lora-alpha 16

Requirements: 3.1, 3.6, 3.7
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from ir_recognition.inference.embeddings import compute_all_embeddings
from ir_recognition.models import TrainingConfig
from ir_recognition.signature_db.database import SignatureDatabase
from ir_recognition.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
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
        description="Train the IR Signature Recognition model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="Path to the signature database directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best"),
        help="Path to save the best checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for the training script."""
    args = parse_args()

    # Validate database path
    if not args.database.exists():
        logger.error(f"Database directory not found: {args.database}")
        sys.exit(1)

    manifest_path = args.database / "manifest.json"
    if not manifest_path.exists():
        logger.error(
            f"No manifest.json found in {args.database}. "
            "Run generate_dataset.py first to create the database."
        )
        sys.exit(1)

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. Training requires GPU access. "
            "Please ensure an NVIDIA GPU with CUDA support is available."
        )
        sys.exit(1)

    print("=" * 60)
    print("IR SIGNATURE RECOGNITION - MODEL TRAINING")
    print("=" * 60)
    print()

    # Load database
    print(f"Loading database from: {args.database}")
    database = SignatureDatabase(args.database)
    stats = database.get_stats()
    print(f"  Total images: {stats['total']}")
    print(f"  Classes: {list(stats['per_class'].keys())}")
    print()

    # Verify splits exist
    train_split = database.get_split("train")
    val_split = database.get_split("val")
    if not train_split:
        logger.error(
            "No training split found. Run generate_dataset.py with split "
            "generation or call database.generate_splits() first."
        )
        sys.exit(1)
    if not val_split:
        logger.error("No validation split found in the database.")
        sys.exit(1)

    print(f"  Train split: {len(train_split)} images")
    print(f"  Val split:   {len(val_split)} images")
    print()

    # Create training config
    config = TrainingConfig(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
    )

    print("Training configuration:")
    print(f"  Base model:       {config.base_model}")
    print(f"  LoRA rank:        {config.lora_rank}")
    print(f"  LoRA alpha:       {config.lora_alpha}")
    print(f"  Quantization:     {config.quantization_bits}-bit (QLoRA)")
    print(f"  Embedding dim:    {config.embedding_dim}")
    print(f"  Num classes:      {config.num_classes}")
    print(f"  Batch size:       {config.batch_size}")
    print(f"  Learning rate:    {config.learning_rate}")
    print(f"  Epochs:           {config.num_epochs}")
    print(f"  Triplet margin:   {config.triplet_margin}")
    print(f"  Loss weights:     {config.loss_weights}")
    print(f"  Max VRAM:         {config.max_vram_gb} GB")
    print(f"  Checkpoint path:  {args.checkpoint}")
    print()

    # Initialize trainer (loads model with QLoRA)
    print("Initializing trainer (loading model with QLoRA)...")
    start_load = time.time()
    trainer = Trainer(config=config, database=database)
    load_elapsed = time.time() - start_load
    print(f"  Model loaded in {load_elapsed:.1f}s")
    print(f"  Current VRAM usage: {trainer.get_vram_usage():.2f} GB")
    print()

    # Run training loop
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print()

    start_train = time.time()
    metrics_history = trainer.train()
    train_elapsed = time.time() - start_train

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total training time: {train_elapsed:.1f}s")
    print(f"  Epochs completed: {len(metrics_history)}")
    print()

    # Print final metrics
    if metrics_history:
        final = metrics_history[-1]
        print("Final epoch metrics:")
        print(f"  Train loss:       {final.train_loss:.4f}")
        print(f"  Val loss:         {final.val_loss:.4f}")
        print(f"  Accuracy:         {final.classification_accuracy:.4f}")
        print(f"  Embedding quality:{final.embedding_quality:.4f}")
        print(f"  Peak VRAM:        {final.vram_peak_gb:.2f} GB")
        print(f"  Learning rate:    {final.learning_rate:.2e}")
        print()

        # Find best epoch by validation loss
        best_epoch = min(metrics_history, key=lambda m: m.val_loss)
        print(f"Best epoch (by val loss): {best_epoch.epoch}")
        print(f"  Val loss:         {best_epoch.val_loss:.4f}")
        print(f"  Accuracy:         {best_epoch.classification_accuracy:.4f}")
        print()

    # Save checkpoint
    print(f"Saving checkpoint to: {args.checkpoint}")
    trainer.save_checkpoint(args.checkpoint)
    print("  Checkpoint saved successfully.")
    print()

    # Compute embeddings for all database images
    print("Computing embeddings for all database images...")
    start_embed = time.time()

    # Use the trained model as the backend for embedding computation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer.model.eval()
    compute_all_embeddings(
        model_backend=trainer.model,
        database=database,
        device=device,
    )

    embed_elapsed = time.time() - start_embed
    print(f"  Embeddings computed in {embed_elapsed:.1f}s")
    print()

    # Print summary
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Database:         {args.database}")
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Epochs:           {len(metrics_history)}")
    print(f"  Training time:    {train_elapsed:.1f}s")
    if metrics_history:
        print(f"  Final accuracy:   {metrics_history[-1].classification_accuracy:.4f}")
        print(f"  Best val loss:    {best_epoch.val_loss:.4f} (epoch {best_epoch.epoch})")
    print(f"  Embeddings:       computed and stored")
    print(f"  Peak VRAM:        {trainer.get_vram_usage():.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
