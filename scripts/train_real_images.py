#!/usr/bin/env python
"""Train the IR Recognition model on real vehicle images.

Imports images from a folder structure (one subfolder per class),
creates a signature database, generates splits, trains the model,
and computes embeddings.

Usage:
    python scripts/train_real_images.py --images data/real_images --epochs 10
    python scripts/train_real_images.py --images data/real_images --checkpoint checkpoints/real

Folder structure expected:
    data/real_images/
    ├── T-72/
    │   ├── img1.jpeg
    │   └── ...
    ├── Challenger-2/
    │   └── ...
    └── Leopard-2/
        └── ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on real vehicle images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/real_images"),
        help="Path to images directory (subfolders = classes).",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/real_signature_db"),
        help="Path to output signature database.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/real"),
        help="Path to save the trained checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
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
    return parser.parse_args(argv)


def import_images_to_database(images_dir: Path, db_path: Path) -> None:
    """Import images from folder structure into a signature database.

    Args:
        images_dir: Directory with one subfolder per vehicle class.
        db_path: Output path for the signature database.
    """
    from ir_recognition.signature_db.database import SignatureDatabase

    # Create database directory
    db_path.mkdir(parents=True, exist_ok=True)
    db_images_dir = db_path / "images"
    db_images_dir.mkdir(exist_ok=True)

    # Discover classes from subfolders
    class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {images_dir}")

    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    # Import images
    manifest_images = []
    total_imported = 0

    for class_dir in class_dirs:
        vehicle_class = class_dir.name
        class_output_dir = db_images_dir / vehicle_class
        class_output_dir.mkdir(exist_ok=True)

        # Find all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        print(f"  {vehicle_class}: {len(image_files)} images")

        for idx, img_path in enumerate(image_files):
            # Generate image_id
            image_id = f"{vehicle_class.lower().replace('-', '')}_{idx:03d}"

            # Load, convert to RGB, resize to 224x224, save as PNG
            try:
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((224, 224), Image.LANCZOS)

                output_path = class_output_dir / f"{image_id}.png"
                img_resized.save(output_path, format="PNG")

                manifest_images.append({
                    "image_id": image_id,
                    "vehicle_class": vehicle_class,
                    "azimuth": 0.0,
                    "elevation": 0.0,
                    "source_type": "real",
                    "file_path": f"images/{vehicle_class}/{image_id}.png",
                })
                total_imported += 1
            except Exception as e:
                logger.warning(f"Failed to import {img_path}: {e}")

    # Create manifest
    manifest = {
        "version": "1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vehicle_classes": [d.name for d in class_dirs],
        "images": manifest_images,
        "splits": {"train": 0.70, "val": 0.15, "test": 0.15},
    }

    with open(db_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nImported {total_imported} images total")

    # Generate splits
    database = SignatureDatabase(db_path)
    database.generate_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    stats = database.get_stats()
    train_split = database.get_split("train")
    val_split = database.get_split("val")
    test_split = database.get_split("test")
    print(f"Splits: train={len(train_split)}, val={len(val_split)}, test={len(test_split)}")

    return database


def main() -> None:
    args = parse_args()

    if not args.images.exists():
        print(f"Error: Images directory not found: {args.images}", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("Error: CUDA not available. Training requires GPU.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("VEHICLE RECOGNITION - REAL IMAGE TRAINING")
    print("=" * 60)
    print()

    # Step 1: Import images into database
    print("=== Step 1: Importing images ===")
    from ir_recognition.signature_db.database import SignatureDatabase

    import_images_to_database(args.images, args.database)
    database = SignatureDatabase(args.database)
    print()

    # Step 2: Update model config for the new classes
    vehicle_classes = sorted(set(
        m.vehicle_class for m in database.get_split("train")
    ))
    num_classes = len(vehicle_classes)
    print(f"Training classes ({num_classes}): {vehicle_classes}")
    print()

    # Step 3: Train
    print("=== Step 2: Training ===")
    from ir_recognition.models import TrainingConfig
    from ir_recognition.training.trainer import Trainer

    config = TrainingConfig(
        num_classes=num_classes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
    )

    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LoRA rank: {config.lora_rank}")
    print()

    trainer = Trainer(config=config, database=database)

    start_time = time.time()
    metrics_history = trainer.train()
    elapsed = time.time() - start_time

    print(f"\nTraining complete in {elapsed:.1f}s")
    if metrics_history:
        final = metrics_history[-1]
        print(f"  Final accuracy: {final.classification_accuracy:.4f}")
        print(f"  Final val loss: {final.val_loss:.4f}")
        print(f"  Peak VRAM: {final.vram_peak_gb:.2f} GB")
    print()

    # Step 4: Save checkpoint
    print("=== Step 3: Saving checkpoint ===")
    trainer.save_checkpoint(args.checkpoint)
    print(f"  Saved to: {args.checkpoint}")
    print()

    # Step 5: Compute embeddings
    print("=== Step 4: Computing embeddings ===")
    from ir_recognition.inference.embeddings import compute_all_embeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer.model.eval()
    compute_all_embeddings(
        model_backend=trainer.model,
        database=database,
        device=device,
    )
    print("  Embeddings computed and saved")
    print()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Classes: {vehicle_classes}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Database: {args.database}")
    print(f"  To run web UI:")
    print(f"    python -m ir_recognition.web --model-path {args.checkpoint} --db-path {args.database}")
    print("=" * 60)


if __name__ == "__main__":
    main()
