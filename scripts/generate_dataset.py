#!/usr/bin/env python
"""Generate a synthetic IR signature dataset.

Creates a signature database populated with synthetic IR images for all
supported vehicle classes across multiple viewing angles. Generates
train/val/test splits and prints dataset statistics.

Usage:
    python scripts/generate_dataset.py --output data/signature_db
    python scripts/generate_dataset.py --output data/signature_db --num-angles 10 --images-per-angle 15
    python scripts/generate_dataset.py --output data/signature_db --seed 42

Requirements: 1.3, 1.4, 2.2, 2.4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ir_recognition.ir_generator.generator import IRGenerator
from ir_recognition.models import (
    SUPPORTED_VEHICLE_CLASSES,
    ImageMetadata,
    IRGeneratorConfig,
)
from ir_recognition.signature_db.database import SignatureDatabase


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate a synthetic IR signature dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the signature database.",
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=8,
        help="Number of azimuth angles per elevation level.",
    )
    parser.add_argument(
        "--images-per-angle",
        type=int,
        default=13,
        help="Number of images to generate per azimuth/elevation combination.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args(argv)


def generate_image_id(
    vehicle_class: str, azimuth: float, elevation: float, index: int
) -> str:
    """Generate a meaningful image ID.

    Format: {class_short}_synth_az{azimuth:03d}_el{elevation:03d}_{index:03d}

    Args:
        vehicle_class: The vehicle class name.
        azimuth: Azimuth angle in degrees.
        elevation: Elevation angle in degrees.
        index: Image index within this angle combination.

    Returns:
        A unique, descriptive image ID string.
    """
    # Create a short class identifier
    class_short = vehicle_class.lower().replace("-", "").replace("_", "")
    az_int = int(round(azimuth))
    el_int = int(round(elevation))
    return f"{class_short}_synth_az{az_int:03d}_el{el_int:03d}_{index:03d}"


def generate_dataset(
    output_dir: Path,
    num_angles: int = 8,
    images_per_angle: int = 13,
    seed: int = 42,
) -> None:
    """Generate the full synthetic IR dataset.

    Creates images for all supported vehicle classes across multiple
    azimuth and elevation angles, stores them in a SignatureDatabase,
    generates train/val/test splits, and prints statistics.

    Args:
        output_dir: Path to the output database directory.
        num_angles: Number of azimuth angles per elevation level.
        images_per_angle: Number of images per azimuth/elevation combination.
        seed: Random seed for reproducibility.
    """
    print(f"Generating synthetic IR dataset")
    print(f"  Output directory: {output_dir}")
    print(f"  Vehicle classes: {len(SUPPORTED_VEHICLE_CLASSES)}")
    print(f"  Azimuth angles: {num_angles}")
    print(f"  Images per angle: {images_per_angle}")
    print(f"  Seed: {seed}")
    print()

    # Elevation angles to cover multiple viewing perspectives
    elevations = [15.0, 30.0, 45.0]

    # Calculate expected totals
    images_per_class = num_angles * len(elevations) * images_per_angle
    total_images = images_per_class * len(SUPPORTED_VEHICLE_CLASSES)
    print(f"  Elevations: {elevations}")
    print(f"  Images per class: {images_per_class}")
    print(f"  Total images to generate: {total_images}")
    print()

    # Verify we meet the 100+ images per class requirement
    if images_per_class < 100:
        print(
            f"WARNING: {images_per_class} images per class is below the "
            f"minimum of 100. Consider increasing --num-angles or --images-per-angle."
        )

    # Initialize generator and database
    generator = IRGenerator(seed=seed)
    database = SignatureDatabase(output_dir)

    start_time = time.time()
    generated_count = 0

    # Compute evenly-spaced azimuth angles
    azimuths = [360.0 * i / num_angles for i in range(num_angles)]

    for class_idx, vehicle_class in enumerate(SUPPORTED_VEHICLE_CLASSES):
        class_start = time.time()
        class_count = 0

        print(
            f"[{class_idx + 1}/{len(SUPPORTED_VEHICLE_CLASSES)}] "
            f"Generating images for {vehicle_class}..."
        )

        for elevation in elevations:
            for az_idx, azimuth in enumerate(azimuths):
                for img_idx in range(images_per_angle):
                    # Generate image ID
                    image_id = generate_image_id(
                        vehicle_class, azimuth, elevation, img_idx + 1
                    )

                    # Create generator config with slight variation per image
                    # Use a different ambient temp for variety
                    ambient_temp = 293.0 + (img_idx - images_per_angle // 2) * 2.0
                    ambient_temp = max(250.0, ambient_temp)  # Keep reasonable

                    config = IRGeneratorConfig(
                        vehicle_type=vehicle_class,
                        azimuth=azimuth,
                        elevation=elevation,
                        ambient_temp=ambient_temp,
                        image_size=(224, 224),
                    )

                    # Generate the image
                    image = generator.generate(config)

                    # Create metadata
                    metadata = ImageMetadata(
                        image_id=image_id,
                        vehicle_class=vehicle_class,
                        azimuth=azimuth,
                        elevation=elevation,
                        source_type="synthetic",
                        file_path=Path("placeholder"),
                    )

                    # Store in database
                    database.add_image(image, metadata)

                    class_count += 1
                    generated_count += 1

        class_elapsed = time.time() - class_start
        print(
            f"  -> {class_count} images generated in {class_elapsed:.1f}s"
        )

    total_elapsed = time.time() - start_time
    print()
    print(f"Generation complete: {generated_count} images in {total_elapsed:.1f}s")
    print()

    # Generate train/val/test splits
    print("Generating train/val/test splits (70/15/15)...")
    database.generate_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed)
    print("  Splits generated successfully.")
    print()

    # Print dataset statistics
    stats = database.get_stats()
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total images: {stats['total']}")
    print()

    print("Per-class counts:")
    for cls, count in sorted(stats["per_class"].items()):
        print(f"  {cls:15s}: {count:5d} images")
    print()

    print("Per-angle bucket counts:")
    for angle, count in sorted(stats["per_angle"].items()):
        print(f"  {angle:10s}: {count:5d} images")
    print()

    print("Per-source type:")
    for source, count in sorted(stats["per_source_type"].items()):
        print(f"  {source:12s}: {count:5d} images")
    print()

    # Print split sizes
    train_images = database.get_split("train")
    val_images = database.get_split("val")
    test_images = database.get_split("test")
    print("Split sizes:")
    print(f"  Train: {len(train_images):5d} images ({len(train_images)/stats['total']*100:.1f}%)")
    print(f"  Val:   {len(val_images):5d} images ({len(val_images)/stats['total']*100:.1f}%)")
    print(f"  Test:  {len(test_images):5d} images ({len(test_images)/stats['total']*100:.1f}%)")
    print()
    print("=" * 60)
    print(f"Database saved to: {output_dir.resolve()}")
    print("=" * 60)


def main() -> None:
    """Entry point for the dataset generation script."""
    args = parse_args()
    generate_dataset(
        output_dir=args.output,
        num_angles=args.num_angles,
        images_per_angle=args.images_per_angle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
