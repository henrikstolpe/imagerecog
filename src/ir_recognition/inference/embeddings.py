"""Embedding pre-computation for the signature database.

Computes embeddings for all images in the database using the trained model
and stores them as an .npz file for fast similarity search during inference.

Supports:
- Full computation: compute embeddings for all images in the database
- Incremental updates: only compute embeddings for new images not already stored

Storage format:
    {db_path}/embeddings/embeddings.npz
    Keys:
        - "embeddings": (N, 256) float32 array of L2-normalized embeddings
        - "image_ids": (N,) array of image ID strings
        - "labels": (N,) array of vehicle class label strings
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from ir_recognition.inference.preprocessing import preprocess_image

logger = logging.getLogger(__name__)


class ModelBackend(Protocol):
    """Protocol for model backends used in embedding computation.

    Any object implementing forward() with the correct signature can be used.
    """

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Args:
            pixel_values: Input tensor of shape (1, 3, 224, 224).

        Returns:
            Tuple of (logits, embeddings) where:
                - logits: shape (1, num_classes) raw classification logits
                - embeddings: shape (1, embedding_dim) L2-normalized embeddings
        """
        ...


def _get_embeddings_path(database) -> Path:
    """Get the path to the embeddings .npz file for a database.

    Args:
        database: SignatureDatabase instance.

    Returns:
        Path to the embeddings.npz file.
    """
    return Path(database.db_path) / "embeddings" / "embeddings.npz"


def _load_existing_embeddings(embeddings_path: Path) -> tuple[np.ndarray | None, list[str], list[str]]:
    """Load existing embeddings from disk if available.

    Args:
        embeddings_path: Path to the embeddings.npz file.

    Returns:
        Tuple of (embeddings_array, image_ids_list, labels_list).
        Returns (None, [], []) if file doesn't exist or can't be loaded.
    """
    if not embeddings_path.exists():
        return None, [], []

    try:
        data = np.load(embeddings_path, allow_pickle=True)
        embeddings = data["embeddings"].astype(np.float32)
        image_ids = data["image_ids"].tolist()
        labels = data["labels"].tolist()
        return embeddings, image_ids, labels
    except Exception as e:
        logger.warning(f"Failed to load existing embeddings from {embeddings_path}: {e}")
        return None, [], []


def _save_embeddings(
    embeddings_path: Path,
    embeddings: np.ndarray,
    image_ids: list[str],
    labels: list[str],
) -> None:
    """Save embeddings to disk as .npz file.

    Args:
        embeddings_path: Path to save the embeddings.npz file.
        embeddings: (N, embedding_dim) float32 array.
        image_ids: List of N image ID strings.
        labels: List of N vehicle class label strings.
    """
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        embeddings_path,
        embeddings=embeddings.astype(np.float32),
        image_ids=np.array(image_ids, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    logger.info(f"Saved {len(image_ids)} embeddings to {embeddings_path}")


def _compute_embedding_for_image(
    model_backend: ModelBackend,
    image_path: Path,
    device: str = "cpu",
) -> np.ndarray | None:
    """Compute the embedding for a single image.

    Args:
        model_backend: Model backend implementing the forward() protocol.
        image_path: Path to the image file.
        device: Torch device to run inference on.

    Returns:
        L2-normalized embedding as a 1D numpy array, or None if preprocessing fails.
    """
    tensor, error = preprocess_image(image_path)
    if tensor is None:
        logger.warning(f"Failed to preprocess image {image_path}: {error}")
        return None

    tensor = tensor.to(device)

    with torch.no_grad():
        _, embedding = model_backend.forward(tensor)

    # Convert to numpy, squeeze batch dimension
    embedding_np = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return embedding_np


def compute_all_embeddings(
    model_backend: ModelBackend,
    database,
    device: str = "cpu",
) -> None:
    """Compute embeddings for all images in the database and save to disk.

    Iterates over every image in the database manifest, computes its embedding
    using the provided model backend, and saves all embeddings as an .npz file.

    Args:
        model_backend: Model backend implementing the forward() protocol.
        database: SignatureDatabase instance with images to embed.
        device: Torch device for inference. Default "cpu".
    """
    embeddings_path = _get_embeddings_path(database)
    manifest = database.get_manifest()

    if manifest.empty:
        logger.warning("Database is empty, no embeddings to compute.")
        return

    all_embeddings = []
    all_image_ids = []
    all_labels = []

    total = len(manifest)
    logger.info(f"Computing embeddings for {total} images...")

    for idx, row in manifest.iterrows():
        image_id = row["image_id"]
        vehicle_class = row["vehicle_class"]
        file_path = Path(database.db_path) / row["file_path"]

        embedding = _compute_embedding_for_image(model_backend, file_path, device)
        if embedding is not None:
            all_embeddings.append(embedding)
            all_image_ids.append(image_id)
            all_labels.append(vehicle_class)
        else:
            logger.warning(f"Skipping image {image_id}: embedding computation failed")

        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/{total} images")

    if not all_embeddings:
        logger.warning("No embeddings were computed successfully.")
        return

    embeddings_array = np.stack(all_embeddings, axis=0)
    _save_embeddings(embeddings_path, embeddings_array, all_image_ids, all_labels)


def compute_incremental_embeddings(
    model_backend: ModelBackend,
    database,
    device: str = "cpu",
) -> int:
    """Compute embeddings only for new images not already in the embeddings file.

    Loads existing embeddings (if any), identifies images in the database manifest
    that don't have stored embeddings, computes embeddings for those new images,
    and saves the combined result.

    Args:
        model_backend: Model backend implementing the forward() protocol.
        database: SignatureDatabase instance with images to embed.
        device: Torch device for inference. Default "cpu".

    Returns:
        Number of new embeddings that were computed and added.
    """
    embeddings_path = _get_embeddings_path(database)
    manifest = database.get_manifest()

    if manifest.empty:
        logger.warning("Database is empty, no embeddings to compute.")
        return 0

    # Load existing embeddings
    existing_embeddings, existing_ids, existing_labels = _load_existing_embeddings(
        embeddings_path
    )

    # Determine which images are new
    existing_id_set = set(existing_ids)
    new_rows = manifest[~manifest["image_id"].isin(existing_id_set)]

    if new_rows.empty:
        logger.info("All images already have embeddings. Nothing to compute.")
        return 0

    logger.info(
        f"Found {len(new_rows)} new images to embed "
        f"(existing: {len(existing_ids)}, total in DB: {len(manifest)})"
    )

    # Compute embeddings for new images
    new_embeddings = []
    new_image_ids = []
    new_labels = []

    for idx, (_, row) in enumerate(new_rows.iterrows()):
        image_id = row["image_id"]
        vehicle_class = row["vehicle_class"]
        file_path = Path(database.db_path) / row["file_path"]

        embedding = _compute_embedding_for_image(model_backend, file_path, device)
        if embedding is not None:
            new_embeddings.append(embedding)
            new_image_ids.append(image_id)
            new_labels.append(vehicle_class)
        else:
            logger.warning(f"Skipping image {image_id}: embedding computation failed")

        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/{len(new_rows)} new images")

    if not new_embeddings:
        logger.info("No new embeddings were computed successfully.")
        return 0

    new_count = len(new_embeddings)
    new_embeddings_array = np.stack(new_embeddings, axis=0)

    # Merge with existing embeddings
    if existing_embeddings is not None and len(existing_ids) > 0:
        combined_embeddings = np.concatenate(
            [existing_embeddings, new_embeddings_array], axis=0
        )
        combined_ids = existing_ids + new_image_ids
        combined_labels = existing_labels + new_labels
    else:
        combined_embeddings = new_embeddings_array
        combined_ids = new_image_ids
        combined_labels = new_labels

    _save_embeddings(embeddings_path, combined_embeddings, combined_ids, combined_labels)

    logger.info(f"Added {new_count} new embeddings (total: {len(combined_ids)})")
    return new_count
