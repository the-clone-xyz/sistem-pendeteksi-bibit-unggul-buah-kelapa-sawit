from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(image_source: str | Path | BinaryIO) -> Image.Image:
    """Load an image, apply EXIF orientation, and convert it to RGB."""
    image = Image.open(image_source)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def preprocess_image(
    image: Image.Image,
    image_size: tuple[int, int] = IMAGE_SIZE,
    add_batch: bool = True,
) -> np.ndarray:
    """Resize and normalize image pixels to the 0..1 range."""
    resized = image.convert("RGB").resize(image_size, Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0

    if add_batch:
        array = np.expand_dims(array, axis=0)

    return array


def count_images(directory: str | Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> int:
    """Count image files in a directory recursively."""
    directory = Path(directory)
    normalized_extensions = {ext.lower() for ext in extensions}

    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )


def dataset_split_summary(
    dataset_root: str | Path,
    class_names: Iterable[str],
    splits: Iterable[str] = ("train", "validation", "test"),
) -> dict[str, dict[str, int]]:
    """Return image counts for each split and class."""
    dataset_root = Path(dataset_root)
    summary: dict[str, dict[str, int]] = {}

    for split in splits:
        summary[split] = {}
        for class_name in class_names:
            summary[split][class_name] = count_images(dataset_root / split / class_name)

    return summary
