from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


COLOR_LABELS = ("merah", "oranye", "kuning", "hijau", "coklat", "gelap")


def _classify_hsv_pixels(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    labels = np.full(hue.shape, "lain", dtype=object)
    labels[value < 45] = "gelap"

    chromatic = (saturation >= 35) & (value >= 45)
    brown = chromatic & (value < 130) & ((hue < 35) | (hue >= 225))
    labels[brown] = "coklat"

    red = chromatic & ~brown & ((hue < 12) | (hue >= 235))
    orange = chromatic & ~brown & (hue >= 12) & (hue < 28)
    yellow = chromatic & ~brown & (hue >= 28) & (hue < 55)
    green = chromatic & ~brown & (hue >= 55) & (hue < 115)

    labels[red] = "merah"
    labels[orange] = "oranye"
    labels[yellow] = "kuning"
    labels[green] = "hijau"
    return labels


def extract_visual_parameters(image: Image.Image) -> dict[str, Any]:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    analysis_size = (224, 224)
    resized = rgb_image.resize(analysis_size, Image.Resampling.LANCZOS)
    hsv = np.asarray(resized.convert("HSV"), dtype=np.float32)

    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    object_mask = (saturation > 35) & (value > 35)
    if int(object_mask.sum()) < 100:
        object_mask = value > 20

    object_area_percent = float(object_mask.mean() * 100.0)
    selected_hue = hue[object_mask]
    selected_saturation = saturation[object_mask]
    selected_value = value[object_mask]

    if selected_hue.size == 0:
        dominant_color = "tidak diketahui"
        dominant_color_percent = 0.0
    else:
        labels = _classify_hsv_pixels(
            selected_hue,
            selected_saturation,
            selected_value,
        )
        counts = {
            label: int(np.count_nonzero(labels == label))
            for label in COLOR_LABELS
        }
        dominant_color = max(counts, key=counts.get)
        dominant_count = counts[dominant_color]
        if dominant_count == 0:
            dominant_color = "tidak diketahui"
            dominant_color_percent = 0.0
        else:
            dominant_color_percent = dominant_count / float(selected_hue.size) * 100.0

    return {
        "width": width,
        "height": height,
        "megapixels": round(width * height / 1_000_000, 2),
        "object_area_percent": round(object_area_percent, 2),
        "dominant_color": dominant_color,
        "dominant_color_percent": round(dominant_color_percent, 2),
    }
