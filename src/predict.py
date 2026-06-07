from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .preprocessing import IMAGE_SIZE, load_image, preprocess_image
except ImportError:  # Allows: python src/predict.py image.jpg
    from preprocessing import IMAGE_SIZE, load_image, preprocess_image


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_CANDIDATES = (
    ROOT_DIR / "models" / "model_sawit.keras",
    ROOT_DIR / "models" / "model_sawit_retrained.keras",
    ROOT_DIR / "models" / "model_sawit.h5",
)

CLASS_NAMES = ("matang", "setengah_matang", "mentah")
DISPLAY_LABELS = {
    "matang": "Matang",
    "setengah_matang": "Setengah Matang",
    "mentah": "Mentah",
}
RECOMMENDATIONS = {
    "matang": "Layak / Direkomendasikan",
    "setengah_matang": "Perlu Pemeriksaan Lanjutan",
    "mentah": "Tidak Direkomendasikan",
}


@dataclass(frozen=True)
class PredictionResult:
    class_name: str
    display_label: str
    confidence: float
    recommendation: str
    probabilities: dict[str, float]
    source: str
    model_path: str | None = None
    note: str | None = None


def find_model_path(preferred_path: str | Path | None = None) -> Path | None:
    if preferred_path:
        path = Path(preferred_path)
        return path if path.exists() else None

    for path in MODEL_CANDIDATES:
        if path.exists():
            return path

    return None


def make_recommendation(class_name: str) -> str:
    return RECOMMENDATIONS.get(class_name, "Perlu Pemeriksaan Lanjutan")


@lru_cache(maxsize=2)
def _load_model_cached(model_path: str):
    from tensorflow.keras.models import load_model

    return load_model(model_path)


def _normalize_model_scores(raw_scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)

    if scores.size < len(CLASS_NAMES):
        raise ValueError(
            f"Output model harus memiliki minimal {len(CLASS_NAMES)} nilai kelas."
        )

    scores = scores[: len(CLASS_NAMES)]

    if np.any(scores < 0) or not np.isclose(float(scores.sum()), 1.0, atol=1e-3):
        shifted = scores - float(np.max(scores))
        exp_scores = np.exp(shifted)
        scores = exp_scores / float(np.sum(exp_scores))

    return {class_name: float(score) for class_name, score in zip(CLASS_NAMES, scores)}


def predict_with_model(image: Image.Image, model_path: str | Path) -> PredictionResult:
    model_path = Path(model_path)
    model = _load_model_cached(str(model_path))
    processed = preprocess_image(image, image_size=IMAGE_SIZE, add_batch=True)

    raw_prediction = model.predict(processed, verbose=0)[0]
    probabilities = _normalize_model_scores(raw_prediction)
    class_name = max(probabilities, key=probabilities.get)
    confidence = probabilities[class_name] * 100

    return PredictionResult(
        class_name=class_name,
        display_label=DISPLAY_LABELS[class_name],
        confidence=confidence,
        recommendation=make_recommendation(class_name),
        probabilities=probabilities,
        source="model",
        model_path=str(model_path),
    )


def predict_with_heuristic(image: Image.Image) -> PredictionResult:
    """Temporary color-based estimate used only when no trained model exists."""
    resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    hsv = np.asarray(resized.convert("HSV"), dtype=np.float32)

    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    mask = (saturation > 35) & (value > 35)
    if int(mask.sum()) < 100:
        mask = value > 20

    selected_hue = hue[mask]
    selected_sat = saturation[mask]
    selected_value = value[mask]

    if selected_hue.size == 0:
        probabilities = {
            "matang": 0.34,
            "setengah_matang": 0.33,
            "mentah": 0.33,
        }
    else:
        green = np.mean((selected_hue >= 55) & (selected_hue <= 115))
        yellow = np.mean((selected_hue >= 28) & (selected_hue < 55))
        orange = np.mean((selected_hue >= 12) & (selected_hue < 28))
        red = np.mean((selected_hue < 12) | (selected_hue > 240))
        brown_dark = np.mean(
            (((selected_hue < 32) | (selected_hue > 235)) & (selected_value < 120))
            & (selected_sat > 45)
        )

        raw_scores = {
            "matang": float(red + (0.90 * orange) + (0.50 * brown_dark)),
            "setengah_matang": float(yellow + (0.55 * orange) + (0.15 * green)),
            "mentah": float(green + (0.15 * yellow)),
        }

        # Small prior keeps the estimate stable for low-saturation images.
        raw_scores = {key: value + 0.05 for key, value in raw_scores.items()}
        total = sum(raw_scores.values())
        probabilities = {key: value / total for key, value in raw_scores.items()}

    class_name = max(probabilities, key=probabilities.get)
    evidence = probabilities[class_name]
    confidence = min(75.0, max(35.0, evidence * 100))

    return PredictionResult(
        class_name=class_name,
        display_label=DISPLAY_LABELS[class_name],
        confidence=confidence,
        recommendation=make_recommendation(class_name),
        probabilities=probabilities,
        source="heuristic",
        note="Model belum ditemukan. Hasil ini adalah estimasi sementara berbasis warna.",
    )


def predict_image(
    image: Image.Image,
    model_path: str | Path | None = None,
    allow_heuristic: bool = True,
) -> PredictionResult:
    resolved_model_path = find_model_path(model_path)

    if resolved_model_path:
        return predict_with_model(image, resolved_model_path)

    if allow_heuristic:
        return predict_with_heuristic(image)

    raise FileNotFoundError(
        "Model belum ditemukan. Simpan model ke models/model_sawit_retrained.keras, "
        "models/model_sawit.keras, atau models/model_sawit.h5."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediksi kematangan sawit dari gambar.")
    parser.add_argument("image", help="Path gambar yang akan diprediksi.")
    parser.add_argument("--model", help="Path model .keras atau .h5.", default=None)
    parser.add_argument(
        "--no-heuristic",
        action="store_true",
        help="Matikan estimasi sementara jika model belum tersedia.",
    )
    args = parser.parse_args()

    image = load_image(args.image)
    result = predict_image(
        image,
        model_path=args.model,
        allow_heuristic=not args.no_heuristic,
    )

    print(f"Kelas Prediksi : {result.display_label}")
    print(f"Confidence     : {result.confidence:.2f}%")
    print(f"Rekomendasi    : {result.recommendation}")
    print(f"Sumber         : {result.source}")
    if result.note:
        print(f"Catatan        : {result.note}")


if __name__ == "__main__":
    main()
