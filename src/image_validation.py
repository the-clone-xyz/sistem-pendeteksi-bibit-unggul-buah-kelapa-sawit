from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .predict import PredictionResult
from .preprocessing import IMAGE_SIZE


ROOT_DIR = Path(__file__).resolve().parents[1]
NON_SAWIT_STATUS = "Yang kamu upload bukan gambar buah sawit"
SAWIT_STATUS = "Gambar buah sawit"
DEFAULT_VALIDATION_THRESHOLD = 50.0
VALIDATION_MODEL_CANDIDATES = (
    ROOT_DIR / "models" / "model_validasi_sawit.keras",
    ROOT_DIR / "models" / "model_validasi_sawit.h5",
    ROOT_DIR / "models" / "model_sawit_validator.keras",
    ROOT_DIR / "models" / "model_sawit_validator.h5",
)


def find_validation_model_path(preferred_path: str | Path | None = None) -> Path | None:
    if preferred_path:
        path = Path(preferred_path)
        return path if path.exists() else None

    for path in VALIDATION_MODEL_CANDIDATES:
        if path.exists():
            return path

    return None


@lru_cache(maxsize=2)
def _load_validation_model(model_path: str):
    from tensorflow.keras.models import load_model

    return load_model(model_path)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-value)))


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - float(np.max(scores))
    exp_scores = np.exp(shifted)
    return exp_scores / float(np.sum(exp_scores))


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise ValueError("Output model validasi kosong.")

    if scores.size == 1:
        value = float(scores[0])
        if 0.0 <= value <= 1.0:
            return np.asarray([1.0 - value, value], dtype=np.float32)
        probability = _sigmoid(value)
        return np.asarray([1.0 - probability, probability], dtype=np.float32)

    scores = scores[:2]
    if np.any(scores < 0) or not np.isclose(float(scores.sum()), 1.0, atol=1e-3):
        scores = _softmax(scores)
    else:
        scores = scores / float(scores.sum())

    return scores.astype(np.float32)


def _preprocess_validation_image(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def _validation_response(
    sawit_probability: float,
    threshold: float,
    source: str,
    message: str,
    model_path: str | None = None,
) -> dict[str, Any]:
    score = round(float(sawit_probability) * 100.0, 2)
    is_valid = score >= threshold
    return {
        "is_valid": is_valid,
        "status": SAWIT_STATUS if is_valid else NON_SAWIT_STATUS,
        "message": message,
        "score": score,
        "source": source,
        "model_path": model_path,
        "details": {
            "sawit_probability": score,
            "threshold": float(threshold),
        },
    }


def validate_oil_palm_image(
    image: Image.Image,
    validator_model_path: str | Path | None = None,
    threshold: float = DEFAULT_VALIDATION_THRESHOLD,
    fallback_result: PredictionResult | None = None,
) -> dict[str, Any]:
    resolved_model_path = find_validation_model_path(validator_model_path)

    if resolved_model_path:
        model = _load_validation_model(str(resolved_model_path))
        processed = _preprocess_validation_image(image)
        raw_prediction = model.predict(processed, verbose=0)[0]
        probabilities = _normalize_scores(raw_prediction)
        sawit_probability = float(probabilities[1])
        return _validation_response(
            sawit_probability,
            threshold,
            source="validation_model",
            message="Validasi memakai model sawit/bukan-sawit.",
            model_path=str(resolved_model_path),
        )

    if fallback_result is not None:
        score = round(max(min(float(fallback_result.confidence), 100.0), 0.0), 2)
        return {
            "is_valid": True,
            "status": SAWIT_STATUS,
            "message": (
                "Model validasi sawit/bukan-sawit belum ditemukan; "
                "gambar diproses sebagai kandidat sawit berdasarkan model klasifikasi. "
                "Tambahkan model validasi untuk menolak gambar bukan-sawit."
            ),
            "score": score,
            "source": "classification_model_fallback",
            "model_path": fallback_result.model_path,
            "details": {
                "classification_confidence": score,
                "threshold": float(threshold),
            },
        }

    return {
        "is_valid": False,
        "status": NON_SAWIT_STATUS,
        "message": "Model validasi sawit/bukan-sawit belum ditemukan.",
        "score": 0.0,
        "source": "missing_validation_model",
        "model_path": None,
        "details": {
            "sawit_probability": 0.0,
            "threshold": float(threshold),
        },
    }
