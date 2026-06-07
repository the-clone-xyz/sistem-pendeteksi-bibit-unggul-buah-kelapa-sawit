from __future__ import annotations

from typing import Any

from .predict import PredictionResult


SUPERIOR_STATUS = "Unggul"
REVIEW_STATUS = "Perlu Pemeriksaan Lanjutan"
REJECTED_STATUS = "Tidak Unggul"


def _to_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _target_values(value: str) -> set[str]:
    return {
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    }


def _ratio_score(current_value: float, target_value: float) -> float:
    if target_value <= 0:
        return 100.0
    return max(min(current_value / target_value * 100.0, 100.0), 0.0)


def assess_superior_seed(
    result: PredictionResult,
    parameters: list[dict[str, Any]],
    visual_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    visual_parameters = visual_parameters or {}
    snapshot: list[dict[str, Any]] = []
    weighted_score = 0.0
    total_weight = 0.0
    class_passed = False
    confidence_passed = False
    weighted_parameter_failed = False

    for parameter in parameters:
        code = str(parameter.get("code", ""))
        target_value = str(parameter.get("target_value", ""))
        weight = max(_to_float(parameter.get("weight"), 1.0), 0.0)
        unit = str(parameter.get("unit", ""))
        current_value = ""
        passed = False
        score = 0.0
        recognized = True

        if code == "kelas_prediksi":
            current_value = result.class_name
            passed = result.class_name == target_value
            score = 100.0 if passed else 0.0
            class_passed = passed
        elif code == "confidence_minimum":
            minimum_confidence = _to_float(target_value, 75.0)
            current_value = f"{result.confidence:.2f}"
            passed = result.confidence >= minimum_confidence
            score = 100.0 if passed else _ratio_score(result.confidence, minimum_confidence)
            confidence_passed = passed
        elif code == "rekomendasi":
            current_value = result.recommendation
            passed = result.recommendation == target_value
            score = 100.0 if passed else 0.0
        elif code == "ukuran_area_minimum":
            object_area = _to_float(visual_parameters.get("object_area_percent"), 0.0)
            minimum_area = _to_float(target_value, 20.0)
            current_value = f"{object_area:.2f}"
            passed = object_area >= minimum_area
            score = 100.0 if passed else _ratio_score(object_area, minimum_area)
        elif code == "warna_dominan":
            dominant_color = str(
                visual_parameters.get("dominant_color", "tidak diketahui")
            ).lower()
            allowed_colors = _target_values(target_value)
            current_value = dominant_color
            passed = not allowed_colors or dominant_color in allowed_colors
            score = 100.0 if passed else 0.0
        else:
            recognized = False
            current_value = "-"

        if recognized and weight > 0 and not passed:
            weighted_parameter_failed = True

        weighted_score += score * weight
        total_weight += weight
        snapshot.append(
            {
                "code": code,
                "label": parameter.get("label", code),
                "target_value": target_value,
                "current_value": current_value,
                "unit": unit,
                "weight": weight,
                "passed": passed,
                "score": round(score, 2),
            }
        )

    quality_score = round(weighted_score / total_weight, 2) if total_weight else 0.0

    if class_passed and confidence_passed and not weighted_parameter_failed:
        quality_status = SUPERIOR_STATUS
    elif result.class_name == "setengah_matang" or class_passed:
        quality_status = REVIEW_STATUS
    else:
        quality_status = REJECTED_STATUS

    return {
        "status": quality_status,
        "score": quality_score,
        "parameters": snapshot,
    }
