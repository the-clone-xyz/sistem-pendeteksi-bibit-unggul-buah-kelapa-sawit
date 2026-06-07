from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import streamlit as st

from src.database import (
    DEFAULT_DB_PATH,
    clear_predictions,
    count_predictions,
    fetch_recent_predictions,
    fetch_superior_parameters,
    init_db,
    insert_prediction,
    update_superior_parameter,
)
from src.image_parameters import COLOR_LABELS, extract_visual_parameters
from src.image_validation import (
    DEFAULT_VALIDATION_THRESHOLD,
    NON_SAWIT_STATUS,
    VALIDATION_MODEL_CANDIDATES,
    find_validation_model_path,
    validate_oil_palm_image,
)
from src.predict import MODEL_CANDIDATES, predict_image
from src.preprocessing import load_image
from src.quality import SUPERIOR_STATUS, assess_superior_seed


ROOT_DIR = Path(__file__).resolve().parent
CLASS_OPTIONS = ("matang", "setengah_matang", "mentah")
MODEL_FILE_EXTENSIONS = {".keras", ".h5"}
MANUAL_MODEL_OPTION = "__manual_path__"
MODEL_DISPLAY_NAMES = {
    "model_sawit.keras": "Model 1 Flash",
    "model_sawit_fine_tune.keras": "Model 1 Pro",
    "model_sawit_retrained.keras": "Model 1 Lite",
    "model_sawit_retrained_fine_tune.keras": "Model 1 Pro Legacy",
    "model_validasi_sawit.keras": "Model 1 Pro",
    "model_validasi_sawit_before_hardpositives.keras": "Model 1 Flash",
    "model_validasi_sawit_before_highres.keras": "Model 1 Lite",
}
MODEL_SELECTION_PRIORITY = {
    "model_sawit.keras": 100,
    "model_sawit_fine_tune.keras": 90,
    "model_sawit_retrained_fine_tune.keras": 70,
    "model_sawit_retrained.keras": 60,
    "model_validasi_sawit.keras": 100,
    "model_validasi_sawit_before_hardpositives.keras": 20,
    "model_validasi_sawit_before_highres.keras": 10,
}


def relative_path_label(path: Path) -> str:
    try:
        return path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def discover_model_files() -> list[Path]:
    models_dir = ROOT_DIR / "models"
    if not models_dir.exists():
        return []

    return sorted(
        path
        for path in models_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MODEL_FILE_EXTENSIONS
    )


def unique_existing_model_paths(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() not in MODEL_FILE_EXTENSIONS:
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        unique_paths.append(path)
        seen.add(key)

    return unique_paths


def is_validation_model_path(path: Path) -> bool:
    name = path.name.lower()
    return "validasi" in name or "validator" in name


def model_display_name(path: Path) -> str:
    configured_name = MODEL_DISPLAY_NAMES.get(path.name)
    if configured_name:
        return configured_name
    return path.stem.replace("_", " " ).replace("-", " " ).title()


def model_priority(path: Path) -> int:
    return MODEL_SELECTION_PRIORITY.get(path.name, 0)


def prioritize_model_options(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: (-model_priority(path), relative_path_label(path)))


def classification_model_options() -> list[Path]:
    discovered = [
        path for path in discover_model_files() if not is_validation_model_path(path)
    ]
    return prioritize_model_options(
        unique_existing_model_paths([*MODEL_CANDIDATES, *discovered])
    )


def validation_model_options() -> list[Path]:
    discovered = [path for path in discover_model_files() if is_validation_model_path(path)]
    return prioritize_model_options(
        unique_existing_model_paths([*VALIDATION_MODEL_CANDIDATES, *discovered])
    )


def model_option_label(option: str) -> str:
    if option == MANUAL_MODEL_OPTION:
        return "Manual path"
    return model_display_name(Path(option))


def render_model_path_selector(
    label: str,
    paths: list[Path],
    fallback_path: str,
    key: str,
) -> str:
    options = [str(path) for path in paths] + [MANUAL_MODEL_OPTION]
    selected_path = st.selectbox(
        label,
        options=options,
        index=0,
        format_func=model_option_label,
        key=f"{key}_selection",
    )
    if selected_path == MANUAL_MODEL_OPTION:
        return st.text_input(
            f"Path {label.lower()} manual",
            value=fallback_path,
            key=f"{key}_manual_path",
        )

    st.caption(f"Path: {relative_path_label(Path(selected_path))}")
    return selected_path


def existing_model_label() -> str:
    paths = classification_model_options()
    if paths:
        return str(paths[0])
    return str(MODEL_CANDIDATES[0])


def existing_validation_model_label() -> str:
    paths = validation_model_options()
    if paths:
        return str(paths[0])
    return str(VALIDATION_MODEL_CANDIDATES[0])


def parameter_by_code(parameters: list[dict], code: str) -> dict | None:
    for parameter in parameters:
        if parameter["code"] == code:
            return parameter
    return None


def parameter_target(parameters: list[dict], code: str, fallback: str) -> str:
    parameter = parameter_by_code(parameters, code)
    return str(parameter["target_value"]) if parameter else fallback


def parameter_weight(parameters: list[dict], code: str, fallback: float) -> float:
    parameter = parameter_by_code(parameters, code)
    return float(parameter["weight"]) if parameter else fallback


def parse_target_colors(value: str) -> list[str]:
    selected = [item.strip() for item in value.split(",") if item.strip()]
    return [color for color in selected if color in COLOR_LABELS]


def render_parameter_editor(parameters: list[dict]) -> None:
    current_class = parameter_target(parameters, "kelas_prediksi", "matang")
    class_index = CLASS_OPTIONS.index(current_class) if current_class in CLASS_OPTIONS else 0
    current_colors = parse_target_colors(
        parameter_target(parameters, "warna_dominan", "merah,oranye,coklat")
    )

    with st.form("superior_parameter_form"):
        target_class = st.selectbox(
            "Tingkat kematangan unggul",
            options=CLASS_OPTIONS,
            index=class_index,
            format_func=lambda value: value.replace("_", " ").title(),
        )
        minimum_confidence = st.number_input(
            "Confidence minimum (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(parameter_target(parameters, "confidence_minimum", "75")),
            step=1.0,
        )
        minimum_object_area = st.number_input(
            "Ukuran area objek minimum (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(parameter_target(parameters, "ukuran_area_minimum", "20")),
            step=1.0,
        )
        target_colors = st.multiselect(
            "Warna dominan unggul",
            options=COLOR_LABELS,
            default=current_colors or ["merah", "oranye", "coklat"],
        )
        target_recommendation = st.text_input(
            "Rekomendasi unggul",
            value=parameter_target(
                parameters,
                "rekomendasi",
                "Layak / Direkomendasikan",
            ),
        )
        class_weight = st.number_input(
            "Bobot kematangan",
            min_value=0.0,
            max_value=1.0,
            value=parameter_weight(parameters, "kelas_prediksi", 0.35),
            step=0.05,
        )
        confidence_weight = st.number_input(
            "Bobot confidence",
            min_value=0.0,
            max_value=1.0,
            value=parameter_weight(parameters, "confidence_minimum", 0.25),
            step=0.05,
        )
        size_weight = st.number_input(
            "Bobot ukuran",
            min_value=0.0,
            max_value=1.0,
            value=parameter_weight(parameters, "ukuran_area_minimum", 0.15),
            step=0.05,
        )
        color_weight = st.number_input(
            "Bobot warna",
            min_value=0.0,
            max_value=1.0,
            value=parameter_weight(parameters, "warna_dominan", 0.10),
            step=0.05,
        )
        recommendation_weight = st.number_input(
            "Bobot rekomendasi",
            min_value=0.0,
            max_value=1.0,
            value=parameter_weight(parameters, "rekomendasi", 0.10),
            step=0.05,
        )

        if st.form_submit_button("Simpan parameter", use_container_width=True):
            update_superior_parameter("kelas_prediksi", target_class, class_weight)
            update_superior_parameter(
                "confidence_minimum",
                f"{minimum_confidence:g}",
                confidence_weight,
            )
            update_superior_parameter(
                "ukuran_area_minimum",
                f"{minimum_object_area:g}",
                size_weight,
            )
            update_superior_parameter(
                "warna_dominan",
                ",".join(target_colors),
                color_weight,
            )
            update_superior_parameter(
                "rekomendasi",
                target_recommendation,
                recommendation_weight,
            )
            st.success("Parameter disimpan.")
            st.rerun()


def format_parameter_status(passed: bool) -> str:
    return "Memenuhi" if passed else "Tidak Memenuhi"


def format_class_label(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def format_color_label(color_name: str) -> str:
    return color_name.replace("_", " ").title()


def parameter_key(parameters: list[dict]) -> tuple[tuple[str, str, float], ...]:
    return tuple(
        (
            parameter["code"],
            parameter["target_value"],
            round(float(parameter["weight"]), 4),
        )
        for parameter in parameters
    )


def save_prediction_once(
    filename: str,
    image_hash: str,
    result,
    assessment: dict,
    superior_parameters: list[dict],
) -> None:
    prediction_key = (
        image_hash,
        result.source,
        result.model_path,
        result.class_name,
        round(float(result.confidence), 4),
        parameter_key(superior_parameters),
    )
    if prediction_key in st.session_state.saved_prediction_keys:
        return

    insert_prediction(
        filename,
        image_hash,
        result,
        quality_status=assessment["status"],
        quality_score=assessment["score"],
        parameter_snapshot=assessment["parameters"],
    )
    st.session_state.saved_prediction_keys.add(prediction_key)


def make_invalid_upload_item(
    filename: str,
    image_hash: str,
    image,
    visual_parameters: dict,
    validation: dict,
) -> dict:
    return {
        "filename": filename,
        "image": image,
        "image_hash": image_hash,
        "visual_parameters": visual_parameters,
        "validation": validation,
        "is_valid_sawit": False,
        "result": None,
        "assessment": {
            "status": NON_SAWIT_STATUS,
            "score": 0.0,
            "parameters": [],
        },
    }


def evaluate_uploaded_file(
    uploaded_file,
    model_path: str,
    use_heuristic: bool,
    superior_parameters: list[dict],
    validator_model_path: str,
    validation_threshold: float,
) -> dict:
    image_bytes = uploaded_file.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    image = load_image(BytesIO(image_bytes))
    visual_parameters = extract_visual_parameters(image)

    if find_validation_model_path(validator_model_path):
        validation = validate_oil_palm_image(
            image,
            validator_model_path=validator_model_path,
            threshold=validation_threshold,
        )
        if not validation["is_valid"]:
            return make_invalid_upload_item(
                uploaded_file.name,
                image_hash,
                image,
                visual_parameters,
                validation,
            )

        result = predict_image(
            image,
            model_path=model_path,
            allow_heuristic=use_heuristic,
        )
    else:
        result = predict_image(
            image,
            model_path=model_path,
            allow_heuristic=use_heuristic,
        )
        validation = validate_oil_palm_image(
            image,
            validator_model_path=validator_model_path,
            threshold=validation_threshold,
            fallback_result=result,
        )
        if not validation["is_valid"]:
            return make_invalid_upload_item(
                uploaded_file.name,
                image_hash,
                image,
                visual_parameters,
                validation,
            )

    assessment = assess_superior_seed(
        result,
        superior_parameters,
        visual_parameters=visual_parameters,
    )
    save_prediction_once(
        uploaded_file.name,
        image_hash,
        result,
        assessment,
        superior_parameters,
    )

    return {
        "filename": uploaded_file.name,
        "image": image,
        "image_hash": image_hash,
        "visual_parameters": visual_parameters,
        "validation": validation,
        "is_valid_sawit": True,
        "result": result,
        "assessment": assessment,
    }


def render_status_message(status: str, recommendation: str) -> None:
    if status == SUPERIOR_STATUS:
        st.success(recommendation)
    elif status == "Perlu Pemeriksaan Lanjutan":
        st.warning(recommendation)
    else:
        st.error(recommendation)


def render_visual_parameters(item: dict) -> None:
    result = item.get("result")
    visual = item["visual_parameters"]
    validation = item["validation"]
    st.subheader("Parameter Visual")
    first_col, second_col = st.columns(2)
    first_col.metric(
        "Ukuran Gambar",
        f"{visual['width']} x {visual['height']} px",
    )
    second_col.metric(
        "Area Objek",
        f"{visual['object_area_percent']:.2f}%",
    )
    third_col, fourth_col = st.columns(2)
    third_col.metric(
        "Tingkat Kematangan",
        result.display_label if result else "-",
    )
    fourth_col.metric(
        "Warna Dominan",
        format_color_label(visual["dominant_color"]),
        delta=f"{visual['dominant_color_percent']:.2f}%",
        delta_color="off",
    )
    validation_col, score_col = st.columns(2)
    validation_col.metric("Validasi Gambar", validation["status"])
    score_col.metric("Skor Validasi", f"{validation['score']:.2f}%")
    if validation.get("message"):
        st.caption(validation["message"])


def render_parameter_table(assessment: dict) -> None:
    st.dataframe(
        [
            {
                "Parameter": parameter["label"],
                "Nilai Saat Ini": parameter["current_value"],
                "Target": f"{parameter['target_value']} {parameter['unit']}".strip(),
                "Bobot": parameter["weight"],
                "Status": format_parameter_status(parameter["passed"]),
            }
            for parameter in assessment["parameters"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_prediction_detail(item: dict, expanded: bool) -> None:
    result = item.get("result")
    assessment = item["assessment"]
    title = f"{item['filename']} - {assessment['status']}"

    with st.expander(title, expanded=expanded):
        left, right = st.columns([1, 1], vertical_alignment="top")

        with left:
            st.image(item["image"], caption="Preview", use_container_width=True)

        with right:
            if result is None:
                st.metric("Kelas", "-")
                st.metric("Confidence", "-")
                st.metric("Status", NON_SAWIT_STATUS)
                st.metric("Skor Validasi", f"{item['validation']['score']:.2f}%")
                st.error(NON_SAWIT_STATUS)
                if item["validation"]["message"]:
                    st.caption(item["validation"]["message"])
            else:
                st.metric("Kelas", result.display_label)
                st.metric("Confidence", f"{result.confidence:.2f}%")
                st.metric("Status Unggul", assessment["status"])
                st.metric("Skor Parameter", f"{assessment['score']:.2f}%")
                render_status_message(assessment["status"], result.recommendation)

                if result.source == "model":
                    st.caption(f"Model: {result.model_path}")
                elif result.note:
                    st.caption(result.note)

        render_visual_parameters(item)

        if result is None:
            return

        st.subheader("Skor Kelas")
        for class_name, probability in result.probabilities.items():
            label = result.display_label if class_name == result.class_name else format_class_label(class_name)
            st.progress(float(probability), text=f"{label}: {probability * 100:.2f}%")

        st.subheader("Parameter Bibit Unggul")
        render_parameter_table(assessment)


def render_search_summary(items: list[dict]) -> None:
    superior_count = sum(
        1 for item in items if item["assessment"]["status"] == SUPERIOR_STATUS
    )
    invalid_count = sum(1 for item in items if not item["is_valid_sawit"])
    total_count = len(items)
    not_superior_count = total_count - superior_count - invalid_count

    st.subheader("Ringkasan Pencarian")
    total_col, superior_col, other_col, invalid_col = st.columns(4)
    total_col.metric("Total File", total_count)
    superior_col.metric("Bibit Unggul", superior_count)
    other_col.metric("Belum Unggul", not_superior_count)
    invalid_col.metric("Bukan Sawit", invalid_count)

    st.dataframe(
        [
            {
                "File": item["filename"],
                "Validasi": item["validation"]["status"],
                "Ukuran": f"{item['visual_parameters']['width']} x {item['visual_parameters']['height']} px",
                "Area Objek": f"{item['visual_parameters']['object_area_percent']:.2f}%",
                "Tingkat Kematangan": item["result"].display_label if item["result"] else "-",
                "Warna Dominan": format_color_label(
                    item["visual_parameters"]["dominant_color"]
                ),
                "Confidence": f"{item['result'].confidence:.2f}%" if item["result"] else "-",
                "Status": item["assessment"]["status"],
                "Skor": f"{item['assessment']['score']:.2f}%",
                "Rekomendasi": item["result"].recommendation if item["result"] else NON_SAWIT_STATUS,
            }
            for item in items
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_prediction_history() -> None:
    st.subheader("Riwayat Prediksi")
    history = fetch_recent_predictions(limit=25)
    if history:
        formatted_history = [
            {
                "Waktu": row["created_at"],
                "File": row["filename"],
                "Kelas": row["display_label"],
                "Confidence": f"{row['confidence']:.2f}%",
                "Status Unggul": row["quality_status"] or "-",
                "Skor Parameter": (
                    f"{row['quality_score']:.2f}%"
                    if row["quality_score"] is not None
                    else "-"
                ),
                "Rekomendasi": row["recommendation"],
                "Sumber": row["source"],
            }
            for row in history
        ]
        st.dataframe(formatted_history, use_container_width=True, hide_index=True)
    else:
        st.caption("Belum ada riwayat.")


def main() -> None:
    st.set_page_config(
        page_title="Cari Bibit Unggul Sawit",
        page_icon="S",
        layout="centered",
    )
    init_db()
    superior_parameters = fetch_superior_parameters()

    st.title("Cari Bibit Unggul Sawit")

    with st.sidebar:
        st.header("Model")
        model_path = render_model_path_selector(
            "Model klasifikasi",
            classification_model_options(),
            existing_model_label(),
            "classification_model",
        )
        validator_model_path = render_model_path_selector(
            "Model validasi sawit",
            validation_model_options(),
            existing_validation_model_label(),
            "validation_model",
        )
        validation_threshold = st.slider(
            "Threshold validasi sawit (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(DEFAULT_VALIDATION_THRESHOLD),
            step=1.0,
        )
        use_heuristic = st.toggle("Estimasi sementara", value=True)
        st.header("Parameter Unggul")
        render_parameter_editor(superior_parameters)
        st.header("Database")
        st.caption(str(DEFAULT_DB_PATH))
        st.metric("Riwayat", count_predictions())
        if st.button("Hapus riwayat", use_container_width=True):
            clear_predictions()
            st.session_state.saved_prediction_keys = set()
            st.rerun()

    if "saved_prediction_keys" not in st.session_state:
        st.session_state.saved_prediction_keys = set()

    uploaded_files = st.file_uploader(
        "Upload gambar sawit untuk mencari bibit unggul",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload satu atau beberapa gambar sawit untuk mulai mencari bibit unggul.")
        render_prediction_history()
        return

    evaluated_items: list[dict] = []
    for uploaded_file in uploaded_files:
        try:
            evaluated_items.append(
                evaluate_uploaded_file(
                    uploaded_file,
                    model_path,
                    use_heuristic,
                    superior_parameters,
                    validator_model_path,
                    validation_threshold,
                )
            )
        except Exception as exc:
            st.error(f"{uploaded_file.name}: {exc}")

    if not evaluated_items:
        render_prediction_history()
        return

    render_search_summary(evaluated_items)

    st.subheader("Detail Hasil")
    expand_all = len(evaluated_items) == 1
    for item in evaluated_items:
        render_prediction_detail(
            item,
            expanded=expand_all or item["assessment"]["status"] in (SUPERIOR_STATUS, NON_SAWIT_STATUS),
        )

    render_prediction_history()


if __name__ == "__main__":
    main()
