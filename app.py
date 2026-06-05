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
    init_db,
    insert_prediction,
)
from src.predict import MODEL_CANDIDATES, predict_image
from src.preprocessing import load_image


ROOT_DIR = Path(__file__).resolve().parent


def existing_model_label() -> str:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return str(path)
    return str(MODEL_CANDIDATES[0])


def main() -> None:
    st.set_page_config(
        page_title="Deteksi Bibit Sawit",
        page_icon="S",
        layout="centered",
    )
    init_db()

    st.title("Deteksi Bibit Sawit")

    with st.sidebar:
        st.header("Model")
        model_path = st.text_input("Path model", value=existing_model_label())
        use_heuristic = st.toggle("Estimasi sementara", value=True)
        st.header("Database")
        st.caption(str(DEFAULT_DB_PATH))
        st.metric("Riwayat", count_predictions())
        if st.button("Hapus riwayat", use_container_width=True):
            clear_predictions()
            st.session_state.saved_prediction_keys = set()
            st.rerun()

    if "saved_prediction_keys" not in st.session_state:
        st.session_state.saved_prediction_keys = set()

    uploaded_file = st.file_uploader(
        "Gambar sawit",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Unggah gambar sawit untuk mulai prediksi.")
        return

    image_bytes = uploaded_file.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    image = load_image(BytesIO(image_bytes))
    left, right = st.columns([1, 1], vertical_alignment="top")

    with left:
        st.image(image, caption="Preview", use_container_width=True)

    with right:
        try:
            result = predict_image(
                image,
                model_path=model_path,
                allow_heuristic=use_heuristic,
            )
        except Exception as exc:
            st.error(str(exc))
            return

        st.metric("Kelas", result.display_label)
        st.metric("Confidence", f"{result.confidence:.2f}%")

        if result.class_name == "matang":
            st.success(result.recommendation)
        elif result.class_name == "setengah_matang":
            st.warning(result.recommendation)
        else:
            st.error(result.recommendation)

        if result.source == "model":
            st.caption(f"Model: {result.model_path}")
        elif result.note:
            st.caption(result.note)

    prediction_key = (
        image_hash,
        result.source,
        result.model_path,
        result.class_name,
        round(float(result.confidence), 4),
    )
    if prediction_key not in st.session_state.saved_prediction_keys:
        insert_prediction(uploaded_file.name, image_hash, result)
        st.session_state.saved_prediction_keys.add(prediction_key)

    st.subheader("Skor Kelas")
    for class_name, probability in result.probabilities.items():
        label = result.display_label if class_name == result.class_name else class_name.replace("_", " ").title()
        st.progress(float(probability), text=f"{label}: {probability * 100:.2f}%")

    st.subheader("Riwayat Prediksi")
    history = fetch_recent_predictions(limit=25)
    if history:
        formatted_history = [
            {
                "Waktu": row["created_at"],
                "File": row["filename"],
                "Kelas": row["display_label"],
                "Confidence": f"{row['confidence']:.2f}%",
                "Rekomendasi": row["recommendation"],
                "Sumber": row["source"],
            }
            for row in history
        ]
        st.dataframe(formatted_history, use_container_width=True, hide_index=True)
    else:
        st.caption("Belum ada riwayat.")


if __name__ == "__main__":
    main()
