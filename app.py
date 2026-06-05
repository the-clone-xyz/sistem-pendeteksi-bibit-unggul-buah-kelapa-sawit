from __future__ import annotations

from pathlib import Path

import streamlit as st

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

    st.title("Deteksi Bibit Sawit")

    with st.sidebar:
        st.header("Model")
        model_path = st.text_input("Path model", value=existing_model_label())
        use_heuristic = st.toggle("Estimasi sementara", value=True)

    uploaded_file = st.file_uploader(
        "Gambar sawit",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Unggah gambar sawit untuk mulai prediksi.")
        return

    image = load_image(uploaded_file)
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

    st.subheader("Skor Kelas")
    for class_name, probability in result.probabilities.items():
        label = result.display_label if class_name == result.class_name else class_name.replace("_", " ").title()
        st.progress(float(probability), text=f"{label}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
