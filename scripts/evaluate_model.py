from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from src.predict import CLASS_NAMES
    from src.preprocessing import IMAGE_EXTENSIONS, load_image, preprocess_image
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.predict import CLASS_NAMES
    from src.preprocessing import IMAGE_EXTENSIONS, load_image, preprocess_image


def iter_images(dataset_dir: Path, split: str):
    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / split / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield class_name, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained sawit classifier.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--model", default="models/model_sawit.keras")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    import tensorflow as tf

    dataset_dir = Path(args.dataset)
    model_path = Path(args.model)
    model = tf.keras.models.load_model(model_path)

    y_true: list[str] = []
    y_pred: list[str] = []
    errors: list[tuple[str, str, float, Path]] = []

    for true_class, path in iter_images(dataset_dir, args.split):
        image = load_image(path)
        batch = preprocess_image(image)
        probabilities = model.predict(batch, verbose=0)[0]
        pred_index = int(np.argmax(probabilities[: len(CLASS_NAMES)]))
        pred_class = CLASS_NAMES[pred_index]
        confidence = float(probabilities[pred_index])

        y_true.append(true_class)
        y_pred.append(pred_class)
        if pred_class != true_class:
            errors.append((true_class, pred_class, confidence, path))

    total = len(y_true)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / total if total else 0.0

    print(f"Split: {args.split}")
    print(f"Total: {total}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print()
    print("Per kelas:")
    for class_name in CLASS_NAMES:
        indices = [idx for idx, true in enumerate(y_true) if true == class_name]
        class_total = len(indices)
        class_correct = sum(1 for idx in indices if y_pred[idx] == class_name)
        class_accuracy = class_correct / class_total if class_total else 0.0
        print(f"- {class_name}: {class_accuracy:.4f} ({class_correct}/{class_total})")

    print()
    print("Confusion matrix (rows=true, cols=pred):")
    header = "true\\pred," + ",".join(CLASS_NAMES)
    print(header)
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    for true in CLASS_NAMES:
        row = [str(matrix[true][pred]) for pred in CLASS_NAMES]
        print(true + "," + ",".join(row))

    if args.show_errors and errors:
        print()
        print("Salah prediksi:")
        for true, pred, confidence, path in errors:
            print(f"- true={true}, pred={pred}, conf={confidence:.4f}, file={path}")


if __name__ == "__main__":
    main()
