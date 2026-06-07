from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

try:
    from .predict import CLASS_NAMES
    from .preprocessing import IMAGE_SIZE, dataset_split_summary
except ImportError:  # Allows: python src/train.py
    from predict import CLASS_NAMES
    from preprocessing import IMAGE_SIZE, dataset_split_summary


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT_DIR / "dataset"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "model_sawit.keras"


def _total_counts(summary: dict[str, dict[str, int]], split: str) -> int:
    return sum(summary.get(split, {}).values())


def validate_dataset_or_exit(dataset_dir: Path) -> None:
    summary = dataset_split_summary(dataset_dir, CLASS_NAMES)

    print("Ringkasan dataset:")
    for split, counts in summary.items():
        total = sum(counts.values())
        detail = ", ".join(f"{class_name}={count}" for class_name, count in counts.items())
        print(f"- {split}: {total} gambar ({detail})")

    if _total_counts(summary, "train") == 0 or _total_counts(summary, "validation") == 0:
        raise SystemExit(
            "\nDataset train dan validation belum berisi gambar. "
            "Isi folder dataset terlebih dahulu sebelum training."
        )


def normalize_dataset(tf, dataset):
    return dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def augmentation_layers(tf):
    layers = tf.keras.layers
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.12),
        ],
        name="image_augmentation",
    )


def build_simple_cnn(tf, num_classes: int):
    layers = tf.keras.layers

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation_layers(tf)(inputs)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="sawit_simple_cnn")


def build_mobilenetv2(tf, num_classes: int, weights: str | None):
    layers = tf.keras.layers
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation_layers(tf)(inputs)
    x = preprocess_input(x * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="sawit_mobilenetv2")


def compile_model(tf, model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks(tf, checkpoint_path: Path, patience: int):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
        ),
    ]


def find_mobilenetv2_base(tf, model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2"):
            return layer
    return None


def enable_mobilenetv2_fine_tuning(tf, model, fine_tune_at: int) -> int:
    base_model = find_mobilenetv2_base(tf, model)
    if base_model is None:
        raise ValueError("Layer dasar MobileNetV2 tidak ditemukan untuk fine-tuning.")

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    return sum(1 for layer in base_model.layers if layer.trainable)


def merge_histories(histories):
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return SimpleNamespace(history=merged)


def save_training_plot(history, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib tidak tersedia, plot training dilewati.")
        return

    accuracy = history.history.get("accuracy", [])
    val_accuracy = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plot_path = output_path.with_name("training_history.png")
    epochs = range(1, len(accuracy) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, label="train")
    plt.plot(epochs, val_accuracy, label="validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="train")
    plt.plot(epochs, val_loss, label="validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot training disimpan: {plot_path}")


def select_best_checkpoint(tf, checkpoint_paths: list[Path], validation_ds):
    best_model = None
    best_path = None
    best_accuracy = -1.0
    best_loss = 0.0

    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            continue
        candidate_model = tf.keras.models.load_model(checkpoint_path)
        val_loss, val_accuracy = candidate_model.evaluate(validation_ds, verbose=0)
        print(
            f"Checkpoint {checkpoint_path.name}: "
            f"val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}"
        )
        if val_accuracy > best_accuracy:
            best_model = candidate_model
            best_path = checkpoint_path
            best_accuracy = float(val_accuracy)
            best_loss = float(val_loss)

    if best_model is None:
        raise RuntimeError("Tidak ada checkpoint model yang berhasil dibuat.")

    print(
        f"Checkpoint terbaik: {best_path} "
        f"(val_accuracy={best_accuracy:.4f}, val_loss={best_loss:.4f})"
    )
    return best_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Training model klasifikasi sawit.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR), help="Folder dataset.")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH), help="Path output model.")
    parser.add_argument("--epochs", type=int, default=20, help="Jumlah epoch training awal.")
    parser.add_argument("--batch-size", type=int, default=16, help="Ukuran batch.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate awal.")
    parser.add_argument(
        "--architecture",
        choices=("simple_cnn", "mobilenetv2"),
        default="simple_cnn",
        help="Arsitektur model.",
    )
    parser.add_argument(
        "--weights",
        choices=("none", "imagenet"),
        default="none",
        help="Bobot awal untuk MobileNetV2.",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=0,
        help="Epoch tambahan untuk fine-tuning MobileNetV2.",
    )
    parser.add_argument(
        "--fine-tune-at",
        type=int,
        default=100,
        help="Indeks layer MobileNetV2 mulai dibuka saat fine-tuning.",
    )
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate saat fine-tuning.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    validate_dataset_or_exit(dataset_dir)

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow belum terpasang. Jalankan: pip install -r requirements.txt"
        ) from exc

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "train",
        labels="inferred",
        label_mode="categorical",
        class_names=list(CLASS_NAMES),
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "validation",
        labels="inferred",
        label_mode="categorical",
        class_names=list(CLASS_NAMES),
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        shuffle=False,
    )

    train_ds = normalize_dataset(tf, train_ds).prefetch(tf.data.AUTOTUNE)
    validation_ds = normalize_dataset(tf, validation_ds).prefetch(tf.data.AUTOTUNE)

    if args.architecture == "mobilenetv2":
        weights = None if args.weights == "none" else "imagenet"
        model = build_mobilenetv2(tf, len(CLASS_NAMES), weights)
    else:
        model = build_simple_cnn(tf, len(CLASS_NAMES))

    histories = []
    checkpoint_paths = [output_path]

    compile_model(tf, model, args.learning_rate)
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=make_callbacks(tf, output_path, patience=5),
    )
    histories.append(history)

    if args.fine_tune_epochs > 0:
        if args.architecture != "mobilenetv2":
            print("Fine-tuning dilewati karena hanya didukung untuk MobileNetV2.")
        else:
            model = tf.keras.models.load_model(output_path)
            trainable_layers = enable_mobilenetv2_fine_tuning(
                tf,
                model,
                args.fine_tune_at,
            )
            print(f"Fine-tuning MobileNetV2: {trainable_layers} layer dasar trainable.")
            compile_model(tf, model, args.fine_tune_learning_rate)
            fine_tune_checkpoint = output_path.with_name(
                f"{output_path.stem}_fine_tune{output_path.suffix}"
            )
            checkpoint_paths.append(fine_tune_checkpoint)
            fine_tune_history = model.fit(
                train_ds,
                validation_data=validation_ds,
                epochs=args.fine_tune_epochs,
                callbacks=make_callbacks(tf, fine_tune_checkpoint, patience=4),
            )
            histories.append(fine_tune_history)

    model = select_best_checkpoint(tf, checkpoint_paths, validation_ds)
    model.save(output_path)
    print(f"Model disimpan: {output_path}")
    save_training_plot(merge_histories(histories), output_path)

    test_dir = dataset_dir / "test"
    test_count = sum(dataset_split_summary(dataset_dir, CLASS_NAMES)["test"].values())
    if test_count > 0:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=list(CLASS_NAMES),
            image_size=IMAGE_SIZE,
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_ds = normalize_dataset(tf, test_ds).prefetch(tf.data.AUTOTUNE)
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss    : {test_loss:.4f}")
    else:
        print("Dataset test kosong, evaluasi test dilewati.")


if __name__ == "__main__":
    main()
