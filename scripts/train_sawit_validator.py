from __future__ import annotations

import argparse
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("bukan_sawit", "sawit")
IMAGE_SIZE = (224, 224)
DEFAULT_DATASET_DIR = ROOT_DIR / "dataset_validasi"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "models" / "model_validasi_sawit.keras"


def augmentation_layers(tf):
    layers = tf.keras.layers
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.04, 0.04),
            layers.RandomContrast(0.10),
        ],
        name="validator_augmentation",
    )


def build_model(tf, weights: str | None):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation_layers(tf)(inputs)
    x = preprocess_input(x)
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="sawit_binary_validator")


def load_split(tf, dataset_dir: Path, split: str, batch_size: int):
    split_dir = dataset_dir / split
    if not split_dir.exists():
        return None

    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=list(CLASS_NAMES),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=(split == "train"),
        seed=42,
    )


def image_count(directory: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not directory.exists():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def validate_dataset_or_exit(dataset_dir: Path) -> None:
    missing: list[str] = []
    print("Ringkasan dataset validasi:")
    for split in ("train", "validation", "test"):
        detail = []
        for class_name in CLASS_NAMES:
            count = image_count(dataset_dir / split / class_name)
            detail.append(f"{class_name}={count}")
            if split in {"train", "validation"} and count == 0:
                missing.append(f"{split}/{class_name}")
        print(f"- {split}: {', '.join(detail)}")

    if missing:
        raise SystemExit(
            "Dataset validasi belum lengkap. Folder berikut harus berisi gambar: "
            + ", ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train binary sawit/bukan-sawit validation model."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Gunakan 'none' jika tidak ingin download bobot ImageNet.",
    )
    args = parser.parse_args()

    import tensorflow as tf

    dataset_dir = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    validate_dataset_or_exit(dataset_dir)

    train_ds = load_split(tf, dataset_dir, "train", args.batch_size)
    validation_ds = load_split(tf, dataset_dir, "validation", args.batch_size)
    if train_ds is None or validation_ds is None:
        raise SystemExit(
            "Dataset validasi harus memiliki folder train dan validation dengan "
            "kelas bukan_sawit dan sawit."
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(tf, None if args.weights == "none" else args.weights)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            output_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
        ),
    ]
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(output_path)
    print(f"Model validasi disimpan: {output_path}")

    test_ds = load_split(tf, dataset_dir, "test", args.batch_size)
    if test_ds is not None:
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss    : {test_loss:.4f}")


if __name__ == "__main__":
    main()
