from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SAWIT_SOURCE = ROOT_DIR / "dataset"
DEFAULT_NEGATIVE_SOURCE = ROOT_DIR / "dataset_bukan_sawit"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "dataset_validasi"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ("bukan_sawit", "sawit")


@dataclass(frozen=True)
class SplitCounts:
    train: int
    validation: int
    test: int


def collect_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_counts(total: int, validation_ratio: float, test_ratio: float) -> SplitCounts:
    if total < 2:
        raise ValueError("Minimal butuh 2 gambar per kelas untuk train dan validation.")

    test_count = int(round(total * test_ratio)) if test_ratio > 0 else 0
    validation_count = max(1, int(round(total * validation_ratio)))
    train_count = total - validation_count - test_count

    if train_count < 1:
        overflow = 1 - train_count
        if test_count >= overflow:
            test_count -= overflow
        else:
            validation_count = max(1, validation_count - (overflow - test_count))
            test_count = 0
        train_count = total - validation_count - test_count

    if train_count < 1 or validation_count < 1:
        raise ValueError("Jumlah gambar tidak cukup untuk membentuk train dan validation.")

    return SplitCounts(train=train_count, validation=validation_count, test=test_count)


def stable_name(label: str, index: int, source: Path) -> str:
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{label}_{index:05d}_{digest}{source.suffix.lower()}"


def copy_split(label: str, images: list[Path], output_dir: Path, counts: SplitCounts) -> None:
    split_plan = (
        ("train", counts.train),
        ("validation", counts.validation),
        ("test", counts.test),
    )
    cursor = 0
    for split, count in split_plan:
        target_dir = output_dir / split / label
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in images[cursor : cursor + count]:
            target_name = stable_name(label, cursor, source)
            shutil.copy2(source, target_dir / target_name)
            cursor += 1


def ensure_empty_output(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return

    existing_images = collect_images(output_dir)
    if existing_images and not overwrite:
        raise SystemExit(
            f"Folder output sudah berisi {len(existing_images)} gambar: {output_dir}. "
            "Jalankan dengan --overwrite jika ingin membuat ulang."
        )

    if overwrite:
        shutil.rmtree(output_dir)


def prepare_dataset(
    sawit_source: Path,
    negative_source: Path,
    output_dir: Path,
    validation_ratio: float,
    test_ratio: float,
    limit_per_class: int | None,
    seed: int,
    overwrite: bool,
) -> None:
    sawit_images = collect_images(sawit_source)
    negative_images = collect_images(negative_source)

    if not sawit_images:
        raise SystemExit(f"Tidak ada gambar sawit di {sawit_source}.")
    if not negative_images:
        raise SystemExit(
            f"Tidak ada gambar bukan-sawit di {negative_source}. "
            "Isi folder itu dengan gambar selain buah sawit terlebih dahulu."
        )

    random_generator = random.Random(seed)
    random_generator.shuffle(sawit_images)
    random_generator.shuffle(negative_images)

    per_class = min(len(sawit_images), len(negative_images))
    if limit_per_class is not None:
        per_class = min(per_class, limit_per_class)

    counts = split_counts(per_class, validation_ratio, test_ratio)
    sawit_images = sawit_images[:per_class]
    negative_images = negative_images[:per_class]

    ensure_empty_output(output_dir, overwrite=overwrite)
    for split in ("train", "validation", "test"):
        for class_name in CLASS_NAMES:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    copy_split("sawit", sawit_images, output_dir, counts)
    copy_split("bukan_sawit", negative_images, output_dir, counts)

    print(f"Dataset validasi dibuat: {output_dir}")
    print(f"- train: {counts.train} sawit, {counts.train} bukan_sawit")
    print(f"- validation: {counts.validation} sawit, {counts.validation} bukan_sawit")
    print(f"- test: {counts.test} sawit, {counts.test} bukan_sawit")


def ratio(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed >= 1:
        raise argparse.ArgumentTypeError("Rasio harus >= 0 dan < 1.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buat dataset binary sawit/bukan_sawit untuk model validasi."
    )
    parser.add_argument("--sawit-source", default=str(DEFAULT_SAWIT_SOURCE))
    parser.add_argument("--negative-source", default=str(DEFAULT_NEGATIVE_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--validation-ratio", type=ratio, default=0.15)
    parser.add_argument("--test-ratio", type=ratio, default=0.15)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Hapus dan buat ulang folder output jika sudah berisi gambar.",
    )
    args = parser.parse_args()

    prepare_dataset(
        sawit_source=Path(args.sawit_source),
        negative_source=Path(args.negative_source),
        output_dir=Path(args.output),
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        limit_per_class=args.limit_per_class,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
