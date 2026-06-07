from __future__ import annotations

import argparse
import csv
import hashlib
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ORDER = ("matang", "setengah_matang", "mentah")
SOURCE_CLASS_TO_LOCAL = {
    2: "matang",
    3: "setengah_matang",
    4: "mentah",
}
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_DOI = "10.57760/sciencedb.12647"
SOURCE_URL = "https://doi.org/10.57760/sciencedb.12647"


@dataclass(frozen=True)
class CropCandidate:
    image_path: str
    label_path: str
    local_class: str
    source_class: int
    source_split: str
    crop_index: int
    x_center: float
    y_center: float
    width: float
    height: float


def image_to_label_path(image_path: str) -> str:
    label_path = image_path.replace("/images/", "/labels/")
    return str(Path(label_path).with_suffix(".txt"))


def source_split_from_path(path: str) -> str:
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "yolov8":
        return parts[1]
    return ""


def parse_yolo_rows(zf: zipfile.ZipFile, label_path: str) -> list[tuple[int, float, float, float, float]]:
    try:
        content = zf.read(label_path).decode("utf-8")
    except KeyError:
        return []

    rows: list[tuple[int, float, float, float, float]] = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            rows.append((int(parts[0]), *(float(value) for value in parts[1:5])))
        except ValueError:
            continue
    return rows


def collect_candidates(zip_path: Path) -> list[CropCandidate]:
    candidates: list[CropCandidate] = []
    target_classes = set(SOURCE_CLASS_TO_LOCAL)

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for image_path in sorted(names):
            if "/images/" not in image_path or Path(image_path).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = image_to_label_path(image_path)
            if label_path not in names:
                continue

            rows = parse_yolo_rows(zf, label_path)
            label_classes = {row[0] for row in rows}
            image_target_classes = label_classes & target_classes
            if len(image_target_classes) != 1:
                continue
            if label_classes - target_classes:
                continue

            source_class = next(iter(image_target_classes))
            local_class = SOURCE_CLASS_TO_LOCAL[source_class]
            for crop_index, (_, x_center, y_center, width, height) in enumerate(rows):
                candidates.append(
                    CropCandidate(
                        image_path=image_path,
                        label_path=label_path,
                        local_class=local_class,
                        source_class=source_class,
                        source_split=source_split_from_path(image_path),
                        crop_index=crop_index,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                    )
                )

    return candidates


def stable_key(candidate: CropCandidate) -> tuple[str, int]:
    digest = hashlib.sha1(
        f"{candidate.image_path}:{candidate.crop_index}".encode("utf-8")
    ).hexdigest()
    return digest, candidate.crop_index


def choose_candidates(
    candidates: list[CropCandidate],
    train_per_class: int,
    validation_per_class: int,
    test_per_class: int,
) -> tuple[list[tuple[str, CropCandidate]], list[str]]:
    by_class_split: dict[str, dict[str, list[CropCandidate]]] = {
        class_name: defaultdict(list) for class_name in CLASS_ORDER
    }
    for candidate in candidates:
        by_class_split[candidate.local_class][candidate.source_split].append(candidate)

    for split_map in by_class_split.values():
        for split, split_candidates in split_map.items():
            split_map[split] = sorted(split_candidates, key=stable_key)

    selected: list[tuple[str, CropCandidate]] = []
    warnings: list[str] = []

    for class_name in CLASS_ORDER:
        used: set[tuple[str, int]] = set()

        def take(split_name: str, source_splits: tuple[str, ...], needed: int) -> None:
            pool: list[CropCandidate] = []
            for source_split in source_splits:
                pool.extend(by_class_split[class_name].get(source_split, []))
            available = [
                candidate
                for candidate in sorted(pool, key=stable_key)
                if (candidate.image_path, candidate.crop_index) not in used
            ]
            picked = available[:needed]
            if len(picked) < needed:
                warnings.append(
                    f"{split_name}/{class_name}: butuh {needed}, tersedia {len(picked)}."
                )
            for candidate in picked:
                used.add((candidate.image_path, candidate.crop_index))
                selected.append((split_name, candidate))

        if class_name == "mentah":
            # The source archive has no clean unripe samples in train. Use source valid
            # for training, then hold source test out for validation/test.
            take("train", ("valid",), train_per_class)
            take("validation", ("test",), validation_per_class)
            take("test", ("test",), test_per_class)
        else:
            take("train", ("train",), train_per_class)
            take("validation", ("valid",), validation_per_class)
            take("test", ("test",), test_per_class)

    return selected, warnings


def safe_filename(candidate: CropCandidate) -> str:
    basename = Path(candidate.image_path).name
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(basename).stem)
    digest = hashlib.sha1(
        f"{candidate.image_path}:{candidate.crop_index}".encode("utf-8")
    ).hexdigest()[:10]
    return f"scidb_crop_{candidate.local_class}_{digest}_{stem}.jpg"


def crop_box(candidate: CropCandidate, image_width: int, image_height: int, padding: float) -> tuple[int, int, int, int]:
    box_width = candidate.width * image_width
    box_height = candidate.height * image_height
    x_center = candidate.x_center * image_width
    y_center = candidate.y_center * image_height

    half_width = box_width * (1.0 + padding) / 2.0
    half_height = box_height * (1.0 + padding) / 2.0

    left = max(0, int(round(x_center - half_width)))
    top = max(0, int(round(y_center - half_height)))
    right = min(image_width, int(round(x_center + half_width)))
    bottom = min(image_height, int(round(y_center + half_height)))
    return left, top, right, bottom


def extract_crops(
    zip_path: Path,
    dataset_dir: Path,
    selected: list[tuple[str, CropCandidate]],
    padding: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for split, candidate in selected:
            dest_dir = dataset_dir / split / candidate.local_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / safe_filename(candidate)

            with zf.open(candidate.image_path) as handle:
                image = Image.open(handle)
                image = ImageOps.exif_transpose(image).convert("RGB")
                left, top, right, bottom = crop_box(candidate, image.width, image.height, padding)
                crop = image.crop((left, top, right, bottom))
                crop.save(dest_path, format="JPEG", quality=92)

            rows.append(
                {
                    "split": split,
                    "local_class": candidate.local_class,
                    "dest_path": str(dest_path),
                    "source_split": candidate.source_split,
                    "source_class": str(candidate.source_class),
                    "image_path": candidate.image_path,
                    "label_path": candidate.label_path,
                    "crop_index": str(candidate.crop_index),
                    "x_center": f"{candidate.x_center:.8f}",
                    "y_center": f"{candidate.y_center:.8f}",
                    "width": f"{candidate.width:.8f}",
                    "height": f"{candidate.height:.8f}",
                    "source_doi": SOURCE_DOI,
                    "source_license": SOURCE_LICENSE,
                }
            )
    return rows


def write_manifest(dataset_dir: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "split",
        "local_class",
        "dest_path",
        "source_split",
        "source_class",
        "image_path",
        "label_path",
        "crop_index",
        "x_center",
        "y_center",
        "width",
        "height",
        "source_doi",
        "source_license",
    ]
    with (dataset_dir / "scidb_crop_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sources(dataset_dir: Path, padding: float) -> None:
    text = f"""# Dataset Sources

Generated/updated: {date.today().isoformat()}

## Oil Palm Fruit Dataset on Plantations for Harvesting Estimation

- DOI: {SOURCE_DOI}
- URL: {SOURCE_URL}
- License: {SOURCE_LICENSE}
- Local mapping: Ripe -> matang, Underripe -> setengah_matang, Unripe -> mentah.
- Selection rule: only images whose YOLO labels contain exactly one target class and no non-target classes.
- Processing: each selected YOLO bounding box was cropped with {padding:.2f} padding.
"""
    (dataset_dir / "SOURCES.md").write_text(text, encoding="utf-8")


def count_selected(selected: list[tuple[str, CropCandidate]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for split, candidate in selected:
        counts[split][candidate.local_class] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a cropped ScienceDB classification dataset.")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--dataset", default=Path("dataset_crops"), type=Path)
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--validation-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--padding", type=float, default=0.18)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates = collect_candidates(args.zip)
    available = Counter(candidate.local_class for candidate in candidates)
    print("Kandidat crop bersih:")
    for class_name in CLASS_ORDER:
        print(f"- {class_name}: {available[class_name]}")

    selected, warnings = choose_candidates(
        candidates,
        args.train_per_class,
        args.validation_per_class,
        args.test_per_class,
    )
    counts = count_selected(selected)
    print("Dataset yang akan dibuat:")
    for split in ("train", "validation", "test"):
        detail = ", ".join(f"{class_name}={counts[split][class_name]}" for class_name in CLASS_ORDER)
        print(f"- {split}: {detail}")
    for warning in warnings:
        print(f"PERINGATAN: {warning}")

    if args.dry_run:
        return

    args.dataset.mkdir(parents=True, exist_ok=True)
    rows = extract_crops(args.zip, args.dataset, selected, args.padding)
    write_manifest(args.dataset, rows)
    write_sources(args.dataset, args.padding)


if __name__ == "__main__":
    main()
