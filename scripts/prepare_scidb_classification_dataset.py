from __future__ import annotations

import argparse
import csv
import hashlib
import re
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_CLASS_TO_LOCAL = {
    2: "matang",
    3: "setengah_matang",
    4: "mentah",
}
SOURCE_SPLIT_TO_TARGET = {
    "train": "train",
    "valid": "validation",
    "test": "test",
}
CLASS_ORDER = ("matang", "setengah_matang", "mentah")
DEFAULT_TARGETS_PER_SPLIT = {
    "train": 80,
    "validation": 10,
    "test": 10,
}
SOURCE_TITLE = "Oil Palm Fruit Dataset on Plantations for Harvesting Estimation"
SOURCE_DOI = "10.57760/sciencedb.12647"
SOURCE_URL = "https://doi.org/10.57760/sciencedb.12647"
SOURCE_LICENSE = "CC BY 4.0"


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_to_label_path(image_path: str) -> str:
    label_path = image_path.replace("/images/", "/labels/")
    return str(Path(label_path).with_suffix(".txt"))


def parse_label_classes(zf: zipfile.ZipFile, label_path: str) -> set[int]:
    try:
        content = zf.read(label_path).decode("utf-8")
    except KeyError:
        return set()

    classes: set[int] = set()
    for line in content.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            classes.add(int(parts[0]))
        except ValueError:
            continue
    return classes


def source_split_from_path(path: str) -> str:
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "yolov8":
        return parts[1]
    return ""


def collect_candidates(zip_path: Path) -> dict[str, list[dict[str, str]]]:
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for name in sorted(names):
            suffix = Path(name).suffix.lower()
            if "/images/" not in name or suffix not in IMAGE_EXTENSIONS:
                continue

            label_path = image_to_label_path(name)
            if label_path not in names:
                continue

            label_classes = parse_label_classes(zf, label_path)
            target_classes = label_classes & set(SOURCE_CLASS_TO_LOCAL)
            if len(target_classes) != 1:
                continue
            if label_classes - set(SOURCE_CLASS_TO_LOCAL):
                continue

            source_class = next(iter(target_classes))
            local_class = SOURCE_CLASS_TO_LOCAL[source_class]
            source_split = source_split_from_path(name)
            target_split = SOURCE_SPLIT_TO_TARGET.get(source_split, "train")
            candidates[local_class].append(
                {
                    "image_path": name,
                    "label_path": label_path,
                    "local_class": local_class,
                    "source_class": str(source_class),
                    "source_split": source_split,
                    "target_split": target_split,
                }
            )

    return candidates


def current_counts(
    dataset_dir: Path,
    targets_per_split: dict[str, int],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in targets_per_split:
        result[split] = {}
        for class_name in CLASS_ORDER:
            result[split][class_name] = count_images(dataset_dir / split / class_name)
    return result


def existing_scidb_filenames(dataset_dir: Path) -> set[str]:
    filenames: set[str] = set()
    for split in SOURCE_SPLIT_TO_TARGET.values():
        for class_name in CLASS_ORDER:
            class_dir = dataset_dir / split / class_name
            if not class_dir.exists():
                continue
            for image_path in class_dir.rglob("scidb_*"):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    filenames.add(image_path.name)
    return filenames


def safe_filename(source_path: str, local_class: str) -> str:
    basename = Path(source_path).name
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(basename).stem)
    digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:10]
    return f"scidb_{local_class}_{digest}_{stem}.jpg"


def is_available_candidate(
    item: dict[str, str],
    class_name: str,
    used_paths: set[str],
    unavailable_filenames: set[str],
) -> bool:
    return (
        item["image_path"] not in used_paths
        and safe_filename(item["image_path"], class_name) not in unavailable_filenames
    )


def build_selection(
    candidates: dict[str, list[dict[str, str]]],
    counts: dict[str, dict[str, int]],
    targets_per_split: dict[str, int],
    unavailable_filenames: set[str],
) -> tuple[list[dict[str, str]], list[str]]:
    used_paths: set[str] = set()
    selected: list[dict[str, str]] = []
    warnings: list[str] = []

    for class_name in CLASS_ORDER:
        class_candidates = candidates[class_name]
        for split, target_count in targets_per_split.items():
            existing = counts[split][class_name]
            needed = max(0, target_count - existing)
            if needed == 0:
                if existing > target_count:
                    warnings.append(
                        f"{split}/{class_name} sudah berisi {existing}, "
                        f"lebih dari target {target_count}; tidak ada file yang dihapus."
                    )
                continue

            preferred = [
                item
                for item in class_candidates
                if item["target_split"] == split
                and is_available_candidate(
                    item,
                    class_name,
                    used_paths,
                    unavailable_filenames,
                )
            ]
            fallback = [
                item
                for item in class_candidates
                if item["target_split"] != split
                and is_available_candidate(
                    item,
                    class_name,
                    used_paths,
                    unavailable_filenames,
                )
            ]
            picked = (preferred + fallback)[:needed]
            if len(picked) < needed:
                warnings.append(
                    f"Kandidat {class_name} untuk {split} kurang: "
                    f"butuh {needed}, tersedia {len(picked)}."
                )

            for item in picked:
                used_paths.add(item["image_path"])
                unavailable_filenames.add(safe_filename(item["image_path"], class_name))
                selected_item = dict(item)
                selected_item["dest_split"] = split
                selected.append(selected_item)

    return selected, warnings


def extract_selection(zip_path: Path, dataset_dir: Path, selected: list[dict[str, str]]) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for item in selected:
            dest_dir = dataset_dir / item["dest_split"] / item["local_class"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / safe_filename(item["image_path"], item["local_class"])
            item["dest_path"] = str(dest_path)

            if dest_path.exists():
                continue

            with zf.open(item["image_path"]) as source, dest_path.open("wb") as dest:
                dest.write(source.read())


def write_manifest(dataset_dir: Path, selected: list[dict[str, str]]) -> None:
    manifest_path = dataset_dir / "scidb_selection_manifest.csv"
    fieldnames = [
        "dest_split",
        "local_class",
        "dest_path",
        "source_split",
        "source_class",
        "image_path",
        "label_path",
        "source_doi",
        "source_license",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected:
            row = {key: item.get(key, "") for key in fieldnames}
            row["source_doi"] = SOURCE_DOI
            row["source_license"] = SOURCE_LICENSE
            writer.writerow(row)


def write_sources(dataset_dir: Path) -> None:
    sources_path = dataset_dir / "SOURCES.md"
    block = f"""# Dataset Sources

Generated/updated: {date.today().isoformat()}

## {SOURCE_TITLE}

- DOI: {SOURCE_DOI}
- URL: {SOURCE_URL}
- License: {SOURCE_LICENSE}
- Local mapping: Ripe -> matang, Underripe -> setengah_matang, Unripe -> mentah.
- Selection rule: images from the YOLO archive were used only when all object labels in an image belonged to exactly one of the three mapped target classes.
"""
    sources_path.write_text(block, encoding="utf-8")


def print_summary(
    title: str,
    counts: dict[str, dict[str, int]],
    targets_per_split: dict[str, int],
    candidates: dict[str, list[dict[str, str]]] | None = None,
) -> None:
    print(title)
    for split in targets_per_split:
        detail = ", ".join(
            f"{class_name}={counts[split][class_name]}" for class_name in CLASS_ORDER
        )
        print(f"- {split}: {detail}")
    if candidates is not None:
        print("Kandidat bersih dari zip:")
        for class_name in CLASS_ORDER:
            print(f"- {class_name}: {len(candidates[class_name])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a balanced classification dataset from the ScienceDB YOLO zip."
    )
    parser.add_argument("--zip", required=True, type=Path, help="Path to preprocessed-ffb.zip.")
    parser.add_argument("--dataset", default=Path("dataset"), type=Path)
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=DEFAULT_TARGETS_PER_SPLIT["train"],
        help="Target jumlah gambar train per kelas.",
    )
    parser.add_argument(
        "--validation-per-class",
        type=int,
        default=DEFAULT_TARGETS_PER_SPLIT["validation"],
        help="Target jumlah gambar validation per kelas.",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=DEFAULT_TARGETS_PER_SPLIT["test"],
        help="Target jumlah gambar test per kelas.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets_per_split = {
        "train": args.train_per_class,
        "validation": args.validation_per_class,
        "test": args.test_per_class,
    }

    candidates = collect_candidates(args.zip)
    before = current_counts(args.dataset, targets_per_split)
    print_summary("Sebelum ekstraksi:", before, targets_per_split, candidates)

    selected, warnings = build_selection(
        candidates,
        before,
        targets_per_split,
        existing_scidb_filenames(args.dataset),
    )
    print(f"File yang akan ditambahkan: {len(selected)}")
    for warning in warnings:
        print(f"PERINGATAN: {warning}")

    if args.dry_run:
        return

    extract_selection(args.zip, args.dataset, selected)
    write_manifest(args.dataset, selected)
    write_sources(args.dataset)

    after = current_counts(args.dataset, targets_per_split)
    print_summary("Sesudah ekstraksi:", after, targets_per_split)


if __name__ == "__main__":
    main()
