"""Read-only audit of the lumen dataset. Run: python src/data/audit_dataset.py."""

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "data/raw/lumen_dataset"
OUTPUT = PROJECT / "metadata"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def inspect_file(path):
    """Verify the container, then fully decode pixels; never write to the source."""
    relative = path.relative_to(DATASET)
    parts = relative.parts
    role = "image" if "image" in parts[:-1] else "mask" if "label" in parts[:-1] else "unknown"
    split = {"train": "train", "val": "validation", "test": "test"}.get(parts[0], "unspecified")
    group = "/".join(parts[:-2]) if role != "unknown" else str(relative.parent)
    record = dict(path=path.relative_to(PROJECT).as_posix(), role=role, split=split,
                  group=group, stem=path.stem, width="", height="", format="", mode="",
                  sha256="", pixel_sha256="", error="", mask_values="", empty_mask="")
    try:
        data = path.read_bytes()
        record["sha256"] = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as image:
            record.update(width=image.width, height=image.height, format=image.format, mode=image.mode)
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            # RGB normalization also catches identical images stored with different encodings.
            rgb = image.convert("RGB")
            payload = f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()
            record["pixel_sha256"] = hashlib.sha256(payload).hexdigest()
            if role == "mask":
                colors = image.getcolors(maxcolors=65536)
                record["mask_values"] = json.dumps(sorted(value for _, value in colors)) if colors else "more than 65536 values"
                record["empty_mask"] = not any(rgb.getextrema()[channel][1] for channel in range(3))
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def duplicate_groups(records, field):
    groups = defaultdict(list)
    for record in records:
        if record[field]:
            groups[record[field]].append(record)
    return [group for group in groups.values() if len(group) > 1]


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET}")
    paths = sorted(DATASET.rglob("*"))
    files = [path for path in paths if path.is_file()]
    # Snapshot file names, sizes, and modification times to check for changes during the audit.
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in files}
    records, other_files = [], []
    for index, path in enumerate(files, 1):
        # Inspect all files, including image files with unexpected extensions.
        record = inspect_file(path)
        if record["format"] or path.suffix.lower() in IMAGE_EXTENSIONS or record["role"] != "unknown":
            records.append(record)
        else:
            other_files.append(record)
        if index % 500 == 0:
            print(f"Inspected {index}/{len(files)} files", flush=True)

    images = [r for r in records if r["role"] == "image"]
    masks = [r for r in records if r["role"] == "mask"]
    unknown = [r for r in records if r["role"] == "unknown"]
    groups = defaultdict(lambda: {"image": [], "mask": []})
    for record in images + masks:
        groups[(record["group"], record["stem"])][record["role"]].append(record)

    duplicate_images = duplicate_groups(images, "pixel_sha256")
    duplicate_paths = {r["path"] for group in duplicate_images for r in group}
    cross_split = [group for group in duplicate_images if len({r["split"] for r in group}) > 1]
    cross_paths = {r["path"] for group in cross_split for r in group}
    rows = []

    def add_row(image, mask, issues):
        record = image or mask
        if image and image["error"]:
            issues.append("unreadable_image")
        if mask and mask["error"]:
            issues.append("unreadable_mask")
        if image and mask and not image["error"] and not mask["error"]:
            if (image["width"], image["height"]) != (mask["width"], mask["height"]):
                issues.append("dimension_mismatch")
        if image and image["path"] in duplicate_paths:
            issues.append("duplicate_image")
        if image and image["path"] in cross_paths:
            issues.append("cross_split_duplicate")
        if mask and mask["empty_mask"] is True:
            issues.append("empty_mask_review")
        identity = (image or mask)["path"]
        row = dict(sample_id="lumen_" + hashlib.sha256(identity.encode()).hexdigest(),
                   image_path=image["path"] if image else "", mask_path=mask["path"] if mask else "",
                   split=record["split"], subgroup=record["group"],
                   width=image["width"] if image else "", height=image["height"] if image else "",
                   image_format=image["format"] if image else "",
                   validation_status=";".join(issues) or "valid")
        for prefix, item in (("image", image), ("mask", mask)):
            for field in ("mode", "sha256", "pixel_sha256", "error"):
                row[f"{prefix}_{field}"] = item[field] if item else ""
        for field in ("width", "height", "format", "mask_values", "empty_mask"):
            row[field if field.startswith("mask_") or field == "empty_mask" else "mask_" + field] = mask[field] if mask else ""
        rows.append(row)

    for key, pair in sorted(groups.items()):
        if len(pair["image"]) == 1 and len(pair["mask"]) == 1:
            add_row(pair["image"][0], pair["mask"][0], [])
        elif len(pair["image"]) > 1 or len(pair["mask"]) > 1:
            # Do not silently choose a match when the pairing is ambiguous.
            for image in pair["image"]:
                add_row(image, None, ["ambiguous_pair"])
            for mask in pair["mask"]:
                add_row(None, mask, ["ambiguous_pair"])
        else:
            for image in pair["image"]:
                add_row(image, None, ["missing_mask"])
            for mask in pair["mask"]:
                add_row(None, mask, ["unmatched_mask"])

    after_files = sorted(path for path in DATASET.rglob("*") if path.is_file())
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in after_files}
    if before != after:
        raise RuntimeError("Dataset changed during audit; rerun against a stable dataset.")
    assert len({r["sample_id"] for r in rows}) == len(rows), "Sample IDs must be unique"
    assert sum(bool(r["image_path"]) for r in rows) == len(images)
    assert sum(bool(r["mask_path"]) for r in rows) == len(masks)
    OUTPUT.mkdir(exist_ok=True)
    manifest = OUTPUT / "dataset_manifest.csv"
    if not rows:
        raise RuntimeError("No image/mask samples were identified; check the dataset layout.")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["LUMEN DATASET AUDIT", f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
             f"Dataset: {DATASET}", f"Total files: {len(files)}", f"Images: {len(images)}",
             f"Masks: {len(masks)}", f"Unclassified image files: {len(unknown)}",
             f"Other files: {len(other_files)}", f"Manifest rows: {len(rows)}",
             "Raw file names, sizes, and modification times unchanged during audit: YES", "",
             "METHOD", "Recursively inspect every file with Pillow verify() and full pixel decoding.",
             "Pair image/label files by exact filename stem within the same split/subgroup.",
             "Paths are relative to the project root; val is standardized to validation.",
             "Sample IDs are deterministic SHA-256 hashes of the image path (mask path for orphan masks).",
             "Duplicate checks: SHA-256 of file bytes and normalized RGB pixels plus dimensions.",
             "Near-duplicate frames and semantic annotation accuracy are not assessed.",
             "Mask values are stored values; class meanings require dataset documentation.", "",
             "FOLDER STRUCTURE (direct file counts)"]
    counts = Counter(path.parent for path in files)
    for directory in [DATASET] + [path for path in paths if path.is_dir()]:
        lines.append(f"{directory.relative_to(DATASET).as_posix()}/: {counts[directory]} files")
    lines.extend(["", "SPLIT COUNTS"])
    for split in sorted({r["split"] for r in records}):
        subset = [r for r in records if r["split"] == split]
        lines.append(f"{split}: images={sum(r['role'] == 'image' for r in subset)}, masks={sum(r['role'] == 'mask' for r in subset)}, total_files={sum(r['split'] == split for r in records + other_files)}")
    for role, subset in (("Images", images), ("Masks", masks)):
        lines.extend(["", role.upper(), f"Formats: {dict(Counter(r['format'] or 'unreadable' for r in subset))}",
                      f"Dimensions (width x height): {dict(Counter(str(r['width']) + 'x' + str(r['height']) for r in subset))}",
                      f"Modes: {dict(Counter(r['mode'] for r in subset))}"])
    color_counts = [len(json.loads(r["mask_values"])) for r in masks if r["mask_values"].startswith("[")]
    lines.extend([f"Distinct stored colors/values per mask: {min(color_counts) if color_counts else 'unknown'} to {max(color_counts) if color_counts else 'unknown'}",
                  "Per-mask value sets are recorded in the CSV mask_values column.",
                  "Masks with more than two values require an explicit label interpretation before training.", "", "VALIDATION STATUS",
                  *[f"{status}: {count}" for status, count in sorted(Counter(r['validation_status'] for r in rows).items())]])
    for issue in ("missing_mask", "unmatched_mask", "ambiguous_pair", "dimension_mismatch", "empty_mask_review"):
        lines.append(f"{issue}: {sum(issue in r['validation_status'].split(';') for r in rows)}")
    lines.append("Dimension mismatches by split: " + str(dict(Counter(r["split"] for r in rows if "dimension_mismatch" in r["validation_status"]))))
    lines.append("Cross-split duplicate group combinations: " + str(dict(Counter(" + ".join(sorted({r["split"] for r in group})) for group in cross_split))))
    errors = [r for r in records if r["error"]]
    lines.append(f"Unreadable/corrupted files: {len(errors)}")
    lines.extend(f"{r['path']}: {r['error']}" for r in errors)
    lines.extend(["", "PAIRING / DIMENSION / MASK ISSUES"])
    problems = [r for r in rows if any(s in r["validation_status"] for s in ("missing", "unmatched", "ambiguous", "mismatch", "empty_mask"))]
    lines.extend(f"{r['sample_id']}: {r['validation_status']} | {r['image_path']} | {r['mask_path']}" for r in problems)
    if not problems:
        lines.append("None")
    for label, groups_found in (("Byte-identical image", duplicate_groups(images, "sha256")),
                                ("Pixel-identical image", duplicate_images),
                                ("Cross-split pixel-identical image", cross_split)):
        lines.extend(["", f"{label} duplicate groups: {len(groups_found)}",
                      f"Files involved: {sum(len(g) for g in groups_found)}; extra copies: {sum(len(g)-1 for g in groups_found)}"])
        for number, group in enumerate(groups_found, 1):
            lines.append(f"Group {number}: " + " | ".join(r["path"] for r in group))
    lines.extend(["", "UNCLASSIFIED / OTHER FILES"])
    lines.extend(r["path"] for r in unknown + other_files)
    if not unknown and not other_files:
        lines.append("None")
    lines.extend(["", "LIMITATIONS", "No patient-level or video-level independence is established by exact duplicate checks.",
                  "Filename pairing and valid dimensions do not prove that annotations are correct.",
                  "No model was trained, and no raw files were written."])
    report = OUTPUT / "dataset_audit.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Finished: {len(images)} images, {len(masks)} masks; {len(errors)} unreadable files.")
    print(f"Validation: {dict(Counter(r['validation_status'] for r in rows))}")
    print(f"Pixel duplicate groups: {len(duplicate_images)}; cross-split groups: {len(cross_split)}")
    print(f"Manifest: {manifest}\nReport: {report}")


if __name__ == "__main__":
    main()
