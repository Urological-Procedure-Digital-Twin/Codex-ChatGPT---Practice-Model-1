"""Manifest-only, leakage-aware baseline preprocessing; raw files are read-only.

Run from any directory: python src/data/preprocess_dataset.py
Requires the already-installed Pillow and NumPy packages.
"""

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT / "metadata/dataset_manifest.csv"
OUTPUT = PROJECT / "data/processed/lumen_baseline_256"
SIZE = (256, 256)
THRESHOLD = 127
PRIORITY = {"test": 0, "validation": 1, "train": 2}


def digest(array):
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def source_group(path):
    """Conservatively group named patients; combine both parts of video 7."""
    stem = Path(path).stem
    patient = re.match(r"p_?(\d+)_video_", stem)
    if patient:
        return "patient_" + patient.group(1).zfill(3)
    video = re.match(r"(video_\d+)(?:_pt\d+)?_\d+$", stem)
    if video:
        return video.group(1)
    raise ValueError(f"Unknown source naming convention: {path}")


def read_source(relative_path, expected_hash):
    # File locations come exclusively from the input manifest, never directory scans.
    path = (PROJECT / relative_path).resolve()
    path.relative_to((PROJECT / "data/raw").resolve())
    data = path.read_bytes()
    if not expected_hash or hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"Raw file differs from the audit: {relative_path}; rerun audit first")
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        return np.array(source.convert("RGB"))


def read_pair(row):
    return (read_source(row["image_path"], row["image_sha256"]),
            read_source(row["mask_path"], row["mask_sha256"]))


def transform(image, mask):
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask dimensions differ; alignment cannot be assumed")
    # RGB masks must be grayscale replicated over channels; colored labels need a mapping.
    if not np.array_equal(mask[:, :, 0], mask[:, :, 1]) or not np.array_equal(mask[:, :, 0], mask[:, :, 2]):
        raise ValueError("Mask contains colored labels without a defined class mapping")
    binary = (mask[:, :, 0] > THRESHOLD).astype(np.uint8)
    image_out = np.asarray(Image.fromarray(image).resize(SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / np.float32(255)
    mask_out = np.asarray(Image.fromarray(binary).resize(SIZE, Image.Resampling.NEAREST), dtype=np.uint8)
    return image_out, mask_out


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_preview(records):
    """Inspect three saved samples per split, evenly spaced in filename order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = []
    for split in ("train", "validation", "test"):
        members = sorted((r for r in records if r["split"] == split), key=lambda r: r["image_path"])
        chosen.extend(members[i] for i in (0, len(members) // 2, len(members) - 1))
    figure, axes = plt.subplots(9, 3, figsize=(9, 25))
    for index, record in enumerate(chosen):
        with np.load(PROJECT / record["processed_path"], allow_pickle=False) as saved:
            image, mask = saved["image"], saved["mask"]
        axes[index, 0].imshow(image)
        axes[index, 1].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[index, 2].imshow(image)
        overlay = np.zeros((256, 256, 4))
        overlay[:, :, 0] = 1
        overlay[:, :, 3] = mask * 0.45
        axes[index, 2].imshow(overlay)
        axes[index, 0].set_title(record["split"] + " | " + Path(record["image_path"]).stem, fontsize=8)
        axes[index, 1].set_title("Binary mask", fontsize=9)
        axes[index, 2].set_title("Overlay", fontsize=9)
        for axis in axes[index]:
            axis.axis("off")
    figure.suptitle("Processed validation preview: 3 pairs per split, 256 x 256", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.99))
    figure.savefig(OUTPUT / "processed_preview.png", dpi=120)
    plt.close(figure)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {OUTPUT}")
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Manifest is empty or sample IDs are not unique")
    if any(r["split"] not in PRIORITY for r in rows):
        raise ValueError("Manifest contains an unknown split")
    for row in rows:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["sample_id"]):
            raise ValueError("Unsafe sample ID")

    parent = {}

    def find(group):
        parent.setdefault(group, group)
        if parent[group] != group:
            parent[group] = find(parent[group])
        return parent[group]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    records = []
    for index, row in enumerate(rows, 1):
        record = dict(row)
        record.update(reason="", source_group="", source_component="", target_split="",
                      raw_pixel_hash="", binary_mask_hash="", processed_pixel_hash="",
                      processed_mask_hash="", processed_path="")
        if not row["image_path"] or not row["mask_path"]:
            record["reason"] = "unpaired_sample"
            records.append(record)
            continue
        record["source_group"] = source_group(row["image_path"])
        find(record["source_group"])
        # A stale manifest is a fatal error, rather than silently using different raw data.
        image, mask = read_pair(row)
        record["raw_pixel_hash"] = digest(image)
        record["binary_mask_hash"] = digest((mask > THRESHOLD).astype(np.uint8))
        try:
            image_out, mask_out = transform(image, mask)
            record["processed_pixel_hash"] = digest(image_out)
            record["processed_mask_hash"] = digest(mask_out)
        except ValueError as error:
            record["reason"] = str(error)
        records.append(record)
        if index % 500 == 0:
            print(f"Validated {index}/{len(rows)} manifest pairs", flush=True)

    # Link source aliases using exact original OR resized-image matches, including excluded pairs.
    for key in ("raw_pixel_hash", "processed_pixel_hash"):
        grouped = defaultdict(list)
        for record in records:
            if record[key]:
                grouped[record[key]].append(record)
        for group in grouped.values():
            for record in group[1:]:
                union(group[0]["source_group"], record["source_group"])
            mask_key = "binary_mask_hash" if key == "raw_pixel_hash" else "processed_mask_hash"
            if len({r[mask_key] for r in group}) > 1:
                for record in group:
                    record["reason"] = (record["reason"] + "; " if record["reason"] else "") + "conflicting_masks_for_identical_image"

    components = defaultdict(list)
    for record in records:
        if record["source_group"]:
            record["source_component"] = find(record["source_group"])
            components[record["source_component"]].append(record)
    ownership = {}
    for component, members in components.items():
        sources = {r["source_group"] for r in members}
        if any(r["split"] == "test" for r in members):
            ownership[component] = "test"
        elif sources == {"video_3"}:
            ownership[component] = "train"
        elif sources == {"video_7"}:
            ownership[component] = "validation"
        else:
            raise ValueError(f"Source component needs an explicit split policy: {sources}")
    selected = []
    seen_raw, seen_processed = set(), set()
    for record in sorted(records, key=lambda r: (PRIORITY[r["split"]], r["image_path"], r["sample_id"])):
        record["target_split"] = ownership.get(record["source_component"], "")
        if record["target_split"] and record["split"] != record["target_split"]:
            record["reason"] = (record["reason"] + "; " if record["reason"] else "") + "source_reserved_for_" + record["target_split"]
        if record["reason"]:
            continue
        if record["raw_pixel_hash"] in seen_raw or record["processed_pixel_hash"] in seen_processed:
            record["reason"] = "duplicate_image_removed"
            continue
        seen_raw.add(record["raw_pixel_hash"])
        seen_processed.add(record["processed_pixel_hash"])
        selected.append(record)
    counts = Counter(r["split"] for r in selected)
    if set(counts) != set(PRIORITY):
        raise ValueError(f"Policy leaves an empty split: {counts}")
    for key in ("source_component", "raw_pixel_hash", "processed_pixel_hash"):
        split_sets = defaultdict(set)
        for record in selected:
            split_sets[record[key]].add(record["split"])
        if any(len(splits) != 1 for splits in split_sets.values()):
            raise AssertionError(f"Cross-split leakage: {key}")

    OUTPUT.mkdir(parents=True)
    for split in PRIORITY:
        (OUTPUT / split).mkdir()
    for record in selected:
        image, mask = read_pair(record)
        image_out, mask_out = transform(image, mask)
        destination = OUTPUT / record["split"] / (record["sample_id"] + ".npz")
        np.savez_compressed(destination, image=image_out, mask=mask_out)
        record["processed_path"] = destination.relative_to(PROJECT).as_posix()
        # Validate every saved pair, not just a few examples.
        with np.load(destination, allow_pickle=False) as saved:
            x, y = saved["image"], saved["mask"]
            assert x.shape == (256, 256, 3) and x.dtype == np.float32
            assert y.shape == (256, 256) and y.dtype == np.uint8
            assert np.isfinite(x).all() and 0 <= x.min() <= x.max() <= 1
            assert set(np.unique(y)).issubset({0, 1})
            assert digest(x) == record["processed_pixel_hash"]
            assert digest(y) == record["processed_mask_hash"]
    fields = ["sample_id", "split", "image_path", "mask_path", "processed_path", "source_group", "source_component", "raw_pixel_hash", "processed_pixel_hash", "processed_mask_hash"]
    write_csv(OUTPUT / "processed_manifest.csv", [{key: r[key] for key in fields} for r in selected])
    write_csv(OUTPUT / "preprocessing_decisions.csv", [dict(sample_id=r["sample_id"], original_split=r["split"], source_group=r["source_group"], source_component=r["source_component"], reserved_split=r["target_split"], status="excluded" if r["reason"] else "processed", reason=r["reason"], image_path=r["image_path"], mask_path=r["mask_path"], processed_path=r["processed_path"]) for r in records])
    report = {
        "input_manifest": MANIFEST.relative_to(PROJECT).as_posix(),
        "input_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "input_samples": len(rows), "processed_counts": dict(counts),
        "processed_total": len(selected), "excluded_total": len(rows) - len(selected),
        "exclusion_reasons_combined": dict(Counter(r["reason"] for r in records if r["reason"])),
        "resolution": [256, 256], "image": "RGB, bilinear resize, float32 HWC, divide by 255, range [0,1]",
        "mask": "Require grayscale or replicated grayscale RGB; intensity >127 before nearest-neighbor resize; uint8 HW {0,1}",
        "mask_assumption": "Bright pixels denote foreground lumen; requires confirmation from annotation documentation.",
        "geometry": "Exclude mismatched image/mask dimensions. Direct square resize applies identical scaling to each pair; aspect ratio may change.",
        "split_policy": "Keep original split membership; test owns linked source components, video_3 owns train, both video_7 parts own validation. Exclude incompatible original memberships.",
        "leakage_checks": "No shared source components, original image hashes, or processed image hashes between splits; repeated images removed within splits too.",
        "limitations": "Unknown patient identity for video_3/video_7 prevents proof of patient independence. No general perceptual near-duplicate detection. Training/validation membership differs from the original benchmark; all 966 test samples are retained in this run.",
        "source_assignments": {component: {"split": ownership[component], "sources": sorted({r['source_group'] for r in members})} for component, members in components.items()},
        "validation": f"All {len(selected)} saved pairs reopened and checked for dimensions, dtypes, finite/ranged images, binary masks, and expected hashes.",
    }
    (OUTPUT / "preprocessing_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    create_preview(selected)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
