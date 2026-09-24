"""Show 20 reproducibly sampled image/mask pairs without changing raw data."""

import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[2]


def main():
    with (PROJECT / "metadata/dataset_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(42)
    selected = []
    for split, count in (("train", 7), ("validation", 7), ("test", 6)):
        candidates = [r for r in rows if r["split"] == split and r["image_path"] and r["mask_path"]]
        selected.extend(rng.sample(candidates, count))

    fig, axes = plt.subplots(10, 6, figsize=(20, 30))
    fig.suptitle("Ureteroscopy: 20 randomly selected image–mask pairs", fontsize=22, y=0.995)
    fig.text(0.5, 0.979, "Seed 42 | 7 train, 7 validation, 6 test | Overlay: mask intensity >127 in red (45% opacity)", ha="center", fontsize=12)
    fig.text(0.5, 0.969, "Native pixel coordinates; no mask resizing or alignment correction. Orange titles flag dimension mismatches.", ha="center", fontsize=12)
    for index, row in enumerate(selected):
        grid_row, block = divmod(index, 2)
        panels = axes[grid_row, block * 3:block * 3 + 3]
        with Image.open(PROJECT / row["image_path"]) as source:
            image = np.array(source.convert("RGB"))
        with Image.open(PROJECT / row["mask_path"]) as source:
            mask = np.array(source.convert("L"))
        ih, iw = image.shape[:2]
        mh, mw = mask.shape
        mismatch = (iw, ih) != (mw, mh)
        color = "darkorange" if mismatch else "black"
        name = Path(row["image_path"]).stem
        panels[0].imshow(image, interpolation="nearest")
        panels[1].imshow(mask, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        panels[2].imshow(image, interpolation="nearest")
        overlay = np.zeros((mh, mw, 4))
        overlay[:, :, 0] = 1
        overlay[:, :, 3] = (mask > 127) * 0.45
        panels[2].imshow(overlay, interpolation="nearest")
        panels[0].set_title(f"{index + 1:02d} | {row['split']} | Original {iw}×{ih}\n{name}", fontsize=8, color=color)
        panels[1].set_title(f"Mask {mw}×{mh}", fontsize=9, color=color)
        panels[2].set_title("Overlay" + (" | SIZE MISMATCH" if mismatch else ""), fontsize=9, color=color)
        for ax in panels:
            ax.set_xlim(-0.5, max(iw, mw) - 0.5)
            ax.set_ylim(max(ih, mh) - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.axis("off")
        print(f"{index+1:02d} {row['split']} {name}: image={iw}x{ih}, mask={mw}x{mh}" + (" SIZE MISMATCH" if mismatch else ""))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.008, top=0.952, wspace=0.07, hspace=0.28)
    output = PROJECT / "metadata/sample_visualization.png"
    fig.savefig(output, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
