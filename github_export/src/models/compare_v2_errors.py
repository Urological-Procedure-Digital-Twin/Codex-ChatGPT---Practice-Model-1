"""Paired inference only: compare saved models on identical processed test inputs."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_unet import PROJECT, MANIFEST, LumenDataset, SmallUNet, read_splits, seed_everything, write_csv


def plot_rows(rows, lookup, predictions, output, heading, offset=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(rows), 4, figsize=(12, 2.8 * len(rows)), squeeze=False)
    for index, row in enumerate(rows):
        image, mask, sid = LumenDataset([lookup[row["sample_id"]]])[0]
        images = [image.permute(1, 2, 0).numpy(), mask[0].numpy(),
                  predictions["baseline"][sid], predictions["version2"][sid]]
        titles = [f"{offset+index+1}. {Path(row['image_path']).name}\nOriginal (resized)",
                  "Ground truth", f"Baseline Dice {row['baseline_dice']:.3f}",
                  f"V2 Dice {row['version2_dice']:.3f} | change {row['dice_change']:+.3f}"]
        for column, axis in enumerate(axes[index]):
            axis.imshow(images[column], **({} if column == 0 else dict(cmap="gray", vmin=0, vmax=1)), interpolation="nearest")
            axis.set_title(titles[column], fontsize=8)
            axis.axis("off")
    figure.suptitle(heading + " | same inputs, threshold 0.5", y=0.995, fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(output, dpi=120)
    plt.close(figure)


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = PROJECT / "metadata/v2_error_comparison"
    output.mkdir(exist_ok=True)
    paths = {"baseline": PROJECT / "models/baseline_unet_best.pt", "version2": PROJECT / "models/unet_v2_best.pt"}
    fingerprints = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    models = {}
    for name, path in paths.items():
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        assert checkpoint["config"]["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        model = SmallUNet(checkpoint["config"]["base_channels"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.requires_grad_(False)
        models[name] = model
    records = read_splits()["test"]
    lookup = {r["sample_id"]: r for r in records}
    loader = DataLoader(LumenDataset(records), batch_size=4, shuffle=False, num_workers=0)
    predictions = {name: {} for name in models}
    rows = []
    border = np.ones((256, 256), dtype=bool)
    border[8:-8, 8:-8] = False
    with torch.inference_mode():
        for image, mask, ids in loader:
            inputs = image.to(device)
            binary = {name: (model(inputs)[:, 0] >= 0).cpu().numpy() for name, model in models.items()}
            for index, sid in enumerate(ids):
                source = lookup[sid]
                truth = mask[index, 0].numpy().astype(bool)
                row = dict(sample_id=sid, image_path=source["image_path"], mask_path=source["mask_path"],
                           source_group=source["source_group"], true_foreground_pixels=int(truth.sum()))
                for name in models:
                    prediction = binary[name][index]
                    predictions[name][sid] = prediction
                    tp = int((prediction & truth).sum())
                    fp = int((prediction & ~truth).sum())
                    fn = int((~prediction & truth).sum())
                    denominator = 2 * tp + fp + fn
                    row[f"{name}_dice"] = 2 * tp / denominator if denominator else 1.0
                    row[f"{name}_false_positive_pixels"] = fp
                    row[f"{name}_false_negative_pixels"] = fn
                    row[f"{name}_outer_border_false_positive_pixels"] = int((prediction & ~truth & border).sum())
                row["dice_change"] = row["version2_dice"] - row["baseline_dice"]
                rows.append(row)
            if len(rows) % 200 == 0:
                print(f"Compared {len(rows)}/{len(records)} identical test inputs", flush=True)
    assert len(rows) == len(lookup) == len({r["sample_id"] for r in rows})
    descending = sorted(rows, key=lambda r: (-r["dice_change"], r["image_path"]))
    ascending = sorted(rows, key=lambda r: (r["dice_change"], r["image_path"]))
    improved, worsened = descending[:15], ascending[:15]
    assert all(r["dice_change"] > 0 for r in improved)
    assert all(r["dice_change"] < 0 for r in worsened)
    write_csv(output / "all_test_dice_comparison.csv", descending)
    write_csv(output / "most_improved_15.csv", improved)
    write_csv(output / "most_worsened_15.csv", worsened)
    for name, selected, heading in (("most_improved", improved, "Largest Dice improvements"),
                                    ("most_worsened", worsened, "Largest Dice regressions")):
        plot_rows(selected, lookup, predictions, output / f"{name}_15.png", heading)
        for page in range(3):
            plot_rows(selected[page*5:(page+1)*5], lookup, predictions,
                      output / f"{name}_page_{page+1}.png", heading, offset=page*5)
    summary = dict(test_images=len(rows), threshold=0.5, baseline_checkpoint_sha256=fingerprints["baseline"],
                   version2_checkpoint_sha256=fingerprints["version2"],
                   improved=sum(r["dice_change"] > 0 for r in rows),
                   worsened=sum(r["dice_change"] < 0 for r in rows),
                   unchanged=sum(r["dice_change"] == 0 for r in rows),
                   baseline_mean_dice=float(np.mean([r["baseline_dice"] for r in rows])),
                   version2_mean_dice=float(np.mean([r["version2_dice"] for r in rows])),
                   mean_dice_change=float(np.mean([r["dice_change"] for r in rows])),
                   improvement_range=[min(r["dice_change"] for r in improved), max(r["dice_change"] for r in improved)],
                   regression_range=[min(r["dice_change"] for r in worsened), max(r["dice_change"] for r in worsened)],
                   improved_selection_sources=dict(Counter(r["source_group"] for r in improved)),
                   worsened_selection_sources=dict(Counter(r["source_group"] for r in worsened)),
                   border_definition="Outermost 8 pixels of the rectangular 256x256 image; excludes ground-truth foreground. Does not measure the entire circular endoscope boundary.")
    for name in models:
        summary[f"{name}_total_outer_border_false_positive_pixels"] = sum(r[f"{name}_outer_border_false_positive_pixels"] for r in rows)
    small = [r for r in rows if r["true_foreground_pixels"] <= 0.02 * 256 * 256]
    summary["small_lumen_definition"] = "Ground-truth foreground <=2% of processed image area; descriptive, not a clinical size measure"
    summary["small_lumen_images"] = len(small)
    summary["small_lumen_mean_dice_change"] = float(np.mean([r["dice_change"] for r in small])) if small else None
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == fingerprints[name] for name, path in paths.items())
    summary["checkpoints_unchanged"] = True
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
