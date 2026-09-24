"""Evaluate the saved U-Net without training; plot the best/worst test cases.

Run: .venv/Scripts/python.exe src/models/evaluate_test.py
Uses the existing processed manifest and fixed probability threshold 0.5.
"""

import hashlib
import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_unet import (PROJECT, MANIFEST, LumenDataset, SmallUNet,
                        overlap_scores, read_splits, seed_everything, write_csv)


def plot_cases(cases, lookup, predictions, destination, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(cases), 4, figsize=(12, 2.8 * len(cases)), squeeze=False)
    for index, case in enumerate(cases):
        sample_id = case["sample_id"]
        image, truth, _ = LumenDataset([lookup[sample_id]])[0]
        original = image.permute(1, 2, 0).numpy()
        prediction = predictions[sample_id]
        panels = axes[index]
        panels[0].imshow(original)
        panels[1].imshow(truth[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        panels[2].imshow(prediction, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        panels[3].imshow(original)
        overlay = np.zeros((256, 256, 4))
        overlay[:, :, 0] = 1
        overlay[:, :, 3] = prediction * 0.45
        panels[3].imshow(overlay)
        titles = [f"{index+1}. Original (resized)\n{case['image_name']}", "Ground truth",
                  f"Prediction\nDice {case['dice']:.4f} | IoU {case['iou']:.4f}", "Prediction overlay (red)"]
        for panel, text in zip(panels, titles):
            panel.set_title(text, fontsize=8)
            panel.axis("off")
    figure.suptitle(title + " | fixed threshold 0.5", fontsize=14, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.978))
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def main():
    checkpoint_path = PROJECT / "models/baseline_unet_best.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"]["manifest_sha256"] != hashlib.sha256(MANIFEST.read_bytes()).hexdigest():
        raise ValueError("Processed manifest differs from the checkpoint's training manifest")
    seed_everything(checkpoint["config"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = read_splits()["test"]
    lookup = {r["sample_id"]: r for r in records}
    model = SmallUNet(checkpoint["config"]["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = DataLoader(LumenDataset(records), batch_size=checkpoint["config"]["batch_size"],
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    metrics, predictions = [], {}
    print(f"Evaluating {len(records)} test images on {device}, checkpoint epoch {checkpoint['epoch']}", flush=True)
    with torch.inference_mode():
        for image, mask, ids in loader:
            logits = model(image.to(device))
            dice, iou = overlap_scores(logits, mask.to(device))
            # logit >=0 is the same decision boundary as probability >=0.5.
            binary = (logits[:, 0] >= 0).cpu().numpy()
            for index, sample_id in enumerate(ids):
                truth = mask[index, 0].numpy().astype(bool)
                prediction = binary[index]
                intersection = int(np.count_nonzero(prediction & truth))
                predicted = int(prediction.sum())
                actual = int(truth.sum())
                union = int(np.count_nonzero(prediction | truth))
                # Independently cross-check GPU overlap metrics against NumPy pixel counts.
                expected_dice = 2 * intersection / (predicted + actual) if predicted + actual else 1.0
                expected_iou = intersection / union if union else 1.0
                assert abs(dice[index].item() - expected_dice) < 1e-6
                assert abs(iou[index].item() - expected_iou) < 1e-6
                source = lookup[sample_id]
                metrics.append(dict(sample_id=sample_id, image_name=source["image_path"].split("/")[-1],
                                    image_path=source["image_path"], mask_path=source["mask_path"],
                                    processed_path=source["processed_path"], split="test",
                                    source_group=source["source_group"], dice=expected_dice, iou=expected_iou,
                                    true_foreground_pixels=actual, predicted_foreground_pixels=predicted,
                                    true_positive_pixels=intersection, false_positive_pixels=predicted-intersection,
                                    false_negative_pixels=actual-intersection, threshold=0.5))
                predictions[sample_id] = prediction
            if len(metrics) % 200 == 0:
                print(f"Evaluated {len(metrics)}/{len(records)}", flush=True)
    assert len(metrics) == len(records) == len({r["sample_id"] for r in metrics})
    # IoU is monotonic in Dice for an individual image. Filename makes ties deterministic.
    best_order = sorted(metrics, key=lambda r: (-r["dice"], -r["iou"], r["image_path"]))
    worst_order = sorted(metrics, key=lambda r: (r["dice"], r["iou"], r["image_path"]))
    ranks = {r["sample_id"]: index + 1 for index, r in enumerate(best_order)}
    for record in metrics:
        record["rank_by_dice_descending"] = ranks[record["sample_id"]]
    best, worst = best_order[:10], worst_order[:10]
    output = PROJECT / "metadata/error_analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(PROJECT / "metadata/test_prediction_metrics.csv", best_order)
    write_csv(output / "best_10.csv", best)
    write_csv(output / "worst_10.csv", worst)
    plot_cases(best, lookup, predictions, output / "best_10.png", "10 best test images by Dice")
    plot_cases(worst, lookup, predictions, output / "worst_10.png", "10 worst test images by Dice")
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
    summary = dict(checkpoint=checkpoint_path.relative_to(PROJECT).as_posix(),
                   checkpoint_sha256=checkpoint_hash, checkpoint_epoch=checkpoint["epoch"],
                   checkpoint_unchanged=True, device=str(device), test_samples=len(metrics),
                   mean_dice=float(np.mean([r["dice"] for r in metrics])),
                   mean_iou=float(np.mean([r["iou"] for r in metrics])),
                   median_dice=float(np.median([r["dice"] for r in metrics])),
                   zero_dice_samples=sum(r["dice"] == 0 for r in metrics),
                   best_dice_range=[min(r["dice"] for r in best), max(r["dice"] for r in best)],
                   worst_dice_range=[min(r["dice"] for r in worst), max(r["dice"] for r in worst)],
                   best_sources=dict(Counter(r["source_group"] for r in best)),
                   worst_sources=dict(Counter(r["source_group"] for r in worst)),
                   ranking="Dice descending/ascending; IoU then image path break ties. Worst selection is one deterministic subset when more than 10 tie.",
                   metrics="Per-image foreground Dice and IoU at probability 0.5; both masks empty=1, only one empty=0. Mean is unweighted across images.",
                   visualization="Processed original images and masks at 256x256, with exactly the predictions used to calculate the CSV scores.",
                   limitations="Extreme-case examples are not representative; adjacent video frames are correlated. No retraining, threshold tuning, or checkpoint changes.")
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
