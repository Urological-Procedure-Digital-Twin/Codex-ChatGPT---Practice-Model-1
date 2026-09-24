"""Controlled augmentation experiment. Run with --smoke first, then no arguments.

Only training inputs change. Reuses the original architecture, loss, metrics,
optimizer settings, epoch implementation, and deterministic seeding.
"""
import argparse
import csv
import hashlib
import json
import random
import time

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader

from train_unet import (PROJECT, MANIFEST, LumenDataset, SmallUNet, read_splits,
                        seed_everything, run_epoch, plot_history, write_csv)


AUGMENTATION = dict(rotation_degrees=[-7.5, 7.5], brightness=[0.9, 1.1],
                    contrast=[0.9, 1.1], blur_probability=0.2, blur_radius=[0.3, 0.7],
                    image_interpolation="bilinear", mask_interpolation="nearest",
                    fill=0, expand=False, order="rotation, brightness, contrast, blur")


def rotate_pair(image, mask, angle):
    """Same center, angle, canvas, and zero fill; nearest-neighbor for labels."""
    return (image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0),
            mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0))


class AugmentedLumenDataset(LumenDataset):
    def __init__(self, records, seed):
        if any(r["split"] != "train" for r in records):
            raise ValueError("Augmentation is permitted only for training records")
        super().__init__(records)
        # Independent RNG: augmentation must not change initialization or shuffle order.
        self.rng = random.Random(seed)

    def __getitem__(self, index):
        image, mask, sample_id = super().__getitem__(index)
        rgb = Image.fromarray(np.rint(image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        label = Image.fromarray(mask[0].numpy().astype(np.uint8))
        rgb, label = rotate_pair(rgb, label, self.rng.uniform(*AUGMENTATION["rotation_degrees"]))
        rgb = ImageEnhance.Brightness(rgb).enhance(self.rng.uniform(*AUGMENTATION["brightness"]))
        rgb = ImageEnhance.Contrast(rgb).enhance(self.rng.uniform(*AUGMENTATION["contrast"]))
        if self.rng.random() < AUGMENTATION["blur_probability"]:
            rgb = rgb.filter(ImageFilter.GaussianBlur(self.rng.uniform(*AUGMENTATION["blur_radius"])))
        x = np.asarray(rgb, dtype=np.float32) / np.float32(255)
        y = np.asarray(label, dtype=np.float32)
        assert np.isin(y, [0, 1]).all()
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(y[None].copy()), sample_id


def verify_augmentation(records):
    """Check paired geometry, binary labels, reproducibility, and RNG isolation."""
    square = np.zeros((256, 256), dtype=np.uint8)
    square[80:176, 80:176] = 1
    image = Image.fromarray(np.repeat((square * 255)[:, :, None], 3, axis=2))
    mask = Image.fromarray(square)
    for angle in (-7.5, 0, 7.5):
        rotated, label = rotate_pair(image, mask, angle)
        a, b = np.asarray(rotated)[:, :, 0] > 127, np.asarray(label).astype(bool)
        assert (a == b).mean() > 0.999
        assert set(np.unique(np.asarray(label))) <= {0, 1}
    first, second = AugmentedLumenDataset(records, 42), AugmentedLumenDataset(records, 42)
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    for index in range(min(8, len(records))):
        x, y, _ = first[index]
        x2, y2, _ = second[index]
        assert torch.equal(x, x2) and torch.equal(y, y2)
        assert x.shape == (3, 256, 256) and y.shape == (1, 256, 256)
        assert 0 <= x.min() <= x.max() <= 1
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()


def comparison_plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, split in zip(axes, ("validation", "test")):
        positions = np.arange(2)
        for offset, row, color in zip((-0.18, 0.18), rows, ("#3969ac", "#11a579")):
            values = [row[f"{split}_dice"], row[f"{split}_iou"]]
            bars = axis.bar(positions + offset, values, width=0.36, label=row["experiment"], color=color)
            axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.set(xticks=positions, xticklabels=["Dice", "IoU"], ylim=(0, 1), title=split.capitalize())
        axis.legend()
    fig.suptitle("Training augmentation experiment | best validation-Dice checkpoints")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    baseline_checkpoint = PROJECT / "models/baseline_unet_best.pt"
    protected = [baseline_checkpoint, PROJECT / "metadata/training_config.json",
                 PROJECT / "metadata/training_results.csv", PROJECT / "metadata/training_summary.json"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    baseline = json.loads(protected[1].read_text())
    assert baseline["seed"] == 42 and baseline["batch_size"] == 4 and baseline["epochs_requested"] == 8
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == baseline["manifest_sha256"]
    seed_everything(baseline["seed"])
    splits = read_splits()
    verify_augmentation(splits["train"])
    print("Augmentation checks passed: paired geometry, binary masks, reproducibility, RNG isolation", flush=True)
    # Reset after verification so full training starts exactly like the baseline.
    seed_everything(baseline["seed"])
    if args.smoke:
        rng = random.Random(42)
        splits["train"] = rng.sample(splits["train"], 8)
        splits["validation"] = rng.sample(splits["validation"], 4)
    output = PROJECT / "metadata/version2" / "smoke" if args.smoke else PROJECT / "metadata/version2"
    checkpoint_path = PROJECT / "models" / ("unet_v2_smoke_best.pt" if args.smoke else "unet_v2_best.pt")
    if checkpoint_path.exists() or (output / "training_results.csv").exists():
        raise FileExistsError("Version 2 output already exists; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert str(device) == baseline["device"], "Device differs from baseline; comparison needs review"
    generator = torch.Generator().manual_seed(42)
    loaders = {}
    for split, records in splits.items():
        dataset = AugmentedLumenDataset(records, 42) if split == "train" else LumenDataset(records)
        loaders[split] = DataLoader(dataset, batch_size=4, shuffle=split == "train", num_workers=0,
                                   generator=generator if split == "train" else None,
                                   pin_memory=device.type == "cuda")
    model = SmallUNet(base_channels=baseline["base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=baseline["learning_rate"])
    assert sum(p.numel() for p in model.parameters()) == baseline["parameters"]
    config = dict(baseline, augmentation=AUGMENTATION, experiment="Version 2", smoke=args.smoke,
                  epochs_requested=1 if args.smoke else 8,
                  split_counts={s: len(r) for s, r in splits.items()},
                  epoch_policy="Exactly 8 epochs as requested; same realized epoch count as baseline",
                  augmented_splits=["train"], untouched_splits=["validation", "test"])
    (output / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    history, best_dice, best_epoch = [], -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, config["epochs_requested"] + 1):
        train, _ = run_epoch(model, loaders["train"], device, optimizer)
        validation, _ = run_epoch(model, loaders["validation"], device)
        improved = validation["dice"] > best_dice
        if improved:
            best_dice, best_epoch = validation["dice"], epoch
            torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                            epoch=epoch, validation_dice=best_dice, config=config), checkpoint_path)
        history.append(dict(epoch=epoch, **{f"train_{k}": v for k, v in train.items()},
                            **{f"validation_{k}": v for k, v in validation.items()}, best_checkpoint=improved))
        write_csv(output / "training_results.csv", history)
        print(f"Epoch {epoch}/{config['epochs_requested']} | train loss {train['loss']:.4f} | validation Dice {validation['dice']:.4f}, IoU {validation['iou']:.4f}", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state_dict"])
    validation, _ = run_epoch(model, loaders["validation"], device)
    test, per_image = run_epoch(model, loaders["validation" if args.smoke else "test"], device)
    write_csv(output / "test_metrics_per_image.csv", per_image)
    plot_history(history, output / "training_curves.png")
    summary = dict(best_epoch=best_epoch, validation=validation, test=test, epochs_completed=len(history),
                   smoke_evaluation_uses_validation=args.smoke, elapsed_seconds=time.perf_counter()-start,
                   checkpoint=checkpoint_path.relative_to(PROJECT).as_posix())
    if not args.smoke:
        # Re-evaluate the original checkpoint with the very same untouched loaders.
        original = SmallUNet(base_channels=baseline["base_channels"]).to(device)
        original.load_state_dict(torch.load(baseline_checkpoint, map_location=device, weights_only=True)["model_state_dict"])
        base_val, _ = run_epoch(original, loaders["validation"], device)
        base_test, _ = run_epoch(original, loaders["test"], device)
        base_summary = json.loads(protected[3].read_text())
        assert abs(base_val["dice"] - base_summary["best_validation_dice"]) < 1e-6
        assert abs(base_test["dice"] - base_summary["test"]["dice"]) < 1e-6
        rows = [dict(experiment=name, best_epoch=epoch, validation_dice=val["dice"], validation_iou=val["iou"],
                     test_dice=held["dice"], test_iou=held["iou"])
                for name, epoch, val, held in (("Baseline", base_summary["best_epoch"], base_val, base_test),
                                              ("Version 2", best_epoch, validation, test))]
        write_csv(output / "baseline_vs_version2.csv", rows)
        comparison_plot(rows, output / "baseline_vs_version2.png")
        summary["comparison"] = rows
        summary["delta_v2_minus_baseline"] = {k: rows[1][k]-rows[0][k] for k in ("validation_dice", "validation_iou", "test_dice", "test_iou")}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value for p, value in before.items())
    summary["baseline_artifacts_unchanged"] = True
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
