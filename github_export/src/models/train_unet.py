"""Reproducible proof-of-concept training on the existing processed splits.

Smoke test: .venv/Scripts/python.exe src/models/train_unet.py --smoke
Baseline:  .venv/Scripts/python.exe src/models/train_unet.py --epochs 8

This script never opens raw images. It consumes the preprocessing manifest and
NPZ arrays as-is. Test labels do not select checkpoints or training settings.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from unet import SmallUNet


PROJECT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT / "data/processed/lumen_baseline_256"
MANIFEST = PROCESSED / "processed_manifest.csv"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def array_hash(array):
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


class LumenDataset(Dataset):
    """Convert saved HWC images to PyTorch CHW; masks become 1 x H x W.

    Values are already normalized/binarized by preprocess_dataset.py. No second
    normalization, resizing, augmentation, or resplitting occurs here.
    """

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = (PROJECT / record["processed_path"]).resolve()
        path.relative_to(PROCESSED.resolve())
        with np.load(path, allow_pickle=False) as saved:
            image, mask = saved["image"], saved["mask"]
        if image.shape != (256, 256, 3) or mask.shape != (256, 256):
            raise ValueError(f"Unexpected dimensions: {path}")
        if image.dtype != np.float32 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f"Invalid normalized image: {path}")
        if mask.dtype != np.uint8 or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Invalid binary mask: {path}")
        if array_hash(image) != record["processed_pixel_hash"] or array_hash(mask) != record["processed_mask_hash"]:
            raise ValueError(f"Processed content differs from manifest: {path}")
        return (torch.from_numpy(image.transpose(2, 0, 1).copy()),
                torch.from_numpy(mask[None].astype(np.float32)), record["sample_id"])


def read_splits():
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Repeated sample IDs")
    if {r["split"] for r in records} != {"train", "validation", "test"}:
        raise ValueError("Expected existing train, validation, test splits")
    for key in ("source_component", "raw_pixel_hash", "processed_pixel_hash"):
        groups = defaultdict(set)
        for record in records:
            if not record[key]:
                raise ValueError(f"Missing leakage-check field {key}")
            groups[record[key]].add(record["split"])
        if any(len(splits) > 1 for splits in groups.values()):
            raise ValueError(f"Cross-split overlap: {key}")
    return {split: [r for r in records if r["split"] == split] for split in ("train", "validation", "test")}


def segmentation_loss(logits, target):
    """Equal mixture of stable binary cross-entropy and per-image soft Dice loss."""
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    intersection = (probability * target).sum(axes)
    soft_dice = (2 * intersection + 1e-6) / (probability.sum(axes) + target.sum(axes) + 1e-6)
    return 0.5 * nn.functional.binary_cross_entropy_with_logits(logits, target) + 0.5 * (1 - soft_dice.mean())


def overlap_scores(logits, target):
    """Per-image foreground Dice and IoU at fixed probability threshold 0.5.

    Both empty: score 1. Only one empty: score 0. Background isn't a class in
    these averages. Aggregation gives each image equal weight, not each batch.
    """
    prediction = logits >= 0
    truth = target >= 0.5
    axes = (1, 2, 3)
    intersection = (prediction & truth).sum(axes).float()
    total = prediction.sum(axes) + truth.sum(axes)
    union = (prediction | truth).sum(axes)
    dice = torch.where(total > 0, 2 * intersection / total.clamp(min=1), torch.ones_like(intersection))
    iou = torch.where(union > 0, intersection / union.clamp(min=1), torch.ones_like(intersection))
    return dice, iou


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, dice_sum, iou_sum, count = 0.0, 0.0, 0.0, 0
    details = []
    start = time.perf_counter()
    with torch.set_grad_enabled(training):
        for image, mask, ids in loader:
            image, mask = image.to(device), mask.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = segmentation_loss(logits, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            dice, iou = overlap_scores(logits.detach(), mask)
            batch = len(ids)
            loss_sum += loss.item() * batch
            dice_sum += dice.sum().item()
            iou_sum += iou.sum().item()
            count += batch
            details.extend(dict(sample_id=sample_id, dice=float(d), iou=float(j)) for sample_id, d, j in zip(ids, dice.cpu(), iou.cpu()))
    return dict(loss=loss_sum / count, dice=dice_sum / count, iou=iou_sum / count,
                samples=count, seconds=time.perf_counter() - start), details


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_history(history, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, title in zip(axes, ("loss", "dice", "iou"), ("BCE + soft Dice loss", "Foreground Dice", "Foreground IoU")):
        for split in ("train", "validation"):
            axis.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], marker="o", label=split)
        axis.set(xlabel="Epoch", title=title)
        if metric != "loss":
            axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def predict_examples(model, records, device, destination, seed):
    """Pick up to four samples per known source group, without using scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        groups[record["source_group"]].append(record)
    chosen = []
    for group in sorted(groups):
        chosen.extend(rng.sample(groups[group], min(4, len(groups[group]))))
    destination.mkdir(exist_ok=True)
    figure, axes = plt.subplots(len(chosen), 4, figsize=(12, 3 * len(chosen)), squeeze=False)
    selection = []
    model.eval()
    with torch.inference_mode():
        for index, record in enumerate(chosen):
            image, mask, sample_id = LumenDataset([record])[0]
            logits = model(image[None].to(device))
            probability = logits.sigmoid()[0, 0].cpu().numpy()
            prediction = (probability >= 0.5).astype(np.uint8)
            dice, iou = overlap_scores(logits, mask[None].to(device))
            original = image.permute(1, 2, 0).numpy()
            Image.fromarray(prediction * 255).save(destination / f"{sample_id}_mask.png")
            np.save(destination / f"{sample_id}_probability.npy", probability)
            panels = axes[index]
            panels[0].imshow(original)
            panels[1].imshow(mask[0], cmap="gray", vmin=0, vmax=1)
            panels[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            panels[3].imshow(original)
            overlay = np.zeros((256, 256, 4))
            overlay[:, :, 0] = 1
            overlay[:, :, 3] = prediction * 0.45
            panels[3].imshow(overlay)
            titles = ["Original (resized)\n" + Path(record["image_path"]).stem,
                      "Ground truth", f"Prediction\nDice {dice.item():.3f} | IoU {iou.item():.3f}", "Prediction overlay"]
            for axis, title in zip(panels, titles):
                axis.set_title(title, fontsize=9)
                axis.axis("off")
            selection.append(dict(sample_id=sample_id, image_path=record["image_path"], source_group=record["source_group"], dice=dice.item(), iou=iou.item()))
    figure.suptitle("Small U-Net predictions | fixed threshold 0.5 | examples chosen without looking at scores", fontsize=12, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(destination.parent / "test_predictions.png", dpi=120)
    plt.close(figure)
    write_csv(destination / "selected_samples.csv", selection)
    return len(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 8 train/4 validation; evaluation uses validation only")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = read_splits()
    if args.smoke:
        rng = random.Random(args.seed)
        splits["train"] = rng.sample(splits["train"], 8)
        splits["validation"] = rng.sample(splits["validation"], 4)
        # Exercise evaluation code without consulting the held-out test labels.
        splits["test"] = splits["validation"]
    metadata = PROJECT / "metadata" / "smoke_test" if args.smoke else PROJECT / "metadata"
    model_dir = PROJECT / "models" / "smoke_test" if args.smoke else PROJECT / "models"
    history_path = metadata / "training_results.csv"
    checkpoint_path = model_dir / "baseline_unet_best.pt"
    if history_path.exists() or checkpoint_path.exists():
        raise FileExistsError("Results already exist; preserve them before starting a new run")
    metadata.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(LumenDataset(records), batch_size=args.batch_size,
                                shuffle=split == "train", num_workers=0,
                                generator=generator if split == "train" else None,
                                pin_memory=device.type == "cuda") for split, records in splits.items()}
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 1 if args.smoke else args.epochs
    config = dict(seed=args.seed, epochs_requested=epochs, batch_size=args.batch_size,
                  learning_rate=1e-3, base_channels=8, threshold=0.5, device=str(device),
                  device_name=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                  parameters=sum(p.numel() for p in model.parameters()), torch_version=str(torch.__version__),
                  numpy_version=np.__version__, manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  split_counts={split: len(records) for split, records in splits.items()}, smoke=args.smoke)
    print(json.dumps(config, indent=2), flush=True)
    (metadata / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    history, best_dice, best_epoch, stale = [], -1.0, 0, 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train, _ = run_epoch(model, loaders["train"], device, optimizer)
        validation, _ = run_epoch(model, loaders["validation"], device)
        improved = validation["dice"] > best_dice
        if improved:
            best_dice, best_epoch, stale = validation["dice"], epoch, 0
            torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                            epoch=epoch, validation_dice=best_dice, config=config), checkpoint_path)
        else:
            stale += 1
        record = dict(epoch=epoch, **{f"train_{k}": v for k, v in train.items()},
                      **{f"validation_{k}": v for k, v in validation.items()}, best_checkpoint=improved)
        history.append(record)
        write_csv(history_path, history)
        print(f"Epoch {epoch}/{epochs} | train loss {train['loss']:.4f}, Dice {train['dice']:.4f} | val loss {validation['loss']:.4f}, Dice {validation['dice']:.4f}, IoU {validation['iou']:.4f} | {train['seconds'] + validation['seconds']:.1f}s", flush=True)
        if stale >= 3:
            print("Early stopping: validation Dice did not improve for 3 epochs", flush=True)
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Evaluating best checkpoint from epoch {best_epoch}", flush=True)
    test, per_image = run_epoch(model, loaders["test"], device)
    lookup = {r["sample_id"]: r for r in splits["test"]}
    for record in per_image:
        record["source_group"] = lookup[record["sample_id"]]["source_group"]
    write_csv(metadata / "test_metrics_per_image.csv", per_image)
    group_metrics = {}
    for group in sorted({r["source_group"] for r in per_image}):
        members = [r for r in per_image if r["source_group"] == group]
        group_metrics[group] = dict(samples=len(members), dice=float(np.mean([r["dice"] for r in members])), iou=float(np.mean([r["iou"] for r in members])))
    plot_history(history, metadata / "training_curves.png")
    examples = predict_examples(model, splits["test"], device, metadata / "test_predictions", args.seed)
    report = dict(best_epoch=best_epoch, best_validation_dice=best_dice, epochs_completed=len(history),
                  test=test, per_source=group_metrics, prediction_examples=examples,
                  elapsed_seconds=time.perf_counter() - started,
                  peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0,
                  evaluation_split="validation smoke subset, not test" if args.smoke else "test",
                  metric_definition="Mean per-image foreground Dice and IoU, probability >=0.5; both empty=1, one empty=0",
                  checkpoint=str(checkpoint_path.relative_to(PROJECT)),
                  limitations="Proof of concept. Only one known video group each for train/validation. Unknown patient identities prevent proof of patient independence. Mask threshold and original annotations need review.")
    (metadata / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
