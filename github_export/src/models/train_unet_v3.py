"""Version 3: identical V2 pipeline, replacing only BCE+Dice with Tversky loss.

Run --smoke first, then run without arguments for exactly eight epochs.
Alpha weights false positives; beta weights false negatives. No focal exponent
or BCE term is used. Metrics and threshold remain identical to prior versions.
"""
import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_unet import (PROJECT, MANIFEST, LumenDataset, SmallUNet, seed_everything,
                        read_splits, overlap_scores, write_csv)
from train_unet_v2 import AUGMENTATION, AugmentedLumenDataset, verify_augmentation


ALPHA, BETA, SMOOTH = 0.4, 0.6, 1e-6


def tversky_loss(logits, target):
    """Mean per-image 1 - (TP+eps)/(TP + 0.4*FP + 0.6*FN + eps).

    Counts use sigmoid probabilities during training, rather than thresholded
    predictions, so gradients can flow through the loss.
    """
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    tp = (probability * target).sum(axes)
    fp = (probability * (1 - target)).sum(axes)
    fn = ((1 - probability) * target).sum(axes)
    return (1 - (tp + SMOOTH) / (tp + ALPHA * fp + BETA * fn + SMOOTH)).mean()


def check_loss():
    target = torch.tensor([1., 0.]).reshape(1, 1, 1, 2)
    correct = target * 40 - 20
    assert tversky_loss(correct, target).item() < 1e-5
    assert tversky_loss(-correct, target).item() > 0.99
    # Equal TP, then one FN versus one FP: FN must receive the larger penalty.
    prediction = torch.tensor([1., 0., 0.]).reshape(1, 1, 1, 3) * 40 - 20
    missed = torch.tensor([1., 1., 0.]).reshape(1, 1, 1, 3)
    assert tversky_loss(prediction, missed).item() > tversky_loss(-prediction, 1-missed).item()
    for fill in (0., 1.):
        logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
        loss = tversky_loss(logits, torch.full_like(logits, fill))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def run_epoch(model, loader, device, optimizer=None):
    """Original epoch logic; only the differentiable loss call changes.

    Extra pixel counts in the output support recall/precision analysis; they
    never participate in optimization, checkpoint selection, or thresholding.
    """
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
            loss = tversky_loss(logits, mask)
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
            prediction, truth = logits.detach() >= 0, mask >= 0.5
            axes = (1, 2, 3)
            tp = (prediction & truth).sum(axes).cpu().tolist()
            fp = (prediction & ~truth).sum(axes).cpu().tolist()
            fn = (~prediction & truth).sum(axes).cpu().tolist()
            details.extend(dict(sample_id=sid, dice=float(d), iou=float(j),
                                true_positive_pixels=t, false_positive_pixels=p, false_negative_pixels=n)
                           for sid, d, j, t, p, n in zip(ids, dice.cpu(), iou.cpu(), tp, fp, fn))
    return dict(loss=loss_sum/count, dice=dice_sum/count, iou=iou_sum/count,
                samples=count, seconds=time.perf_counter()-start), details


def plot_history(history, output, loss_title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, title in zip(axes, ("loss", "dice", "iou"), (loss_title, "Foreground Dice", "Foreground IoU")):
        for split in ("train", "validation"):
            axis.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], marker="o", label=split)
        axis.set(xlabel="Epoch", title=title)
        if metric != "loss":
            axis.set_ylim(0, 1)
        axis.grid(alpha=.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_comparison(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, split in zip(axes, ("validation", "test")):
        for offset, row, color in zip((-0.25, 0, 0.25), rows, ("#3969ac", "#11a579", "#e68310")):
            bars = axis.bar(np.arange(2)+offset, [row[f"{split}_dice"], row[f"{split}_iou"]],
                            width=0.24, label=row["experiment"], color=color)
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axis.set(xticks=[0, 1], xticklabels=["Dice", "IoU"], ylim=(0, 1), title=split.capitalize())
        axis.legend()
    fig.suptitle("Saved best validation-Dice checkpoints | identical held-out inputs")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    paths = {"Baseline": PROJECT / "models/baseline_unet_best.pt", "Version 2": PROJECT / "models/unet_v2_best.pt"}
    protected = list(paths.values()) + [PROJECT / p for p in (
        "metadata/training_results.csv", "metadata/training_config.json", "metadata/training_summary.json",
        "metadata/version2/training_results.csv", "metadata/version2/training_config.json", "metadata/version2/training_summary.json",
        "src/models/unet.py", "src/models/train_unet.py", "src/models/train_unet_v2.py")]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    v2 = json.loads((PROJECT / "metadata/version2/training_config.json").read_text())
    assert v2["augmentation"] == AUGMENTATION
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == v2["manifest_sha256"]
    seed_everything(v2["seed"])
    splits = read_splits()
    check_loss()
    verify_augmentation(splits["train"])
    print("Loss and unchanged augmentation checks passed", flush=True)
    seed_everything(v2["seed"])
    if args.smoke:
        rng = random.Random(v2["seed"])
        splits["train"] = rng.sample(splits["train"], 8)
        splits["validation"] = rng.sample(splits["validation"], 4)
    output = PROJECT / "metadata/version3" / "smoke" if args.smoke else PROJECT / "metadata/version3"
    checkpoint_path = PROJECT / "models" / ("unet_v3_smoke_best.pt" if args.smoke else "unet_v3_best.pt")
    if checkpoint_path.exists() or (output / "training_results.csv").exists():
        raise FileExistsError("V3 output exists; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert str(device) == v2["device"]
    generator = torch.Generator().manual_seed(v2["seed"])
    loaders = {}
    for split, records in splits.items():
        dataset = AugmentedLumenDataset(records, v2["seed"]) if split == "train" else LumenDataset(records)
        loaders[split] = DataLoader(dataset, batch_size=v2["batch_size"], shuffle=split == "train",
                                   num_workers=0, generator=generator if split == "train" else None,
                                   pin_memory=device.type == "cuda")
    model = SmallUNet(v2["base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=v2["learning_rate"])
    config = dict(v2, experiment="Version 3", smoke=args.smoke, epochs_requested=1 if args.smoke else 8,
                  split_counts={s: len(r) for s, r in splits.items()},
                  loss=dict(name="mean soft Tversky loss", alpha_false_positive=ALPHA,
                            beta_false_negative=BETA, smoothing=SMOOTH, bce_term=False, focal_exponent=None))
    (output / "training_config.json").write_text(json.dumps(config, indent=2)+"\n")
    history, best_dice, best_epoch = [], -1., 0
    start = time.perf_counter()
    for epoch in range(1, config["epochs_requested"]+1):
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
        print(f"Epoch {epoch}/{config['epochs_requested']} | train Tversky loss {train['loss']:.4f} | validation Dice {validation['dice']:.4f}, IoU {validation['iou']:.4f}", flush=True)
    plot_history(history, output / "training_curves.png", loss_title="Tversky loss (alpha=0.4, beta=0.6)")
    paths["Version 3"] = checkpoint_path
    if args.smoke:
        paths = {"Version 3": checkpoint_path}
    overall, group_rows, detection = [], [], []
    lookup = {r["sample_id"]: r for r in splits["test"]+splits["validation"]}
    for name, path in paths.items():
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        val, val_rows = run_epoch(model, loaders["validation"], device)
        test, test_rows = run_epoch(model, loaders["validation" if args.smoke else "test"], device)
        for row in val_rows+test_rows:
            row["source_group"] = lookup[row["sample_id"]]["source_group"]
        overall.append(dict(experiment=name, best_epoch=ckpt["epoch"], validation_dice=val["dice"],
                            validation_iou=val["iou"], test_dice=test["dice"], test_iou=test["iou"]))
        if name == "Version 3":
            write_csv(output / "test_metrics_per_image.csv", test_rows)
            write_csv(output / "validation_metrics_per_image.csv", val_rows)
        for label, subset in [("p_003", [r for r in test_rows if r["source_group"] == "patient_003"]),
                              ("p001", [r for r in test_rows if r["source_group"] == "patient_001"]),
                              ("p006", [r for r in test_rows if r["source_group"] == "patient_006"]),
                              ("validation Patient 4", val_rows)]:
            if subset:
                group_rows.append(dict(group=label, experiment=name, number_of_images=len(subset),
                                       mean_dice=float(np.mean([r["dice"] for r in subset])),
                                       mean_iou=float(np.mean([r["iou"] for r in subset]))))
        # Additional detection diagnostics, separate from the unchanged Dice/IoU metrics.
        for label, subset in [("all test", test_rows), ("validation", val_rows),
                              ("small test openings (area <=2%)", [r for r in test_rows if r["true_positive_pixels"]+r["false_negative_pixels"] <= .02*256*256])]:
            if not subset:
                continue
            tp = sum(r["true_positive_pixels"] for r in subset)
            fp = sum(r["false_positive_pixels"] for r in subset)
            fn = sum(r["false_negative_pixels"] for r in subset)
            detection.append(dict(experiment=name, subset=label, images=len(subset),
                                  mean_dice=float(np.mean([r["dice"] for r in subset])),
                                  pooled_foreground_recall=tp/(tp+fn) if tp+fn else 1.,
                                  pooled_foreground_precision=tp/(tp+fp) if tp+fp else 1.,
                                  false_negative_pixels=fn, false_positive_pixels=fp,
                                  zero_overlap_images=sum(r["dice"] == 0 for r in subset)))
        print(f"Evaluated {name}: validation Dice {val['dice']:.4f}; test Dice {test['dice']:.4f}", flush=True)
    write_csv(output / "model_comparison.csv", overall)
    write_csv(output / "group_performance_comparison.csv", group_rows)
    write_csv(output / "detection_diagnostics.csv", detection)
    if not args.smoke:
        plot_comparison(overall, output / "model_comparison.png")
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value for p, value in hashes.items())
    summary = dict(best_epoch=best_epoch, epochs_completed=len(history), elapsed_seconds=time.perf_counter()-start,
                   checkpoint=checkpoint_path.relative_to(PROJECT).as_posix(), comparison=overall,
                   protected_artifacts_unchanged=True, smoke_evaluation_uses_validation=args.smoke)
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
