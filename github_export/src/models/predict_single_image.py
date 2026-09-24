"""Predict one image with the finalized Version 2 U-Net; never train or edit inputs.

From the project folder, use the Python environment containing PyTorch:
  .venv/Scripts/python.exe src/models/predict_single_image.py --image "path/to/image.png"

Optional: --output-dir "path/to/results"
Outputs are PNGs at the original image dimensions. White mask pixels indicate
lumen; the overlay highlights predicted lumen in red. Existing files are kept.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from unet import SmallUNet


PROJECT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT / "models/unet_v2_best.pt"
DEFAULT_OUTPUT = PROJECT / "metadata/single_image_predictions"


def preprocess_image(original):
    """Exactly match training: RGB, bilinear 256x256 resize, float32 / 255.

    No augmentation, auto-orientation, additional normalization, or cropping.
    The tensor layout is changed from HWC to the model's NCHW format.
    """
    resized = original.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / np.float32(255)
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0)


def predict(image_path, output_dir):
    image_path, output_dir = image_path.resolve(), output_dir.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Version 2 checkpoint not found: {CHECKPOINT}")
    # Protect the project's datasets and saved models even with custom output paths.
    if any(output_dir.is_relative_to(PROJECT / folder) for folder in ("data", "models")):
        raise ValueError("Choose an output directory outside the project's data/ and models/ folders.")
    with Image.open(image_path) as source:
        source.load()
        original = source.convert("RGB")
    inputs = preprocess_image(original)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config.get("experiment") != "Version 2" or config.get("threshold") != 0.5:
        raise ValueError("Checkpoint does not match the finalized Version 2 configuration.")
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet(base_channels=config["base_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    with torch.inference_mode():
        logits = model(inputs.to(device))
        # A logit >=0 is the fixed sigmoid probability >=0.5 decision boundary.
        binary = (logits[0, 0] >= 0).cpu().numpy().astype(np.uint8)
    # Return the mask to the original canvas without creating intermediate labels.
    mask = Image.fromarray(binary * 255).resize(original.size, Image.Resampling.NEAREST)
    original_pixels = np.array(original)
    overlay_pixels = original_pixels.copy()
    foreground = np.asarray(mask) > 0
    overlay_pixels[foreground] = np.rint(
        original_pixels[foreground].astype(np.float32) * 0.55
        + np.array([255, 0, 0], dtype=np.float32) * 0.45
    ).astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    number = 1
    while True:
        mask_path = output_dir / f"{image_path.stem}{suffix}_mask.png"
        overlay_path = output_dir / f"{image_path.stem}{suffix}_overlay.png"
        if not mask_path.exists() and not overlay_path.exists():
            break
        number += 1
        suffix = f"_{number}"
    with mask_path.open("xb") as handle:
        mask.save(handle, format="PNG")
    with overlay_path.open("xb") as handle:
        Image.fromarray(overlay_pixels).save(handle, format="PNG")
    print(f"Device: {device}")
    print(f"Model: {CHECKPOINT}")
    print(f"Output size: {original.width} x {original.height} (original image size)")
    print(f"Predicted mask: {mask_path}")
    print(f"Prediction overlay: {overlay_path}")
    return mask_path, overlay_path


def main():
    parser = argparse.ArgumentParser(description="Predict a lumen mask with the finalized Version 2 U-Net.")
    parser.add_argument("--image", required=True, type=Path, help="Path to one ureteroscopy image")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output folder (default: metadata/single_image_predictions)")
    args = parser.parse_args()
    try:
        predict(args.image, args.output_dir)
    except (OSError, ValueError, UnidentifiedImageError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
