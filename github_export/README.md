# Urological Procedure Digital Twin: first segmentation baseline

This is a proof-of-concept binary lumen segmentation model, not a completed
digital twin. It does not yet model irrigation flow or intrarenal pressure.

## Current model and repository contents

Version 2 is the selected proof-of-concept model, using the local checkpoint
`models/unet_v2_best.pt`. See `metadata/final_model_summary.txt` for its results
and limitations. The baseline documentation below records the original experiment.

Git includes source code, tests, dependency requirements, text/CSV/JSON reports,
and performance charts. Datasets, virtual environments, model checkpoints,
dataset-image galleries, and generated prediction images/arrays stay local.
No dataset or checkpoint is bundled with a fresh clone. Empty folders are not
tracked by Git; scripts create their output folders when run, and input data
must be supplied separately under `data/raw/lumen_dataset/`.

`requirements.txt` records the four direct dependencies from the working Python
3.11 environment. In a new Python 3.11 virtual environment, install them with
`python -m pip install -r requirements.txt`. For the CUDA build used here,
install PyTorch using the command in the environment section first, then install
the requirements. This preparation step has not installed any packages.

With the checkpoint available locally, run inference from the project folder:

```powershell
.\.venv\Scripts\python.exe src/models/predict_single_image.py --image "path/to/image.png"
```

Outputs are saved under `metadata/single_image_predictions/`, which Git ignores.

## Data and preprocessing

Raw data is read-only. `src/data/preprocess_dataset.py` consumes
`metadata/dataset_manifest.csv` and produces normalized 256 x 256 RGB images
and binary masks under `data/processed/lumen_baseline_256/`.
Images use bilinear resizing and values in [0, 1]. Masks are thresholded at
intensity >127 before nearest-neighbor resizing. Bright mask pixels are assumed
to represent lumen. This label interpretation needs dataset documentation review.

Training consumes the existing processed manifest without resplitting:

| Split | Samples | Source policy |
|---|---:|---|
| Train | 311 | video_3 |
| Validation | 99 | both video_7 parts |
| Test | 966 | existing patient-labeled test groups |

Preprocessing excluded dimension mismatches, conflicting labels, duplicates, and
samples violating source separation. The original dataset split counts therefore
differ. Known shared videos and exact pixel duplicates are separated. Published
mappings identify video_3 as Patient 2 and video_7 as Patient 4; exact-image matches
associate test group p_003 with Patient 3. Test groups p001 and p006 remain
unidentified, so complete patient independence cannot be confirmed. See
`metadata/patient_mapping_analysis.txt` for the supporting evidence.

## Python environment

The project `.venv` inherits the existing Anaconda NumPy, Pillow, and Matplotlib
packages. PyTorch is installed locally into this environment; global Anaconda
packages are not replaced. Use this Python executable for model commands:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

To recreate the environment on this computer:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

This official CUDA 11.8 wheel was selected for the installed NVIDIA driver.
Other computers should choose a compatible build from the
[PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).
The training script automatically selects CUDA when available, otherwise CPU.

## Small U-Net

`src/models/unet.py` uses encoder channels 8, 16, 32, then a 64-channel
bottleneck. Three max-pooling stages gather context; transpose convolutions
restore resolution. Skip connections concatenate encoder features into the
decoder to retain spatial detail. Group normalization works with small batches.
One output channel gives a lumen logit per pixel. Sigmoid maps logits to
probabilities; a fixed threshold of 0.5 produces the predicted binary mask.

Training uses Adam (learning rate 0.001), batch size 4, and an equal mixture of
binary cross-entropy and soft Dice loss. No pretrained weights, augmentation,
mixed precision, or test-driven tuning are used. Seed 42 controls Python, NumPy,
PyTorch, and training sample order. Deterministic algorithms are enabled;
results may still differ across hardware/software versions, as described in
[PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Run and validate

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_baseline.py -v
.\.venv\Scripts\python.exe src/models/train_unet.py --smoke
.\.venv\Scripts\python.exe src/models/train_unet.py --epochs 8
```

The smoke test uses 8 training and 4 validation samples for one epoch. Its
evaluation/plotting uses the validation subset, not the held-out test labels.
Smoke artifacts are separate under `models/smoke_test/` and `metadata/smoke_test/`.
The full run starts from a fresh seed and initialization, trains for at most
8 epochs, and stops after 3 consecutive epochs without improved validation Dice.
Existing checkpoints/history are protected against accidental overwriting;
preserve previous run outputs before intentionally repeating a run.

`LumenDataset` loads the existing NPZ pairs, verifies their hashes and values,
then changes RGB array layout from HWC to CHW. DataLoaders shuffle training only
and use zero background workers for a straightforward Windows-compatible setup.

## Outputs and interpretation

- `models/baseline_unet_best.pt`: checkpoint selected by highest validation Dice,
  with model weights, optimizer state, epoch, and configuration.
- `metadata/training_results.csv`: each epoch's train/validation loss, Dice, IoU,
  sample counts, timings, and checkpoint decision.
- `metadata/training_curves.png`: loss, Dice, and IoU curves.
- `metadata/training_config.json`: settings, device, versions, and manifest hash.
- `metadata/training_summary.json`: best epoch, final held-out metrics, timings,
  GPU memory use, and per-source results.
- `metadata/test_metrics_per_image.csv`: all held-out per-image metrics.
- `metadata/test_predictions.png`: original resized image, truth, prediction,
  and overlay for 12 test examples (4 from each source group, seed 42).
- `metadata/test_predictions/`: binary PNG predictions, probability arrays,
  and the selected sample list.

Dice = 2 * intersection / (predicted foreground + true foreground).
IoU = intersection / union. Scores are averaged per image; 1 is perfect and
0 means no foreground overlap. Two empty masks score 1; only one empty scores 0.
Training metrics are measured while weights change; validation uses the fixed
end-of-epoch model. The test set is evaluated after reloading the best validation
checkpoint. Examples are chosen without considering prediction quality.

With only one known source video group each for training and validation,
performance should be treated as an initial technical baseline. Annotation
quality and patient independence remain unresolved; high scores would not
establish clinical reliability.

## First recorded run

The four automated tests and the one-epoch smoke test passed before training.
The full run used an NVIDIA GeForce RTX 3050 Laptop GPU, 121,177 model parameters,
seed 42, and 8 epochs. Training, test evaluation, and plot generation took about
89 seconds (excluding dependency installation); peak PyTorch tensor allocation
was approximately 183 MB, which excludes driver/runtime memory overhead.

The checkpoint selected at epoch 6 achieved validation Dice 0.5313 and IoU
0.3813. On all 966 test images it achieved mean Dice 0.4558 and IoU 0.3363.
Test Dice differed by source: patient_001 0.3037, patient_003 0.3848, and
patient_006 0.5972. These are image-level averages, not independent patient-level
estimates. Visual inspection found incomplete lumen coverage and false-positive
regions around image borders. No settings were changed based on test results.

All 4,362 raw file hashes were checked against the original audit after the run
and were unchanged. Saved predictions, test sample membership, test metric
counts, and best-checkpoint selection were also checked.
