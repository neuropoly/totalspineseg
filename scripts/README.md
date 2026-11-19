# Training

This folder contains scripts used to download the data, prepare the nnUNet datasets and train the TotalSpineSeg models.

## Setup

1. Install TotalSpineSeg with the training dependencies (`auglab` provides the data augmentation used by the custom trainer):
```bash
git clone https://github.com/neuropoly/totalspineseg.git
python3 -m pip install -e "./totalspineseg[training]"
```

2. Make sure `jq` and `git-annex` are installed (used to read the data list and download the images).

3. Set the environment variables (optional):
```bash
export TOTALSPINESEG="$(realpath totalspineseg)"  # Path to this repository, default: ./totalspineseg
export TOTALSPINESEG_DATA="$(realpath data)"      # Path to the data folder, default: ./data
export TOTALSPINESEG_JOBS=16                      # Number of CPU cores, default: all available cores
export TOTALSPINESEG_JOBSNN=8                     # Number of nnUNet workers, default: min(JOBS, RAM in GB / 8)
```

## Scripts

### `download_datasets.sh`

Clones the training datasets into `$TOTALSPINESEG_DATA/bids` at fixed commits and downloads the images and labels listed in `totalspineseg/resources/data/training_data.json` with `git annex get`:
> Make sure you have access to the internal datasets before running this script.
- `git@data.neuro.polymtl.ca:datasets/whole-spine.git` (internal)
- `git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git` (internal)
- [data-multi-subject](https://github.com/spine-generic/data-multi-subject)

```bash
bash "$TOTALSPINESEG"/scripts/download_datasets.sh
```

### `prepare_datasets.sh`

Converts the BIDS data into the nnUNet raw format under `$TOTALSPINESEG_DATA/nnUNet/raw`: images are reoriented to canonical space and resampled to 1x1x1 mm, and labels are mapped for each step.
- `Dataset101_TotalSpineSeg_step1`: step 1 model
- `Dataset102_TotalSpineSeg_step2`: step 2 model

```bash
bash "$TOTALSPINESEG"/scripts/prepare_datasets.sh [DATASET]
```
- `DATASET`: `101`, `102` or `all` (default: `101` and `102`)

### `train.sh`

For each dataset: extracts the fingerprint, plans and preprocesses (skipped if already done), trains the model, evaluates it on the test set and exports it as a zip in `$TOTALSPINESEG_DATA/nnUNet/exports`. The custom trainer is copied into the installed `nnunetv2` package before training, and the augmentation parameters are read from `totalspineseg/models/transforms_gpu.json`.

```bash
bash "$TOTALSPINESEG"/scripts/train.sh [DATASET] [FOLD] [TRAINER] [PLANNER] [PLANS]
```
- `DATASET`: `101`, `102` or `all` (default: `101` and `102`)
- `FOLD`: nnUNet fold (default: `0`)
- `TRAINER`: nnUNet trainer (default: `nnUNetTrainerDAExtGPU`)
- `PLANNER`: nnUNet planner (default: `nnUNetPlannerResEncL`)
- `PLANS`: nnUNet plans name (default: `nnUNetPlans`)

> Note: The custom trainer only changes the training (data augmentation). At inference, TotalSpineSeg always uses the default `nnUNetTrainer` to rebuild the network, so `auglab` is not needed to run a trained model.

### `active_learning/`

Scripts and steps used to improve the ground truth segmentations (SPINEPS + canal segmentation + TotalSpineSeg). See [active_learning/README.md](active_learning/README.md).

