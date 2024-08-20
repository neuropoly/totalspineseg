# TotalSpineSeg

TotalSpineSeg is a tool for automatic instance segmentation and labeling of all vertebrae, intervertebral discs (IVDs), spinal cord, and spinal canal in MRI images. It follows the [TotalSegmentator classes](https://github.com/wasserth/TotalSegmentator/tree/v1.5.7#class-details) with additional classes for IVDs, spinal cord, and spinal canal (see [list of classes](#list-of-classes)). The model is based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet) as the backbone for training and inference.

If you use this model, please cite our work:
> Warszawer Y, Molinier N, Valošek J, Shirbint E, Benveniste PL, Achiron A, Eshaghi A and Cohen-Adad J. _Fully Automatic Vertebrae and Spinal Cord Segmentation Using a Hybrid Approach Combining nnU-Net and Iterative Algorithm_.	Proceedings of the 32th Annual Meeting of ISMRM. 2024

Please also cite nnUNet since our work is heavily based on it:
> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

## Table of Contents

- [Model Description](#model-description)
- [Datasets](#datasets)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [List of Classes](#list-of-classes)

![Thumbnail](https://github.com/neuropoly/totalspineseg/assets/36595323/550a159f-de6c-4817-abee-d98d9ce0c106)

## Model Description

TotalSpineSeg uses a hybrid approach that integrates nnU-Net with an iterative algorithm for instance segmentation and labeling of vertebrae, intervertebral discs (IVDs), spinal cord, and spinal canal. The process involves two main steps:

**Step 1**: An nnUnet model (`Dataset101`) was trained to identify 8 classes in total (Figure 1A). This includes 4 main classes: spinal cord, spinal canal, IVDs, and vertebrae. Additionally, it identifies 4 specific IVDs: C2-C3, C7-T1, T12-L1, and L5-S, which represent key anatomical landmarks along the spine. The output segmentation was then processed using an iterative algorithm. This algorithm extracts odd IVDs segmentation based on the C2-C3, C7-T1, T12-L1, and L5-S IVD labels produced by the model (Figure 1B).

**Step 2:** A second nnUNet model (`Dataset102`) was trained to identify 14 classes in total (Figure 1C). This includes 6 main classes: spinal cord, spinal canal, odd IVDs, even IVDs, odd vertebrae, and even vertebrae. Additionally, it identifies 4 specific IVDs: C2-C3, C7-T1, T12-L1, and L5-S, and 4 specific vertebrae: C2, T1, T12, and Sacrum. This model uses two input channels: the MRI image and the odd IVDs extracted from the first step. The output segmentation was then processed using an algorithm that assigns an individual label value to each vertebra and IVD in the final segmentation mask (Figure 1D).

For comparison, we also trained a single model (`Dataset103`) that outputs individual label values for each vertebra and IVD in a single step.

![Figure 1](https://github.com/neuropoly/totalspineseg/assets/36595323/84fae79f-442b-48c3-bcdb-ce4ea857ac59)

**Figure 1**: Illustration of the hybrid method for automatic segmentation of the spine and spinal cord structures. T1w image (A) is used to train model 1, which outputs 8 classes (B). These output labels are processed to extract odd IVDs (C). The T1w and odd IVDs are used as two input channels to train model 2, which outputs 14 classes (D). These output labels are processed to extract individual IVDs and vertebrae (E).

## Datasets

Our model is trained on those datasets:
- Private whole-spine dataset (Internal access: `git@data.neuro.polymtl.ca:datasets/whole-spine.git`).
- [SPIDER](https://doi.org/10.5281/zenodo.10159290) project dataset (Internal access: `git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git`)
- [Spine Generic Project](https://github.com/spine-generic), including single and multi subject datasets (Public access: `git@github.com:spine-generic/data-single-subject.git` and `git@github.com:spine-generic/data-multi-subject.git`).

## Dependencies

- Bash terminal
- [Python](https://www.python.org/) >= 3.9, with pip >= 23 and setuptools >= 67.

## Installation

1. Open a `bash` terminal in the directory where you want to work.

1. Create and activate a virtual environment (highly recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

1. Clone and install this repository:
   ```bash
   git clone https://github.com/neuropoly/totalspineseg.git
   python -m pip install -e totalspineseg
   ```

1. For CUDA GPU support, install **PyTorch** following the instructions on their [website](https://pytorch.org/). Be sure to add the `--upgrade` flag to your installation command to replace any existing PyTorch installation.
   Example:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
     ```

1. Set the path to TotalSpineSeg and data folders in the virtual environment:
   ```bash
   export TOTALSPINESEG="$(realpath totalspineseg)" && echo "export TOTALSPINESEG=\"$TOTALSPINESEG\"" >> venv/bin/activate
   export TOTALSPINESEG_DATA="$(realpath data)" && echo "export TOTALSPINESEG_DATA=\"$TOTALSPINESEG_DATA\"" >> venv/bin/activate
   ```

## Training

To train the TotalSpineSeg model, you will need the following hardware specifications:
- Approximately 3.5TB of available disk space (for training with data augmentation)
- RAM capacity of at least 32GB
- CUDA GPU with at least 8GB of VRAM

Please ensure that your system meets these requirements before proceeding with the training process.

1. Make sure that the `bash` terminal is opened with the virtual environment (if used) activated (using `source <path to installation directory>/venv/bin/activate`).

1. Ensure training dependencies are installed:
   ```bash
   apt-get install git git-annex jq -y
   ```

1. Get the required datasets into `$TOTALSPINESEG_DATA/bids` (make sure you have access to the specified repositories):
   ```bash
   bash "$TOTALSPINESEG"/scripts/download_datasets.sh
   ```

1. Temporary step (until all labels are pushed into the repositories) - Download labels into `$TOTALSPINESEG_DATA/bids`:
   ```bash
   curl -L -O https://github.com/neuropoly/totalspineseg/releases/download/labels/labels_iso_bids_0524.zip
   unzip -qo labels_iso_bids_0524.zip -d "$TOTALSPINESEG_DATA"
   rm labels_iso_bids_0524.zip
   ```

1. Prepare datasets in nnUNetv2 structure into `$TOTALSPINESEG_DATA/nnUnet`:
   ```bash
   bash "$TOTALSPINESEG"/scripts/prepare_datasets.sh [DATASET_ID] [-noaug]
   ```

   The script optionally accepts `DATASET_ID` as the first positional argument to specify the dataset to prepare. It can be either 101, 102, 103, or all. If `all` is specified, it will prepare all datasets (101, 102, 103). By default, it will prepare datasets 101 and 102.

   Additionally, you can use the `-noaug` parameter to prepare the datasets without data augmentations.

1. Train the model:
   ```bash
   bash "$TOTALSPINESEG"/scripts/train.sh [DATASET_ID [FOLD]]
   ```

   The script optionally accepts `DATASET_ID` as the first positional argument to specify the dataset to train. It can be either 101, 102, 103, or all. If `all` is specified, it will train all datasets (101, 102, 103). By default, it will train datasets 101 and 102.

   Additionally, you can specify `FOLD` as the second positional argument to specify the fold. It can be either 0, 1, 2, 3, 4, 5 or all. By default, it will train with fold 0.

## Inference

1. Make sure that the `bash` terminal is opened with the virtual environment (if used) activated (using `source <path to installation directory>/venv/bin/activate`).

1. If you didn't train the model yourself, you should download the model zip file from the release into `$TOTALSPINESEG_DATA/nnUNet/exports` (do not extract the zip files). You can run `mkdir -p "$TOTALSPINESEG_DATA"/nnUNet/exports` to create the folder.
1. Run the model on a folder containing the images in .nii.gz format:
   ```bash
   bash "$TOTALSPINESEG"/scripts/inference.sh INPUT_FOLDER OUTPUT_FOLDER [-step1]
   ```

   This will process all .nii.gz files in the INPUT_FOLDER and save the results in the OUTPUT_FOLDER.

   Additionally, you can use the `-step1` parameter to run only the step 1 model, which outputs a single label for all vertebrae, including the sacrum.

## List of Classes

| Label | Name |
|:------|:-----|
| 18 | vertebrae_L5 |
| 19 | vertebrae_L4 |
| 20 | vertebrae_L3 |
| 21 | vertebrae_L2 |
| 22 | vertebrae_L1 |
| 23 | vertebrae_T12 |
| 24 | vertebrae_T11 |
| 25 | vertebrae_T10 |
| 26 | vertebrae_T9 |
| 27 | vertebrae_T8 |
| 28 | vertebrae_T7 |
| 29 | vertebrae_T6 |
| 30 | vertebrae_T5 |
| 31 | vertebrae_T4 |
| 32 | vertebrae_T3 |
| 33 | vertebrae_T2 |
| 34 | vertebrae_T1 |
| 35 | vertebrae_C7 |
| 36 | vertebrae_C6 |
| 37 | vertebrae_C5 |
| 38 | vertebrae_C4 |
| 39 | vertebrae_C3 |
| 40 | vertebrae_C2 |
| 41 | vertebrae_C1 |
| 92 | sacrum |
| 200 | spinal_cord |
| 201 | spinal_canal |
| 202 | disc_L5_S |
| 203 | disc_L4_L5 |
| 204 | disc_L3_L4 |
| 205 | disc_L2_L3 |
| 206 | disc_L1_L2 |
| 207 | disc_T12_L1 |
| 208 | disc_T11_T12 |
| 209 | disc_T10_T11 |
| 210 | disc_T9_T10 |
| 211 | disc_T8_T9 |
| 212 | disc_T7_T8 |
| 213 | disc_T6_T7 |
| 214 | disc_T5_T6 |
| 215 | disc_T4_T5 |
| 216 | disc_T3_T4 |
| 217 | disc_T2_T3 |
| 218 | disc_T1_T2 |
| 219 | disc_C7_T1 |
| 220 | disc_C6_C7 |
| 221 | disc_C5_C6 |
| 222 | disc_C4_C5 |
| 223 | disc_C3_C4 |
| 224 | disc_C2_C3 |
