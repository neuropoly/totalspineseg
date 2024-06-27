# TotalSpineSeg

TotalSpineSeg is a tool for automatic instance segmentation and labeling of all vertebrae, intervertebral discs (IVDs), spinal cord, and spinal canal in MRI images. It follows the [TotalSegmentator classes](https://github.com/wasserth/TotalSegmentator/tree/v1.5.7#class-details) with additional classes for IVDs, spinal cord, and spinal canal (see [list of classes](#list-of-classes)). The model is based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet) as the backbone for training and inference.

If you use this model, please cite our work:
> Warszawer Y, Molinier N, Valošek J, Shirbint E, Benveniste PL, Achiron A, Eshaghi A and Cohen-Adad J. _Fully Automatic Vertebrae and Spinal Cord Segmentation Using a Hybrid Approach Combining nnU-Net and Iterative Algorithm_.	Proceedings of the 32th Annual Meeting of ISMRM. 2024

- [Model Description](#model-description)
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

## Installation

1. Open a terminal in the directory where you want to work.

1. Create and activate a virtual environment (highly recommended):
   ```
   python -m venv venv
   source venv/bin/activate
   ```

1. Install [PyTorch](https://pytorch.org/) as described on their website.

1. Clone and install this repository:
   ```
   git clone https://github.com/neuropoly/totalspineseg.git
   python -m pip install -e totalspineseg
   ```

1. Set the path to TotalSpineSeg and data folders:
   ```
   export TOTALSPINESEG="$(realpath totalspineseg)"
   export TOTALSPINESEG_DATA="$(realpath data)"
   ```

## Training

1. Ensure training dependencies are installed:
   ```
   apt-get install git git-annex jq -y
   ```

1. Get the required datasets into `$TOTALSPINESEG_DATA/bids` (make sure you have access to the specified repositories):
   ```
   bash "$TOTALSPINESEG"/scripts/get_datasets.sh
   ```

1. Temporary step (until all labels are pushed into the repositories): Extract [labels_iso_bids_0524.zip](https://github.com/neuropoly/totalspineseg/releases/download/labels/labels_iso_bids_0524.zip) and merge the `bids` folder (containing the labels) into `$TOTALSPINESEG_DATA/bids`.

1. Prepare datasets in nnUNetv2 structure into `$TOTALSPINESEG_DATA/nnUnet`:
   ```
   bash "$TOTALSPINESEG"/scripts/prepare_nnunet_datasets.sh
   ```

1. Train the model. By default, this will train all datasets using fold 0. You can specify DATASET_ID (101, 102, or 103) and optionally a fold (only if DATASET_ID is specified, can be one of: 0, 1, 2, 3, 4, 5 or all):
   ```
   bash "$TOTALSPINESEG"/scripts/train_nnunet.sh [DATASET_ID [FOLD]]
   ```

## Inference

Run the model on a folder containing the images in .nii.gz format. If you didn't train the model yourself, you should download the model zip file from the release into `$TOTALSPINESEG_DATA/nnUNet/exports` (without extracting, you can run `mkdir -p "$TOTALSPINESEG_DATA"/nnUNet/exports` before):

```
bash "$TOTALSPINESEG"/scripts/inference_nnunet.sh INPUT_FOLDER OUTPUT_FOLDER
```

This will process all .nii.gz files in the INPUT_FOLDER and save the results in the OUTPUT_FOLDER.

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