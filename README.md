# TotalSpineSeg

TotalSegMRI is a tool for automatic instance segmentation and labelling of all vertebrae and intervertebral discs (IVDs), spinal cord, and spinal canal. We follow [TotalSegmentator classes](https://github.com/wasserth/TotalSegmentator/tree/v1.5.7#class-details) with an additional class for IVDs, spinal cord and spinal canal (See list of class [here](#list-of-class)). The model is based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet) as the backbone for training and inference.

If you use this model, please cite our work:
> Warszawer Y, Molinier N, Valošek J, Shirbint E, Benveniste PL, Achiron A, Eshaghi A and Cohen-Adad J. _Fully Automatic Vertebrae and Spinal Cord Segmentation Using a Hybrid Approach Combining nnU-Net and Iterative Algorithm_.	Proceedings of the 32th Annual Meeting of ISMRM. 2024

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [List of class](#list-of-class)

![Thumbnail](https://github.com/neuropoly/totalsegmentator-mri/assets/36595323/c7a4a951-fcb9-43a2-8c9c-9fafa33e4d67)

## Model description

TotalSegMRI uses a hybrid approach that integrates nnU-Net with an iterative algorithm for instance segmentation and labeling of vertebrae, intervertebral discs (IVDs), spinal cord, and spinal canal. The process involves two main steps:

1. The first model (Dataset101) was trained using a single input channel (image) to identify 8 classes in total (Figure 1A):

   - 4 main classes: spinal cord, spinal canal, IVDs, and vertebrae
   - 4 specific IVDs: C2-C3, C7-T1, T12-L1, and L5-S, representing key anatomical landmarks along the spine

1. The output segmentation was processed using an algorithm that distinguished odd and even IVDs based on the C2-C3, C7-T1, T12-L1, and L5-S IVD labels produced by the model (Figure 1B).

1. A second nnU-Net model was trained using two input channels: MRI image and odd IVDs extracted from the first step. This model outputs 14 classes in total (Figure 1C):

   - 6 main classes: spinal cord, spinal canal, odd IVDs, even IVDs, odd vertebrae, and even vertebrae.
   - 4 specific IVDs: C2-C3, C7-T1, T12-L1, and L5-S.
   - 4 specific vertebrae: C2, T1, T12, and Sacrum.

1. Finally, the model's output was processed to assign an individual label value to each vertebra and IVD in the final segmentation mask (Figure 1D).

For comparison, we also trained a single model (Dataset103) that output all the final labels in a single step.

![Figure 1](https://github.com/neuropoly/totalsegmentator-mri/assets/36595323/3958cbc6-a059-4ccf-b3b1-02dbc3a4a62d)

**Figure 1**: Illustration of the hybrid method for automatic segmentation of the spine and spinal cord structures. T1w image (A) is used to train model 1, which outputs 8 classes (B). These output labels are processed to extract odd IVDs (C). The T1w and odd IVDs are used as two input channels to train model 2, which outputs 16 classes (D). These output labels are processed to extract individual IVDs and vertebrae (E).

## Installation

1. Open Terminal in a directory you want to work on.

1. Create and activate Virtual Environment (Highly recommended):
    ```
    python -m venv venv
    source venv/bin/activate
    ```

1. Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website.

1. Clone and install this repository:
    ```
    git clone https://github.com/neuropoly/totalsegmentator-mri.git
    python -m pip install -e totalsegmentator-mri
    ```

1. Install requirements:
    ```
    python -m pip install -r totalsegmentator-mri/requirements.txt
    ```

## Training

1. Ensure training dependencies are installed:
    ```
    apt-get install git git-annex jq -y
    ```

1. Get the required datasets into `data/bids` (make sure you have access to `git@data.neuro.polymtl.ca:datasets/whole-spine.git` and `git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git`):
    ```
    bash totalsegmentator-mri/scripts/get_datasets.sh
    ```

1. Temporary!!! (until all labels will be pushed into the repositories): Extract [labels_iso_bids_0524.zip](https://github.com/neuropoly/totalsegmentator-mri/releases/download/labels/labels_iso_bids_0524.zip) and merge the `bids` folder (containing the labels) into `data/bids`.

1. Prepare datasets in nnUNetv2 structure into `data/nnUnet`:
    ```
    bash totalsegmentator-mri/scripts/prepare_nnunet_datasets.sh
    ```

1. Train the model. By default, this will train all datasets using fold 0. You can specify DATASET_ID (101, 102, or 103) and optionally a fold (only if DATASET_ID is specified, can be one of: 0, 1, 2, 3, 4, 5 or `all`):
    ```
    bash totalsegmentator-mri/scripts/train_nnunet.sh [DATASET_ID [FOLD]]
    ```

## Inference
Run the model. If you didn't train the model yourself, you should download models zip from release into `data/nnUNet/exports` (Without extracting, you can run `mkdir -p data/nnUNet/exports` before). This will process all .nii.gz files in the INPUT_FOLDER and save the results in the OUTPUT_FOLDER:
```
bash totalsegmentator-mri/scripts/inference_nnunet.sh INPUT_FOLDER OUTPUT_FOLDER
```

## List of class

|Label|Name|
|:-----|:-----|
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
