# TotalSpineSeg
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13894354.svg)](https://doi.org/10.5281/zenodo.13894354)

TotalSpineSeg is a tool for automatic instance segmentation of all vertebrae, intervertebral discs (IVDs), spinal cord, and spinal canal in MRI images. It is robust to various MRI contrasts, acquisition orientations, and resolutions. The model used in TotalSpineSeg is based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) as the backbone for training and inference.

If you use this model, please cite our work:
> Warszawer Y, Molinier N, Valošek J, Shirbint E, Benveniste PL, Achiron A, Eshaghi A and Cohen-Adad J. _Fully Automatic Vertebrae and Spinal Cord Segmentation Using a Hybrid Approach Combining nnU-Net and Iterative Algorithm_.	Proceedings of the 32th Annual Meeting of ISMRM. 2024

Please also cite nnU-Net since our work is heavily based on it:
> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

![Thumbnail](https://github.com/user-attachments/assets/2c1b1ff9-daaa-479f-8d21-01a66b9c9cb4)

## Table of Contents

- [Model Description](#model-description)
- [Datasets](#datasets)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Localizer based labeling](#localizer-based-labeling)
- [Results](#results)
- [List of Classes](#list-of-classes)

## Model Description

TotalSpineSeg uses an hybrid approach that integrates nnU-Net with an iterative algorithm for instance segmentation and labeling of vertebrae, intervertebral discs (IVDs), spinal cord, and spinal canal. The process involves two main steps:

**Step 1**: An nnU-Net model (`Dataset101`) is used to identify nine classes in total. This includes four semantic classes: spinal cord, spinal canal, IVDs, and vertebrae and five landmark classes: C2-C3, C7-T1, T12-L1, and L5-S, which represent key IVDs along the spine, as well as the C1 vertebra to determine whether the MRI images cover C1 (Figure 1B). The output segmentation is then processed using an iterative algorithm to map individual IVDs, from which odd IVDs segmentation are extracted (Figure 1C).

**Step 2:** A second nnU-Net model (`Dataset102`) is then used to identify ten classes in total. This includes five semantic classes: spinal cord, spinal canal, IVDs, odd vertebrae, and even vertebrae and five landmark classes: C2-C3, C7-T1, T12-L1, and L5-S, which represent the same key IVDs along the spine, as well as the sacrum (Figure 1D). This model uses two input channels: the MRI image (Figure 1A) and the odd IVDs extracted from the first step (Figure 1C). The output segmentation is finally processed using an algorithm that assigns individual labels to each vertebra and IVD to generate the final segmentation (Figure 1F).

For comparison, we also trained a single model (`Dataset103`) that outputs individual label values for each vertebra and IVD in a single step.

![Figure 1](https://github.com/user-attachments/assets/9017fb8e-bed5-413f-a80f-b123a97f5735)

**Figure 1**: Illustration of the hybrid method for automatic segmentation of spinal structures. (A) Input MRI image. (B) Step 1 model prediction. (C) Odd IVDs extraction from the Step1 prediction. (D) Step 2 model prediction. (E) Final segmentation with individual labels for each vertebra and IVD.

## Datasets

The totalspineseg model was trained on these 3 main datasets:
- [whole-spine](https://openneuro.org/datasets/ds005616/versions/1.0.0) dataset (Internal access: `git@data.neuro.polymtl.ca:datasets/whole-spine.git`).
- [SPIDER](https://doi.org/10.5281/zenodo.10159290) project dataset (Internal access: `git@data.neuro.polymtl.ca:datasets/spider-challenge-2023.git`)
- [Spine Generic Project](https://github.com/spine-generic), including single and multi subject datasets (Public access: `git@github.com:spine-generic/data-single-subject.git` and `git@github.com:spine-generic/data-multi-subject.git`).

We used manual labels from the SPIDER dataset. For other datasets, we generated initial labels by registering MRIs to the PAM50 template using [Spinal Cord Toolbox (SCT)](https://spinalcordtoolbox.com/). We trained an initial segmentation model with these labels, applied it to the datasets, and manually corrected the outputs using [3D Slicer](https://www.slicer.org/).

Additional public datasets were used during this project to generate sacrum segmentations:
- [GoldAtlas](https://zenodo.org/records/583096) (Internal access: `git@data.neuro.polymtl.ca:datasets/goldatlas.git`)
- [SynthRAD2023](https://synthrad2023.grand-challenge.org/) (Internal access: `git@data.neuro.polymtl.ca:datasets/synthrad-challenge-2023.git`)
- [MRSpineSeg](https://paperswithcode.com/dataset/mrspineseg-challenge) (Internal access: `git@data.neuro.polymtl.ca:datasets/mrspineseg-challenge-2021.git`)

When not available, sacrum segmentations were generated using the [totalsegmentator](https://github.com/wasserth/TotalSegmentator) model. For more information, please see [this issue](https://github.com/neuropoly/totalspineseg/issues/18).

## Dependencies

- `bash` terminal
- [Python](https://www.python.org/) >= 3.10, with pip >= 23 and setuptools >= 67

## Installation

1. Open a `bash` terminal in the directory where you want to work.

2. Create the installation directory:
```bash
mkdir TotalSpineSeg
cd TotalSpineSeg
```

3. Create and activate a virtual environment using one of the following options (highly recommended):
   - venv
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   - conda env
   ```
   conda create -n myenv python=3.10
   conda activate myenv
   ```

4. Install this repository using one of the following options:
   - Git clone (for developpers)
   > **Note:** If you pull a new version from GitHub, make sure to rerun this command with the flag `--upgrade`
   ```bash
   git clone https://github.com/neuropoly/totalspineseg.git
   python3 -m pip install -e totalspineseg[nnunetv2]
   ```
   - PyPI installation (for inference only)
   ```
   python3 -m pip install totalspineseg[nnunetv2]
   ```
   - PyPI installation (with specific nnU-Net version)
   ```bash
   # Note: Use "[nnunetv2]" to stick to tested versions of nnunetv2
   python3 -m pip install totalspineseg nnunetv2==2.6.2
   ```

5. For CUDA GPU support, install **PyTorch<2.6** following the instructions on their [website](https://pytorch.org/). Be sure to add the `--upgrade` flag to your installation command to replace any existing PyTorch installation.
   Example:
```bash
python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

6. **OPTIONAL STEP:** Define a folder where weights will be stored:
> By default, weights will be stored in the package under `totalspineseg/models`
 ```bash
 mkdir data
 export TOTALSPINESEG_DATA="$(realpath data)"
 ```

## Training

To train the TotalSpineSeg model, you will need the following hardware specifications:
- Approximately 3.5TB of available disk space (for training with data augmentation)
- RAM capacity of at least 32GB
- CUDA GPU with at least 8GB of VRAM

Please ensure that your system meets these requirements before proceeding with the training process.

1. Make sure that the `bash` terminal is opened with the virtual environment activated (see [Installation](#installation)).

2. Ensure training dependencies are installed:
```bash
apt-get install git git-annex jq -y
```

3. Set the path to TotalSpineSeg and data folders in the virtual environment:
```bash
mkdir data
export TOTALSPINESEG="$(realpath totalspineseg)"
export TOTALSPINESEG_DATA="$(realpath data)"
echo "export TOTALSPINESEG=\"$TOTALSPINESEG\"" >> venv/bin/activate
echo "export TOTALSPINESEG_DATA=\"$TOTALSPINESEG_DATA\"" >> venv/bin/activate
```

4. Download the required datasets into `$TOTALSPINESEG_DATA/bids` (make sure you have access to the specified repositories):
```bash
bash "$TOTALSPINESEG"/scripts/download_datasets.sh
```

5. Temporary step (until all labels are pushed into the repositories) - Download labels into `$TOTALSPINESEG_DATA/bids`:
```bash
curl -L -O https://github.com/neuropoly/totalspineseg/releases/download/labels/labels_iso_bids_0924.zip
unzip -qo labels_iso_bids_0924.zip -d "$TOTALSPINESEG_DATA"
rm labels_iso_bids_0924.zip
```

6. Prepare datasets in nnUNetv2 structure into `$TOTALSPINESEG_DATA/nnUnet`:
```bash
bash "$TOTALSPINESEG"/scripts/prepare_datasets.sh [DATASET_ID] [-noaug]
```

   The script optionally accepts `DATASET_ID` as the first positional argument to specify the dataset to prepare. It can be either 101, 102, 103, or all. If `all` is specified, it will prepare all datasets (101, 102, 103). By default, it will prepare datasets 101 and 102.

   Additionally, you can use the `-noaug` parameter to prepare the datasets without data augmentations.

7. Train the model:
```bash
bash "$TOTALSPINESEG"/scripts/train.sh [DATASET_ID [FOLD]]
```

   The script optionally accepts `DATASET_ID` as the first positional argument to specify the dataset to train. It can be either 101, 102, 103, or all. If `all` is specified, it will train all datasets (101, 102, 103). By default, it will train datasets 101 and 102.

   Additionally, you can specify `FOLD` as the second positional argument to specify the fold. It can be either 0, 1, 2, 3, 4, 5 or all. By default, it will train with fold 0.

## Inference

1. Make sure that the `bash` terminal is opened with the virtual environment activated (see [Installation](#installation)).

2. Run the model on a folder containing niftii images (`.nii.gz` or `.nii`), or on a single niftii file:
> If you haven't trained the model, the script will automatically download the pre-trained models from the GitHub release.
```bash
totalspineseg INPUT OUTPUT_FOLDER [--step1] [--iso]
```

   This will process the images in INPUT or the single image and save the results in OUTPUT_FOLDER.

   **Important Note:** By default, the output segmentations are resampled back to the input image space. If you prefer to obtain the outputs in the model's original 1mm isotropic resolution, especially useful for visualization purposes, we strongly recommend using the `--iso` argument.

   Additionally, you can use the `--step1` parameter to run only the step 1 model, which outputs a single label for all vertebrae, including the sacrum.

   For more options, you can use the `--help` parameter:
```bash
totalspineseg --help
```

**Output Data Structure:**

```
output_folder/
├── input/                   # Preprocessed input images
├── preview/                 # Preview images for all steps
├── step1_raw/               # Raw outputs from step 1 model
├── step1_output/            # Results of iterative labeling algorithm for step 1
├── step1_cord/              # Spinal cord soft segmentations
├── step1_canal/             # Spinal canal soft segmentations
├── step1_levels/            # Single voxel in canal centerline at each IVD level
├── step2_raw/               # Raw outputs from step 2 model
└── step2_output/            # Results of iterative labeling algorithm for step 2 (final output)
```

**Important Note:** While TotalSpineSeg provides spinal cord segmentation, it is not intended to replace validated methods for cross-sectional area (CSA) analysis. The spinal cord segmentation from TotalSpineSeg has not been validated for CSA measurements, nor has it been tested on cases involving spinal cord compressions, MS lesions, or other spinal cord abnormalities. For accurate CSA analysis, we strongly recommend using the validated algorithms available in the [Spinal Cord Toolbox](https://spinalcordtoolbox.com/user_section/tutorials/segmentation.html).

Key points:
- All segmentations in NIfTI format (`.nii.gz`)
- Preview images in JPEG format
- step1_levels: single voxel in canal centerline at each IVD level, numbered from C1 (1 above C1, 2 above C2, etc.)
- step2_output: final labeled vertebrae, discs, cord, and canal

## Localizer based labeling

TotalSpineSeg supports using localizer images to improve the labeling process, particularly useful for images with different fields of view (FOV) where landmarks like C1 and sacrum may not be visible. It uses localizer information to accurately label vertebrae and discs in the main image.

![Localizer](https://github.com/user-attachments/assets/c00ec3b6-2f04-4bbc-be08-b7ae1373b6ae)

Example of directory structure:

```
.
├── images/
│   ├── sub-01_T2w.nii.gz
│   └── sub-02_T2w.nii.gz
└── localizers/
    ├── sub-01_T1w.nii.gz
    └── sub-02_T1w.nii.gz
```

In this example, main images are placed in the `images` folder and corresponding localizer images in the `localizers` folder.

To use localizer-based labeling:

```bash
# Process localizer images. We recommend using the --iso flag for the localizer to ensure consistent resolution.
totalspineseg localizers localizers_output --iso

# Run model on main images using localizer output
totalspineseg images output --loc localizers_output/step2_output --suffix _T2w --loc-suffix _T1w
```

- `--loc`: Specifies the path to the localizer output
- `--suffix`: Suffix for the main images (e.g., "_T2w")
- `--loc-suffix`: Suffix for the localizer images (e.g., "_T1w")

Note: If the localizer and main image files have the same names, you can omit the `--suffix` and `--loc-suffix` arguments.

## Results

TotalSpineSeg demonstrates robust performance across a wide range of imaging parameters. Here are some examples of the model output:

![Model Output Preview](https://github.com/user-attachments/assets/b4c85ce8-c59b-4ab1-b02a-37638c9ac375)

The examples shown above include segmentation results on various contrasts (T1w, T2w, STIR, MTS, T2star, and even CT images), acquisition orientations (sagittal, axial), and resolutions.

## List of Classes

| Label | Name |
|:------|:-----|
| 1 | spinal_cord |
| 2 | spinal_canal |
| 11 | vertebrae_C1 |
| 12 | vertebrae_C2 |
| 13 | vertebrae_C3 |
| 14 | vertebrae_C4 |
| 15 | vertebrae_C5 |
| 16 | vertebrae_C6 |
| 17 | vertebrae_C7 |
| 21 | vertebrae_T1 |
| 22 | vertebrae_T2 |
| 23 | vertebrae_T3 |
| 24 | vertebrae_T4 |
| 25 | vertebrae_T5 |
| 26 | vertebrae_T6 |
| 27 | vertebrae_T7 |
| 28 | vertebrae_T8 |
| 29 | vertebrae_T9 |
| 30 | vertebrae_T10 |
| 31 | vertebrae_T11 |
| 32 | vertebrae_T12 |
| 41 | vertebrae_L1 |
| 42 | vertebrae_L2 |
| 43 | vertebrae_L3 |
| 44 | vertebrae_L4 |
| 45 | vertebrae_L5 |
| 50 | sacrum |
| 63 | disc_C2_C3 |
| 64 | disc_C3_C4 |
| 65 | disc_C4_C5 |
| 66 | disc_C5_C6 |
| 67 | disc_C6_C7 |
| 71 | disc_C7_T1 |
| 72 | disc_T1_T2 |
| 73 | disc_T2_T3 |
| 74 | disc_T3_T4 |
| 75 | disc_T4_T5 |
| 76 | disc_T5_T6 |
| 77 | disc_T6_T7 |
| 78 | disc_T7_T8 |
| 79 | disc_T8_T9 |
| 80 | disc_T9_T10 |
| 81 | disc_T10_T11 |
| 82 | disc_T11_T12 |
| 91 | disc_T12_L1 |
| 92 | disc_L1_L2 |
| 93 | disc_L2_L3 |
| 94 | disc_L3_L4 |
| 95 | disc_L4_L5 |
| 100 | disc_L5_S |

## How to cite us

If you find this work and/or code useful for your research, please cite our paper:

```
@article{warszawer2025totalspineseg,
   title={TotalSpineSeg: Robust Spine Segmentation with Landmark-Based Labeling in MRI},
   author={Warszawer, Yehuda and Molinier, Nathan and Valosek, Jan and Benveniste, Pierre-Louis and Bédard, Sandrine and Shirbint, Emanuel and Mohamed, Feroze and Tsagkas, Charidimos and Kolind, Shannon and Lynd, Larry and Oh, Jiwon and Prat, Alexandre and Tam, Roger and Traboulsee, Anthony and Patten, Scott and Lee, Lisa Eunyoung and Achiron, Anat and Cohen-Adad, Julien},
   year={2025},
   journal={ResearchGate preprint},
   url={https://www.researchgate.net/publication/389881289_TotalSpineSeg_Robust_Spine_Segmentation_with_Landmark-Based_Labeling_in_MRI}
}
```
