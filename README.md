# TotalSegMRI

Tool for automatic segmentation and labelling of all vertebrae and intervertebral discs (IVDs), spinal cord, and spinal canal. We follow [TotalSegmentator classes](https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file#class-details) with an additional class for IVDs, spinal cord and spinal canal (See list of class [here](#list-of-class)). We used [nnUNet](https://github.com/MIC-DKFZ/nnUNet) as our backbone for model training and inference.

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Model description](#model-description)
- [Training](#training)
- [Inference](#inference)
- [List of class](#list-of-class)

![Thumbnail](https://github.com/neuropoly/totalsegmentator-mri/assets/36595323/c7a4a951-fcb9-43a2-8c9c-9fafa33e4d67)

## Dependencies

- [Spinal Cord Toolbox (SCT)](https://github.com/neuropoly/spinalcordtoolbox)

## Installation

1. Open Terminal in a directory you want to work on.

1. Create and activate Virtual Environment (Highly recommanded):
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

## Model description
A hybrid approach integrating nnU-Net with an iterative algorithm for segmenting vertebrae, IVDs, spinal cord, and spinal canal. To tackle the challenge of having many classes and class imbalance, we developed a two-step training process. A first model (Dataset206) was trained (single input channel: image) to identify 4 classes (IVDs, vertebrae, spinal cord and spinal canal) as well as specific IVDs (C2-C3, C7-T1 and L5-S1) representing key anatomical landmarks along the spine, so 7 classes in total (Figure 1A). The output segmentation was processed using an algorithm that distinguished odd and even IVDs based on the C2-C3, C7-T1 and L5-S1 IVD labels output by the model (Figure 1B). Then, a second nnU-Net model (Dataset210) was trained (two input channels: 1=image, 2=odd IVDs), to output 12 classes (Figure 1C). Finally, the output of the model was processed in order to assign an individual label value to each vertebrae and IVD in the final segmentation mask (Figure 1D).

![Figure 1](https://github.com/neuropoly/totalsegmentator-mri/assets/36595323/3958cbc6-a059-4ccf-b3b1-02dbc3a4a62d)

**Figure 1**: Illustration of the hybrid method for automatic segmentation of the spine and spinal cord structures. T1w image (A) is used to train model 1, which outputs 7 classes (B). These output labels are processed to extract odd IVDs (C). The T1w and odd IVDs are used as two input channels to train model 2, which outputs 12 classes (D). These output labels are processed to extract individual IVDs and vertebrae (E).

## Training

1. Download the corresponding content from [SPIDER dataset](https://doi.org/10.5281/zenodo.10159290) into 'data/raw/spider/images' and 'data/raw/spider/masks' (you can use `mkdir -p data/raw/spider` to create the folder first).

1. Make sure `git` and `git-annex` are installed (You can install with `sudo apt-get install git-annex -y`).

1. Extract [data-multi-subject_PAM50_seg.zip](https://drive.google.com/file/d/1Sq38xLHnVxhLr0s1j27ywbeshNUjo3IP) into 'data/bids/data-multi-subject'.

1. Extract [data-single-subject_PAM50_seg.zip](https://drive.google.com/file/d/1YvuFHL8GDJ5SXlMLORWDjR5SNkDL6TUU) into 'data/bids/data-single-subject'.

1. Extract [whole-spine.zip](https://drive.google.com/file/d/143i0ODmeqohpc4vu5Aa5lnv8LLEyOU0F) (private dataset) into 'data/bids/whole-spine'.

1. Get the required datasets from [Spine Generic Project](https://github.com/spine-generic/):
    ```
    source totalsegmentator-mri/run/get_spine_generic_datasets.sh
    ```

1. Prepares SPIDER datasets in [BIDS](https://bids.neuroimaging.io/) structure:
    ```
    source totalsegmentator-mri/run/prepare_spider_bids_datasets.sh
    ```

1. Prepares datasets in nnUNetv2 structure:
    ```
    source totalsegmentator-mri/run/prepare_nnunet_datasets.sh
    ```

1. Train the model:
    ```
    source totalsegmentator-mri/run/train_nnunet.sh
    ```

## Inference
Run the model on a folder containing the images in .nii.gz format (Make sure to train the model or extract the trained `results` into `data/nnUNet/results` befor running):
```
source totalsegmentator-mri/run/inference_nnunet.sh INPUT_FOLDER OUTPUT_FOLDER
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
