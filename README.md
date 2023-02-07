# totalsegmentator-mri
Code for the TotalSegmentator MRI project.

## Steps to install

1. Clone this repository
    ```
    git clone https://github.com/neuropoly/totalsegmentator-mri.git
    ```

1. Clone SynthSeg repository
    ```
    git clone https://github.com/BBillot/SynthSeg.git
    ```

1. Download [this google folder](https://drive.google.com/drive/folders/11F8q3jhZR0KfHhBpyKygXMo-alTDbp0U?usp=sharing) (the TotalSegmentator example image was downloaded from [here](https://zenodo.org/record/6802614)).

1. Create Virtual Environment (Please make sure you're using python 3.8 !!!)
    ```
    python -m venv venv
    ```

1. Add SynthSeg to Virtual Environment (If not using bash change '$(pwd)' to the working directory):
    ```
    echo "$(pwd)/SynthSeg" > venv/lib/python3.8/site-packages/SynthSeg.pth
    ```

1. Activate Virtual Environment
    ```
    source venv/bin/activate
    ```

1. Install requirements:
    ```
    pip install -r SynthSeg/requirements_python3.8.txt
    ```

## To run scripsts

`resources/labels.json` - Contain mapping of each mask to unique number.
`resources/classes.json` - Contain mapping of each mask to class of masks with similar statistics.

### Option 1 - Run script for all TotalSegmentator labels

1. Combine all MPRAGE 'blob' masks for each subject into a single segmentation file:
    ```
    python totalsegmentator-mri/scripts/combine_masks.py -d TotalSegmentatorMRI_SynthSeg/data/derivatives/manual_masks -o TotalSegmentatorMRI_SynthSeg/output/MP-RAGE_Masks_Combined -m totalsegmentator-mri/resources/labels.json
    ```

1. Calculate signal statistics (mean + std) for each masks (group masks into classes of similar statistics):
    ```
    python totalsegmentator-mri/scripts/build_intensity_stats.py -d TotalSegmentatorMRI_SynthSeg/data -s TotalSegmentatorMRI_SynthSeg/output/MP-RAGE_Masks_Combined -o TotalSegmentatorMRI_SynthSeg/output/MP-RAGE_priors -m totalsegmentator-mri/resources/labels.json -c totalsegmentator-mri/resources/classes.json
    ```

1. Combine all TotalSegmentator masks for each subject into a single segmentation file:
    ```
    python totalsegmentator-mri/scripts/combine_masks.py -d TotalSegmentatorMRI_SynthSeg/Totalsegmentator_dataset -o TotalSegmentatorMRI_SynthSeg/output/TotalSegmentator_Masks_Combined -m totalsegmentator-mri/resources/labels.json --subject-prefix s --subject-subdir segmentations --seg-suffix _ct_seg
    ```

1. Create a synthetic image using TotalSegmentator segmentation and the calculated MPRAGE signal statistics:
    ```
    python totalsegmentator-mri/scripts/generate_image.py -s /content/TotalSegmentatorMRI_SynthSeg/output/TotalSegmentator_Masks_Combined/sub-0287/anat/sub-0287_ct_seg.nii.gz -p TotalSegmentatorMRI_SynthSeg/output/MP-RAGE_priors -o TotalSegmentatorMRI_SynthSeg/output/MP-RAGE_Synthetic/test1
    ```

### Option 2 - Run script with TotalSegmentator labels reduced to 15 labels (using `resources/classes.json`)

1. Combine all MPRAGE 'blob' masks for each subject into a single segmentation file:
    ```
    python totalsegmentator-mri/scripts/combine_masks.py -d TotalSegmentatorMRI_SynthSeg/data/derivatives/manual_masks -o TotalSegmentatorMRI_SynthSeg/output/MP-RAGE_Masks_Combined -m totalsegmentator-mri/resources/classes.json
    ```

1. Calculate signal statistics (mean + std) for each masks:
    ```
    python totalsegmentator-mri/scripts/build_intensity_stats.py -d TotalSegmentatorMRI_SynthSeg/data -s TotalSegmentatorMRI_SynthSeg/output/MP-RAGE_Masks_Combined -o TotalSegmentatorMRI_SynthSeg/output/MP-RAGE_priors -m totalsegmentator-mri/resources/classes.json
    ```

1. Combine all TotalSegmentator masks for each subject into a single segmentation file:
    ```
    python totalsegmentator-mri/scripts/combine_masks.py -d TotalSegmentatorMRI_SynthSeg/Totalsegmentator_dataset -o TotalSegmentatorMRI_SynthSeg/output/TotalSegmentator_Masks_Combined -m totalsegmentator-mri/resources/classes.json --subject-prefix s --subject-subdir segmentations --seg-suffix _ct_seg
    ```

1. Create a synthetic image using TotalSegmentator segmentation and the calculated MPRAGE signal statistics:
    ```
    python totalsegmentator-mri/scripts/generate_image.py -s /content/TotalSegmentatorMRI_SynthSeg/output/TotalSegmentator_Masks_Combined/sub-0287/anat/sub-0287_ct_seg.nii.gz -p TotalSegmentatorMRI_SynthSeg/output/MP-RAGE_priors -o TotalSegmentatorMRI_SynthSeg/output/MP-RAGE_Synthetic/test1
    ```
## Data organization

As a starting point, a few MPRAGE data are under our private [google folder](https://drive.google.com/drive/folders/1CAkz4ZuxQjWza7GAXhXxTkKcyB9p3yME).

We will follow the BIDS structure:
```
├── derivatives
│   └── manual_masks
│       └── sub-errsm37
│           └── anat
└── sub-errsm37
    └── anat
        ├── sub-errsm37_T1w.json
        └── sub-errsm37_T1w.nii.gz
```
