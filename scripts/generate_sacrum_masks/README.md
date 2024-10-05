# Generate sacrum masks guides

This file provides the different steps that were carried out to generate sacrum masks for the MRI totalspineseg project.

More information about the context and problems can be found in this [issue](https://github.com/neuropoly/totalspineseg/issues/18).

The main idea was to use open-source datasets with "easy to make" sacrum masks on MRI (T1w and T2w) scans to train a [nnUNetV2](https://github.com/MIC-DKFZ/nnUNet) model that will be able to segment sacrums on the whole-spine dataset and the spider dataset.

# Training

If you want to retrain the model, you can follow these steps.

## I - Dowload the training datasets

To generate the sacrum masks, 3 open-source datasets were used:

| [GoldAtlas](https://zenodo.org/records/583096) | [SynthRAD2023](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16529) | [MRSpineSeg](https://paperswithcode.com/dataset/mrspineseg-challenge) |
| :---: | :---: | :---: |
| <img width="780" alt="Screenshot 2024-01-02 at 5 10 39 AM" src="https://github.com/neuropoly/totalsegmentator-mri/assets/68945192/a324ba05-1118-4eb3-bd4f-f9aabd077477">  | <img width="628" alt="Screenshot 2024-01-02 at 5 10 53 AM" src="https://github.com/neuropoly/totalsegmentator-mri/assets/68945192/10ddd780-ec42-4540-a091-19d2b2dc3e53"> | <img width="671" alt="Screenshot 2024-01-02 at 5 11 19 AM" src="https://github.com/neuropoly/totalsegmentator-mri/assets/68945192/a0069483-ad59-48bd-9c3e-a436888a39d7"> |

> These datasets were chosen because they had these sacrum masks available or because they had co-registered MRI and CT images that allowed us to rely on the [CT total segmentator network](https://github.com/wasserth/TotalSegmentator) to generate these labels.

These datasets were BIDSified and stored on our internal servers:
- SynthRAD2023 (Internal access: `git@data.neuro.polymtl.ca:datasets/synthrad-challenge-2023.git`)
- MRSpineSeg (Internal access: `git@data.neuro.polymtl.ca:datasets/mrspineseg-challenge-2021.git`)
- GoldAtlas (Internal access: `git@data.neuro.polymtl.ca:datasets/goldatlas.git`)

## II - Register CT labels to MRI

> For this step, the https://github.com/spinalcordtoolbox/spinalcordtoolbox (SCT) was used.

As specified before, some sacrum masks were generated using the [CT total segmentator network](https://github.com/wasserth/TotalSegmentator) but due to slightly different image shape between MRI (T1w and T2w) and CT scans (see [issue](https://github.com/neuropoly/totalspineseg/issues/18)), CT segmentations were registered to MRI space. To do that, the script `totalspineseg/utils/register_CT_seg_to_MR.py` was used on the three datasets.

> Registration was also performed on the dataset `MRSpineSeg` due to slightly different q-form and s-form between segmentations and images.

```bash
python "$TOTALSPINESEG/totalspineseg/utils/register_CT_to_MR.py" --path-img <PATH-TO-BIDS-FOLDER>
```

## III - Generate a config file to select the data for training

To select the data used for training, a [config file](https://github.com/spinalcordtoolbox/disc-labeling-hourglass/issues/25#issuecomment-1695818382) was used. 

First fetch the paths to all the sacrum masks that will be used for TRAINING/VALIDATION/TESTING. The datasets should be stored inside the same parent folder

> Run this following command in the parent folder folder of the datasets.

```bash
find ~+ -type f -name *_seg.nii.gz | grep -v CT | sort > train_sacrum.txt
```

Then run this command to generate the JSON config file.

```bash
python "$TOTALSPINESEG/totalspineseg/data_management/init_data_config.py" --txt train_sacrum.txt --type LABEL --split-validation SPLIT_VAL --split-test SPLIT_TEST
```

With `SPLIT_VAL` the fraction of the data used for validation and `SPLIT_TEST` the fraction of the data used for testing.

Finally, to organize your data according to nnUNetV2 format, run this last command.

```bash
export nnUNet_raw="$TOTALSPINESEG_DATA"/nnUNet/raw
python "$TOTALSPINESEG/totalspineseg/data_management/convert_config_to_nnunet.py" --config train_sacrum.json --path-out "$nnUNet_raw" -dnum 300
```

## IV - Train with nnUNetV2

> Regarding nnUNetV2 installation and general usage, please check https://github.com/ivadomed/utilities/blob/main/quick_start_guides/nnU-Net_quick_start_guide.md

Now that your data is ready, you can run nnUNetV2 preprocessing

```bash
export nnUNet_preprocessed="$TOTALSPINESEG_DATA"/nnUNet/preprocessed
export nnUNet_results="$TOTALSPINESEG_DATA"/nnUNet/results/sacrum
nnUNetv2_plan_and_preprocess -d 300 --verify_dataset_integrity -c 3d_fullres
```

Then train using this command

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> nnUNetv2_train 300 3d_fullres 0
```

# Inference on whole-spine and spider dataset

To run nnUNetV2's inference, keep the largest component and store the data according to BIDS standard you must run:

> Before running you must download the datasets `whole-spine` and `spider-challenge-2023` and update the variable `DATASETS_PATH` inside the config file `totalspineseg/resources/configs/test_sacrum.json`. 
> This last path corresponds to the parent folder of the two datasets `whole-spine` and `spider-challenge-2023`.

```bash
bash "$TOTALSPINESEG/scripts/generate_sacrum_masks/generate_sacrum.sh"
```





