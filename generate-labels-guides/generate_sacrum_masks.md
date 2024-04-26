# Generate sacrum masks guides

This file provides the different steps that were carried out to generate sacrum masks for the MRI total segmentator project.

More information about the context and problems can be found in this [issue](https://github.com/neuropoly/totalsegmentator-mri/issues/18).

The main idea was to use open-source datasets with "easy to make" sacrum masks on MRI (T1w and T2w) scans to train a [nnUNetV2](https://github.com/MIC-DKFZ/nnUNet) model that will be able to segment sacrums on the whole-spine dataset and the spider dataset.

If you just want the labels they were already git-annexed on our private servers.

## I - Dowload the training datasets

To generate the sacrum masks, 3 open-source datasets were used:

| GoldAtlas | SynthRAD2023 | MRSpineSeg |
| :---: | :---: | :---: |
| <img width="780" alt="Screenshot 2024-01-02 at 5 10 39 AM" src="https://github.com/neuropoly/totalsegmentator-mri/assets/68945192/a324ba05-1118-4eb3-bd4f-f9aabd077477">  | <img width="628" alt="Screenshot 2024-01-02 at 5 10 53 AM" src="https://github.com/neuropoly/totalsegmentator-mri/assets/68945192/10ddd780-ec42-4540-a091-19d2b2dc3e53"> | <img width="671" alt="Screenshot 2024-01-02 at 5 11 19 AM" src="https://github.com/neuropoly/totalsegmentator-mri/assets/68945192/a0069483-ad59-48bd-9c3e-a436888a39d7"> |
| [link](https://zenodo.org/records/583096) | [link](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.16529) | [link](https://paperswithcode.com/dataset/mrspineseg-challenge) |

> These datasets were chosen because they had these sacrum masks available or because they had co-registered MRI and CT images that allowed us to rely on the [CT total segmentator network](https://github.com/wasserth/TotalSegmentator) to generate these labels.

These datasets were BIDSified and stored on our private google drive, under `reg2pam50`:
- SynthRAD2023
- MRSpineSeg.zip
- GoldAtlas.zip

## II - Register CT labels to MRI

As specified before, some sacrum masks were generated using the [CT total segmentator network](https://github.com/wasserth/TotalSegmentator) but due to slightly different image shape between MRI (T1w and T2w) and CT scans segmentations still needed to me registered. To do that, the script `src/totalsegmri/utils/register_CT_seg_to_MR.py` was used with `GoldAtlas.zip` and `SynthRAD2023_pelvis.zip`.

> Registration could be needed aswell for some images of the dataset `MRSpineSeg`

```bash
python <PATH-TO-MRITOTALSEG-REPO>/src/totalsegmri/utils/register_CT_seg_to_MR.py --path-img <PATH-TO-BIDS-FOLDER>
```

## III - Generate a config file to select the data for training

To select the data used for training, a [config file](https://github.com/spinalcordtoolbox/disc-labeling-hourglass/issues/25#issuecomment-1695818382) was used. 

First fetch the paths to all the sacrum masks that will be used for TRAINING/VALIDATION/TESTING

> Run this following command in the previous datasets: the datasets should be stored inside the same parent folder.

```bash
find ~+ -type f -name *_seg.nii.gz | grep -v CT | sort > <PATH-TO-MRITOTALSEG-REPO>/src/totalsegmri/resources/configs/train_sacrum.txt
```

Then run this command to generate the JSON config file.

```bash
python src/totalsegmri/data_management/init_data_config.py --txt <PATH-TO-MRITOTALSEG-REPO>/src/totalsegmri/resources/configs/train_sacrum.txt --type LABEL --split-validation SPLIT_VAL --split-test SPLIT_TEST
```

`SPLIT_VAL` corresponds to the fraction of the data used for validation.

`SPLIT_TEST` corresponds to the fraction of the data used for testing.

Finally, to organize your data according to nnUNetV2 format, run this last command.

```bash
python src/totalsegmri/utils/convert_BIDS_config_to_nnunet_format.py --config <PATH-TO-MRITOTALSEG-REPO>/src/totalsegmri/resources/configs/train_sacrum.json --path-out <PATH-TO-NNUNET-DATASET> -dnum 300
```

`PATH-TO-NNUNET-DATASET` corresponds to the path where the generated nnUNetV2 dataset will be stored.

## IV - Train with nnUNetV2

> Regarding nnUNetV2 installation and general usage, please check https://github.com/ivadomed/utilities/blob/main/quick_start_guides/nnU-Net_quick_start_guide.md

Now that your data is ready, you can run nnUNetV2 preprocessing

```bash
nnUNetv2_plan_and_preprocess -d 300 --verify_dataset_integrity -c 3d_fullres
```

Then train using this command

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> nnUNetv2_train 300 3d_fullres 0
```

## V - Run nnUNetV2 inference on whole-spine and spider dataset

To run nnUNetV2's inference, keep the largest component and store the data according to BIDS standard, the script `run/gen_sacrum_seg_BIDS_from_config.sh` was used with the config file `src/totalsegmri/resources/configs/test_sacrum.json`.

Before running the following line, a few variables MUST be updated:
- the datasets path (key: DATASETS_PATH) in the JSON config `add_sacrum.json` (src/totalsegmri/resources/configs/add_sacrum.json)
- the repository path in `gen_sacrum_seg_BIDS_from_config.sh` (Variable: PATH_REPO)
- the nnunet model path in `gen_sacrum_seg_BIDS_from_config.sh` (Variable: PATH_NNUNET_MODEL)
- the environment path in `gen_sacrum_seg_BIDS_from_config.sh` (Variable: PATH_NNUNET_ENV)
- the author in `gen_sacrum_seg_BIDS_from_config.sh` (Variable: AUTHOR)

```bash
./run/gen_sacrum_seg_BIDS_from_config.sh
```





