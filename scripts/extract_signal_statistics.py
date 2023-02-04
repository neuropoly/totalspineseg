
import os, sys, argparse, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

def dir_path(path: str) -> Path:
    '''
    Get path parameter from argparse and retuen it as pathlib Path object.
    '''

    # Test if path exists
    if os.path.isdir(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')

def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script extracts signal statistics (mean + std) for each mask, for each subject, and saves it to a CSV file. The data is assumed to follow the BIDS structure:

        ├── derivatives
        │   └── manual_masks
        │       ├── sub-errsm37
        │       │   └── anat
        │       │       ├── organ_spleen.nii.gz
        │       │       └── vertebrae_L3.nii.gz
        │       └── sub-errsm38
        │           └── anat
        │               ├── organ_spleen.nii.gz
        │               └── vertebrae_L3.nii.gz
        ├── sub-errsm37
        │   └── anat
        │       ├── sub-errsm37_T1w.json
        │       └── sub-errsm37_T1w.nii.gz
        └── sub-errsm38
            └── anat
                ├── sub-errsm38_T1w.json
                └── sub-errsm38_T1w.nii.gz
                
        The output csv will contain row for each subject with mean and std for each mask:
        
        ----------------------------------------------------------------------------------------
        |subject     |organ_spleen_mean |organ_spleen_std |vertebrae_L3_mean |vertebrae_L3_std |
        ----------------------------------------------------------------------------------------
        |sub-errsm37 |236.818181818181  |42.1278390252412 |271.25            |37.86010433      |
        |sub-errsm38 |252.354838709677  |29.04340288      |223.341234123412  |29.23423432      |
        ----------------------------------------------------------------------------------------

        '''),
        epilog=textwrap.dedent('''
        Example:
        extract_signal_statistics -d data -o signal_statistics.csv
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--data-dir', '-d', type= dir_path, required=True,
        help='The data folder to process.'
    )
    parser.add_argument(
        '--output-csv', '-o', type=argparse.FileType('w',encoding='utf-8'), required=True,
        help='The output csv file.'
    )
    parser.add_argument(
        '--subject-prefix', type= str, default='sub-',
        help='Subject prefix, defaults to "sub-" which is the prefix used for BIDS directories.'
    )
    parser.add_argument(
        '--image-suffix', type= str, default='_T1w.nii.gz',
        help='Image suffix, defaults to "_T1w.nii.gz".'
    )
    parser.add_argument(
        '--mask-suffix', type= str, default='.nii.gz',
        help='Mask suffix Subject prefix, defaults to ".nii.gz".'
    )
    parser.add_argument(
        '--verbosity', '-v', type= int, default=1, choices=[0, 1],
        help='Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages (default: 1)'
    )

    try:
        args = parser.parse_args()
    except BaseException as e:
        sys.exit()

    # Get args
    data_dir = args.data_dir
    output_csv = args.output_csv
    subject_prefix = args.subject_prefix
    image_suffix = args.image_suffix
    mask_suffix = args.mask_suffix
    verbosity = args.verbosity

    if verbosity:
        print(textwrap.dedent(f''' 
            Running with the following arguments:
            data_dir = "{data_dir}"
            output_csv = "{output_csv.name}"
            subject_prefix = "{subject_prefix}"
            image_suffix = "{image_suffix}"
            mask_suffix = "{mask_suffix}"
            verbosity = {verbosity}
        '''))

    # manual_masks directory
    manual_masks_dir = data_dir / 'derivatives' / 'manual_masks'

    # Initialize an Array to store results for each subject
    subjects_stats = []

    # Loop across BIDS structure (sub-*)
    if verbosity: print(f'Looking for subjects in "{data_dir}"')
    for subject_dir in data_dir.glob(f'{subject_prefix}*'):

        if verbosity: print(f'Working on "{subject_dir.name}":')

        # Init dict for subject statistics
        subject_stats = {}
        subjects_stats.append(subject_stats)

        # Get subject
        subject = subject_stats['subject'] = subject_dir.name
        
        # Get image path
        img_path = subject_dir / 'anat' / f'{subject}{image_suffix}'

        # Check if image exist. If not, go to next subject (all values for this subject row will be empty)
        if not img_path.exists():
            if verbosity: print(f'Image not found: "{img_path}"')
            continue

        if verbosity: print(f'Extracting masks signal statistics for "{img_path.name}"')

        # For each subject (sub-xxx), open the image with nibabel
        img = nib.load(img_path)
        img_data = img.get_fdata()

        subject_manual_masks_dir = manual_masks_dir / subject / 'anat'

        if verbosity: print(f'Looking for masks in "{subject_manual_masks_dir}"')

        # Loop across the masks under derivatives/manual_masks/sub-xxx/anat/*
        for mask_path in sorted((subject_manual_masks_dir).glob(f'*{mask_suffix}')):
            
            if verbosity: print(f'Extracting signal statistics for "{mask_path.name}"')

            # Get mask name without suffix
            mask_name = mask_path.name.replace(mask_suffix, '')
            # Open mask
            mask = nib.load(mask_path)
            mask_data = mask.get_fdata()

            # For each mask, compute the mean and STD and update the dataframe (nan on empty mask)
            mask_img_values = img_data[np.where(mask_data)]
            subject_stats[f'{mask_name}_mean'] = np.mean(mask_img_values) if mask_img_values.size > 0 else np.nan
            subject_stats[f'{mask_name}_std'] = np.std(mask_img_values) if mask_img_values.size > 0 else np.nan

    # Initialize a pandas dataframe (row: subject, column: mask)
    df = pd.DataFrame(subjects_stats)

    # Save dataframe as CSV.
    if verbosity: print(f'Saving output to "{output_csv.name}"')
    df.to_csv(output_csv, index=False, sep=',', lineterminator='\n')

if __name__ == '__main__':
    main()
