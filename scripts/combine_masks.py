
import os, sys, argparse, textwrap, re, json
from pathlib import Path
import numpy as np
import nibabel as nib


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script combine multiple masks and save the result as a single segmentation file, for each subject. '''
        '''The script sould be provided with a dictionary contained mapping of each mask to its value in the result '''
        '''segmentation.
        The data is assumed to follow the BIDS structure (or similar):

        masks
        ├── sub-errsm37
        │   └── anat
        │       ├── organ_spleen.nii.gz
        │       └── vertebrae_L3.nii.gz
        └── sub-errsm38
            └── anat
                ├── organ_spleen.nii.gz
                └── vertebrae_L3.nii.gz
                
        The output segmentations for this structure will be:

        segmentations
        ├── sub-errsm37
        │   └── anat
        │       └── sub-errsm37_seg.nii.gz
        └── sub-errsm38
            └── anat
                └── sub-errsm38_seg.nii.gz

        '''),
        epilog=textwrap.dedent('''
        Example:
        combine_masks -d data -o segmentations -m classes.json
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--masks-dir', '-d', type= DirPath(), required=True,
        help='The folder where input masks are located for each subject.'
    )
    parser.add_argument(
        '--output-dir', '-o', type= DirPath(True), required=True,
        help='The folder where output combined segmentations will be saved for each subject.'
    )
    parser.add_argument(
        '--masks-ids', '-m', type=argparse.FileType('r',encoding='utf-8'), required=True,
        help='json file contaning mapping of each mask to a unique number. for example: {"vertebrae_L3": 2, "organ_spleen": 15}'
    )
    parser.add_argument(
        '--subject-prefix', type= str, default='sub-',
        help='Subject prefix, defaults to "sub-" which is the prefix used for BIDS directories.'
    )
    parser.add_argument(
        '--subject-subdir', type= str, default='anat',
        help='The subfolder inside subject folder contaning the masks, defaults to "anat" which is used for BIDS directories.'
    )
    parser.add_argument(
        '--seg-suffix', type= str, default='_T1w_seg',
        help='The suffix for the output segmentation, defaults to "_T1w_seg".'
    )
    parser.add_argument(
        '--verbose', '-v', type= int, default=1, choices=[0, 1],
        help='verbose. 0: Display only errors/warnings, 1: Errors/warnings + info messages (default: 1)'
    )

    try:
        args = parser.parse_args()
    except BaseException as e:
        sys.exit()

    # Get args
    masks_path = args.masks_dir
    output_path = args.output_dir
    masks_ids_file = args.masks_ids
    subject_prefix = args.subject_prefix
    subject_subdir = args.subject_subdir
    seg_suffix = args.seg_suffix
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f''' 
            Running with the following arguments:
            masks_dir = "{masks_path}"
            output_dir = "{output_path}"
            masks_ids = "{masks_ids_file.name}"
            subject_prefix = "{subject_prefix}"
            subject_subdir = "{subject_subdir}"
            seg_suffix = "{seg_suffix}"
            verbose = {verbose}
        '''))

    # Load masks unique values from the json file
    masks_ids = json.load(masks_ids_file)
    masks_ids_file.close()

    # Loop over all subjects
    for sub_path in masks_path.glob(f'{subject_prefix}*'):

        # Ensure its a folder
        if not sub_path.is_dir():
            continue
        
        # Get subject name
        subject_name = sub_path.name
        # Make output subject name follow the BIDS.
        output_subject_name = re.sub(f'^{re.escape(subject_prefix)}', 'sub-', subject_name)
        
        # Combine masks
        masks_path = sub_path / subject_subdir
        output_file_path = output_path / output_subject_name / 'anat' / f'{output_subject_name}{seg_suffix}.nii.gz'
        if masks_path.exists():
            combine_masks(masks_path, output_file_path, masks_ids)


class DirPath(object):
    """
    Get path parameter from argparse and return it as pathlib Path object.

    Args:
    create (bool): Indicate if the directorie should be created. Default: False.
    """

    def __init__(self, create:bool=False):
        self.create = create

    def __call__(self, dir):

        path = Path(dir)

        # Create dir if create was specify
        if self.create and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except: pass

        # Test if path exists
        if path.is_dir():
            return path
        else:
            raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')


def combine_masks(input_masks_dir, output_file, masks_ids, verbose=True):
    """
    Combine multiple masks and save the result as a single segmentation file.
    The function get dictionary with mapping of each mask to its value in the result segmentation.

    Args:
    input_masks_dir (str): Directory where input masks are located.
    output_file (str): Output segmentation file path.
    masks_ids (dict): Dictionary contain all masks names and their unique id.
        The masks should be save to a file with the same name, for example
        masks for vertebrae_L3 should be save to vertebrae_L3.nii.gz
    verbose (bool, optional): If True, print status updates. Default is True.
    """

    output_data = None

    if verbose: print(f'Looking for masks in {input_masks_dir}')

    input_masks_path = Path(input_masks_dir)

    # Loop over the masks dictionary and look for mask file
    for mask_name, id in masks_ids.items():

        mask_path = input_masks_path / f'{mask_name}.nii.gz'

        # Ensure that the mask exists
        if not mask_path.exists():
            continue

        # Load mask
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata()

        # Convert data type to int to prevent problem with segmentation ids
        mask_data = mask_data.astype(np.uint8)

        if output_data is None:
            # On the first time init the output segmentation
            output_data = mask_data
        output_data[np.where(mask_data > 0)] = id

    # Create the result segmentation
    combined_seg = nib.Nifti1Image(output_data, mask.affine, mask.header)
    combined_seg.set_data_dtype(np.uint8)

    if verbose: print(f'Saving combined segmentation to {output_file}')

    #Make sure output path exists and save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    nib.save(combined_seg, output_file)


if __name__ == '__main__':
    main()
