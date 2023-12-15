import sys, argparse, textwrap, json
from pathlib import Path

import numpy as np
import nibabel as nib

import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
from functools import partial

def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
            Generate random disc labels channel for training from the labels.
            1. 50% of files - select only one voxel from all labels
            2. Leave 10% of discs as is.
            3. Remove 10% of discs.
            4. 50% of discs with single voxel.
            5. Randomally choose disc voxels.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            generate_random_disc_labels -i labels -o images
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--seg-dir', '-i', type=DirPath(), required=True,
        help='Folder containing input segmentations.'
    )
    parser.add_argument(
        '--output-dir', '-o', type=DirPath(True), required=True, 
        help='Folder to save output segmentations.'
    )
    parser.add_argument(
        '--subject-dir', '-d', type=str, default=None, nargs='?', const='',
        help=textwrap.dedent('''
            Is every subject has its oen direcrory.
            If this argument will be provided without value it will look for any directory in the segmentation directory.
            If value also provided it will be used as a prefix to subject directory (for example "sub-"), defaults to False (no subjet directory).
        '''),
    )
    parser.add_argument(
        '--subject-subdir', '-s', type=str, default='', 
        help='Subfolder inside subject folder containing masks (for example "anat"), defaults to no subfolder.'
    )
    parser.add_argument(
        '--seg-suffix', type=str, default='',
        help='Segmentation suffix, defaults to "".'
    )
    parser.add_argument(
        '--output-seg-suffix', type=str, default='_0001',
        help='Suffix for output segmentation, defaults to "_0001".'
    )
    parser.add_argument(
        '--add-to-discs-labels', type=int, required=True,
        help='Add this number to existing disk label to get the output disk label.'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=min(32, mp.cpu_count() + 4),
        help='Max worker to run in parallel proccess, defaults to min(32, mp.cpu_count() + 4).'
    )
    parser.add_argument(
        '--verbose', '-v', type=int, default=1, choices=[0, 1],
        help='Verbosity level. 0: Errors/warnings only, 1: Errors/warnings + info (default: 1)'
    )

    try:
        args = parser.parse_args()
    except BaseException as e:
        sys.exit()

    # Get arguments
    seg_path = args.seg_dir
    output_path = args.output_dir
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    add_to_discs_labels = args.add_to_discs_labels
    max_workers = args.max_workers
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f'''
            Running with arguments:
            seg_dir = "{seg_path}"
            output_dir = "{output_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            add_to_discs_labels = "{add_to_discs_labels}"
            max_workers = "{max_workers}"
            verbose = {verbose}
        '''))
    
    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    segs_path_list = list(seg_path.glob(glob_pattern))

    # Create a partially-applied function with the extra arguments
    partial_generate_random_disc_labels = partial(generate_random_disc_labels, add_to_discs_labels=add_to_discs_labels, output_path=output_path, seg_suffix=seg_suffix, output_seg_suffix=output_seg_suffix)

    with mp.Pool() as pool:
        process_map(partial_generate_random_disc_labels, segs_path_list, max_workers=max_workers)
    

def generate_random_disc_labels(seg_path, add_to_discs_labels, output_path, seg_suffix, output_seg_suffix):
    
    output_seg_path = output_path / seg_path.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz')

    # Load segmentation
    seg = nib.load(seg_path)
    seg_data = seg.get_fdata()
    
    # translate into SCT labels -2 (we do not have labels 1, 2)
    seg_data = seg_data + add_to_discs_labels - 2

    # Convert data to uint8 to avoid issues with segmentation IDs
    seg_data = seg_data.astype(np.uint8)
    
    # Randomly decide whether to limit the label to a single voxel for 50% of files
    single_voxel = np.random.choice(2)

    # Init output (put 25 in the even places to normalize 0-25 to 0-1)
    seg_data_out = np.zeros_like(seg_data)
    seg_data_out[::2, ::2, ::2] = 25

    for d in range(1, 23 + 1):
        # Get mask of this disc and delete it
        discs_mask = seg_data == d

        # Get indices of non-zero voxels
        non_zero_indices = np.argwhere(discs_mask)

        # If this disc not exists
        if len(non_zero_indices) == 0:
            continue
        
        # Randomally skip this disc as is
        if np.random.choice(2, p=(0.9, 0.1)):
            continue
        
        # Randomally leave this disc as is
        if not single_voxel and np.random.choice(2, p=(0.9, 0.1)):
            seg_data_out[discs_mask] = d
            continue

        # Decide the number of voxels to select from this disc
        if single_voxel or len(non_zero_indices) == 1:
            # If single_voxel flag is set, select only one voxel
            num_voxels = 1
        elif np.random.choice(2):
            # 50% chance of selecting only one voxel
            num_voxels = 1
        else:
            # Randomly select the number of voxels to be chosen, range starts from 1
            num_voxels = np.random.choice(len(non_zero_indices) - 1) + 1

        # Randomly select num_voxels from non-zero indices
        selected_indices = non_zero_indices[np.random.choice(len(non_zero_indices), num_voxels, replace=False)]

        # Randomly move this disc in the y axis
        selected_indices[:, 1] -= np.random.choice(5, len(selected_indices))

        # Set selected voxels to label value
        seg_data_out[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]] = d


    # Create result segmentation
    seg = nib.Nifti1Image(seg_data_out, seg.affine, seg.header)
    seg.set_data_dtype(np.uint8)
    # Make sure output directory exists
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    # Save mapped segmentation
    nib.save(seg, output_seg_path)


class DirPath:
    """
    Get path from argparse and return as Path object.
    
    Args:
        create: Create directory if it doesn't exist
        
    """

    def __init__(self, create=False):
        self.create = create

    def __call__(self, dir_path):
        path = Path(dir_path)
        
        if self.create and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True) 
            except:
                pass
                
        if path.is_dir():
            return path
        else:
            raise argparse.ArgumentTypeError(
                f"readble_dir:{path} is not a valid path")


if __name__ == '__main__':
    main()