
import os, sys, argparse, textwrap, re, json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import label


def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script aim to remove outliers vilues in the warping field.
        The data is assumed to follow the BIDS structure (or similar):

        labels
        └── sub-errsm37
            └── anat
                └── sub-errsm37_T1w/warp_template2anat.nii.gz

        '''),
        epilog=textwrap.dedent('''
        Example:
        remove_warp_outliers.py -s warp_template2anat -b _orig
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--input_path', '-i', type= PathArg(), default=Path.cwd(),
        help='The working directory where the subjects are located or a warp path.'
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
        '--warp-suffix', '-s', type= str, default='warp_template2anat',
        help='Suffix for identifying the warping field files, defaults to "warp_template2anat".'
    )
    parser.add_argument(
        '--backup-suffix', '-b', type= str, default='_orig',
        help='Suffix for the backup files (leave empty remove original file), defaults to "_orig".'
    )
    parser.add_argument(
        '--src-z-voxel-size', '-z', type= float, default=0.5,
        help='The size of the source image voxel in the z-axis, defaults to 0.5.'
    )
    parser.add_argument(
        '--max-allowed-val', '-m', type= int, default=10000,
        help='Maximum allowed value for the components in the warping field, defaults to 10000.'
    )
    parser.add_argument(
        '--outliers-values', action='store_true', default=False,
        help='Filter outliers greater then 2 times the 75th percentile, defaults to False.'
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
    input_path = args.input_path
    subject_prefix = args.subject_prefix
    subject_subdir = args.subject_subdir
    warp_suffix = args.warp_suffix
    backup_suffix = args.backup_suffix
    src_z_voxel_size = args.src_z_voxel_size
    max_allowed_val = args.max_allowed_val
    outliers_values = args.outliers_values
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f''' 
            Running with the following arguments:
            input_path = "{input_path}"
            subject_prefix = "{subject_prefix}"
            subject_subdir = "{subject_subdir}"
            warp_suffix = "{warp_suffix}"
            backup_suffix = "{backup_suffix}"
            src_z_voxel_size = "{src_z_voxel_size}"
            max_allowed_val = "{max_allowed_val}"
            outliers_values = "{outliers_values}"
            verbose = {verbose}
        '''))

    if input_path.is_file():
        remove_warp_outliers(input_path, src_z_voxel_size, max_allowed_val, backup_suffix, outliers_values)

    elif input_path.is_dir():
        # Loop over all subjects
        for sub_path in input_path.glob(f'{subject_prefix}*'):
            # Ensure its a folder
            if not (sub_path / subject_subdir).is_dir():
                continue

            if verbose: print(f'Looking for warp files for: {sub_path.name}')

            for warp_path in (sub_path / subject_subdir).rglob(f'*{warp_suffix}.nii.gz'):
            
                if verbose: print(f'Working on: {warp_path}')
                remove_warp_outliers(warp_path, src_z_voxel_size, max_allowed_val, backup_suffix, outliers_values)


def remove_warp_outliers(warp_path, src_z_voxel_size, max_allowed_val, backup_suffix, outliers_values):

    # Load the nii.gz file
    img = nib.load(warp_path)

    # Extract the data
    data = img.get_fdata()

    # Work only on the z-axis
    z_component = data[..., 2]
    z_component_abs = np.abs(z_component)

    # Calculate the maximum value within the z displacement field
    z_component_max = np.max(z_component)
    
    # Create a mask that excludes values greater than max_allowed_val
    mask = z_component_abs < max_allowed_val
    
    # Create an array of z indices
    z_indices = np.tile(np.arange(data.shape[2]), (data.shape[0], data.shape[1], 1))[..., np.newaxis]

    # Make an array of the absulute value
    z_component_abs_displacement = np.where(mask, z_indices + z_component * src_z_voxel_size / img.header.get_zooms()[2], z_component_max)

    # Initialize a validation mask for the next steps
    mask_largest_label = np.zeros_like(mask)

    # Compute differences in absolute displacement in both directions
    diffs = [
        np.diff(z_component_abs_displacement, axis=2, prepend=-max_allowed_val),
        np.diff(z_component_abs_displacement, axis=2, append=max_allowed_val)
    ]

    # Process differences to create a mask that preserves spatial continuity
    # Ensure the displacement field increase with z axis
    for i, diff in enumerate(diffs):
        mask_spaced = mask & (diff >= 0) & (np.abs(diff) < max_allowed_val)

        # Analyzing the masked area: dividing it into continuous parts and keep only the largest part
        mask_spaced_labeled, num_labels = label(mask_spaced)
        mask_sizes = [np.sum(mask_spaced_labeled == (i + 1)) for i in range(num_labels)]
        mask_largest_label_index = np.argmax(mask_sizes) + 1
        mask_largest_label |= (mask_spaced_labeled == mask_largest_label_index)
    
    if outliers_values:
        # Define a threshold for identifying outliers based on 2 times Q3 (75th percentile).
        outliers = mask & (z_component_abs > 2 * np.percentile(z_component_abs[mask], 75))

        # Remove outliers from mask
        mask &= ~outliers

    # Leave only the largest connected part in the mask
    mask &= mask_largest_label

    # Update the z-component with the maximum value where the mask is false
    z_component[~mask] = z_component_max
    data[..., 2] = z_component

    # Backup the original warping field
    if backup_suffix:
        Path(warp_path).rename(str(warp_path).replace(".nii.gz", f"{backup_suffix}.nii.gz"))

    # Save the edited warping field
    nib.save(nib.Nifti1Image(data, img.affine, img.header), warp_path)


class PathArg(object):
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
        if path.exists():
            return path
        else:
            raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')

if __name__ == '__main__':
    main()
