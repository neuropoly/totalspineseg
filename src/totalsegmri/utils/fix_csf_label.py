import sys, argparse, textwrap
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
from functools import partial
from pathlib import Path
import numpy as np
import nibabel as nib

def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
            Fix csf label to include all non cord spinal canal, this will put the spinal canal label in all the voxels (labeled as a backgroupn) between the spinal canal and the spinal cord.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            fix_csf_label -s labels -o labels_fixed
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--segs-dir', '-s', type=Path, required=True,
        help='Folder containing input segmentations.'
    )
    parser.add_argument(
        '--output-dir', '-o', type=Path, required=True, 
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
        '--subject-subdir', '-u', type=str, default='', 
        help='Subfolder inside subject folder containing masks (for example "anat"), defaults to no subfolder.'
    )
    parser.add_argument(
        '--prefix', '-p', type=str, default='', 
        help='File prefix to work on.'
    )
    parser.add_argument(
        '--seg-suffix', type=str, default='',
        help='Segmentation suffix, defaults to "".'
    )
    parser.add_argument(
        '--output-seg-suffix', type=str, default='',
        help='Suffix for output segmentation, defaults to "".'
    )
    parser.add_argument(
        '--cord-label', type=int, default=200,
        help='Label used for spinal cord, defaults to 200.'
    )
    parser.add_argument(
        '--csf-label', type=int, default=201,
        help='Label used for csf, defaults to 201.'
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
    prefix = args.prefix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    cord_label = args.cord_label
    csf_label = args.csf_label
    max_workers = args.max_workers
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            seg_dir = "{seg_path}"
            output_dir = "{output_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            cord_label = "{cord_label}"
            csf_label = "{csf_label}"
            max_workers = "{max_workers}"
            verbose = {verbose}
        '''))
    
    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    segs_path_list = list(seg_path.glob(glob_pattern))

    # Create a partially-applied function with the extra arguments
    partial_fix_csf_label = partial(fix_csf_label, cord_label=cord_label, csf_label=csf_label, output_path=output_path, seg_suffix=seg_suffix, output_seg_suffix=output_seg_suffix)

    with mp.Pool() as pool:
        process_map(partial_fix_csf_label, segs_path_list, max_workers=max_workers)
    

def fix_csf_label(seg_path, cord_label, csf_label, output_path, seg_suffix, output_seg_suffix):
    
    output_seg_path = output_path / seg_path.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz')

    # Load segmentation
    seg = nib.load(seg_path)
    seg_data = seg.get_fdata()

    # Convert data to uint8 to avoid issues with segmentation IDs
    seg_data = seg_data.astype(np.uint8)

    # Create an array of x indices
    x_indices = np.broadcast_to(np.arange(seg_data.shape[0])[..., np.newaxis, np.newaxis], seg_data.shape)
    # Create an array of y indices
    y_indices = np.broadcast_to(np.arange(seg_data.shape[1])[..., np.newaxis], seg_data.shape)

    canal_mask = np.isin(seg_data, [cord_label, csf_label])
    canal_mask_min_x = np.min(np.where(canal_mask, x_indices, np.inf), axis=0)[np.newaxis, ...]
    canal_mask_max_x = np.max(np.where(canal_mask, x_indices, -np.inf), axis=0)[np.newaxis, ...]
    canal_mask_min_y = np.min(np.where(canal_mask, y_indices, np.inf), axis=1)[:, np.newaxis, :]
    canal_mask_max_y = np.max(np.where(canal_mask, y_indices, -np.inf), axis=1)[:, np.newaxis, :]
    canal_mask = \
        (canal_mask_min_x <= x_indices) & \
            (x_indices <= canal_mask_max_x) & \
            (canal_mask_min_y <= y_indices) & \
            (y_indices <= canal_mask_max_y)
    seg_data[canal_mask & (seg_data != cord_label)] = csf_label

    # Create result segmentation 
    mapped_seg = nib.Nifti1Image(seg_data, seg.affine, seg.header)
    mapped_seg.set_data_dtype(np.uint8)

    # Make sure output directory exists
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save mapped segmentation
    nib.save(mapped_seg, output_seg_path)

if __name__ == '__main__':
    main()