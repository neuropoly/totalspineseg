import sys, argparse, textwrap
from scipy.ndimage import label
from pathlib import Path
import numpy as np
import nibabel as nib
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from totalsegmri.utils.dirpath import DirPath

def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
            This script processes NIfTI segmentation files, leaving the largest component for each label.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            generate_largest_labels -s labels_init -o labels
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--seg-dir', '-s', type=DirPath(), required=True,
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
    partial_generate_largest_labels = partial(
        generate_largest_labels,
        output_path=output_path,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
   )

    with mp.Pool() as pool:
        process_map(partial_generate_largest_labels, segs_path_list, max_workers=max_workers)
    

def generate_largest_labels(
            seg_path,
            output_path,
            seg_suffix,
            output_seg_suffix,
        ):
    
    output_seg_path = output_path / seg_path.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz')

    # Load segmentation
    seg = nib.load(seg_path)
    seg_data = seg.get_fdata()

    # Convert data to uint8 to avoid issues with segmentation IDs
    seg_data_src = seg_data.astype(np.uint8)

    seg_data = np.zeros_like(seg_data_src)

    for l in [_ for _ in np.unique(seg_data_src) if _ != 0]:
        mask = seg_data_src == l
        mask_labeled, num_labels = label(mask, np.ones((3, 3, 3)))
        # Find the label of the largest component
        label_sizes = np.bincount(mask_labeled.ravel())[1:]  # Skip 0 label size
        largest_label = label_sizes.argmax() + 1  # +1 because bincount labels start at 0
        seg_data[mask_labeled == largest_label] = l

    # Create result segmentation
    seg = nib.Nifti1Image(seg_data, seg.affine, seg.header)
    seg.set_data_dtype(np.uint8)
    # Make sure output directory exists
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    # Save mapped segmentation
    nib.save(seg, output_seg_path)

if __name__ == '__main__':
    main()