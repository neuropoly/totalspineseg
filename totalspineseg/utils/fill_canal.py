import sys, argparse, textwrap
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
from functools import partial
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import label
import warnings

warnings.filterwarnings("ignore")

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
            Fix Spinal Canal label to include all non cord spinal canal, this will put the spinal canal label in all the voxels (labeled as a backgroupn) between the spinal canal and the spinal cord.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            fill_canal -s labels -o labels_fixed
            For BIDS:
            fill_canal -s derivatives/labels -o derivatives/labels --seg-suffix "_seg" --output-seg-suffix "_seg_fixed" -d "sub-" -u "anat"
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--segs-dir', '-s', type=Path, required=True,
        help='Folder containing input segmentations.'
    )
    parser.add_argument(
        '--output-segs-dir', '-o', type=Path, required=True,
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
        '--canal-label', type=int, default=201,
        help='Label used for Spinal Canal, defaults to 201.'
    )
    parser.add_argument(
        '--largest-cord', action="store_true", default=False,
        help='Take the largest spinal cord component.'
    )
    parser.add_argument(
        '--largest-canal', action="store_true", default=False,
        help='Take the largest spinal canal component.'
    )
    parser.add_argument(
        '--override', '-r', action="store_true", default=False,
        help='Override existing output files, defaults to false (Do not override).'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
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
    segs_path = args.segs_dir
    output_segs_path = args.output_segs_dir
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    cord_label = args.cord_label
    canal_label = args.canal_label
    largest_cord = args.largest_cord
    largest_canal = args.largest_canal
    override = args.override
    max_workers = args.max_workers
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            segs_dir = "{segs_path}"
            output_segs_dir = "{output_segs_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            cord_label = {cord_label}
            canal_label = {canal_label}
            largest_cord = {largest_cord}
            largest_canal = {largest_canal}
            override = {override}
            max_workers = {max_workers}
            verbose = {verbose}
        '''))

    fill_canal_mp(
        segs_path=segs_path,
        output_segs_path=output_segs_path,
        subject_dir=subject_dir,
        subject_subdir=subject_subdir,
        prefix=prefix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        cord_label=cord_label,
        canal_label=canal_label,
        largest_cord=largest_cord,
        largest_canal=largest_canal,
        override=override,
        max_workers=max_workers,
    )

def fill_canal_mp(
        segs_path,
        output_segs_path,
        subject_dir=None,
        subject_subdir='',
        prefix='',
        seg_suffix='',
        output_seg_suffix='',
        cord_label=200,
        canal_label=201,
        largest_cord=False,
        largest_canal=False,
        override=False,
        max_workers=mp.cpu_count(),
    ):
    segs_path = Path(segs_path)
    output_segs_path = Path(output_segs_path)

    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    seg_path_list = list(segs_path.glob(glob_pattern))
    output_seg_path_list = [output_segs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz') for _ in seg_path_list]

    process_map(
        partial(
            _fill_canal,
            cord_label=cord_label,
            canal_label=canal_label,
            largest_cord=largest_cord,
            largest_canal=largest_canal,
            override=override,
        ),
        seg_path_list,
        output_seg_path_list,
        max_workers=max_workers,
    )

def _fill_canal(
        seg_path,
        output_seg_path,
        cord_label=200,
        canal_label=201,
        largest_cord=False,
        largest_canal=False,
        override=False,
    ):
    seg_path = Path(seg_path)
    output_seg_path = Path(output_seg_path)

    # If the output image already exists and we are not overriding it, return
    if not override and output_seg_path.exists():
        return

    # Load segmentation
    seg = nib.load(seg_path)

    output_seg = fill_canal(
        seg,
        cord_label=cord_label,
        canal_label=canal_label,
        largest_cord=largest_cord,
        largest_canal=largest_canal,
    )

    # Ensure correct segmentation dtype, affine and header
    output_seg = nib.Nifti1Image(
        np.asanyarray(output_seg.dataobj).round().astype(np.uint8),
        output_seg.affine, output_seg.header
    )
    output_seg.set_data_dtype(np.uint8)
    output_seg.set_qform(output_seg.affine)
    output_seg.set_sform(output_seg.affine)

    # Make sure output directory exists and save the segmentation
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_seg, output_seg_path)

def fill_canal(
        seg,
        cord_label=200,
        canal_label=201,
        largest_cord=False,
        largest_canal=False,
    ):
    output_seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    # Take the largest spinal cord component
    if largest_cord and cord_label in output_seg_data:
        cord_mask = output_seg_data == cord_label
        cord_mask_largest = largest(cord_mask)
        output_seg_data[cord_mask & ~cord_mask_largest] = canal_label

    if canal_label in output_seg_data:

        # Take the largest spinal canal component
        if largest_canal:
            canal_mask = np.isin(output_seg_data, [cord_label, canal_label])
            canal_mask_largest = largest(canal_mask)
            output_seg_data[canal_mask & ~canal_mask_largest] = 0

        # Create an array of x indices
        x_indices = np.broadcast_to(np.arange(output_seg_data.shape[0])[..., np.newaxis, np.newaxis], output_seg_data.shape)
        # Create an array of y indices
        y_indices = np.broadcast_to(np.arange(output_seg_data.shape[1])[..., np.newaxis], output_seg_data.shape)

        canal_mask = np.isin(output_seg_data, [cord_label, canal_label])
        canal_mask_min_x = np.min(np.where(canal_mask, x_indices, np.inf), axis=0)[np.newaxis, ...]
        canal_mask_max_x = np.max(np.where(canal_mask, x_indices, -np.inf), axis=0)[np.newaxis, ...]
        canal_mask_min_y = np.min(np.where(canal_mask, y_indices, np.inf), axis=1)[:, np.newaxis, :]
        canal_mask_max_y = np.max(np.where(canal_mask, y_indices, -np.inf), axis=1)[:, np.newaxis, :]
        canal_mask = \
            (canal_mask_min_x <= x_indices) & \
                (x_indices <= canal_mask_max_x) & \
                (canal_mask_min_y <= y_indices) & \
                (y_indices <= canal_mask_max_y)
        output_seg_data[canal_mask & (output_seg_data != cord_label)] = canal_label

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

def largest(mask):
    mask_labeled, num_labels = label(mask, np.ones((3, 3, 3)))
    # Find the label of the largest component
    label_sizes = np.bincount(mask_labeled.ravel())[1:]  # Skip 0 label size
    largest_label = label_sizes.argmax() + 1  # +1 because bincount labels start at 0
    return mask_labeled == largest_label

if __name__ == '__main__':
    main()