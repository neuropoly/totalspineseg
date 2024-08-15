import sys, argparse, textwrap
from pathlib import Path
import numpy as np
import nibabel as nib
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import scipy.ndimage as ndi
import warnings

warnings.filterwarnings("ignore")

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
            Extract vertebrae levels from Spinal Canal and Discs.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            extract_levels -s labels_init -o labels --disc-labels 1 2 3 4 5 6 7 --vertebrea-labels 9 10 11 12 13 14 --vertebrea-extra-labels 8 --init-disc 4:224 7:202 5:219 6:207 --init-vertebrae 11:40 14:17 12:34 13:23 --step-diff-label --step-diff-disc --output-disc-step -1 --output-vertebrea-step -1 --map-output 17:92 --map-input 14:92 16:201 17:200 -r
            For BIDS:
            extract_levels -s derivatives/labels -o derivatives/labels --seg-suffix "_seg" --output-seg-suffix "_seg_seq" -d "sub-" -u "anat" --disc-labels 1 2 3 4 5 6 7 --vertebrea-labels 9 10 11 12 13 14 --vertebrea-extra-labels 8 --init-disc 4:224 7:202 5:219 6:207 --init-vertebrae 11:40 14:17 12:34 13:23 --step-diff-label --step-diff-disc --output-disc-step -1 --output-vertebrea-step -1 --map-output 17:92 --map-input 14:92 16:201 17:200 -r
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
        '--canal-labels', type=int, nargs='+', required=True,
        help='The canal labels.'
    )
    parser.add_argument(
        '--c2c3-label', type=int, required=True,
        help='The label for C2-C3 disc.'
    )
    parser.add_argument(
        '--step', type=int, default=1,
        help='The step to take between discs labels in the input, defaults to 1.'
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
    canal_labels = args.canal_labels
    c2c3_label = args.c2c3_label
    step = args.step
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
            canal_labels = {canal_labels}
            c2c3_label = {c2c3_label}
            step = {step}
            override = {override}
            max_workers = {max_workers}
            verbose = {verbose}
        '''))

    extract_levels_mp(
        segs_path=segs_path,
        output_segs_path=output_segs_path,
        subject_dir=subject_dir,
        subject_subdir=subject_subdir,
        prefix=prefix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        canal_labels=canal_labels,
        c2c3_label=c2c3_label,
        step=step,
        override=override,
        max_workers=max_workers,
    )

def extract_levels_mp(
        segs_path,
        output_segs_path,
        subject_dir=None,
        subject_subdir='',
        prefix='',
        seg_suffix='',
        output_seg_suffix='',
        canal_labels=[],
        c2c3_label=3,
        step=1,
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
            _extract_levels,
            canal_labels=canal_labels,
            step=step,
            c2c3_label=c2c3_label,
            override=override,
        ),
        seg_path_list,
        output_seg_path_list,
        max_workers=max_workers,
    )

def _extract_levels(
        seg_path,
        output_seg_path,
        canal_labels=[],
        c2c3_label=3,
        step=1,
        override=False,
    ):
    seg_path = Path(seg_path)
    output_seg_path = Path(output_seg_path)

    # If the output image already exists and we are not overriding it, return
    if not override and output_seg_path.exists():
        return

    # Load segmentation
    seg = nib.load(seg_path)

    try:
        output_seg = extract_levels(
            seg,
            canal_labels=canal_labels,
            c2c3_label=c2c3_label,
            step=step,
        )
    except ValueError as e:
        output_seg_path.is_file() and output_seg_path.unlink()
        print(f'Error: {seg_path}, {e}')
        return

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

def extract_levels(
        seg: nib.Nifti1Image,
        canal_labels: list,
        c2c3_label: dict,
        step: int,
    ) -> nib.Nifti1Image:
    '''
    Extract vertebrae levels from Spinal Canal and Discs.

    The function extracts the vertebrae levels from the input segmentation by finding the closest voxel in the canal centerline to the middle of each disc.
    The superior voxels in the canal centerline are set to 1 and the middle voxels between C2-C3 and the superior voxels are set to 2.

    Parameters
    ----------
    seg : nibabel.Nifti1Image
        The input segmentation.
    canal_labels : list
        The canal labels.
    c2c3_label : int
        The label for C2-C3 disc.
    step : int
        The step to take between discs labels in the input.

    Returns
    -------
    nibabel.Nifti1Image
        The output segmentation with the vertebrae levels.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    output_seg_data = np.zeros_like(seg_data)

    # Get array of indices for x, y, and z axes
    x_indices, y_indices, z_indices = np.indices(seg_data.shape)

    # Create a mask of the canal
    mask_canal = np.isin(seg_data, canal_labels)

    # If cancl is empty raise an error
    if not np.any(mask_canal):
        raise ValueError(f"No canal labels found in the segmentation.")

    # Create a mask the canal centerline by finding the middle voxels in x and y axes for each z index
    mask_min_x_indices = np.min(x_indices, where=mask_canal, initial=np.iinfo(x_indices.dtype).max, keepdims=True, axis=(0, 1))
    mask_max_x_indices = np.max(x_indices, where=mask_canal, initial=np.iinfo(x_indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_mid_x = x_indices == ((mask_min_x_indices + mask_max_x_indices) // 2)
    mask_min_y_indices = np.min(y_indices, where=mask_canal, initial=np.iinfo(x_indices.dtype).max, keepdims=True, axis=(0, 1))
    mask_max_y_indices = np.max(y_indices, where=mask_canal, initial=np.iinfo(x_indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_mid_y = y_indices == ((mask_min_y_indices + mask_max_y_indices) // 2)
    mask_canal_centerline = mask_canal * mask_mid_x * mask_mid_y

    # Compute the distance transform of the canal centerline
    distances_from_centerline, closest_centerline_indices = ndi.distance_transform_edt(~mask_canal_centerline, return_indices=True)

    # Get the labels of the discs and the output labels
    disc_labels = list(range(c2c3_label, c2c3_label + step * 23, step))
    out_labels = list(range(3, 26))

    # Filter the discs that are in the segmentation
    in_seg = np.isin(disc_labels, seg_data)
    map_labels = [(d, o) for d, o, i in zip(disc_labels, out_labels, in_seg) if i]

    # If no disc labels found in the segmentation raise an error
    if len(map_labels) == 0:
        raise ValueError(f"No disc labels found in the segmentation.")

    # Loop over the discs from C2-C3 to L5-S1 and find the closest voxel in the canal centerline
    for disc_label, out_label in map_labels:
        # Create a mask of the disc
        mask_disc = seg_data == disc_label

        # Find the location of the minimum distance in disc from the canal centerline
        voxel_in_disc_closest_to_centerline = np.unravel_index(np.argmin(np.where(mask_disc, distances_from_centerline, np.inf)), mask_canal.shape)

        # Get the corresponding closest voxel in the canal centerline
        voxel_in_centerline_closest_to_disc = tuple(closest_centerline_indices[(...,) + tuple(voxel_in_disc_closest_to_centerline)])

        # Set the output label
        output_seg_data[voxel_in_centerline_closest_to_disc] = out_label

    # If C2-C3 is in the segmentation, set 1 and 2 to the superior voxels in the canal centerline and the middle voxels between C2-C3 and the superior voxels
    if 3 in output_seg_data:
        # Find the location of the C2-C3 disc
        c2c3_index = np.unravel_index(np.argmax(output_seg_data == 3), seg_data.shape)

        # Find the location of the superior voxels in mask_canal_centerline
        canal_superior_index = np.unravel_index(np.argmax(mask_canal_centerline * z_indices), seg_data.shape)

        if canal_superior_index[2] - c2c3_index[2] >= 4:
            # If the distance between the superior voxels and C2-C3 is more than 4
            # Set 1 to the superior voxels
            output_seg_data[canal_superior_index] = 1

            # Set 2 to the middle voxels between C2-C3 and the superior voxels
            c1c2_z_index = (canal_superior_index[2] + c2c3_index[2]) // 2
            c1c2_index = np.unravel_index(np.argmax(mask_canal_centerline * (z_indices == c1c2_z_index)), seg_data.shape)
            output_seg_data[c1c2_index] = 2

        elif canal_superior_index[2] - c2c3_index[2] >= 2:
            # If the distance between the superior voxels and C2-C3 is more than 2 set 2 to the superior voxels
            output_seg_data[canal_superior_index] = 2

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

if __name__ == '__main__':
    main()