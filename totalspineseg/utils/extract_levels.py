import sys, argparse, textwrap
from pathlib import Path
import numpy as np
import nibabel as nib
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import warnings

warnings.filterwarnings("ignore")

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            Extract vertebrae levels from Spinal Canal and Discs.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            extract_levels -s labels -o levels --canal-labels 1 2 --disc-labels 60-64 70-81 90-94 100 -r
            For BIDS:
            extract_levels -s derivatives/labels -o derivatives/labels --seg-suffix "_seg" --output-seg-suffix "_levels" -p "sub-*/anat/" --canal-labels  1 2 --disc-labels 60-64 70-81 90-94 100 -r
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
        '--canal-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', required=True,
        help='The canal labels.'
    )
    parser.add_argument(
        '--disc-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', required=True,
        help='The disc labels starting at C2C3 ordered from superior to inferior.'
    )
    parser.add_argument(
        '--c1-label', type=int, default=0,
        help='The label for C1 vertebra in the segmentation, if provided it will be used to determine if C1 is in the segmentation.'
    )
    parser.add_argument(
        '--overwrite', '-r', action="store_true", default=False,
        help='Overwrite existing output files, defaults to false (Do not overwrite).'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get arguments
    segs_path = args.segs_dir
    output_segs_path = args.output_segs_dir
    prefix = args.prefix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    canal_labels = args.canal_labels
    disc_labels = [l for raw in args.disc_labels for l in (raw if isinstance(raw, list) else [raw])]
    c1_label = args.c1_label
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            segs_dir = "{segs_path}"
            output_segs_dir = "{output_segs_path}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            canal_labels = {canal_labels}
            disc_labels = {disc_labels}
            c1_label = {c1_label}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    extract_levels_mp(
        segs_path=segs_path,
        output_segs_path=output_segs_path,
        prefix=prefix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        canal_labels=canal_labels,
        disc_labels=disc_labels,
        c1_label=c1_label,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def extract_levels_mp(
        segs_path,
        output_segs_path,
        prefix='',
        seg_suffix='',
        output_seg_suffix='',
        canal_labels=[],
        disc_labels=[],
        c1_label=0,
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    segs_path = Path(segs_path)
    output_segs_path = Path(output_segs_path)

    glob_pattern = f'{prefix}*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    seg_path_list = list(segs_path.glob(glob_pattern))
    output_seg_path_list = [output_segs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz') for _ in seg_path_list]

    process_map(
        partial(
            _extract_levels,
            canal_labels=canal_labels,
            disc_labels=disc_labels,
            c1_label=c1_label,
            overwrite=overwrite,
        ),
        seg_path_list,
        output_seg_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _extract_levels(
        seg_path,
        output_seg_path,
        canal_labels=[],
        disc_labels=[],
        c1_label=0,
        overwrite=False,
    ):
    '''
    Wrapper function to handle IO.
    '''
    seg_path = Path(seg_path)
    output_seg_path = Path(output_seg_path)

    # If the output image already exists and we are not overriding it, return
    if not overwrite and output_seg_path.exists():
        return

    # Load segmentation
    seg = nib.load(seg_path)

    try:
        output_seg = extract_levels(
            seg,
            canal_labels=canal_labels,
            disc_labels=disc_labels,
            c1_label=c1_label,
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
        seg,
        canal_labels=[],
        disc_labels=[],
        c1_label=0,
    ):
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
    disc_labels : list
        The disc labels starting at C2C3 ordered from superior to inferior.
    c1_label : int
        The label for C1 vertebra in the segmentation, if provided it will be used to determine if C1 is in the segmentation.

    Returns
    -------
    nibabel.Nifti1Image
        The output segmentation with the vertebrae levels.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    output_seg_data = np.zeros_like(seg_data)

    # Get array of indices for x, y, and z axes
    indices = np.indices(seg_data.shape)

    # Create a mask of the canal
    mask_canal = np.isin(seg_data, canal_labels)

    # If cancl is empty raise an error
    if not np.any(mask_canal):
        raise ValueError(f"No canal labels found in the segmentation.")

    # Create a mask the canal centerline by finding the middle voxels in x and y axes for each z index
    mask_min_x_indices = np.min(indices[0], where=mask_canal, initial=np.iinfo(indices.dtype).max, keepdims=True, axis=(0, 1))
    mask_max_x_indices = np.max(indices[0], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_mid_x = indices[0] == ((mask_min_x_indices + mask_max_x_indices) // 2)
    mask_min_y_indices = np.min(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).max, keepdims=True, axis=(0, 1))
    mask_max_y_indices = np.max(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_mid_y = indices[1] == ((mask_min_y_indices + mask_max_y_indices) // 2)
    mask_canal_centerline = mask_canal * mask_mid_x * mask_mid_y

    # Get the indices of the canal centerline
    canal_centerline_indices = np.array(np.nonzero(mask_canal_centerline)).T

    # Get the labels of the discs in the segmentation
    disc_labels_in_seg = np.array(disc_labels)[np.isin(disc_labels, seg_data)]

    # If no disc labels found in the segmentation raise an error
    if len(disc_labels_in_seg) == 0:
        raise ValueError(f"No disc labels found in the segmentation.")

    # Get the labels of the discs and the output labels
    first_disk_idx = disc_labels.index(disc_labels_in_seg[0])
    out_labels = list(range(3 + first_disk_idx, 3 + first_disk_idx + len(disc_labels_in_seg)))

    # Filter the discs that are in the segmentation
    map_labels = dict(zip(disc_labels_in_seg, out_labels))

    # Create mask of the discs
    mask_discs = np.isin(seg_data, disc_labels_in_seg)

    # Get list of indices for all the discs voxels
    discs_indices = np.nonzero(mask_discs)
    
    # Get the matching labels for the discs indices
    discs_indices_labels = seg_data[discs_indices]

    # Make the discs_indices 2D array
    discs_indices = np.array(discs_indices).T

    # Calculate the distance of each disc voxel to each canal centerline voxel
    discs_distances_from_all_centerline = np.linalg.norm(discs_indices[:, None, :] - canal_centerline_indices[None, ...], axis=2)

    # Find the minimum distance for each disc voxel and the corresponding canal centerline index
    discs_distance_from_centerline = np.min(discs_distances_from_all_centerline, axis=1)
    discs_centerline_indices = canal_centerline_indices[np.argmin(discs_distances_from_all_centerline, axis=1)]

    # Find the closest canal centerline voxel to each disc label
    disc_labels_centerline_indices = [discs_centerline_indices[discs_indices_labels == label][np.argmin(discs_distance_from_centerline[discs_indices_labels == label])] for label in disc_labels_in_seg]

    # Set the output labels to the closest canal centerline voxels
    for idx, label in zip(disc_labels_centerline_indices, disc_labels_in_seg):
        output_seg_data[tuple(idx)] = map_labels[label]

    # If C2-C3 is in the segmentation, set 1 and 2 to the superior voxels in the canal centerline and the middle voxels between C2-C3 and the superior voxels
    if 3 in output_seg_data:
        # Find the location of the C2-C3 disc
        c2c3_index = np.unravel_index(np.argmax(output_seg_data == 3), seg_data.shape)

        # Find the location of the superior voxels in the canal centerline
        canal_superior_index = np.unravel_index(np.argmax(mask_canal_centerline * indices[2]), seg_data.shape)

        if (c1_label > 0 and c1_label in seg_data) or (c1_label == 0 and canal_superior_index[2] - c2c3_index[2] >= 8 and output_seg_data.shape[2] - canal_superior_index[2] >= 2):
            # If C1 is in the segmentation or C2-C3 at least 8 voxels below the top of the canal and the top of the canal is at least 2 voxels from the top of the image
            # Set 1 to the superior voxels
            output_seg_data[canal_superior_index] = 1

            # Set 2 to the middle voxels between C2-C3 and the superior voxels
            c1c2_z_index = (canal_superior_index[2] + c2c3_index[2]) // 2
            c1c2_index = np.unravel_index(np.argmax(mask_canal_centerline * (indices[2] == c1c2_z_index)), seg_data.shape)
            output_seg_data[c1c2_index] = 2

        elif canal_superior_index[2] - c2c3_index[2] >= 4:
            # If C2-C3 at least 4 voxels below the top of the canal
            output_seg_data[canal_superior_index] = 2

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

if __name__ == '__main__':
    main()