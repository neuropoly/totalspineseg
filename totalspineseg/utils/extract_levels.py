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
        help='The label for C1 vertebra in the segmentation, if provided it will be used to extract the level 1.'
    )
    parser.add_argument(
        '--c2-label', type=int, default=0,
        help='The label for C2 vertebra in the segmentation (this label may also be used with other vertebrae), if provided it will be used to extract the level 1.'
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
    c2_label = args.c2_label
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
            c2_label = {c2_label}
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
        c2_label=c2_label,
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
        c2_label=0,
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
            c2_label=c2_label,
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
        c2_label=0,
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
            c2_label=c2_label,
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
        c2_label=0,
    ):
    '''
    Extract vertebrae levels from Spinal Canal and Discs.

    The function extracts the vertebrae levels from the input segmentation by finding the closest voxel in the canal anteriorline to the middle of each disc.
    The superior voxels of the top vertebrae is set to 1 and the middle voxels between C2-C3 and the superior voxels are set to 2.
    The generated labeling convention follows the one from SCT (https://spinalcordtoolbox.com/stable/user_section/tutorials/vertebral-labeling/labeling-conventions.html)

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

    # Create a canal anteriorline shifted toward the posterior tip by finding the middle voxels in x and the maximum voxels in y for each z index
    mask_min_x_indices = np.min(indices[0], where=mask_canal, initial=np.iinfo(indices.dtype).max, keepdims=True, axis=(0, 1))
    mask_max_x_indices = np.max(indices[0], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_mid_x = indices[0] == ((mask_min_x_indices + mask_max_x_indices) // 2)
    mask_max_y_indices = np.max(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_max_y = indices[1] == mask_max_y_indices
    mask_canal_anteriorline = mask_canal * mask_mid_x * mask_max_y

    # Get the indices of the canal anteriorline
    canal_anteriorline_indices = np.array(np.nonzero(mask_canal_anteriorline)).T

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

    # Calculate the distance of each disc voxel to each canal anteriorline voxel
    discs_distances_from_all_anteriorline = np.linalg.norm(discs_indices[:, None, :] - canal_anteriorline_indices[None, ...], axis=2)

    # Find the minimum distance for each disc voxel and the corresponding canal anteriorline index
    discs_distance_from_anteriorline = np.min(discs_distances_from_all_anteriorline, axis=1)
    discs_anteriorline_indices = canal_anteriorline_indices[np.argmin(discs_distances_from_all_anteriorline, axis=1)]

    # Find the closest voxel to the canal anteriorline for each disc label (posterior tip)
    disc_labels_anteriorline_indices = [discs_anteriorline_indices[discs_indices_labels == label][np.argmin(discs_distance_from_anteriorline[discs_indices_labels == label])] for label in disc_labels_in_seg]

    # Set the output labels to the closest voxel to the canal anteriorline for each disc
    for idx, label in zip(disc_labels_anteriorline_indices, disc_labels_in_seg):
        output_seg_data[tuple(idx)] = map_labels[label]

    # If C2-C3 and C1 are in the segmentation, set 1 and 2
    if 3 in output_seg_data and c2_label != 0 and c1_label != 0 and all(np.isin([c1_label, c2_label], seg_data)):
        # Place 1 at the top of C2 if C1 is visible in the image
        # Find the location of the C2-C3 disc
        c2c3_index = np.unravel_index(np.argmax(output_seg_data == 3), seg_data.shape)

        # Find the maximum coordinate of the vertebra C1
        c1_coords = np.where(seg_data == c1_label)
        c1_z_max_index = np.max(c1_coords[2])

        # Extract coordinate of the vertebrae
        # The coordinate of 1 needs to be in the same slice as 3 but below the max index of C1
        vert_coords = np.where(seg_data[c2c3_index[0],:,:c1_z_max_index] == c2_label)

        # Check if not empty
        if len(vert_coords[1]) > 0:
            # Find top pixel of the vertebrae
            argmax_z = np.argmax(vert_coords[1])
            top_vert_voxel = tuple([c2c3_index[0]]+[vert_coords[i][argmax_z] for i in range(2)])

            # Set 1 to the superior voxels and project onto the anterior line
            top_vert_distances_from_all_anteriorline = np.linalg.norm(top_vert_voxel - canal_anteriorline_indices[None, ...], axis=2)
            top_vert_index_anteriorline = canal_anteriorline_indices[np.argmin(top_vert_distances_from_all_anteriorline, axis=1)]
            output_seg_data[tuple(top_vert_index_anteriorline[0])] = 1

            # Set 2 to the middle voxels between C2-C3 and the superior voxels
            c1c2_index = tuple([(top_vert_voxel[i] + c2c3_index[i]) // 2 for i in range(3)])

            # Project 2 on the anterior line
            c1c2_distances_from_all_anteriorline = np.linalg.norm(c1c2_index - canal_anteriorline_indices[None, ...], axis=2)
            c1c2_index_anteriorline = canal_anteriorline_indices[np.argmin(c1c2_distances_from_all_anteriorline, axis=1)]
            output_seg_data[tuple(c1c2_index_anteriorline[0])] = 2

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

if __name__ == '__main__':
    main()
