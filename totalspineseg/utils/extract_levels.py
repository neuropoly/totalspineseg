import sys, argparse, textwrap
from pathlib import Path
import numpy as np
import nibabel as nib
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import warnings
from scipy import interpolate
from totalspineseg.utils.image import Image, zeros_like

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
    seg = Image(str(seg_path))
    ori_orientation = seg.orientation

    try:
        output_seg = extract_levels(
            seg.change_orientation('LPI'),
            canal_labels=canal_labels,
            disc_labels=disc_labels,
            c1_label=c1_label,
            c2_label=c2_label,
        )
    except ValueError as e:
        output_seg_path.is_file() and output_seg_path.unlink()
        print(f'Error: {seg_path}, {e}')
        return

    # Ensure correct orientation and dtype
    output_seg.change_orientation(ori_orientation)
    output_seg.change_type(np.uint8)

    # Make sure output directory exists and save the segmentation
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    output_seg.save(str(output_seg_path))

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
    seg : Image class
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

    output_seg = zeros_like(seg)

    # Create a mask of the canal
    mask_canal = np.isin(seg.data, canal_labels)

    # If canal is empty raise an error
    if not np.any(mask_canal):
        raise ValueError(f"No canal labels found in the segmentation.")
    
    # Create canal centerline
    canal_seg = zeros_like(seg)
    canal_seg.data[mask_canal] = 1
    canal_centerline = get_centerline(canal_seg)

    # Get the labels of the discs in the segmentation
    disc_labels_in_seg = np.array(disc_labels)[np.isin(disc_labels, seg.data)]

    # If no disc labels found in the segmentation raise an error
    if len(disc_labels_in_seg) == 0:
        raise ValueError(f"No disc labels found in the segmentation.")

    # Get the labels of the discs and the output labels
    first_disk_idx = disc_labels.index(disc_labels_in_seg[0])
    out_labels = list(range(3 + first_disk_idx, 3 + first_disk_idx + len(disc_labels_in_seg)))

    # Filter the discs that are in the segmentation
    map_labels = dict(zip(disc_labels_in_seg, out_labels))

    # Create mask of the discs
    mask_discs = np.isin(seg.data, disc_labels_in_seg)

    # Get list of indices for all the discs voxels
    discs_indices = np.nonzero(mask_discs)
    
    # Get the matching labels for the discs indices
    discs_indices_labels = seg.data[discs_indices]

    # Make the discs_indices 2D array
    discs_indices = np.array(discs_indices).T

    # Calculate the distance of each disc voxel to each canal centerline voxel
    discs_distances_from_all_centerline = np.linalg.norm(discs_indices[:, None, :] - canal_centerline['position'].T[None, ...], axis=2)

    # Find the minimum distance for each disc voxel and the corresponding canal centerline index
    discs_distance_from_centerline = np.min(discs_distances_from_all_centerline, axis=1)
    
    # Find the closest voxel to the canal centerline for each disc label (posterior tip)
    disc_labels_centerline_indices = [discs_indices[discs_indices_labels == label][np.argmin(discs_distance_from_centerline[discs_indices_labels == label])] for label in disc_labels_in_seg]

    # Set the output labels to the closest voxel to the canal centerline for each disc
    for idx, label in zip(disc_labels_centerline_indices, disc_labels_in_seg):
        output_seg.data[tuple(idx)] = map_labels[label]
    
    # If C2-C3 and C1 are in the segmentation, set 1 and 2
    if 3 in output_seg.data and c2_label != 0 and c1_label != 0 and all(np.isin([c1_label, c2_label], seg.data)):
        # Place 1 at the top of C2 if C1 is visible in the image
        # Find the location of the C2-C3 disc
        c2c3_index = np.unravel_index(np.argmax(output_seg.data == 3), seg.data.shape)

        # Find the maximum coordinate of the vertebra C1
        c1_coords = np.where(seg.data == c1_label)
        c1_z_max_index = np.max(c1_coords[2])

        # Extract coordinate of the vertebrae
        # The coordinate of 1 needs to be in the same slice as 3 but below the max index of C1
        vert_coords = np.where(seg.data[c2c3_index[0],:,:c1_z_max_index] == c2_label)
        
        # Init the anterior line
        anteriorline_pos = canal_centerline['position'].T[:]
        dist = np.linalg.norm(np.array(c2c3_index)-anteriorline_pos, axis=1)
        canal_proj_c2c3_proj = anteriorline_pos[np.argmin(dist)]

        # Shift the anterior line towards C2C3 disc position
        anteriorline_pos[:, 0] += (c2c3_index[0] - canal_proj_c2c3_proj[0])
        anteriorline_pos[:, 1] += (c2c3_index[1] - canal_proj_c2c3_proj[1])
        canal_anteriorline_indices = np.round(anteriorline_pos).astype(int)

        # Check if not empty
        if len(vert_coords[1]) > 0:
            # Find top pixel of the vertebrae
            argmax_z = np.argmax(vert_coords[1])
            top_vert_voxel = tuple([c2c3_index[0]]+[vert_coords[i][argmax_z] for i in range(2)])

            # Set 1 to the superior voxels and project onto the anterior line
            top_vert_distances_from_all_anteriorline = np.linalg.norm(top_vert_voxel - canal_anteriorline_indices[None, ...], axis=2)
            top_vert_index_anteriorline = canal_anteriorline_indices[np.argmin(top_vert_distances_from_all_anteriorline, axis=1)]
            output_seg.data[tuple(top_vert_index_anteriorline[0])] = 1

            # Set 2 to the middle voxels between C2-C3 and the superior voxels
            c1c2_index = tuple([(top_vert_voxel[i] + c2c3_index[i]) // 2 for i in range(3)])

            # Project 2 on the anterior line
            c1c2_distances_from_all_anteriorline = np.linalg.norm(c1c2_index - canal_anteriorline_indices[None, ...], axis=2)
            c1c2_index_anteriorline = canal_anteriorline_indices[np.argmin(c1c2_distances_from_all_anteriorline, axis=1)]
            output_seg.data[tuple(c1c2_index_anteriorline[0])] = 2

    return output_seg

def get_centerline(seg):
    '''
    Based on https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/centerline/core.py

    Extract centerline from canal segmentation using center of mass and interpolate with bspline
    Expect orientation RPI
    '''
    arr = np.array(np.where(seg.data))
    # Loop across SI axis and average coordinates within duplicate SI values
    sorted_avg = []
    for i_si in np.unique(arr[2]):
        sorted_avg.append(arr[:, arr[2] == i_si].mean(axis=1))
    x_mean, y_mean, z_mean = np.array(sorted_avg).T
    z_ref = np.array(range(z_mean.min().astype(int), z_mean.max().astype(int) + 1))

    # Interpolate centerline
    px, py, pz = seg.dim[4:7]
    x_centerline_fit, x_centerline_deriv = bspline(z_mean, x_mean, z_ref, smooth=8000, pz=pz) # Increase smoothing...
    y_centerline_fit, y_centerline_deriv = bspline(z_mean, y_mean, z_ref, smooth=8000, pz=pz) # ...a lot

    # Construct output
    arr_ctl = np.array([x_centerline_fit, y_centerline_fit, z_ref])
    arr_ctl_der = np.array([x_centerline_deriv, y_centerline_deriv, np.ones_like(z_ref)])

    # Create centerline dictionary
    centerline = {"position": arr_ctl, "derivative": arr_ctl_der}
    return centerline

def bspline(x, y, xref, smooth, deg_bspline=3, pz=1):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/centerline/curve_fitting.py
    Bspline interpolation.

    The smoothing factor (s) is calculated based on an empirical formula (made by JCA, based on
    preliminary results) and is a function of pz, density of points and an input smoothing parameter (smooth). The
    formula is adjusted such that the parameter (smooth) produces similar smoothing results than a Hanning window with
    length smooth, as implemented in linear().

    :param x:
    :param y:
    :param xref:
    :param smooth: float: Smoothing factor. 0: no smoothing, 5: moderate smoothing, 50: large smoothing
    :param deg_bspline: int: Degree of spline
    :param pz: float: dimension of pixel along superior-inferior direction (z, assuming RPI orientation)
    :return:
    """
    if len(x) <= deg_bspline:
        deg_bspline -= 2
    density = (float(len(x)) / len(xref)) ** 2
    s = density * smooth * pz / float(3)
    # Then, run bspline interpolation
    tck = interpolate.splrep(x, y, s=s, k=deg_bspline)
    y_fit = interpolate.splev(xref, tck, der=0)
    y_fit_der = interpolate.splev(xref, tck, der=1)
    return y_fit, y_fit_der

if __name__ == '__main__':
    _extract_levels(
        seg_path='/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/sexy/test/canproco_sub-cal083_ses-M12_STIR.nii.gz',
        output_seg_path='/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/sexy/canproco_sub-cal083_ses-M12_STIR_levels.nii.gz',
        canal_labels=[1, 2],
        disc_labels=list(range(63, 68)) + list(range(71, 83)) + list(range(91, 96)) + [100],
        c1_label=11,
        c2_label=50,
        overwrite=True,
    )
    #main()
