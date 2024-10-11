import sys, argparse, textwrap
import scipy.ndimage as ndi
from pathlib import Path
import numpy as np
import nibabel as nib
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import torchio as tio
import warnings

warnings.filterwarnings("ignore")

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            Label Vertebrae, IVDs, Spinal Cord and canal from init segmentation.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            iterative_label -s labels_init -o labels --selected-disc-landmarks 2 5 3 4 --disc-labels 1-5 --disc-landmark-labels 2 3 4 5 --disc-landmark-output-labels 63 71 91 100 --canal-labels 7 8 --canal-output-label 2 --cord-labels 9 --cord-output-label 1 -r
            iterative_label -s labels_init -o labels --selected-disc-landmarks 2 5 3 4 --disc-labels 1-7 --disc-landmark-labels 4 5 6 7 --disc-landmark-output-labels 63 71 91 100 --vertebrae-labels 9-14 --vertebrae-landmark-output-labels 13 21 41 50 --vertebrae-extra-labels 8 --canal-labels 15 16 --canal-output-label 2 --cord-labels 17 --cord-output-label 1 --sacrum-labels 14 --sacrum-output-label 50 -r
            iterative_label -s labels_init -o labels -l localizers --selected-disc-landmarks 2 5 --disc-labels 1-5 --disc-landmark-labels 2 3 4 5 --disc-landmark-output-labels 63 71 91 100 --canal-labels 7 8 --canal-output-label 2 --cord-labels 9 --cord-output-label 1 --loc-disc-labels 63-100 -r
            iterative_label -s labels_init -o labels -l localizers --selected-disc-landmarks 4 7 --disc-labels 1-7 --disc-landmark-labels 4 5 6 7 --disc-landmark-output-labels 63 71 91 100 --vertebrae-labels 9-14 --vertebrae-landmark-output-labels 13 21 41 50 --vertebrae-extra-labels 8 --canal-labels 15 16 --canal-output-label 2 --cord-labels 17 --cord-output-label 1 --sacrum-labels 14 --sacrum-output-label 50 --loc-disc-labels 63-100 -r
            For BIDS:
            iterative_label -s derivatives/labels -o derivatives/labels --seg-suffix "_seg" --output-seg-suffix "_seg_seq" -p "sub-*/anat/" --selected-disc-landmarks 2 5 3 4 --disc-labels 1-5 --disc-landmark-labels 2 3 4 5 --disc-landmark-output-labels 63 71 91 100 --canal-labels 7 8 --canal-output-label 2 --cord-labels 9 --cord-output-label 1 -r
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
        '--locs-dir', '-l', type=Path, default=None,
        help=' '.join(f'''
            Folder containing localizers' segmentations to help the labeling if landmarks not found, Optional.
            The algorithm will use the localizers' segmentations to detect the matching vertebrae and discs. The localizer and the image must be aligned.
            Matching will based on the majority of the voxels of the first vertebra or disc in the localizer, that intersect with the input segmentation.
        '''.split())
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
        '--loc-suffix', type=str, default='',
        help='Localizer suffix, defaults to "".'
    )
    parser.add_argument(
        '--selected-disc-landmarks', type=int, nargs='+', default=[],
        help='Discs labels to use as landmarks from the --disc-landmark-labels.'
    )
    parser.add_argument(
        '--disc-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='The disc labels.'
    )
    parser.add_argument(
        '--disc-landmark-labels', type=int, nargs=4,
        help='All disc labels that can be used as a landmark: C2C3, C7T1, T12L1 and L5S1.'
    )
    parser.add_argument(
        '--disc-landmark-output-labels', type=int, nargs=4,
        help='List of output labels for discs C2C3, C7T1, T12L1 and L5S1.'
    )
    parser.add_argument(
        '--disc-output-step', type=int, default=1,
        help='The step to take between disc labels in the output, defaults to 1.'
    )
    parser.add_argument(
        '--vertebrae-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='The vertebrae labels.'
    )
    parser.add_argument(
        '--vertebrae-landmark-output-labels', type=int, nargs=4,
        help='List of output labels for vertebrae C3, T1, L1, Sacrum.'
    )
    parser.add_argument(
        '--vertebrae-output-step', type=int, default=1,
        help='The step to take between vertebrae labels in the output, defaults to 1.'
    )
    parser.add_argument(
        '--vertebrae-extra-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='Extra vertebrae labels to add to adjacent vertebrae labels.'
    )
    parser.add_argument(
        '--region-max-sizes', type=int, nargs=4, default=[5, 12, 6, 1],
        help='The maximum number of discs/vertebrae for each region (Cervical from C3, Thoracic, Lumbar, Sacrum), defaults to [5, 12, 6, 1].'
    )
    parser.add_argument(
        '--region-default-sizes', type=int, nargs=4, default=[5, 12, 5, 1],
        help='The default number of discs/vertebrae for each region (Cervical from C3, Thoracic, Lumbar, Sacrum), defaults to [5, 12, 5, 1].'
    )
    parser.add_argument(
        '--loc-disc-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='The disc labels in the localizer used for detecting first disc.'
    )
    parser.add_argument(
        '--canal-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='The canal labels in the segmentation.'
    )
    parser.add_argument(
        '--canal-output-label', type=int, default=0,
        help='Output label for the canal, defaults to 0 (Do not output).'
    )
    parser.add_argument(
        '--cord-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='The spinal cord labels in the segmentation.'
    )
    parser.add_argument(
        '--cord-output-label', type=int, default=0,
        help='Output label for the spinal cord, defaults to 0 (Do not output).'
    )
    parser.add_argument(
        '--sacrum-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='The sacrum labels in the segmentation.'
    )
    parser.add_argument(
        '--sacrum-output-label', type=int, default=0,
        help='Output label for the sacrum, defaults to 0 (Do not output).'
    )
    parser.add_argument(
        '--map-input', type=str, nargs='+', default=[],
        help=' '.join(f'''
            A dict mapping labels from input into the output segmentation.
            The format should be input_label:output_label without any spaces.
            For example, 7:11 to map the input C1 label to 11 in output segmentation.
        '''.split())
    )
    parser.add_argument(
        '--dilation-size', type=int, default=1,
        help='Number of voxels to dilate before finding connected voxels to label, defaults to 1 (No dilation).'
    )
    # This argument is not used in the current implementation in inference script
    parser.add_argument(
        '--default-superior-disc', type=int, default=0,
        help='Default superior disc label if no init label found, defaults to 0 (Raise error if init label not found).'
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
    locs_path = args.locs_dir
    prefix = args.prefix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    loc_suffix = args.loc_suffix
    selected_disc_landmarks = args.selected_disc_landmarks
    disc_labels = [l for raw in args.disc_labels for l in (raw if isinstance(raw, list) else [raw])]
    disc_landmark_labels = args.disc_landmark_labels
    disc_landmark_output_labels = args.disc_landmark_output_labels
    disc_output_step = args.disc_output_step
    vertebrae_labels = [l for raw in args.vertebrae_labels for l in (raw if isinstance(raw, list) else [raw])]
    vertebrae_landmark_output_labels = args.vertebrae_landmark_output_labels
    vertebrae_output_step = args.vertebrae_output_step
    vertebrae_extra_labels = [l for raw in args.vertebrae_extra_labels for l in (raw if isinstance(raw, list) else [raw])]
    region_max_sizes = args.region_max_sizes
    region_default_sizes = args.region_default_sizes
    loc_disc_labels = [l for raw in args.loc_disc_labels for l in (raw if isinstance(raw, list) else [raw])]
    canal_labels = [l for raw in args.canal_labels for l in (raw if isinstance(raw, list) else [raw])]
    canal_output_label = args.canal_output_label
    cord_labels = [l for raw in args.cord_labels for l in (raw if isinstance(raw, list) else [raw])]
    cord_output_label = args.cord_output_label
    sacrum_labels = [l for raw in args.sacrum_labels for l in (raw if isinstance(raw, list) else [raw])]
    sacrum_output_label = args.sacrum_output_label
    map_input_list = args.map_input
    dilation_size = args.dilation_size
    default_superior_disc = args.default_superior_disc
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            segs_dir = "{segs_path}"
            output_segs_dir = "{output_segs_path}"
            locs_dir = "{locs_path}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            loc_suffix = "{loc_suffix}"
            selected_disc_landmarks = {selected_disc_landmarks}
            disc_labels = {disc_labels}
            disc_landmark_labels = {disc_landmark_labels}
            disc_landmark_output_labels = {disc_landmark_output_labels}
            disc_output_step = {disc_output_step}
            vertebrae_labels = {vertebrae_labels}
            vertebrae_landmark_output_labels = {vertebrae_landmark_output_labels}
            vertebrae_output_step = {vertebrae_output_step}
            vertebrae_extra_labels = {vertebrae_extra_labels}
            region_max_sizes = {region_max_sizes}
            region_default_sizes = {region_default_sizes}
            loc_disc_labels = {loc_disc_labels}
            canal_labels = {canal_labels}
            canal_output_label = {canal_output_label}
            cord_labels = {cord_labels}
            cord_output_label = {cord_output_label}
            sacrum_labels = {sacrum_labels}
            sacrum_output_label = {sacrum_output_label}
            map_input = {map_input_list}
            dilation_size = {dilation_size}
            default_superior_disc = {default_superior_disc}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    # Load maps into a dict
    try:
        map_input_dict = {int(l_in): int(l_out) for l_in, l_out in map(lambda x:x.split(':'), map_input_list)}
    except:
        raise ValueError("Input param map_input is not in the right structure. Make sure it is in the right format, e.g., 1:2 3:5")

    iterative_label_mp(
        segs_path=segs_path,
        output_segs_path=output_segs_path,
        locs_path=locs_path,
        prefix=prefix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        loc_suffix=loc_suffix,
        selected_disc_landmarks=selected_disc_landmarks,
        disc_labels=disc_labels,
        disc_landmark_labels=disc_landmark_labels,
        disc_landmark_output_labels=disc_landmark_output_labels,
        disc_output_step=disc_output_step,
        vertebrae_labels=vertebrae_labels,
        vertebrae_landmark_output_labels=vertebrae_landmark_output_labels,
        vertebrae_output_step=vertebrae_output_step,
        vertebrae_extra_labels=vertebrae_extra_labels,
        region_max_sizes=region_max_sizes,
        region_default_sizes=region_default_sizes,
        loc_disc_labels=loc_disc_labels,
        canal_labels=canal_labels,
        canal_output_label=canal_output_label,
        cord_labels=cord_labels,
        cord_output_label=cord_output_label,
        sacrum_labels=sacrum_labels,
        sacrum_output_label=sacrum_output_label,
        map_input_dict=map_input_dict,
        dilation_size=dilation_size,
        default_superior_disc=default_superior_disc,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def iterative_label_mp(
        segs_path,
        output_segs_path,
        locs_path=None,
        prefix='',
        seg_suffix='',
        output_seg_suffix='',
        loc_suffix='',
        selected_disc_landmarks=[],
        disc_labels=[],
        disc_landmark_labels=[],
        disc_landmark_output_labels=[],
        disc_output_step=1,
        vertebrae_labels=[],
        vertebrae_landmark_output_labels=[],
        vertebrae_output_step=1,
        vertebrae_extra_labels=[],
        region_max_sizes=[5, 12, 6, 1],
        region_default_sizes=[5, 12, 5, 1],
        loc_disc_labels=[],
        canal_labels=[],
        canal_output_label=0,
        cord_labels=[],
        cord_output_label=0,
        sacrum_labels=[],
        sacrum_output_label=0,
        map_input_dict={},
        dilation_size=1,
        default_superior_disc=0,
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    segs_path = Path(segs_path)
    output_segs_path = Path(output_segs_path)
    locs_path = locs_path and Path(locs_path)

    glob_pattern = f'{prefix}*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    seg_path_list = list(segs_path.glob(glob_pattern))
    output_seg_path_list = [output_segs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz') for _ in seg_path_list]
    loc_path_list = [locs_path and locs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{loc_suffix}.nii.gz') for _ in seg_path_list]

    process_map(
        partial(
            _iterative_label,
            selected_disc_landmarks=selected_disc_landmarks,
            disc_labels=disc_labels,
            disc_landmark_labels=disc_landmark_labels,
            disc_landmark_output_labels=disc_landmark_output_labels,
            disc_output_step=disc_output_step,
            vertebrae_labels=vertebrae_labels,
            vertebrae_landmark_output_labels=vertebrae_landmark_output_labels,
            vertebrae_output_step=vertebrae_output_step,
            vertebrae_extra_labels=vertebrae_extra_labels,
            region_max_sizes=region_max_sizes,
            region_default_sizes=region_default_sizes,
            loc_disc_labels=loc_disc_labels,
            canal_labels=canal_labels,
            canal_output_label=canal_output_label,
            cord_labels=cord_labels,
            cord_output_label=cord_output_label,
            sacrum_labels=sacrum_labels,
            sacrum_output_label=sacrum_output_label,
            map_input_dict=map_input_dict,
            dilation_size=dilation_size,
            default_superior_disc=default_superior_disc,
            overwrite=overwrite,
        ),
        seg_path_list,
        output_seg_path_list,
        loc_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _iterative_label(
        seg_path,
        output_seg_path,
        loc_path=None,
        selected_disc_landmarks=[],
        disc_labels=[],
        disc_landmark_labels=[],
        disc_landmark_output_labels=[],
        disc_output_step=1,
        vertebrae_labels=[],
        vertebrae_landmark_output_labels=[],
        vertebrae_output_step=1,
        vertebrae_extra_labels=[],
        region_max_sizes=[5, 12, 6, 1],
        region_default_sizes=[5, 12, 5, 1],
        loc_disc_labels=[],
        canal_labels=[],
        canal_output_label=0,
        cord_labels=[],
        cord_output_label=0,
        sacrum_labels=[],
        sacrum_output_label=0,
        map_input_dict={},
        dilation_size=1,
        default_superior_disc=0,
        overwrite=False,
    ):
    '''
    Wrapper function to handle IO.
    '''
    seg_path = Path(seg_path)
    output_seg_path = Path(output_seg_path)
    loc_path = loc_path and Path(loc_path)

    # If the output image already exists and we are not overriding it, return
    if not overwrite and output_seg_path.exists():
        return

    # Load segmentation and localizer
    seg = nib.load(seg_path)
    loc = loc_path and (loc_path.is_file() or None) and nib.load(loc_path)

    try:
        output_seg = iterative_label(
            seg,
            loc,
            selected_disc_landmarks=selected_disc_landmarks,
            disc_labels=disc_labels,
            disc_landmark_labels=disc_landmark_labels,
            disc_landmark_output_labels=disc_landmark_output_labels,
            disc_output_step=disc_output_step,
            vertebrae_labels=vertebrae_labels,
            vertebrae_landmark_output_labels=vertebrae_landmark_output_labels,
            vertebrae_output_step=vertebrae_output_step,
            vertebrae_extra_labels=vertebrae_extra_labels,
            region_max_sizes=region_max_sizes,
            region_default_sizes=region_default_sizes,
            loc_disc_labels=loc_disc_labels,
            canal_labels=canal_labels,
            canal_output_label=canal_output_label,
            cord_labels=cord_labels,
            cord_output_label=cord_output_label,
            sacrum_labels=sacrum_labels,
            sacrum_output_label=sacrum_output_label,
            map_input_dict=map_input_dict,
            dilation_size=dilation_size,
            disc_default_superior_output=default_superior_disc,
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

def iterative_label(
        seg,
        loc=None,
        selected_disc_landmarks=[],
        disc_labels=[],
        disc_landmark_labels=[],
        disc_landmark_output_labels=[],
        disc_output_step=1,
        vertebrae_labels=[],
        vertebrae_landmark_output_labels=[],
        vertebrae_output_step=1,
        vertebrae_extra_labels=[],
        region_max_sizes=[5, 12, 6, 1],
        region_default_sizes=[5, 12, 5, 1],
        loc_disc_labels=[],
        canal_labels=[],
        canal_output_label=0,
        cord_labels=[],
        cord_output_label=0,
        sacrum_labels=[],
        sacrum_output_label=0,
        map_input_dict={},
        dilation_size=1,
        disc_default_superior_output=0,
    ):
    '''
    Label Vertebrae, IVDs, Spinal Cord and canal from init segmentation.
    The algorithm is iterative and works as follows:
    1. Find connected voxels for each disc label and label them into separate labels
    2. Find connected voxels for each vertebrae label and label them into separate labels
    3. Combine sequential vertebrae labels based on some conditions
    4. Combine extra labels with adjacent vertebrae labels
    5. Find the landmark disc labels and output labels
    6. Label the discs with the output labels
    7. Find the matching vertebrae labels to the discs landmarks
    8. Label the vertebrae with the output labels
    9. Map input labels to the final output (e.g., map the input sacrum, canal and spinal cord labels to the output labels)

    Parameters
    ----------
    seg : nibabel.nifti1.Nifti1Image
        Segmentation image
    loc : nibabel.nifti1.Nifti1Image
        Localizer image to use for detecting first vertebrae and disc (optional)
    selected_disc_landmarks : list
        Discs labels to use as landmarks from the disc_landmark_labels
    disc_labels : list
        The disc labels in the segmentation
    disc_landmark_labels : list
        All disc labels that can be used as a landmark: [C2C3, C7T1, T12L1, L5S1]
    disc_landmark_output_labels : list
        List of output labels for discs [C2C3, C7T1, T12L1, L5S1]
    disc_output_step : int
        The step to take between disc labels in the output
    vertebrae_labels : list
        The vertebrae labels in the segmentation
    vertebrae_landmark_output_labels : list
        List of output labels for vertebrae [C3, T1, L1, Sacrum]
    vertebrae_output_step : int
        The step to take between vertebrae labels in the output
    vertebrae_extra_labels : list
        Extra vertebrae labels to add to adjacent vertebrae labels
    region_max_sizes : list
        The maximum number of discs/vertebrae for each region (Cervical from C3, Thoracic, Lumbar, Sacrum).
    region_default_sizes : list
        The default number of discs/vertebrae for each region (Cervical from C3, Thoracic, Lumbar, Sacrum).
    loc_disc_labels : list
        Localizer labels to use for detecting first disc
    canal_labels : list
        Canal labels in the segmentation
    canal_output_label : int
        Output label for the canal
    cord_labels : list
        Spinal Cord labels in the segmentation
    cord_output_label : int
        Output label for the spinal cord
    sacrum_labels : list
        Sacrum labels in the segmentation
    sacrum_output_label : int
        Output label for the sacrum
    map_input_dict : dict
        A dict mapping labels from input into the output segmentation
    dilation_size : int
        Number of voxels to dilate before finding connected voxels to label
    default_superior_disc : int
        Default superior disc label if no init label found

    Returns
    -------
    nibabel.nifti1.Nifti1Image
        Segmentation image with labeled vertebrae, IVDs, Spinal Cord and canal
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    output_seg_data = np.zeros_like(seg_data)

    # Get the canal centerline indices to use for sorting the discs and vertebrae based on the prjection on the canal centerline
    canal_centerline_indices = _get_canal_centerline_indices(seg_data, canal_labels + cord_labels)

    # Get the mask of the voxels anterior to the canal, this helps in sorting the vertebrae considering only the vertebrae body
    mask_aterior_to_canal = _get_mask_aterior_to_canal(seg_data, canal_labels + cord_labels)

    # Get sorted connected components superio-inferior (SI) for the disc labels
    disc_mask_labeled, disc_num_labels, disc_sorted_labels, disc_sorted_z_indices = _get_si_sorted_components(
        seg,
        disc_labels,
        canal_centerline_indices,
        mask_aterior_to_canal,
        dilation_size,
        combine_labels=True,
    )

    # Get sorted connected components superio-inferior (SI) for the vertebrae labels
    vert_mask_labeled, vert_num_labels, vert_sorted_labels, vert_sorted_z_indices = _get_si_sorted_components(
        seg,
        vertebrae_labels,
        canal_centerline_indices,
        mask_aterior_to_canal,
        dilation_size,
    )

    # Combine sequential vertebrae labels if they have the same value in the original segmentation
    vert_mask_labeled, vert_num_labels, vert_sorted_labels, vert_sorted_z_indices = _merge_vertebrae_with_same_label(
        seg,
        vertebrae_labels,
        vert_mask_labeled,
        vert_num_labels,
        vert_sorted_labels,
        vert_sorted_z_indices,
        canal_centerline_indices,
        mask_aterior_to_canal,
    )

    # Combine extra labels with adjacent vertebrae labels
    vert_mask_labeled, vert_num_labels, vert_sorted_labels, vert_sorted_z_indices = _merge_extra_labels_with_adjacent_vertebrae(
        seg,
        vert_mask_labeled,
        vert_num_labels,
        vert_sorted_labels,
        vert_sorted_z_indices,
        vertebrae_extra_labels,
        canal_centerline_indices,
        mask_aterior_to_canal,
    )

    # Get the landmark disc labels and output labels - {label in sorted labels: output label}
    # TODO Currently only the first 2 landmark from selected_disc_landmarks is used, to get all landmarks see TODO in the function
    map_disc_sorted_labels_landmark2output = _get_landmark_output_labels(
        seg,
        loc,
        disc_mask_labeled,
        disc_sorted_labels,
        selected_disc_landmarks,
        disc_landmark_labels,
        disc_landmark_output_labels,
        loc_disc_labels,
        disc_default_superior_output,
    )

    # Build a list containing all possible labels for the disc ordered superio-inferior
    all_possible_disc_output_labels = []
    for l, s in zip(disc_landmark_output_labels, region_max_sizes):
        for i in range(s):
            all_possible_disc_output_labels.append(l + i * disc_output_step)

    # Make a list containing all possible labels for the disc ordered superio-inferior with the default region sizes
    all_default_disc_output_labels = []
    for l, s in zip(disc_landmark_output_labels, region_default_sizes):
        for i in range(s):
            all_default_disc_output_labels.append(l + i * disc_output_step)

    # Make a dict mapping the sorted disc labels to the output labels
    map_disc_sorted_labels_2output = {}

    # We loop over all the landmarks starting from the most superior
    for l_disc in [_ for _ in disc_sorted_labels if _ in map_disc_sorted_labels_landmark2output]:

        # If this is the most superior landmark, we have to adjust the start indices to start from the most superior label in the image
        if len(map_disc_sorted_labels_2output) == 0:
            # Get the index of the current landmark in the sorted disc labels
            start_l = disc_sorted_labels.index(l_disc)

            # Get the index of the current landmark in the list of all default disc output labels
            start_o_def = all_default_disc_output_labels.index(map_disc_sorted_labels_landmark2output[l_disc])

            # Adjust the start indices
            start_l, start_o_def = max(0, start_l - start_o_def), max(0, start_o_def - start_l)

            # Map the sorted disc labels to the output labels
            for l, o in zip(disc_sorted_labels[start_l:], all_default_disc_output_labels[start_o_def:]):
                map_disc_sorted_labels_2output[l] = o

        # Get the index of the current landmark in the sorted disc labels
        start_l = disc_sorted_labels.index(l_disc)

        # Get the index of the current landmark in the list of all possible disc output labels
        start_o = all_possible_disc_output_labels.index(map_disc_sorted_labels_landmark2output[l_disc])

        # Map the sorted disc labels to the output labels
        # This will ovveride the mapping from the previous landmarks for all labels inferior to the current landmark
        for l, o in zip(disc_sorted_labels[start_l:], all_possible_disc_output_labels[start_o:]):
            map_disc_sorted_labels_2output[l] = o

    # Label the discs with the output labels superio-inferior
    for l, o in map_disc_sorted_labels_2output.items():
        output_seg_data[disc_mask_labeled == l] = o

    if vert_num_labels > 0:
        # Build a list containing all possible labels for the vertebrae ordered superio-inferior
        # We start with the C1 and C2 labels as the first landmark is the C3 vertebrae
        all_possible_vertebrae_output_labels = [
            vertebrae_landmark_output_labels[0] - 2 * vertebrae_output_step, # C1
            vertebrae_landmark_output_labels[0] - vertebrae_output_step # C2
        ]
        for l, s in zip(vertebrae_landmark_output_labels, region_max_sizes):
            for i in range(s):
                all_possible_vertebrae_output_labels.append(l + i * vertebrae_output_step)

        # Make a list containing all possible labels for the vertebrae ordered superio-inferior with the default region sizes
        all_default_vertebrae_output_labels = [
            vertebrae_landmark_output_labels[0] - 2 * vertebrae_output_step, # C1
            vertebrae_landmark_output_labels[0] - vertebrae_output_step # C2
        ]
        for l, s in zip(vertebrae_landmark_output_labels, region_default_sizes):
            for i in range(s):
                all_default_vertebrae_output_labels.append(l + i * vertebrae_output_step)

        # Sort the combined disc+vert labels by their z-index
        sorted_labels = vert_sorted_labels + disc_sorted_labels
        sorted_z_indices = vert_sorted_z_indices + disc_sorted_z_indices
        is_vert = [True] * len(vert_sorted_labels) + [False] * len(disc_sorted_labels)

        # Sort the labels by their z-index (reversed to go from superior to inferior)
        sorted_z_indices, sorted_labels, is_vert = zip(*sorted(zip(sorted_z_indices, sorted_labels, is_vert))[::-1])
        sorted_z_indices, sorted_labels, is_vert = list(sorted_z_indices), list(sorted_labels), list(is_vert)

        # Look for two discs with no vertebrae between them, if found, look if there is there is 2 vertebrae without a disc between them just next to the discs and switch the second disk with the first vertebrae, if not found, look if there is there is 2 vertebrae without a disc between them just above to the discs and switch the first disk with the second vertebrae.
        # This is useful in cases where only the spinous processes is segmented in the last vertebrae and the disc is not segmented
        for idx in range(len(sorted_labels) - 1):
            # Cehck if this is two discs without a vertebrae between them
            if not is_vert[idx] and not is_vert[idx + 1]:
                # Check if there is two vertebrae without a disc between them just next to the discs
                if idx < len(sorted_labels) - 3 and is_vert[idx + 2] and is_vert[idx + 3]:
                    sorted_labels[idx + 1], sorted_labels[idx + 2] = sorted_labels[idx + 2], sorted_labels[idx + 1]
                    sorted_z_indices[idx + 1], sorted_z_indices[idx + 2] = sorted_z_indices[idx + 2], sorted_z_indices[idx + 1]
                    is_vert[idx + 1], is_vert[idx + 2] = is_vert[idx + 2], is_vert[idx + 1]

                # Check if there is two vertebrae without a disc between them just above to the discs
                elif idx > 1 and is_vert[idx - 1] and is_vert[idx - 2]:
                    sorted_labels[idx], sorted_labels[idx - 1] = sorted_labels[idx - 1], sorted_labels[idx]
                    sorted_z_indices[idx], sorted_z_indices[idx - 1] = sorted_z_indices[idx - 1], sorted_z_indices[idx]
                    is_vert[idx], is_vert[idx - 1] = is_vert[idx - 1], is_vert[idx]

        # If there is a disc at the top and 2 vertebrae below it, switch the disc with the first vertebrae
        if len(sorted_labels) > 2 and not is_vert[0] and is_vert[1] and is_vert[2]:
            sorted_labels[0], sorted_labels[1] = sorted_labels[1], sorted_labels[0]
            sorted_z_indices[0], sorted_z_indices[1] = sorted_z_indices[1], sorted_z_indices[0]
            is_vert[0], is_vert[1] = is_vert[1], is_vert[0]

        # If there is a disc at the bottom and 2 vertebrae above it, switch the disc with the second vertebrae
        if len(sorted_labels) > 2 and not is_vert[-1] and is_vert[-2] and is_vert[-3]:
            sorted_labels[-1], sorted_labels[-2] = sorted_labels[-2], sorted_labels[-1]
            sorted_z_indices[-1], sorted_z_indices[-2] = sorted_z_indices[-2], sorted_z_indices[-1]
            is_vert[-1], is_vert[-2] = is_vert[-2], is_vert[-1]

        # Make a dict mapping disc to vertebrae labels
        disc_output_labels_2vert = dict(zip(all_possible_disc_output_labels, all_possible_vertebrae_output_labels[2:]))

        # Make a dict mapping the sorted vertebrae labels to the output labels
        map_vert_sorted_labels_2output = {}

        l_vert_output = 0
        # We loop over all the labels starting from the most superior, and we map the vertebrae labels to the output labels
        for idx, curr_l, curr_is_vert in zip(range(len(sorted_labels)), sorted_labels, is_vert):
            if not curr_is_vert: # This is a disc
                # If the current disc is not in the map, continue
                if curr_l not in map_disc_sorted_labels_2output:
                    continue

                # Get the output label for the disc and vertebrae
                l_disc_output = map_disc_sorted_labels_2output[curr_l]
                l_vert_output = disc_output_labels_2vert[l_disc_output]

                if idx > 0 and len(map_vert_sorted_labels_2output) == 0: # This is the first disc
                    # Get the index of the current vertebrae in the default vertebrae output labels list
                    i = all_default_vertebrae_output_labels.index(l_vert_output)

                    # Get the labels of the vertebrae superior to the current disc
                    prev_vert_ls = [l for l, _is_v in zip(sorted_labels[idx - 1::-1], is_vert[idx - 1::-1]) if _is_v]

                    # Map all the vertebrae superior to the current disc to the default vertebrae output labels
                    for l, o in zip(prev_vert_ls, all_default_vertebrae_output_labels[i - 1::-1]):
                        map_vert_sorted_labels_2output[l] = o

            elif l_vert_output > 0: # This is a vertebrae
                # If this is the last vertebrae and no disc btween it and the prev vertebrae, map it to the next vertebrae output label
                # This is useful in cases where only the spinous processes is segmented in the last vertebrae and the disc is not segmented
                if idx == len(sorted_labels) - 1 and idx > 0 and is_vert[idx - 1] and l_vert_output != all_possible_vertebrae_output_labels[-1]:
                    map_vert_sorted_labels_2output[curr_l] = all_possible_vertebrae_output_labels[all_possible_vertebrae_output_labels.index(l_vert_output) + 1]
                else:
                    map_vert_sorted_labels_2output[curr_l] = l_vert_output

        # Label the vertebrae with the output labels superio-inferior
        for l, o in map_vert_sorted_labels_2output.items():
            output_seg_data[vert_mask_labeled == l] = o

    # Map Spinal Canal to the output label
    if canal_labels is not None and len(canal_labels) > 0 and canal_output_label > 0:
        output_seg_data[np.isin(seg_data, canal_labels)] = canal_output_label

    # Map Spinal Cord to the output label
    if cord_labels is not None and len(cord_labels) > 0 and cord_output_label > 0:
        output_seg_data[np.isin(seg_data, cord_labels)] = cord_output_label

    # Map Sacrum to the output label
    if sacrum_labels is not None and len(sacrum_labels) > 0 and sacrum_output_label > 0:
        output_seg_data[np.isin(seg_data, sacrum_labels)] = sacrum_output_label

    # Use the map to map input labels to the final output
    # This is useful to map the input C1 to the output.
    for orig, new in map_input_dict.items():
        output_seg_data[seg_data == int(orig)] = int(new)

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

def _get_canal_centerline_indices(
        seg_data,
        canal_labels=[],
    ):
    '''
    Get the indices of the canal centerline.
    '''
    # If no canal labels is found in the segmentation, raise an error
    if not np.any(np.isin(canal_labels, seg_data)):
        raise ValueError(f"No canal found in the segmentation (canal labels: {canal_labels})")

    # Get array of indices for x, y, and z axes
    indices = np.indices(seg_data.shape)

    # Create a mask of the canal
    mask_canal = np.isin(seg_data, canal_labels)

    # Create a mask the canal centerline by finding the middle voxels in x and y axes for each z index
    mask_min_x_indices = np.min(indices[0], where=mask_canal, initial=np.iinfo(indices.dtype).max, keepdims=True, axis=(0, 1))
    mask_max_x_indices = np.max(indices[0], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_mid_x = indices[0] == ((mask_min_x_indices + mask_max_x_indices) // 2)
    mask_min_y_indices = np.min(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).max, keepdims=True, axis=(0, 1))
    mask_max_y_indices = np.max(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_mid_y = indices[1] == ((mask_min_y_indices + mask_max_y_indices) // 2)
    mask_canal_centerline = mask_canal * mask_mid_x * mask_mid_y

    # Get the indices of the canal centerline
    return np.array(np.nonzero(mask_canal_centerline)).T

def _sort_labels_si(
        mask_labeled,
        labels,
        canal_centerline_indices,
        mask_aterior_to_canal=None,
    ):
    '''
    Sort the labels by their z-index (reversed to go from superior to inferior).
    '''
    # Get the indices of the center of mass for each label
    labels_indices = np.array(ndi.center_of_mass(np.isin(mask_labeled, labels), mask_labeled, labels))

    # Get the distance of each label indices from the canal centerline
    labels_distances_from_centerline = np.linalg.norm(labels_indices[:, None, :] - canal_centerline_indices[None, ...],axis=2)

    # Get the z-index of the closest canal centerline voxel for each label
    labels_z_indices = canal_centerline_indices[np.argmin(labels_distances_from_centerline, axis=1), -1]

    # If mask_aterior_to_canal is provided, calculate the center of mass in this mask if the label is inside the mask
    if mask_aterior_to_canal is not None:
        # Save the existing labels z-index in a dict
        labels_z_indices_dict = dict(zip(labels, labels_z_indices))

        # Get the part that is anterior to the canal od mask_labeled
        mask_labeled_aterior_to_canal = mask_aterior_to_canal * mask_labeled

        # Get the labels that contain voxels anterior to the canal
        labels_masked = np.isin(labels, mask_labeled_aterior_to_canal)

        # Get the indices of the center of mass for each label
        labels_masked_indices = np.array(ndi.center_of_mass(np.isin(mask_labeled_aterior_to_canal, labels_masked), mask_labeled_aterior_to_canal, labels_masked))

        # Get the distance of each label indices for each voxel in the canal centerline
        labels_masked_distances_from_centerline = np.linalg.norm(labels_masked_indices[:, None, :] - canal_centerline_indices[None, :],axis=2)

        # Get the z-index of the closest canal centerline voxel for each label
        labels_masked_z_indices = canal_centerline_indices[np.argmin(labels_masked_distances_from_centerline, axis=1), -1]

        # Update the dict with the new z-index of the labels anterior to the canal
        for l, z in zip(labels_masked, labels_masked_z_indices):
            labels_z_indices_dict[l] = z

        # Update the labels_z_indices from the dict
        labels_z_indices = [labels_z_indices_dict[l] for l in labels]

    # Sort the labels by their z-index (reversed to go from superior to inferior)
    return zip(*sorted(zip(labels_z_indices, labels))[::-1])

def _get_si_sorted_components(
        seg,
        labels,
        canal_centerline_indices,
        mask_aterior_to_canal=None,
        dilation_size=1,
        combine_labels=False,
    ):
    '''
    Get sorted connected components superio-inferior (SI) for the given labels in the segmentation.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    binary_dilation_structure = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), dilation_size)

    # Skip if no labels are provided
    if len(labels) == 0:
        return None, 0, [], []

    if combine_labels:
        # For discs, combine all labels before label continue voxels since the discs not touching each other
        _labels = [labels]
    else:
        _labels = [[_] for _ in labels]

    # Init labeled segmentation
    mask_labeled, num_labels = np.zeros_like(seg_data, dtype=np.uint32), 0

    # For each label, find connected voxels and label them into separate labels
    for l in _labels:
        mask = np.isin(seg_data, l)

        # Dilate the mask to combine small disconnected regions
        mask_dilated = ndi.binary_dilation(mask, binary_dilation_structure)

        # Label the connected voxels in the dilated mask into separate labels
        tmp_mask_labeled, tmp_num_labels = ndi.label(mask_dilated.astype(np.uint32), np.ones((3, 3, 3)))

        # Undo dilation
        tmp_mask_labeled *= mask

        # Add current labels to the labeled segmentation
        if tmp_num_labels > 0:
            mask_labeled[tmp_mask_labeled != 0] = tmp_mask_labeled[tmp_mask_labeled != 0] + num_labels
            num_labels += tmp_num_labels

    # If no label found, raise error
    if num_labels == 0:
        raise ValueError(f"Some label must be in the segmentation (labels: {labels})")

    # Reduce size of mask_labeled
    if mask_labeled.max() < np.iinfo(np.uint8).max:
        mask_labeled = mask_labeled.astype(np.uint8)
    elif mask_labeled.max() < np.iinfo(np.uint16).max:
        mask_labeled = mask_labeled.astype(np.uint16)

    # Sort the labels by their z-index (reversed to go from superior to inferior)
    sorted_z_indices, sorted_labels = _sort_labels_si(
        mask_labeled, range(1,num_labels+1), canal_centerline_indices, mask_aterior_to_canal
    )
    return mask_labeled, num_labels, list(sorted_labels), list(sorted_z_indices)

def _get_mask_aterior_to_canal(
        seg_data,
        canal_labels=[],
    ):
    '''
    Get the mask of the voxels anterior to the canal.
    '''
    # Get array of indices for x, y, and z axes
    indices = np.indices(seg_data.shape)

    # Create a mask of the canal
    mask_canal = np.isin(seg_data, canal_labels)

    # Create a mask the canal centerline by finding the middle voxels in x and y axes for each z index
    mask_min_y_indices = np.min(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).max, keepdims=True, axis=(0, 1))
    mask_max_y_indices = np.max(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_mid_y_indices = (mask_min_y_indices + mask_max_y_indices) // 2

    return indices[1] > mask_mid_y_indices

def _merge_vertebrae_with_same_label(
        seg,
        labels,
        mask_labeled,
        num_labels,
        sorted_labels,
        sorted_z_indices,
        canal_centerline_indices,
        mask_aterior_to_canal=None,
    ):
    '''
    Combine sequential vertebrae labels if they have the same value in the original segmentation.
    This is useful when parts of the vertebrae are not touching in the segmentation but have the same odd/even value.
    '''
    if num_labels == 0 or len(labels) <= 1:
        return mask_labeled, num_labels, sorted_labels, sorted_z_indices

    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    new_sorted_labels = []

    # Store the previous label and the original label of the previous label
    prev_l, prev_orig_label = 0, 0

    # Loop over the sorted labels
    for l in sorted_labels:
        # Get the original label of the current label
        curr_orig_label = seg_data[mask_labeled == l].flat[0]

        # Combine the current label with the previous label if they have the same original label
        if curr_orig_label == prev_orig_label:
            # Combine the current label with the previous label
            mask_labeled[mask_labeled == l] = prev_l
            num_labels -= 1

        else:
            # Add the current label to the new sorted labels
            new_sorted_labels.append(l)
            prev_l, prev_orig_label = l, curr_orig_label

    sorted_labels = new_sorted_labels

    # Reduce size of mask_labeled
    if mask_labeled.max() < np.iinfo(np.uint8).max:
        mask_labeled = mask_labeled.astype(np.uint8)
    elif mask_labeled.max() < np.iinfo(np.uint16).max:
        mask_labeled = mask_labeled.astype(np.uint16)

    # Sort the labels by their z-index (reversed to go from superior to inferior)
    sorted_z_indices, sorted_labels = _sort_labels_si(
        mask_labeled, sorted_labels, canal_centerline_indices, mask_aterior_to_canal
    )

    return mask_labeled, num_labels, list(sorted_labels), list(sorted_z_indices)

def _merge_extra_labels_with_adjacent_vertebrae(
        seg,
        mask_labeled,
        num_labels,
        sorted_labels,
        sorted_z_indices,
        extra_labels,
        canal_centerline_indices,
        mask_aterior_to_canal,
    ):
    '''
    Combine extra labels with adjacent vertebrae labels.
    This is useful to combine segmentations of vertebrae introduced during region-based training, where the model sometimes outputs a general vertebrae label instead of the specific odd or even vertebrae.
    The process adjusts these remnants by merging them with the closest odd or even vertebrae to ensure correct segmentation.
    '''
    if num_labels == 0 or len(extra_labels) == 0:
        return mask_labeled, num_labels, sorted_labels, sorted_z_indices

    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    mask_extra = np.isin(seg_data, extra_labels)

    # Loop over vertebral labels (from inferior because the transverse process make it steal from above)
    for i in range(num_labels - 1, -1, -1):
        # Mkae mask for the current vertebrae with filling the holes and dilating it
        mask = _fill(mask_labeled == sorted_labels[i])
        mask = ndi.binary_dilation(mask, ndi.iterate_structure(ndi.generate_binary_structure(3, 1), 1))

        # Add the intersection of the mask with the extra labels to the current verebrae
        mask_labeled[mask_extra * mask] = sorted_labels[i]

    # Sort the labels by their z-index (reversed to go from superior to inferior)
    sorted_z_indices, sorted_labels = _sort_labels_si(
        mask_labeled, sorted_labels, canal_centerline_indices, mask_aterior_to_canal
    )

    return mask_labeled, num_labels, list(sorted_labels), list(sorted_z_indices)

def _get_landmark_output_labels(
        seg,
        loc,
        mask_labeled,
        sorted_labels,
        selected_landmarks,
        landmark_labels,
        landmark_output_labels,
        loc_labels,
        default_superior_output,
    ):
    '''
    Get dict mapping labels from sorted_labels to the output labels based on the landmarks in the segmentation or localizer.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    loc_data = loc and np.asanyarray(loc.dataobj).round().astype(np.uint8)

    map_landmark_labels = dict(zip(landmark_labels, landmark_output_labels))

    # If localizer is provided, transform it to the segmentation space
    if loc_data is not None:
        loc_data = tio.Resample(
            tio.ScalarImage(tensor=seg_data[None, ...], affine=seg.affine)
        )(
            tio.LabelMap(tensor=loc_data[None, ...], affine=loc.affine)
        ).data.numpy()[0, ...].astype(np.uint8)

    # Init dict to store the output labels for the landmarks
    map_landmark_outputs = {}

    mask_seg_data_landmarks = np.isin(seg_data, landmark_labels)

    # First we try to set initial output labels from the localizer
    if loc_data is not None:
        # Make mask for the intersection of the localizer labels and the labels in the segmentation
        mask = np.isin(loc_data, loc_labels) * np.isin(mask_labeled, sorted_labels)
        mask_labeled_masked = mask * mask_labeled
        loc_data_masked = mask * loc_data

        # First we try to look for the landmarks in the localizer
        for output_label in np.array(landmark_output_labels)[np.isin(landmark_output_labels, loc_data_masked)].tolist():
            # Map the label with the most voxels in the localizer landmark to the output label
            map_landmark_outputs[np.argmax(np.bincount(mask_labeled_masked[loc_data_masked == output_label].flat))] = output_label

        if len(map_landmark_outputs) == 0:
            # Get the first label from sorted_labels that is in the localizer specified labels
            first_sorted_labels_in_loc = next(np.array(sorted_labels)[np.isin(sorted_labels, mask_labeled_masked)].flat, 0)

            if first_sorted_labels_in_loc > 0:
                # Get the output label for first_sorted_labels_in_loc, the label from the localizer that has the most voxels in it
                map_landmark_outputs[first_sorted_labels_in_loc] = np.argmax(np.bincount(loc_data_masked[mask_labeled_masked == first_sorted_labels_in_loc].flat))

    # If no output label found from the localizer, try to set the output labels from landmarks in the segmentation
    if len(map_landmark_outputs) == 0:
        for l in selected_landmarks:
            ############################################################################################################
            # TODO Remove this reake when we trust all the landmarks to get all landmarks instead of the first 2
            if len(map_landmark_outputs) > 0 and selected_landmarks.index(l) > 1:
                break
            ############################################################################################################
            if l in map_landmark_labels and l in seg_data:
                mask_labeled_l = np.argmax(np.bincount(mask_labeled[seg_data == l].flat))
                # We map only if the landmark cover the majority of the voxels in the mask_labeled label
                if np.argmax(np.bincount(seg_data[mask_seg_data_landmarks & (mask_labeled == mask_labeled_l)].flat)) == l:
                    map_landmark_outputs[mask_labeled_l] = map_landmark_labels[l]

    # If no init label found, set the default superior label
    if len(map_landmark_outputs) == 0 and default_superior_output > 0:
        map_landmark_outputs[sorted_labels[0]] = default_superior_output

    # If no init label found, print error
    if len(map_landmark_outputs) == 0:
        if loc_data is not None:
            raise ValueError(
                f"At least one of the landmarks must be in the segmentation or localizer (landmarks: {selected_landmarks}. "
                f"Check {loc_labels}), make sure the localizer is in the same space as the segmentation"
            )
        raise ValueError(f"At least one of the landmarks must be in the segmentation or localizer (landmarks: {selected_landmarks})")

    return map_landmark_outputs

def _fill(mask):
    '''
    Fill holes in a binary mask

    Parameters
    ----------
    mask : np.ndarray
        Binary mask

    Returns
    -------
    np.ndarray
        Binary mask with holes filled
    '''
    # Get array of indices for x, y, and z axes
    indices = np.indices(mask.shape)

    mask_min_x = np.min(np.where(mask, indices[0], np.inf), axis=0)[np.newaxis, ...]
    mask_max_x = np.max(np.where(mask, indices[0], -np.inf), axis=0)[np.newaxis, ...]
    mask_min_y = np.min(np.where(mask, indices[1], np.inf), axis=1)[:, np.newaxis, :]
    mask_max_y = np.max(np.where(mask, indices[1], -np.inf), axis=1)[:, np.newaxis, :]
    mask_min_z = np.min(np.where(mask, indices[2], np.inf), axis=2)[:, :, np.newaxis]
    mask_max_z = np.max(np.where(mask, indices[2], -np.inf), axis=2)[:, :, np.newaxis]

    return \
        ((mask_min_x <= indices[0]) & (indices[0] <= mask_max_x)) | \
        ((mask_min_y <= indices[1]) & (indices[1] <= mask_max_y)) | \
        ((mask_min_z <= indices[2]) & (indices[2] <= mask_max_z))

if __name__ == '__main__':
    main()