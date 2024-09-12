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
            iterative_label -s labels_init -o labels --disc-labels 1-7 --vertebrae-labels 9-14 --vertebrae-extra-labels 8 --init-disc 4:224 7:202 5:219 6:207 --output-disc-step -1 --output-vertebrae-step -1 --map-output 17:92 --map-input 14:92 16:201 17:200 --output-c2c3 224 --output-c2 40 -r
            iterative_label -s labels_init -o labels -l localizers --disc-labels 1-7 --vertebrae-labels 9-14 --vertebrae-extra-labels 8 --init-disc 4:224 7:202 --output-disc-step -1 --output-vertebrae-step -1 --loc-disc-labels 202-224 --map-output 17:92 --map-input 14:92 16:201 17:200 --output-c2c3 224 --output-c2 40 -r
            For BIDS:
            iterative_label -s derivatives/labels -o derivatives/labels --seg-suffix "_seg" --output-seg-suffix "_seg_seq" -d "sub-" -u "anat" --disc-labels 1 2 3 4 5 6 7 --vertebrae-labels 9 10 11 12 13 14 --vertebrae-extra-labels 8 --init-disc 4:224 7:202 5:219 6:207 --output-disc-step -1 --output-vertebrae-step -1 --map-output 17:92 --map-input 14:92 16:201 17:200 --output-c2c3 224 --output-c2 40 -r
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
            Folder containing localizers segmentations to use for detecting first vertebrae and disc if init label not found, Optional.
            The algorithm will transform the localizer to the segmentation space and use it to detect the matching vertebrae and disc if the init label not found.
            Mathcing will based on the magority of the voxels of the first vertebrae or disc in the localizer, that intersect with the input segmentation.
        '''.split())
    )
    parser.add_argument(
        '--subject-dir', '-d', type=str, default=None, nargs='?', const='',
        help=' '.join(f'''
            Is every subject has its oen direcrory.
            If this argument will be provided without value it will look for any directory in the segmentation directory.
            If value also provided it will be used as a prefix to subject directory (for example "sub-"), defaults to False (no subjet directory).
        '''.split())
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
        '--loc-suffix', type=str, default='',
        help='Localizer suffix, defaults to "".'
    )
    parser.add_argument(
        '--disc-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='The disc labels.'
    )
    parser.add_argument(
        '--init-disc', type=lambda x:map(int, x.split(':')), nargs='+', default=[],
        help='Init labels list for disc ordered by priority (input_label:output_label !!without space!!). for example 4:224 5:219 6:202'
    )
    parser.add_argument(
        '--output-c2c3', type=int, default=0,
        help='The output label for C2C3, used to calculate the first vertebrae label, defaults to 0.'
    )
    parser.add_argument(
        '--output-disc-step', type=int, default=1,
        help='The step to take between disc labels in the output, defaults to 1.'
    )
    parser.add_argument(
        '--loc-disc-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='The disc labels in the localizer used for detecting first disc.'
    )
    parser.add_argument(
        '--vertebrae-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='The vertebrae labels.'
    )
    parser.add_argument(
        '--vertebrae-extra-labels', type=lambda x:list(range(int(x.split('-')[0]), int(x.split('-')[-1]) + 1)), nargs='+', default=[],
        help='Extra vertebrae labels to add to add to adjacent vertebrae labels.'
    )
    parser.add_argument(
        '--output-c2', type=int, default=0,
        help='The output label for C2, used to calculate the first vertebrae label, defaults to 0.'
    )
    parser.add_argument(
        '--output-vertebrae-step', type=int, default=1,
        help='The step to take between vertebrae labels in the output, defaults to 1.'
    )
    parser.add_argument(
        '--map-input', type=str, nargs='+', default=[],
        help=' '.join(f'''
            A dict mapping labels from input into the output segmentation.
            The format should be input_label:output_label without any spaces.
            For example, 14:92 16:201 17:200 to map the input sacrum label 14 to 92, canal label 16 to 201 and spinal cord label 17 to 200.
        '''.split())
    )
    parser.add_argument(
        '--map-output', type=str, nargs='+', default=[],
        help=' '.join(f'''
            A dict mapping labels from the output of the iterative labeling algorithm into different labels in the output segmentation.
            The format should be input_label:output_label without any spaces.
            For example, 17:92 to map the iteratively labeled vertebrae 17 to the sacrum label 92.
        '''.split())
    )
    parser.add_argument(
        '--dilation-size', type=int, default=1,
        help='Number of voxels to dilate before finding connected voxels to label, defaults to 1 (No dilation).'
    )
    parser.add_argument(
        '--default-superior-disc', type=int, default=0,
        help='Default superior disc label if no init label found, defaults to 0 (Raise error if init label not found).'
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
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get arguments
    segs_path = args.segs_dir
    output_segs_path = args.output_segs_dir
    locs_path = args.locs_dir
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    loc_suffix = args.loc_suffix
    disc_labels = [_ for __ in args.disc_labels for _ in (__ if isinstance(__, list) else [__])]
    init_disc = dict(args.init_disc)
    output_c2c3 = args.output_c2c3
    output_disc_step = args.output_disc_step
    loc_disc_labels = [_ for __ in args.loc_disc_labels for _ in (__ if isinstance(__, list) else [__])]
    vertebrae_labels = [_ for __ in args.vertebrae_labels for _ in (__ if isinstance(__, list) else [__])]
    vertebrae_extra_labels = [_ for __ in args.vertebrae_extra_labels for _ in (__ if isinstance(__, list) else [__])]
    output_c2 = args.output_c2
    output_vertebrae_step = args.output_vertebrae_step
    map_input_list = args.map_input
    map_output_list = args.map_output
    dilation_size = args.dilation_size
    default_superior_disc = args.default_superior_disc
    override = args.override
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            segs_dir = "{segs_path}"
            output_segs_dir = "{output_segs_path}"
            locs_dir = "{locs_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            loc_suffix = "{loc_suffix}"
            disc_labels = {disc_labels}
            init_disc = {init_disc}
            output_c2c3 = {output_c2c3}
            output_disc_step = {output_disc_step}
            loc_disc_labels = {loc_disc_labels}
            vertebrae_labels = {vertebrae_labels}
            vertebrae_extra_labels = {vertebrae_extra_labels}
            output_c2 = {output_c2}
            output_vertebrae_step = {output_vertebrae_step}
            map_input = {map_input_list}
            map_output = {map_output_list}
            dilation_size = {dilation_size}
            default_superior_disc = {default_superior_disc}
            override = {override}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    # Load maps into a dict
    try:
        map_input_dict = {int(l_in): int(l_out) for l_in, l_out in map(lambda x:x.split(':'), map_input_list)}
    except:
        raise ValueError("Input param map_input is not in the right structure. Make sure it is in the right format, e.g., 1:2 3:5")

    try:
        map_output_dict = {int(l_in): int(l_out) for l_in, l_out in map(lambda x:x.split(':'), map_output_list)}
    except:
        raise ValueError("Input param map_output is not in the right structure. Make sure it is in the right format, e.g., 1:2 3:5")

    iterative_label_mp(
        segs_path=segs_path,
        output_segs_path=output_segs_path,
        locs_path=locs_path,
        subject_dir=subject_dir,
        subject_subdir=subject_subdir,
        prefix=prefix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        loc_suffix=loc_suffix,
        disc_labels=disc_labels,
        init_disc=init_disc,
        output_c2c3=output_c2c3,
        output_disc_step=output_disc_step,
        loc_disc_labels=loc_disc_labels,
        vertebrae_labels=vertebrae_labels,
        vertebrae_extra_labels=vertebrae_extra_labels,
        output_c2=output_c2,
        output_vertebrae_step=output_vertebrae_step,
        map_input_dict=map_input_dict,
        map_output_dict=map_output_dict,
        dilation_size=dilation_size,
        default_superior_disc=default_superior_disc,
        override=override,
        max_workers=max_workers,
        quiet=quiet,
    )

def iterative_label_mp(
        segs_path,
        output_segs_path,
        locs_path=None,
        subject_dir=None,
        subject_subdir='',
        prefix='',
        seg_suffix='',
        output_seg_suffix='',
        loc_suffix='',
        disc_labels=[],
        init_disc={},
        output_c2c3=0,
        output_disc_step=1,
        loc_disc_labels=[],
        vertebrae_labels=[],
        vertebrae_extra_labels=[],
        output_c2=0,
        output_vertebrae_step=1,
        map_input_dict={},
        map_output_dict={},
        dilation_size=1,
        default_superior_disc=0,
        override=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    segs_path = Path(segs_path)
    output_segs_path = Path(output_segs_path)
    locs_path = locs_path and Path(locs_path)

    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*{seg_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    seg_path_list = list(segs_path.glob(glob_pattern))
    output_seg_path_list = [output_segs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz') for _ in seg_path_list]
    loc_path_list = [locs_path and locs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'{loc_suffix}.nii.gz') for _ in seg_path_list]

    process_map(
        partial(
            _iterative_label,
            disc_labels=disc_labels,
            output_c2c3=output_c2c3,
            output_disc_step=output_disc_step,
            loc_disc_labels=loc_disc_labels,
            init_disc=init_disc,
            vertebrae_labels=vertebrae_labels,
            vertebrae_extra_labels=vertebrae_extra_labels,
            output_c2=output_c2,
            output_vertebrae_step=output_vertebrae_step,
            map_input_dict=map_input_dict,
            map_output_dict=map_output_dict,
            dilation_size=dilation_size,
            default_superior_disc=default_superior_disc,
            override=override,
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
        disc_labels=[],
        init_disc={},
        output_c2c3=0,
        output_disc_step=1,
        loc_disc_labels=[],
        vertebrae_labels=[],
        vertebrae_extra_labels=[],
        output_c2=0,
        output_vertebrae_step=1,
        map_input_dict={},
        map_output_dict={},
        dilation_size=1,
        default_superior_disc=0,
        override=False,
    ):
    '''
    Wrapper function to handle IO.
    '''
    seg_path = Path(seg_path)
    output_seg_path = Path(output_seg_path)
    loc_path = loc_path and Path(loc_path)

    # If the output image already exists and we are not overriding it, return
    if not override and output_seg_path.exists():
        return

    # Load segmentation and localizer images
    seg = nib.load(seg_path)
    loc = loc_path and (loc_path.is_file() or None) and nib.load(loc_path)

    try:
        output_seg = iterative_label(
            seg,
            loc,
            disc_labels=disc_labels,
            init_disc=init_disc,
            output_c2c3=output_c2c3,
            output_disc_step=output_disc_step,
            loc_disc_labels=loc_disc_labels,
            vertebrae_labels=vertebrae_labels,
            vertebrae_extra_labels=vertebrae_extra_labels,
            output_c2=output_c2,
            output_vertebrae_step=output_vertebrae_step,
            map_input_dict=map_input_dict,
            map_output_dict=map_output_dict,
            dilation_size=dilation_size,
            default_superior_disc=default_superior_disc,
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
        disc_labels=[],
        init_disc={},
        output_c2c3=0,
        output_disc_step=1,
        loc_disc_labels=[],
        vertebrae_labels=[],
        vertebrae_extra_labels=[],
        output_c2=0,
        output_vertebrae_step=1,
        map_input_dict={},
        map_output_dict={},
        dilation_size=1,
        default_superior_disc=0,
    ):
    '''
    Label Vertebrae, IVDs, Spinal Cord and canal from init segmentation.
    The algorithm is iterative and works as follows:
    1. Find connected voxels for each disc label and label them into separate labels
    2. Find connected voxels for each vertebrae label and label them into separate labels
    3. Combine sequential vertebrae labels based on some conditions
    4. Combine extra labels with adjacent vertebrae labels
    5. Map labels from the iteative algorithm output, to the final output (e.g., map the vertebrae label from the iteative algorithm output to the special sacrum label)
    6. Map input labels to the final output (e.g., map the input sacrum, canal and spinal cord labels to the output labels)

    Parameters
    ----------
    seg : nibabel.nifti1.Nifti1Image
        Segmentation image
    loc : nibabel.nifti1.Nifti1Image
        Localizer image to use for detecting first vertebrae and disc (optional)
    disc_labels : list
        The disc labels
    init_disc : dict
        Init labels list for disc ordered by priority (input_label:output_label)
    output_c2c3 : int
        The output label for C2C3, used to calculate the first vertebrae label
    output_disc_step : int
        The step to take between disc labels in the output
    loc_disc_labels : list
        Localizer labels to use for detecting first disc
    vertebrae_labels : list
        The vertebrae labels
    vertebrae_extra_labels : list
        Extra vertebrae labels to add to add to adjacent vertebrae labels
    output_c2 : int
        The output label for C2, used to calculate the first vertebrae label
    output_vertebrae_step : int
        The step to take between vertebrae labels in the output
    map_input_dict : dict
        A dict mapping labels from input into the output segmentation
    map_output_dict : dict
        A dict mapping labels from the output of the iterative labeling algorithm into different labels in the output segmentation
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

    # Get sorted connected components superio-inferior (SI) for the disc labels
    disc_mask_labeled, disc_num_labels, disc_sorted_labels, disc_sorted_z_indexes = _get_si_sorted_components(
        seg,
        disc_labels,
        dilation_size,
    )

    # Get sorted connected components superio-inferior (SI) for the vertebrae labels
    vert_mask_labeled, vert_num_labels, vert_sorted_labels, vert_sorted_z_indexes = _get_si_sorted_components(
        seg,
        vertebrae_labels,
        dilation_size,
        combine_labels=True,
    )

    # Combine sequential vertebrae labels based on some conditions
    vert_mask_labeled, vert_num_labels, vert_sorted_labels, vert_sorted_z_indexes = _merge_vertebrae_labels(
        seg,
        vertebrae_labels,
        vert_mask_labeled,
        vert_num_labels,
        vert_sorted_labels,
        vert_sorted_z_indexes,
        disc_sorted_z_indexes,
        vertebrae_extra_labels,
    )

    # Get the first disc label
    superior_disc_output_label = _get_superior_output_label(
        seg,
        loc,
        disc_mask_labeled,
        disc_sorted_labels,
        init_disc,
        output_disc_step,
        loc_disc_labels,
        default_superior_disc,
        map_output_dict,
    )

    # Sort the combined disc+vert labels by their z-index
    sorted_labels = vert_sorted_labels + disc_sorted_labels
    sorted_z_indexes = vert_sorted_z_indexes + disc_sorted_z_indexes
    is_vert = [True] * len(vert_sorted_labels) + [False] * len(disc_sorted_labels)

    # Sort the labels by their z-index (reversed to go from superior to inferior)
    sorted_z_indexes, sorted_labels, is_vert = zip(*sorted(zip(sorted_z_indexes, sorted_labels, is_vert))[::-1])

    # Get the superior output label for the vertebrae based on the superior disc label
    # For C1 and C2 we have to adjust the first vertebrae label by the position of the first disc with substraction of (is_vert.index(False) - 1) * output_vertebrae_step
    superior_vert_output_label = output_c2 + (superior_disc_output_label - output_c2c3) * (output_vertebrae_step / output_disc_step) - (is_vert.index(False) - 1) * output_vertebrae_step

    # Label the vertebrae with the output labels superio-inferior
    for i in range(vert_num_labels):
        output_seg_data[vert_mask_labeled == vert_sorted_labels[i]] = superior_vert_output_label + output_vertebrae_step * i

    # Label the discs with the output labels superio-inferior
    for i in range(disc_num_labels):
        output_seg_data[disc_mask_labeled == disc_sorted_labels[i]] = superior_disc_output_label + output_disc_step * i

    # Use the map to map labels from the iteative algorithm output, to the final output
    # This is useful to map the vertebrae label from the iteative algorithm output to the special sacrum label
    for orig, new in map_output_dict.items():
        if int(orig) in output_seg_data:
            output_seg_data[output_seg_data == int(new)] = 0
            output_seg_data[output_seg_data == int(orig)] = int(new)

    # Use the map to map input labels to the final output
    # This is useful to map the input sacrum, canal and spinal cord labels to the output labels
    for orig, new in map_input_dict.items():
        if int(orig) in seg_data:
            output_seg_data[output_seg_data == int(new)] = 0
            mask = seg_data == int(orig)

            # Map also all labels that are currently in the mask
            # This is useful for example if we addedd from extra_labels to the sacrum and we want them to map together with the sacrum
            mask_labes = [_ for _ in np.unique(output_seg_data[mask]) if _ != 0]
            if len(mask_labes) > 0:
                mask |= np.isin(output_seg_data, mask_labes)
            output_seg_data[mask] = int(new)

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

def _get_si_sorted_components(
        seg,
        labels,
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
        _labels = [[_] for _ in labels]
    else:
        # For discs, combine all labels before label continue voxels since the discs not touching each other
        _labels = [labels]

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

    # Get the z index of the center of mass for each label
    canonical_mask_labeled = np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(mask_labeled, seg.affine, seg.header)).dataobj).round().astype(mask_labeled.dtype)
    mask_labeled_z_indexes = [_[-1] for _ in ndi.center_of_mass(canonical_mask_labeled != 0, canonical_mask_labeled, range(1, num_labels + 1))]

    # Sort the labels by their z-index (reversed to go from superior to inferior)
    sorted_z_indexes, sorted_labels = zip(*sorted(zip(mask_labeled_z_indexes,range(1,num_labels+1)))[::-1])

    return mask_labeled, num_labels, list(sorted_labels), list(sorted_z_indexes)

def _merge_vertebrae_labels(
        seg,
        labels,
        mask_labeled,
        num_labels,
        sorted_labels,
        sorted_z_indexes,
        disc_sorted_z_indexes,
        extra_labels,
    ):
    '''
    Combine sequential vertebrae labels based on some conditions.
    '''
    if num_labels == 0:
        return mask_labeled, num_labels, sorted_labels, sorted_z_indexes

    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)
    # Combine sequential vertebrae labels if they have the same value in the original segmentation
    # This is useful when part part of the vertebrae is not connected to the main part but have the same odd/even value
    if len(labels) > 1:
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

        # Get the z index of the center of mass for each label
        canonical_mask_labeled = np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(mask_labeled, seg.affine, seg.header)).dataobj).round().astype(mask_labeled.dtype)
        mask_labeled_z_indexes = [_[-1] for _ in ndi.center_of_mass(canonical_mask_labeled != 0, canonical_mask_labeled, new_sorted_labels)]

        # Sort the labels by their z-index (reversed to go from superior to inferior)
        sorted_z_indexes, sorted_labels = zip(*sorted(zip(mask_labeled_z_indexes, new_sorted_labels))[::-1])

        # Reduce size of mask_labeled
        if mask_labeled.max() < np.iinfo(np.uint8).max:
            mask_labeled = mask_labeled.astype(np.uint8)
        elif mask_labeled.max() < np.iinfo(np.uint16).max:
            mask_labeled = mask_labeled.astype(np.uint16)

    # Combine sequential vertebrae labels if there is no disc between them
    if len(disc_sorted_z_indexes) > 0:
        new_sorted_labels = []

        # Store the previous label and the z index of the previous label
        prev_l, prev_z = 0, 0

        for l, z in zip(sorted_labels, sorted_z_indexes):
            # Do not combine first and last vertebrae since it can be C1 or only contain the spinous process
            if l not in sorted_labels[:2] and l != sorted_labels[-1] and prev_l > 0 and not any(z < _ < prev_z for _ in disc_sorted_z_indexes):
                # Combine the current label with the previous label
                mask_labeled[mask_labeled == l] = prev_l
                num_labels -= 1

            else:
                # Add the current label to the new sorted labels
                new_sorted_labels.append(l)
                prev_l, prev_z = l, z

        sorted_labels = new_sorted_labels

        # Reduce size of mask_labeled
        if mask_labeled.max() < np.iinfo(np.uint8).max:
            mask_labeled = mask_labeled.astype(np.uint8)
        elif mask_labeled.max() < np.iinfo(np.uint16).max:
            mask_labeled = mask_labeled.astype(np.uint16)

    # Combine extra labels with adjacent vertebrae labels
    if len(extra_labels) > 0:
        mask_extra = np.isin(seg_data, extra_labels)

        # Loop over vertebral labels (from inferior because the transverse process make it steal from above)
        for i in range(num_labels - 1, -1, -1):
            # Mkae mask for the current vertebrae with filling the holes and dilating it
            mask = _fill(mask_labeled == sorted_labels[i])
            mask = ndi.binary_dilation(mask, ndi.iterate_structure(ndi.generate_binary_structure(3, 1), 1))

            # Add the intersection of the mask with the extra labels to the current verebrae
            mask_labeled[mask_extra * mask] = sorted_labels[i]

    # Get the z index of the center of mass for each label
    canonical_mask_labeled = np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(mask_labeled, seg.affine, seg.header)).dataobj).round().astype(mask_labeled.dtype)
    mask_labeled_z_indexes = [_[-1] for _ in ndi.center_of_mass(canonical_mask_labeled != 0, canonical_mask_labeled, sorted_labels)]

    # Sort the labels by their z-index (reversed to go from superior to inferior)
    sorted_z_indexes, sorted_labels = zip(*sorted(zip(mask_labeled_z_indexes, new_sorted_labels))[::-1])

    return mask_labeled, num_labels, list(sorted_labels), list(sorted_z_indexes)

def _get_superior_output_label(
        seg,
        loc,
        mask_labeled,
        sorted_labels,
        init,
        step,
        loc_labels,
        default_superior,
        map_output_dict,
    ):
    '''
    Get the first label for the iterative labeling algorithm.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    loc_data = loc and np.asanyarray(loc.dataobj).round().astype(np.uint8)

    # If localizer is provided, transform it to the segmentation space
    if loc_data is not None:
        loc_data = tio.Resample(
            tio.ScalarImage(tensor=seg_data[None, ...], affine=seg.affine)
        )(
            tio.LabelMap(tensor=loc_data[None, ...], affine=loc.affine)
        ).data.numpy()[0, ...].astype(np.uint8)

    # Find the most superior label in the segmentation
    superior_output_label = 0
    for k, v in init.items():
        if k in seg_data:
            superior_output_label = v - step * sorted_labels.index(np.argmax(np.bincount(mask_labeled[seg_data == k].flat)))
            break

    # If no init label found, set it from the localizer
    if superior_output_label == 0 and loc_data is not None:
        # Make mask for the intersection of the localizer labels and the labels in the segmentation
        mask = np.isin(loc_data, loc_labels) * np.isin(mask_labeled, sorted_labels)

        # Get the first label from sorted_labels that is in the localizer specified labels
        mask_labeled_masked = mask * mask_labeled
        first_sorted_labels_in_loc = next(np.array(sorted_labels)[np.isin(sorted_labels, mask_labeled_masked)].flat, 0)

        if first_sorted_labels_in_loc > 0:
            # Get the target label for first_sorted_labels_in_loc - the label from the localizer that has the most voxels in it
            loc_data_masked = mask * loc_data
            target = np.argmax(np.bincount(loc_data_masked[mask_labeled_masked == first_sorted_labels_in_loc].flat))
            # If target in map_output_dict reverse it from the reversed map
            # TODO Edge case if multiple keys have the same value, not used in the current implementation
            target = {v: k for k, v in map_output_dict.items()}.get(target, target)
            superior_output_label = target - step * sorted_labels.index(first_sorted_labels_in_loc)

    # If no init label found, set the default superior label
    if superior_output_label == 0 and default_superior > 0:
        superior_output_label = default_superior

    # If no init label found, print error
    if superior_output_label == 0:
        raise ValueError(f"Some initiation label must be in the segmentation (init: {list(init.keys())})")

    return superior_output_label

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

    # Create an array of x indices with the same shape as the mask
    x_indices = np.broadcast_to(np.arange(mask.shape[0])[..., np.newaxis, np.newaxis], mask.shape)
    # Create an array of y indices with the same shape as the mask
    y_indices = np.broadcast_to(np.arange(mask.shape[1])[..., np.newaxis], mask.shape)
    # Create an array of z indices with the same shape as the mask
    z_indices = np.broadcast_to(np.arange(mask.shape[2]), mask.shape)

    mask_min_x = np.min(np.where(mask, x_indices, np.inf), axis=0)[np.newaxis, ...]
    mask_max_x = np.max(np.where(mask, x_indices, -np.inf), axis=0)[np.newaxis, ...]
    mask_min_y = np.min(np.where(mask, y_indices, np.inf), axis=1)[:, np.newaxis, :]
    mask_max_y = np.max(np.where(mask, y_indices, -np.inf), axis=1)[:, np.newaxis, :]
    mask_min_z = np.min(np.where(mask, z_indices, np.inf), axis=2)[:, :, np.newaxis]
    mask_max_z = np.max(np.where(mask, z_indices, -np.inf), axis=2)[:, :, np.newaxis]

    return \
        ((mask_min_x <= x_indices) & (x_indices <= mask_max_x)) | \
        ((mask_min_y <= y_indices) & (y_indices <= mask_max_y)) | \
        ((mask_min_z <= z_indices) & (z_indices <= mask_max_z))

if __name__ == '__main__':
    main()