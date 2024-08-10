import sys, argparse, textwrap
from scipy.ndimage import label, binary_dilation, generate_binary_structure, iterate_structure, center_of_mass
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
        description=textwrap.dedent(f'''
            Label Vertebrae, IVDs, Spinal Cord and canal from init segmentation.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            iterative_label -s labels_init -o labels --sacrum-labels 14 --canal-labels 16 --sc-labels 17 --disc-labels 2 3 4 5 6 7 --vertebrea-labels 9 10 11 12 13 14 --init-disc 4:224 7:202 5:219 6:207 --init-vertebrae 11:40 14:17 12:34 13:23 --step-diff-label --step-diff-disc
            For BIDS:
            iterative_label -s derivatives/labels -o derivatives/labels --sacrum-labels 14 --canal-labels 16 --sc-labels 17 --disc-labels 2 3 4 5 6 7 --vertebrea-labels 9 10 11 12 13 14 --init-disc 4:224 7:202 5:219 6:207 --init-vertebrae 11:40 14:17 12:34 13:23 --step-diff-label --step-diff-disc --seg-suffix "_seg" --output-seg-suffix "_seg_seq" -d "sub-" -u "anat"
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
        '--disc-labels', type=int, nargs='+', default=[],
        help='The disc labels.'
    )
    parser.add_argument(
        '--init-disc', type=lambda x:map(int, x.split(':')), nargs='+', default=[],
        help='Init labels list for disc ordered by priority (input_label:output_label !!without space!!). for example 4:224 5:219 6:202'
    )
    parser.add_argument(
        '--output-disc-step', type=int, default=1,
        help='The step to take between disc labels in the output, defaults to 1.'
    )
    parser.add_argument(
        '--vertebrea-labels', type=int, nargs='+', default=[],
        help='The vertebrae labels.'
    )
    parser.add_argument(
        '--init-vertebrae', type=lambda x:map(int, x.split(':')), nargs='+', default=[],
        help='Init labels list for vertebrae ordered by priority (input_label:output_label !!without space!!). for example 10:41 11:34 12:18'
    )
    parser.add_argument(
        '--output-vertebrea-step', type=int, default=1,
        help='The step to take between vertebrae labels in the output, defaults to 1.'
    )
    parser.add_argument(
        '--map-input', type=str, nargs='+', default=[],
        help=textwrap.dedent('''
            A dict mapping labels from input into the output segmentation.
            The format should be input_label:output_label without any spaces.
            For example, 14:92 16:201 17:200 to map the input sacrum label 14 to 92, canal label 16 to 201 and spinal cord label 17 to 200.
        '''),
    )
    parser.add_argument(
        '--map-output', type=str, nargs='+', default=[],
        help=textwrap.dedent('''
            A dict mapping labels from the output of the iterative labeling algorithm into different labels in the output segmentation.
            The format should be input_label:output_label without any spaces.
            For example, 40:92 to map the iteratively labeled vertebrae 40 to the sacrum label 92.
        '''),
    )
    parser.add_argument(
        '--dilation-size', type=int, default=1,
        help='Number of voxels to dilate before finding connected voxels to label, defaults to 1 (No dilation).'
    )
    parser.add_argument(
        '--combine-before-label', action="store_true", default=False,
        help='Combine all labels before label continue voxels, defaults to false (label continue voxels separatly for each label).'
    )
    parser.add_argument(
        '--step-diff-label', action="store_true", default=False,
        help=textwrap.dedent('''
            Make step only for different labels. When looping on the labels on the z axis, it will give a new label to the next label only
             if it is different from the previous label. This is useful if there are labels for odd and even vertebrae, so the next label will
             be for even vertebrae only if the previous label was odd. If it is still odd, it should give the same label.
        '''),
    )
    parser.add_argument(
        '--step-diff-disc', action="store_true", default=False,
        help=textwrap.dedent('''
            Make step only for different discs. When looping on the labels on the z axis, it will give a new label to the next label only
             if there is a disc between them. This exclude the first and last vertebrae since it can be C1 or only contain the spinous process.
        '''),
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
    disc_labels = args.disc_labels
    init_disc = dict(args.init_disc)
    output_disc_step = args.output_disc_step
    vertebrea_labels = args.vertebrea_labels
    init_vertebrae = dict(args.init_vertebrae)
    output_vertebrea_step = args.output_vertebrea_step
    map_input_list = args.map_input
    map_output_list = args.map_output
    dilation_size = args.dilation_size
    combine_before_label = args.combine_before_label
    step_diff_label = args.step_diff_label
    step_diff_disc = args.step_diff_disc
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
            disc_labels = {disc_labels}
            init_disc = {init_disc}
            output_disc_step = {output_disc_step}
            vertebrea_labels = {vertebrea_labels}
            init_vertebrae = {init_vertebrae}
            output_vertebrea_step = {output_vertebrea_step}
            map_input = {map_input_list}
            map_output = {map_output_list}
            dilation_size = {dilation_size}
            combine_before_label = {combine_before_label}
            step_diff_label = {step_diff_label}
            step_diff_disc = {step_diff_disc}
            override = {override}
            max_workers = {max_workers}
            verbose = {verbose}
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
        subject_dir=subject_dir,
        subject_subdir=subject_subdir,
        prefix=prefix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        disc_labels=disc_labels,
        init_disc=init_disc,
        output_disc_step=output_disc_step,
        vertebrea_labels=vertebrea_labels,
        init_vertebrae=init_vertebrae,
        output_vertebrea_step=output_vertebrea_step,
        map_input_dict=map_input_dict,
        map_output_dict=map_output_dict,
        dilation_size=dilation_size,
        combine_before_label=combine_before_label,
        step_diff_label=step_diff_label,
        step_diff_disc=step_diff_disc,
        override=override,
        max_workers=max_workers,
    )

def iterative_label_mp(
        segs_path,
        output_segs_path,
        subject_dir=None,
        subject_subdir='',
        prefix='',
        seg_suffix='',
        output_seg_suffix='',
        disc_labels=[],
        init_disc={},
        output_disc_step=1,
        vertebrea_labels=[],
        init_vertebrae={},
        output_vertebrea_step=1,
        map_input_dict={},
        map_output_dict={},
        dilation_size=1,
        combine_before_label=False,
        step_diff_label=False,
        step_diff_disc=False,
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
            _iterative_label,
            disc_labels=disc_labels,
            output_disc_step=output_disc_step,
            init_disc=init_disc,
            vertebrea_labels=vertebrea_labels,
            init_vertebrae=init_vertebrae,
            output_vertebrea_step=output_vertebrea_step,
            map_input_dict=map_input_dict,
            map_output_dict=map_output_dict,
            dilation_size=dilation_size,
            combine_before_label=combine_before_label,
            step_diff_label=step_diff_label,
            step_diff_disc=step_diff_disc,
            override=override,
        ),
        seg_path_list,
        output_seg_path_list,
        max_workers=max_workers,
    )

def _iterative_label(
        seg_path,
        output_seg_path,
        disc_labels=[],
        init_disc={},
        output_disc_step=1,
        vertebrea_labels=[],
        init_vertebrae={},
        output_vertebrea_step=1,
        map_input_dict={},
        map_output_dict={},
        dilation_size=1,
        combine_before_label=False,
        step_diff_label=False,
        step_diff_disc=False,
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
        output_seg = iterative_label(
            seg,
            disc_labels=disc_labels,
            init_disc=init_disc,
            output_disc_step=output_disc_step,
            vertebrea_labels=vertebrea_labels,
            init_vertebrae=init_vertebrae,
            output_vertebrea_step=output_vertebrea_step,
            map_input_dict=map_input_dict,
            map_output_dict=map_output_dict,
            dilation_size=dilation_size,
            combine_before_label=combine_before_label,
            step_diff_label=step_diff_label,
            step_diff_disc=step_diff_disc,
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
        disc_labels=[],
        init_disc={},
        output_disc_step=1,
        vertebrea_labels=[],
        init_vertebrae={},
        output_vertebrea_step=1,
        map_input_dict={},
        map_output_dict={},
        dilation_size=1,
        combine_before_label=False,
        step_diff_label=False,
        step_diff_disc=False,
    ):
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    output_seg_data = np.zeros_like(seg_data)

    binary_dilation_structure = iterate_structure(generate_binary_structure(3, 1), dilation_size)

    disc_sorted_z_indexes = []

    for labels, step, init, is_vert in (
        (disc_labels, output_disc_step, init_disc, False),
        (vertebrea_labels, output_vertebrea_step, init_vertebrae, True)):

        if len(labels) == 0:
            continue

        # Get labeled
        if combine_before_label or not is_vert: # Always for discs
            # Combine all labels before label continue voxels
            mask_labeled, num_labels = label(binary_dilation(np.isin(seg_data, labels), binary_dilation_structure), np.ones((3, 3, 3)))
        else:
            mask_labeled, num_labels = np.zeros_like(seg_data), 0
            for l in labels:
                mask = seg_data == l
                # Dilate
                tmp_mask_labeled, tmp_num_labels = label(binary_dilation(mask, binary_dilation_structure), np.ones((3, 3, 3)))
                # Undo dilate
                tmp_mask_labeled *= mask
                if tmp_num_labels > 0:
                    mask_labeled[tmp_mask_labeled != 0] = tmp_mask_labeled[tmp_mask_labeled != 0] + num_labels
                    num_labels += tmp_num_labels

        # If no label found, raise error
        if num_labels == 0:
            raise ValueError(f"Some label must be in the segmentation (labels: {labels})")

        # Get the z index of the center of mass for each label
        canonical_mask_labeled = np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(mask_labeled, seg.affine, seg.header)).dataobj).round().astype(np.uint8)
        mask_labeled_z_indexes = [_[-1] for _ in center_of_mass(canonical_mask_labeled != 0, canonical_mask_labeled, range(1, num_labels + 1))]

        # Sort the labels by their z-index (reversed)
        sorted_z_indexes, sorted_labels = zip(*sorted(zip(mask_labeled_z_indexes,range(1,num_labels+1)))[::-1])

        if is_vert:
            # Combine sequential vertebrae labels if they have the same value in the original segmentation
            if step_diff_label and len(labels) - len(init) > 1:
                new_sorted_labels = []
                prev_l, prev_orig_label = 0, 0
                for l in sorted_labels:
                    curr_orig_label = seg_data[mask_labeled == l].flat[0]
                    if curr_orig_label == prev_orig_label:
                        mask_labeled[mask_labeled == l] = prev_l
                        num_labels -= 1
                    else:
                        new_sorted_labels.append(l)
                        prev_l, prev_orig_label = l, curr_orig_label

                # Get the z index of the center of mass for each label
                canonical_mask_labeled = np.asanyarray(nib.as_closest_canonical(nib.Nifti1Image(mask_labeled, seg.affine, seg.header)).dataobj).round().astype(np.uint8)
                mask_labeled_z_indexes = [_[-1] for _ in center_of_mass(canonical_mask_labeled != 0, canonical_mask_labeled, new_sorted_labels)]

                # Sort the labels by their z-index (reversed)
                sorted_z_indexes, sorted_labels = zip(*sorted(zip(mask_labeled_z_indexes, new_sorted_labels))[::-1])

            # Combine sequential vertebrae labels if there is no disc between them
            if step_diff_disc and len(disc_sorted_z_indexes) > 0:
                new_sorted_labels = []
                prev_l, prev_z = 0, 0

                for l, z in zip(sorted_labels, sorted_z_indexes):
                    # Do not combine first and last vertebrae since it can be C1 or only contain the spinous process
                    if l not in sorted_labels[:2] and l != sorted_labels[-1] and prev_l > 0 and not any(z < _ < prev_z for _ in disc_sorted_z_indexes):
                        mask_labeled[mask_labeled == l] = prev_l
                        num_labels -= 1
                    else:
                        new_sorted_labels.append(l)
                        prev_l, prev_z = l, z

                sorted_labels = new_sorted_labels

        else:
            # Save the z indexes of the discs
            disc_sorted_z_indexes = sorted_z_indexes

        # Set the first label
        first_label = 0
        for k, v in init.items():
            if k in seg_data:
                first_label = v - step * sorted_labels.index(mask_labeled[seg_data == k].flat[0])
                break

        # If no init label found, print error
        if first_label == 0:
            raise ValueError(f"Some initiation label must be in the segmentation (init: {list(init.keys())})")

        # Set the output value for the current label
        for i in range(num_labels):
            output_seg_data[mask_labeled == sorted_labels[i]] = first_label + step * i

    # Use the map to change the iteative algorithm output to the final output
    for orig, new in map_output_dict.items():
        if int(orig) in output_seg_data:
            output_seg_data[output_seg_data == int(new)] = 0
            output_seg_data[output_seg_data==int(orig)] = int(new)

    # Use the map to map input labels to the final output
    for orig, new in map_input_dict.items():
        if int(orig) in seg_data:
            output_seg_data[output_seg_data == int(new)] = 0
            output_seg_data[seg_data==int(orig)] = int(new)

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

if __name__ == '__main__':
    main()