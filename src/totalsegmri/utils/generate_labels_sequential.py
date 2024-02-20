import sys, argparse, textwrap
from scipy.ndimage import label, binary_dilation, generate_binary_structure, iterate_structure
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
            Label Vertebrae IVD CSF and Spinal Cord from init segmentation.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            generate_labels_sequential -s labels_init -o labels
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--segs-dir', '-s', type=DirPath(), required=True,
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
        '--disc-labels', type=int, nargs='+', default=[],
        help='The disc labels.'
    )
    parser.add_argument(
        '--init-disc', type=lambda x:map(int, x.split(',')), nargs='+', default=[],
        help='Init labels list for disc ordered by priority (input_label,output_label !!without space!!). for example 4,224 5,219 6,202'
    )
    parser.add_argument(
        '--output-disc-step', type=int, default=-1,
        help='The step to take between disc labels in the output, defaults to -1.'
    )
    parser.add_argument(
        '--vertebrea-labels', type=int, nargs='+', default=[],
        help='The vertebrae labels.'
    )
    parser.add_argument(
        '--init-vertebrae', type=lambda x:map(int, x.split(',')), nargs='+', default=[],
        help='Init labels list for vertebrae ordered by priority (input_label,output_label !!without space!!). for example 10,41 11,34 12,18'
    )
    parser.add_argument(
        '--output-vertebrea-step', type=int, default=-1,
        help='The step to take between vertebrae labels in the output, defaults to -1.'
    )
    parser.add_argument(
        '--csf-labels', type=int, nargs='+', default=[],
        help='The CSF label.'
    )
    parser.add_argument(
        '--output-csf-label', type=int, default=201,
        help='The CSF label in the output, defaults to 201.'
    )
    parser.add_argument(
        '--sc-labels', type=int, nargs='+', default=[],
        help='The spinal cord label.'
    )
    parser.add_argument(
        '--output-sc-label', type=int, default=200,
        help='The spinal cord label in the output, defaults to 200.'
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
    disc_labels = args.disc_labels
    init_disc = dict(args.init_disc)
    output_disc_step = args.output_disc_step
    vertebrea_labels = args.vertebrea_labels
    init_vertebrae = dict(args.init_vertebrae)
    output_vertebrea_step = args.output_vertebrea_step
    csf_labels = args.csf_labels
    output_csf_label = args.output_csf_label
    sc_labels = args.sc_labels
    output_sc_label = args.output_sc_label
    dilation_size = args.dilation_size
    combine_before_label = args.combine_before_label
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
            disc_labels = "{disc_labels}"
            init_disc = "{init_disc}"
            output_disc_step = "{output_disc_step}"
            vertebrea_labels = "{vertebrea_labels}"
            init_vertebrae = "{init_vertebrae}"
            output_vertebrea_step = "{output_vertebrea_step}"
            csf_labels = "{csf_labels}"
            output_csf_label = "{output_csf_label}"
            sc_labels = "{sc_labels}"
            output_sc_label = "{output_sc_label}"
            dilation_size = "{dilation_size}"
            combine_before_label = "{combine_before_label}"
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
    partial_generate_labels_sequential = partial(
        generate_labels_sequential,
        output_path=output_path,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        disc_labels=disc_labels,
        output_disc_step=output_disc_step,
        init_disc=init_disc,
        vertebrea_labels=vertebrea_labels,
        init_vertebrae=init_vertebrae,
        output_vertebrea_step=output_vertebrea_step,
        csf_labels=csf_labels,
        output_csf_label=output_csf_label,
        sc_labels=sc_labels,
        output_sc_label=output_sc_label,
        dilation_size=dilation_size,
        combine_before_label=combine_before_label,
   )

    with mp.Pool() as pool:
        process_map(partial_generate_labels_sequential, segs_path_list, max_workers=max_workers)
    

def generate_labels_sequential(
            seg_path,
            output_path,
            seg_suffix,
            output_seg_suffix,
            disc_labels,
            init_disc,
            output_disc_step,
            vertebrea_labels,
            init_vertebrae,
            output_vertebrea_step,
            csf_labels,
            output_csf_label,
            sc_labels,
            output_sc_label,
            dilation_size,
            combine_before_label,
        ):
    
    output_seg_path = output_path / seg_path.name.replace(f'{seg_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz')

    # Load segmentation
    seg = nib.load(seg_path)
    seg_data = seg.get_fdata()

    # Convert data to uint8 to avoid issues with segmentation IDs
    seg_data_src = seg_data.astype(np.uint8)

    seg_data = np.zeros_like(seg_data_src)

    # Set cord label
    seg_data[np.isin(seg_data_src, sc_labels)] = output_sc_label
    # Set CSF label
    seg_data[np.isin(seg_data_src, csf_labels)] = output_csf_label

    # Create an array of z indices
    z_indices = np.broadcast_to(np.arange(seg_data_src.shape[2]), seg_data_src.shape)

    binary_dilation_structure = iterate_structure(generate_binary_structure(3, 1), dilation_size)

    for labels, step, init in (
        (disc_labels, output_disc_step, init_disc),
        (vertebrea_labels, output_vertebrea_step, init_vertebrae)):

        if len(labels) == 0:
            continue

        # Get labeled
        if combine_before_label:
            mask_labeled, num_labels = label(binary_dilation(np.isin(seg_data_src, labels), binary_dilation_structure), np.ones((3, 3, 3)))
        else:
            mask_labeled, num_labels = np.zeros_like(seg_data_src), 0
            for l in labels:
                mask = seg_data_src == l
                tmp_mask_labeled, tmp_num_labels = label(binary_dilation(mask, binary_dilation_structure), np.ones((3, 3, 3)))
                tmp_mask_labeled *= mask
                if tmp_num_labels > 0:
                    mask_labeled[tmp_mask_labeled != 0] = tmp_mask_labeled[tmp_mask_labeled != 0] + num_labels
                    num_labels += tmp_num_labels

        # Get the z-axis index for each label
        mask_labeled_z_indexes = [np.max(z_indices[mask_labeled==i]) for i in range(1, num_labels+1)]

        # Sort the labels by their z-index (reversed)
        sorted_labels = [x for _,x in sorted(zip(mask_labeled_z_indexes,range(1,num_labels+1)))][::-1]

        first_label = 0
        for k, v in init.items():
            if k in seg_data_src:
                first_label = v - step * sorted_labels.index(np.min(mask_labeled[seg_data_src == k]))
                break

        if first_label == 0:
            print(f"Error: {seg_path}, some initiation label must be in the segmentation (init: {init.keys()})")
            return

        for i in range(num_labels):
            seg_data[mask_labeled == sorted_labels[i]] = first_label + step * i

    # Create result segmentation
    seg = nib.Nifti1Image(seg_data, seg.affine, seg.header)
    seg.set_data_dtype(np.uint8)
    # Make sure output directory exists
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    # Save mapped segmentation
    nib.save(seg, output_seg_path)

if __name__ == '__main__':
    main()