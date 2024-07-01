import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import nibabel as nib
import numpy as np
import torchio as tio
import warnings

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
        It transform segmentations into the image space to have the same origin, spacing, direction and shape as the image.'''
        ),
        epilog=textwrap.dedent('''
            Examples:
            transform_labels2images -i images -s labels -o labels_transformed
            For BIDS:
            transform_labels2images -i . -s derivatives/labels -o derivatives/labels --image-suffix "" --output-image-suffix "" --seg-suffix "_seg" --output-seg-suffix "_seg" -d "sub-" -u "anat"
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--segs-dir', '-s', type=Path, required=True,
        help='The folder where input NIfTI segmentation files are located (required).'
    )
    parser.add_argument(
        '--output-dir', '-o', type=Path, required=True,
        help='The folder where output augmented images will be saved (required).'
    )
    parser.add_argument(
        '--subject-dir', '-d', type=str, default=None, nargs='?', const='',
        help=textwrap.dedent('''
            Is every subject has its oen direcrory.
            If this argument will be provided without value it will look for any directory in the segmentation directory.
            If value also provided it will be used as a prefix to subject directory, defaults to False (no subjet directory).
        '''),
    )
    parser.add_argument(
        '--subject-subdir', '-u', type=str, default='', 
        help='Subfolder inside subject folder containing masks, defaults to no subfolder.'
    )
    parser.add_argument(
        '--prefix', '-p', type=str, default='', 
        help='File prefix to work on.'
    )
    parser.add_argument(
        '--image-suffix', type=str, default='_0000',
        help='Image suffix, defaults to "_0000".'
    )
    parser.add_argument(
        '--seg-suffix', type=str, default='',
        help='Segmentation suffix, defaults to "".'
    )
    parser.add_argument(
        '--output-suffix', type=str, default='',
        help='Image suffix for output, defaults to "".'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--verbose', '-v', type=int, default=1, choices=[0, 1],
        help='verbose. 0: Display only errors/warnings, 1: Errors/warnings + info messages. Default is 1.'
    )

    # Parse the command-line arguments
    try:
        args = parser.parse_args()
    except BaseException as e:
        sys.exit()

    # Get the command-line argument values
    images_path = args.images_dir
    segs_path = args.segs_dir
    output_path = args.output_dir
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    image_suffix = args.image_suffix
    seg_suffix = args.seg_suffix
    output_suffix = args.output_suffix
    max_workers = args.max_workers
    verbose = args.verbose
    
    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f''' 
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            segs_path = "{segs_path}"
            output_path = "{output_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            seg_suffix = "{seg_suffix}"
            output_suffix = "{output_suffix}"
            max_workers = {max_workers}
            verbose = {verbose}
        '''))

    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*{image_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    images_path_list = list(images_path.glob(glob_pattern))

    # Create a partially-applied function with the extra arguments
    partial_transform_labels2images = partial(
        transform_labels2images,
        images_path=images_path,
        segs_path=segs_path,
        output_path=output_path,
        image_suffix=image_suffix,
        seg_suffix=seg_suffix,
        output_suffix=output_suffix,
    )

    with mp.Pool() as pool:
        process_map(partial_transform_labels2images, images_path_list, max_workers=max_workers)


def transform_labels2images(
        image_path,
        images_path,
        segs_path,
        output_path,
        image_suffix,
        seg_suffix,
        output_suffix,
    ):
    
    seg_path = segs_path / image_path.relative_to(images_path).parent /  image_path.name.replace(f'{image_suffix}.nii.gz', f'{seg_suffix}.nii.gz')
    
    if not seg_path.is_file():
        print(f'Segmentation file not found: {seg_path}')
        return
    
    output_seg_path = output_path / image_path.relative_to(images_path).parent / seg_path.name.replace(f'{seg_suffix}.nii.gz', f'{output_suffix}.nii.gz')

    image = nib.load(image_path)
    seg = nib.load(seg_path)

    image_data = np.asanyarray(image.dataobj).astype(np.float64)
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    # Create result
    tio_img=tio.ScalarImage(tensor=image_data[None, ...], affine=image.affine)
    tio_seg=tio.LabelMap(tensor=seg_data[None, ...], affine=seg.affine)
    
    tio_output_seg = tio.Resample(tio_img)(tio_seg)
    output_seg_data = tio_output_seg.data.numpy()[0, ...].astype(np.uint8)
    
    # Make sure output directory exists and save
    output_seg_path.parent.mkdir(parents=True, exist_ok=True)
    output_seg = nib.Nifti1Image(output_seg_data, image.affine, seg.header)
    output_seg.set_qform(image.affine)
    output_seg.set_sform(image.affine)
    output_seg.set_data_dtype(np.uint8)
    nib.save(output_seg, output_seg_path)

if __name__ == '__main__':
    main()