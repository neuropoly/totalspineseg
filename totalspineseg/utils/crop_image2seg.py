import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import nibabel as nib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

rs = np.random.RandomState()

def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
        It crop images to the segmented region. This script requires the images and segementation to be in the asme space.
        '''),
        epilog=textwrap.dedent('''
            Examples:
            crop_image2seg -i images -s labels -o images -m 50
            For BIDS:
            crop_image2seg -i . -s derivatives/labels -o . --image-suffix "" --output-image-suffix "" --seg-suffix "_seg" -d "sub-" -u "anat" -m 50
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
        '--output-images-dir', '-o', type=Path, required=True,
        help='The folder where output augmented images will be saved with _a1, _a2 etc. suffixes (required).'
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
        '--output-image-suffix', type=str, default='_0000',
        help='Image suffix for output, defaults to "_0000".'
    )
    parser.add_argument(
        '--margin', '-m', type=int, default=0,
        help='Margin to add to the cropped region, defaults to 0.'
    )
    parser.add_argument(
        '--override', '-r', action="store_true", default=False,
        help='If provided, override existing output files, defaults to false (Do not override).'
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
    output_images_path = args.output_images_dir
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    image_suffix = args.image_suffix
    seg_suffix = args.seg_suffix
    output_image_suffix = args.output_image_suffix
    margin = args.margin
    override = args.override
    max_workers = args.max_workers
    verbose = args.verbose

    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            segs_path = "{segs_path}"
            output_images_path = "{output_images_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            seg_suffix = "{seg_suffix}"
            output_image_suffix = "{output_image_suffix}"
            margin = {margin}
            override = {override}
            max_workers = {max_workers}
            verbose = {verbose}
        '''))

    crop_image2seg_mp(
        images_path=images_path,
        segs_path=segs_path,
        output_images_path=output_images_path,
        subject_dir=subject_dir,
        subject_subdir=subject_subdir,
        prefix=prefix,
        image_suffix=image_suffix,
        seg_suffix=seg_suffix,
        output_image_suffix=output_image_suffix,
        margin=margin,
        override=override,
        max_workers=max_workers,
    )

def crop_image2seg_mp(
        images_path,
        segs_path,
        output_images_path,
        subject_dir=None,
        subject_subdir='',
        prefix='',
        image_suffix='_0000',
        seg_suffix='',
        output_image_suffix='_0000',
        margin=0,
        override=False,
        max_workers=mp.cpu_count(),
    ):
    images_path = Path(images_path)
    segs_path = Path(segs_path)
    output_images_path = Path(output_images_path)

    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*{image_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    image_path_list = list(images_path.glob(glob_pattern))
    seg_path_list = [segs_path / _.relative_to(images_path).parent / _.name.replace(f'{image_suffix}.nii.gz', f'{seg_suffix}.nii.gz') for _ in image_path_list]
    output_image_path_list = [output_images_path / _.relative_to(images_path).parent / _.name.replace(f'{image_suffix}.nii.gz', f'{output_image_suffix}.nii.gz') for _ in image_path_list]

    process_map(
        partial(
            _crop_image2seg,
            margin=margin,
            override=override,
        ),
        image_path_list,
        seg_path_list,
        output_image_path_list,
        max_workers=max_workers,
    )

def _crop_image2seg(
        image_path,
        seg_path,
        output_image_path,
        margin=0,
        override=False,
    ):
    image_path = Path(image_path)
    seg_path = Path(seg_path)
    output_image_path = Path(output_image_path)

    # If the output image already exists and we are not overriding it, return
    if not override and output_image_path.exists():
        return

    # Check if the segmentation file exists
    if not seg_path.is_file():
        output_image_path.is_file() and output_image_path.unlink()
        print(f'Error: {seg_path}, Segmentation file not found')
        return

    image = nib.load(image_path)
    seg = nib.load(seg_path)

    # Get image dtype from the image data (preferred over header dtype to avoid data loss)
    image_data_dtype = getattr(np, np.asanyarray(image.dataobj).dtype.name)

    # Rescale the image to the output dtype range if necessary
    # Modified from https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/6.3/spinalcordtoolbox/image.py#L1217
    if "int" in np.dtype(image_data_dtype).name:
        image_data = np.asanyarray(image.dataobj).astype(np.float64)
        image_min, image_max = image_data.min(), image_data.max()
        dtype_min, dtype_max = np.iinfo(image_data_dtype).min, np.iinfo(image_data_dtype).max
        if (image_min < dtype_min) or (dtype_max < image_max):
            data_rescaled = image_data * (dtype_max - dtype_min) / (image_max - image_min)
            image_data = data_rescaled - (data_rescaled.min() - dtype_min)
            image = nib.Nifti1Image(image_data.astype(image_data_dtype), image.affine, image.header)

    output_image = crop_image2seg(image, seg, margin=margin)

    # Ensure correct image dtype, affine and header
    output_image = nib.Nifti1Image(
        np.asanyarray(output_image.dataobj).astype(image_data_dtype),
        output_image.affine, output_image.header
    )
    output_image.set_data_dtype(image_data_dtype)
    output_image.set_qform(output_image.affine)
    output_image.set_sform(output_image.affine)

    # Make sure output directory exists and save the image
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_image, output_image_path)

def crop_image2seg(
        image,
        seg,
        margin=0,
    ):
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    # Get bounding box of the segmentation and crop the image
    x, y, z = np.nonzero(seg_data)

    if len(x) > 0 and len(y) > 0 and len(z) > 0:
        # Calculate the bounding box
        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()
        min_z, max_z = z.min(), z.max()

        # Add margin to the bounding box ensuring it does not exceed the image dimensions
        min_x = max(min_x - margin, 0)
        max_x = min(max_x + margin, seg_data.shape[0] - 1)
        min_y = max(min_y - margin, 0)
        max_y = min(max_y + margin, seg_data.shape[1] - 1)
        min_z = max(min_z - margin, 0)
        max_z = min(max_z + margin, seg_data.shape[2] - 1)

        image = image.slicer[
            min_x: max_x + 1,
            min_y: max_y + 1,
            min_z: max_z + 1
        ]

    return image

if __name__ == '__main__':
    main()