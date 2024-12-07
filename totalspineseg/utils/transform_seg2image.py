import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import nibabel as nib
import numpy as np
import torchio as tio
import scipy.ndimage as ndi
import warnings

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
            It transform segmentations into the image space to have the same origin, spacing, direction and shape as the image.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            transform_seg2image -i images -s labels -o labels_transformed
            For BIDS:
            transform_seg2image -i . -s derivatives/labels -o derivatives/labels --image-suffix "" --output-image-suffix "" --seg-suffix "_seg" --output-seg-suffix "_seg" -p "sub-*/anat/"
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
        '--output-segs-dir', '-o', type=Path, required=True,
        help='The folder where output augmented images will be saved (required).'
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
        '--output-seg-suffix', type=str, default='',
        help='Image suffix for output, defaults to "".'
    )
    parser.add_argument(
        '--interpolation', '-x', type=str, default='nearest', choices=['nearest', 'linear', 'label'],
        help='Interpolation method, can be "nearest", "linear" or "label" (for singel voxel labels), defaults to "nearest".'
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

    # Get the command-line argument values
    images_path = args.images_dir
    segs_path = args.segs_dir
    output_segs_path = args.output_segs_dir
    prefix = args.prefix
    image_suffix = args.image_suffix
    seg_suffix = args.seg_suffix
    output_seg_suffix = args.output_seg_suffix
    interpolation = args.interpolation
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            segs_path = "{segs_path}"
            output_segs_path = "{output_segs_path}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            seg_suffix = "{seg_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            interpolation = "{interpolation}"
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    transform_seg2image_mp(
        images_path=images_path,
        segs_path=segs_path,
        output_segs_path=output_segs_path,
        prefix=prefix,
        image_suffix=image_suffix,
        seg_suffix=seg_suffix,
        output_seg_suffix=output_seg_suffix,
        interpolation=interpolation,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def transform_seg2image_mp(
        images_path,
        segs_path,
        output_segs_path,
        prefix='',
        image_suffix='_0000',
        seg_suffix='',
        output_seg_suffix='',
        interpolation='nearest',
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    images_path = Path(images_path)
    segs_path = Path(segs_path)
    output_segs_path = Path(output_segs_path)

    glob_pattern = f'{prefix}*{image_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    image_path_list = list(images_path.glob(glob_pattern))
    seg_path_list = [segs_path / _.relative_to(images_path).parent / _.name.replace(f'{image_suffix}.nii.gz', f'{seg_suffix}.nii.gz') for _ in image_path_list]
    output_segs_path_list = [output_segs_path / _.relative_to(images_path).parent / _.name.replace(f'{image_suffix}.nii.gz', f'{output_seg_suffix}.nii.gz') for _ in image_path_list]

    process_map(
        partial(
            _transform_seg2image,
            interpolation=interpolation,
            overwrite=overwrite,
        ),
        image_path_list,
        seg_path_list,
        output_segs_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _transform_seg2image(
        image_path,
        seg_path,
        output_seg_path,
        interpolation='nearest',
        overwrite=False,
    ):
    '''
    Wrapper function to handle IO.
    '''
    image_path = Path(image_path)
    seg_path = Path(seg_path)
    output_seg_path = Path(output_seg_path)

    # If the output image already exists and we are not overriding it, return
    if not overwrite and output_seg_path.exists():
        return

    # Check if the segmentation file exists
    if not seg_path.is_file():
        output_seg_path.is_file() and output_seg_path.unlink()
        print(f'Error: {seg_path}, Segmentation file not found')
        return

    image = nib.load(image_path)
    seg = nib.load(seg_path)

    output_seg = transform_seg2image(
        image,
        seg,
        interpolation=interpolation,
    )

    # Ensure correct segmentation dtype, affine and header
    if interpolation == 'linear':
        output_seg = nib.Nifti1Image(
            np.asanyarray(output_seg.dataobj).astype(np.float32),
            output_seg.affine, output_seg.header
        )
        output_seg.set_data_dtype(np.float32)
    else:
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

def transform_seg2image(
        image,
        seg,
        interpolation='nearest',
    ):
    '''
    Transform the segmentation to the image space to have the same origin, spacing, direction and shape as the image.

    Parameters
    ----------
    image : nibabel.Nifti1Image
        Image.
    seg : nibabel.Nifti1Image
        Segmentation.
    interpolation : str, optional
        Interpolation method, can be 'nearest', 'linear' or 'label' (for singel voxel labels), defaults to 'nearest'.

    Returns
    -------
    nibabel.Nifti1Image
        Output segmentation.
    '''
    # Check if the input image is 4D and take the first image from the last axis for resampling
    if len(np.asanyarray(image.dataobj).shape) == 4:
        image = image.slicer[..., 0]

    image_data = np.asanyarray(image.dataobj).astype(np.float64)
    image_affine = image.affine.copy()
    seg_data = np.asanyarray(seg.dataobj)
    if interpolation == 'linear':
        seg_data = seg_data.astype(np.float32)
    else:
        seg_data = seg_data.round().astype(np.uint8)
    seg_affine = seg.affine.copy()

    # Dilations size - the maximum of factor by which the image zooms are larger than the segmentation zooms
    dilation_size = int(np.ceil(np.max(np.array(image.header.get_zooms()) / np.array(seg.header.get_zooms()))))

    # Pad width - maximum possible number of voxels the dilation can occupy in the image space
    pad_width = int(dilation_size * int(np.ceil(np.max(np.array(seg.header.get_zooms()) / np.array(image.header.get_zooms())))))

    if interpolation == 'label':
        # Pad the image and segmentation to avoid labels at the edge bing displaced on center of mass calculation
        image_data = np.pad(image_data, pad_width)
        image_affine[:3, 3] -= (image_affine[:3, :3] @ ([pad_width] * 3))
        seg_data = np.pad(seg_data, pad_width)
        seg_affine[:3, 3] -= (seg_affine[:3, :3] @ ([pad_width] * 3))

        # Dilate the segmentation to avoid interpolation artifacts
        seg_data = ndi.grey_dilation(seg_data, footprint=ndi.iterate_structure(ndi.generate_binary_structure(3, 3), dilation_size))

    # Make TorchIO images
    tio_img=tio.ScalarImage(tensor=image_data[None, ...], affine=image_affine)

    # Define the segmentation as a ScalarImage or LabelMap based on the interpolation method
    if interpolation == 'linear':
        tio_seg=tio.ScalarImage(tensor=seg_data[None, ...], affine=seg_affine)
    else:
        tio_seg=tio.LabelMap(tensor=seg_data[None, ...], affine=seg_affine)

    # Resample the segmentation to the image space
    tio_output_seg = tio.Resample(tio_img)(tio_seg)
    output_seg_data = tio_output_seg.data.numpy()[0, ...]
    if interpolation == 'linear':
        output_seg_data = output_seg_data.astype(np.float32)
    else:
        output_seg_data = output_seg_data.round().astype(np.uint8)

    if interpolation == 'label':
        # Initialize the output segmentation to zeros
        com_output_seg_data = np.zeros_like(output_seg_data)

        # Get the labels in the segmentation
        labels = [_ for _ in np.unique(output_seg_data) if _ != 0]

        # Get the center of mass of each label
        com = ndi.center_of_mass(output_seg_data != 0, output_seg_data, labels)

        # Set the labels at the center of mass
        for label, idx in zip(labels, com):
            # Round the center of mass index
            idx = np.round(idx).astype(int)

            # Clip the index to the segmentation shape
            idx = np.maximum(np.minimum(idx, np.array(com_output_seg_data.shape) - pad_width - 1), [pad_width] * 3)

            # Set the label at the index
            com_output_seg_data[tuple(idx)] = label

        # Remove the padding
        output_seg_data = com_output_seg_data[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]

    output_seg = nib.Nifti1Image(output_seg_data, image.affine, seg.header)

    return output_seg

if __name__ == '__main__':
    main()