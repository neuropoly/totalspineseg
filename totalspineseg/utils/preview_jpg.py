import sys, argparse, textwrap
from PIL import Image
import multiprocessing as mp
from functools import partial
from pathlib import Path
from nibabel import freesurfer
import numpy as np
from tqdm.contrib.concurrent import process_map
import torchio as tio
import warnings

warnings.filterwarnings("ignore")

EXT = ['nii.gz', 'mgz', 'nii', 'mgh', 'nrrd', 'nhdr']

def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes .nii.gz or .mgz image with segmentation files.
            It combines the specified slice of the image and segmentation files and saves the result as a JPG image
            in the specified output folder.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            preview_jpg -i images -s labels -o preview
            For BIDS:
            preview_jpg -i . -s derivatives/labels -o derivatives/preview --seg-suffix "_seg" -d "sub-" -u "anat"
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--output-dir', '-o', type=Path, required=True,
        help='The folder where output combined JPG images will be saved (required).'
    )
    parser.add_argument(
        '--segs-dir', '-s', type=Path, default=None,
        help='The folder where input NIfTI segmentation files are located.'
    )
    parser.add_argument(
        '--subject-dir', '-d', type=str, default=None, nargs='?', const='',
        help=' '.join(f'''
            Is every subject has its oen direcrory.
            If this argument will be provided without value it will look for any directory in the segmentation directory.
            If value also provided it will be used as a prefix to subject directory, defaults to False (no subjet directory).
        '''.split())
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
        '--output-suffix', type=str, default='',
        help='Suffix to add to the output jpg, defaults to "".'
    )
    parser.add_argument(
        '--seg-suffix', type=str, default='',
        help='Segmentation suffix, defaults to "".'
    )
    parser.add_argument(
        '--orient', '-t', type=str, choices=['sag', 'ax', 'cor'], default='sag',
        help='Orientation of the output slice (sagittal, axial, or coronal). Default is "sag".'
    )
    parser.add_argument(
        '--sliceloc', '-l', type=float, default=0.5,
        help='Slice location within the specified orientation (0-1). Default is 0.5 (middle slice).'
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

    # Get the command-line argument values
    images_path = args.images_dir
    output_path = args.output_dir
    segs_path = args.segs_dir
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    image_suffix = args.image_suffix
    output_suffix = args.output_suffix
    seg_suffix = args.seg_suffix
    orient = args.orient
    sliceloc = args.sliceloc
    override = args.override
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            output_dir = "{output_path}"
            segs_path = "{segs_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            output_suffix = "{output_suffix}"
            seg_suffix = "{seg_suffix}"
            orient = "{orient}"
            sliceloc = {sliceloc}
            override = {override}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    preview_jpg_mp(
        images_path=images_path,
        output_path=output_path,
        segs_path=segs_path,
        subject_dir=subject_dir,
        subject_subdir=subject_subdir,
        prefix=prefix,
        image_suffix=image_suffix,
        output_suffix=output_suffix,
        seg_suffix=seg_suffix,
        orient=orient,
        sliceloc=sliceloc,
        override=override,
        max_workers=max_workers,
        quiet=quiet,
    )

def preview_jpg_mp(
        images_path,
        output_path,
        segs_path=None,
        subject_dir=None,
        subject_subdir='',
        prefix='',
        image_suffix='_0000',
        output_suffix='',
        seg_suffix='',
        orient='sag',
        sliceloc=0.5,
        override=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    images_path = Path(images_path)
    output_path = Path(output_path)
    segs_path = segs_path and Path(segs_path)

    glob_pattern = ""
    if subject_dir is not None:
        glob_pattern += f"{subject_dir}*/"
    if len(subject_subdir) > 0:
        glob_pattern += f"{subject_subdir}/"
    glob_pattern += f'{prefix}*{image_suffix}'

    # Process the NIfTI image and segmentation files
    image_path_list = [_ for __ in [list(images_path.glob(f'{glob_pattern}.{e}')) for e in EXT] for _ in __]
    image_ext_list = [[e for e in EXT if _.name.endswith(e)][0] for _ in image_path_list]
    output_path_list = [output_path / i.relative_to(images_path).parent / i.name.replace(f'{image_suffix}.{e}', f'_{orient}_{sliceloc}{output_suffix}.jpg') for i, e in zip(image_path_list, image_ext_list)]
    seg_path_list = [segs_path and segs_path / i.relative_to(images_path).parent / i.name.replace(f'{image_suffix}.{e}', f'{seg_suffix}') for i, e in zip(image_path_list, image_ext_list)]
    seg_path_list = [segs_path and ([_.parent / f'{_.name}.{e}' for e in EXT if (_.parent / f'{_.name}.{e}').is_file()] + [None])[0] for _ in seg_path_list]

    process_map(
        partial(
            _preview_jpg,
            orient=orient,
            sliceloc=sliceloc,
            override=override,
        ),
        image_path_list,
        output_path_list,
        seg_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _preview_jpg(
        image_path,
        output_path,
        seg_path=None,
        orient='sag',
        sliceloc=0.5,
        override=False,
    ):
    '''
    Combine the specified slice of the image and possibly the segmentation file and save the result as a JPG image.
    '''
    image_path = Path(image_path)
    output_path = Path(output_path)
    seg_path = seg_path and Path(seg_path)

    # If the output image already exists and we are not overriding it, return
    if not override and output_path.exists():
        return

    try:
        image = tio.ScalarImage(image_path)
        image.affine
    except:
        image = freesurfer.load(image_path)
        image = tio.ScalarImage(tensor=image.get_fdata()[None, ...], affine=image.affine)
    image = tio.ToCanonical()(image)
    image = tio.Resample((1, 1, 1))(image)

    image_data = image.data.squeeze().numpy().astype(np.float64)

    # Find the specified slice
    axis={'sag': 0, 'cor': 1, 'ax': 2}[orient]
    slice_index = int(sliceloc * image_data.shape[axis])
    slice_img = image_data.take(slice_index, axis=axis)

    # Normalize the slice to the range 0-255
    slice_img = (255 * (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))).astype(np.uint8)

    # Repeat the grayscale slice 3 times to create a color image
    slice_img = np.repeat(slice_img[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

    # Create a blank color image with the same dimensions as the input image
    output_data = np.zeros_like(slice_img, dtype=np.uint8)

    if seg_path and seg_path.is_file():
        try:
            seg = tio.LabelMap(seg_path)
            seg.affine
        except:
            seg = freesurfer.load(seg_path)
            seg = tio.LabelMap(tensor=np.asanyarray(seg.dataobj).astype(np.uint8)[None, ...], affine=seg.affine)
        seg = tio.ToCanonical()(seg)
        seg = tio.Resample(image)(seg)

        seg_data = seg.data.squeeze().numpy().round().astype(np.uint8)

        slice_seg = seg_data.take(slice_index, axis=axis)

        # Generate consistent random colors for each label
        unique_labels = np.unique(slice_seg).astype(int)
        colors = {}
        for label in unique_labels:
            rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(label * 10)))
            colors[label] = rs.randint(0, 255, 3)

        # Apply the segmentation mask to the image and assign colors
        for label, color in colors.items():
            if label != 0:  # Ignore the background (label 0)
                mask = slice_seg == label
                output_data[mask] = color

    output_data = np.where(output_data > 0, output_data, slice_img)

    # Rotate the image 90 degrees counter-clockwise and flip it vertically
    output_data = np.flipud(output_data)
    output_data = np.rot90(output_data, k=1)

    # Create an Image object from the output Image object as a JPG file
    output_image = Image.fromarray(output_data, mode="RGB")

    # Make sure output directory exists and save the image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)

if __name__ == '__main__':
    main()