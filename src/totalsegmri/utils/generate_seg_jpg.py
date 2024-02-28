import sys, argparse, textwrap
import numpy as np
import nibabel as nib
from nibabel import freesurfer
from PIL import Image
from nilearn import image as nl_image
import multiprocessing as mp
from functools import partial
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import warnings

warnings.filterwarnings("ignore")

def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
        It combines the specified slice of the image and segmentation files and saves the result as a JPG image
        in the specified output folder.'''
        ),
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
        help='The folder where output combined JPG images will be saved (required).'
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
        '--seg-suffix', type=str, default='',
        help='Segmentation suffix, defaults to "".'
    )
    parser.add_argument(
        '--image-suffix', type=str, default='_0000',
        help='Image suffix, defaults to "_0000".'
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
        '--override', '-r', type=int, default=0, choices=[0, 1],
        help='Override existing output files. 0: Do not override, 1: Override. Default is 0.'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=min(32, mp.cpu_count() + 4),
        help='Max worker to run in parallel proccess, defaults to min(32, mp.cpu_count() + 4).'
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
    seg_suffix = args.seg_suffix
    image_suffix = args.image_suffix
    orient = args.orient
    sliceloc = args.sliceloc
    override = args.override
    max_workers = args.max_workers
    verbose = args.verbose
    
    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f''' 
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            segs_path = "{segs_path}"
            output_dir = "{output_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            seg_suffix = "{seg_suffix}"
            image_suffix = "{image_suffix}"
            orient = "{orient}"
            sliceloc = "{sliceloc}"
            override = "{override}"
            max_workers = "{max_workers}"
            verbose = "{verbose}"
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
    partial_generate_seg_jpg = partial(
        generate_seg_jpg,
        segs_path=segs_path,
        images_path=images_path,
        output_path=output_path,
        seg_suffix=seg_suffix,
        image_suffix=image_suffix,
        orient=orient,
        sliceloc=sliceloc,
        override=override,
    )

    with mp.Pool() as pool:
        process_map(partial_generate_seg_jpg, images_path_list, max_workers=max_workers)
    

def generate_seg_jpg(image_path, segs_path, images_path, output_path, seg_suffix, image_suffix, orient, sliceloc, override):
    
    seg_path = segs_path / image_path.relative_to(images_path).parent /  image_path.name.replace(f'{image_suffix}.nii.gz', f'{seg_suffix}.nii.gz')
    if not seg_path.is_file():
        seg_path = seg_path.parent / seg_path.name.replace('.nii.gz', '.mgz')
    output_jpg_path = output_path / image_path.name.replace(f'{image_suffix}.nii.gz', f'_{orient}_{sliceloc}.jpg')
    
    if not image_path.is_file() or not seg_path.is_file():
        return
    
    # Check if the output file exists and if the override flag is set to 0
    if output_jpg_path.exists() and not override:
        return

    # Set the seed to ensure consistent colors for each label
    np.random.seed(42)

    # Load the NIfTI image file
    img = nib.load(image_path)

    # Load the segmentation file (NIfTI or MGZ)
    if seg_path.suffix == '.mgz':
        seg = freesurfer.load(seg_path)
        # Convert the MGHImage to a NIfTI image
        seg = nib.Nifti1Image(seg.get_fdata(), seg.affine)
    else:
        seg = nib.load(seg_path)

    # Reorient the input image to RAS
    img = nib.as_closest_canonical(img)
    seg = nib.as_closest_canonical(seg)

    # Resample the segmentation to 1X1 mm
    seg = nl_image.resample_img(seg, np.eye(3), interpolation='nearest')

    # Resample the image to the same space as the segmentation
    img = nl_image.resample_to_img(img, seg)

    # Convert the image and segmentation data into numpy arrays
    img_data = img.get_fdata()
    seg_data = seg.get_fdata()

    # Find the specified slice
    axis={'sag': 0, 'cor': 1, 'ax': 2}[orient]
    slice_index = int(sliceloc * img_data.shape[axis])
    slice_img = img_data.take(slice_index, axis=axis)
    slice_seg = seg_data.take(slice_index, axis=axis)

    # Generate consistent random colors for each label
    unique_labels = np.unique(slice_seg).astype(int)
    colors = {}
    for label in unique_labels:
        np.random.seed(label)
        colors[label] = np.random.randint(0, 255, 3)

    # Create a blank color image with the same dimensions as the input image
    color_img = np.zeros((*slice_img.shape, 3), dtype=np.uint8)

    # Apply the segmentation mask to the image and assign colors
    for label, color in colors.items():
        if label != 0:  # Ignore the background (label 0)
            mask = slice_seg == label
            color_img[mask] = color

    # Normalize the slice to the range 0-255
    normalized_slice = (255 * (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))).astype(np.uint8)

    # Combine the original grayscale image with the colored segmentation mask
    grayscale_img = np.repeat(normalized_slice[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    combined_img = np.where(color_img > 0, color_img, grayscale_img)

    # Rotate the image 90 degrees counter-clockwise and flip it vertically
    output_img = np.flipud(combined_img)
    output_img = np.rot90(output_img, k=1)

    # Create an Image object from the output image
    jpg_image = Image.fromarray(output_img, mode="RGB")

    # Make sure output directory exists
    output_jpg_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the Image object as a JPG file
    jpg_image.save(output_jpg_path)

if __name__ == '__main__':
    main()