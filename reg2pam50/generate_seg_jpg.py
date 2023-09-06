import os, sys, argparse, textwrap, json
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel import freesurfer
from PIL import Image
from nilearn import image as nl_image


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
        '--images-dir', '-i', type=DirPath(), default='.',
        help='The folder where input NIfTI images files are located. Default is ".".'
    )
    parser.add_argument(
        '--segs-dir', '-s', type=DirPath(), default='derivatives/labels',
        help='The folder where input NIfTI segmentation files are located. Default is "derivatives/labels".'
    )
    parser.add_argument(
        '--segs-suffix', '-f', type=str, default='_PAM50_seg',
        help='The segmentation suffix to add to the image name. Default is "_PAM50_seg".'
    )
    parser.add_argument(
        '--output-dir', '-o', type=DirPath(True), default='derivatives/preview',
        help='The folder where output combined JPG images will be saved. Default is "derivatives/preview".'
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
    segs_suffix = args.segs_suffix
    output_path = args.output_dir
    orient = args.orient
    sliceloc = args.sliceloc
    override = args.override
    verbose = args.verbose
    
    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f''' 
            Running with the following arguments:
            images_path = "{images_path}"
            segs_path = "{segs_path}"
            segs_suffix = "{segs_suffix}"
            output_dir = "{output_path}"
            orient = "{orient}"
            sliceloc = "{sliceloc}"
            override = "{override}"
            verbose = "{verbose}"
        '''))

    # Process the NIfTI image and segmentation files
    subject_dir_list = list(images_path.glob('sub-*'))
    for i, subject_dir in enumerate(subject_dir_list):
        subject = subject_dir.name
        if verbose: print(f'Working on {subject} ({i + 1}/{len(subject_dir_list)})')
        if not (subject_dir / 'anat').is_dir():
            if verbose: print(f'Warning: "anat" folder not found for {subject}')
            continue
        for image_path in (subject_dir / 'anat').glob('*.nii.gz'):
            image_name = image_path.name.replace('.nii.gz', '')
            segs_suffix_output = segs_suffix if len(segs_suffix) == 0 or segs_suffix[0] == "_" else f'_{segs_suffix}'
            output_jpg_path = output_path / f'{image_name}{segs_suffix_output}_{orient}_{sliceloc}.jpeg'
            
            # Check if the output file exists and if the override flag is set to 0
            if output_jpg_path.exists() and not override:
                if verbose: print(f"Output file {output_jpg_path} already exists, skipping.")
                continue
            
            seg_path = segs_path / subject / 'anat' / f'{image_name}{segs_suffix}.nii.gz'
            if not seg_path.is_file():
                seg_path = segs_path / subject / 'anat' / f'{image_name}{segs_suffix}.mgz'
            if not seg_path.is_file():
                seg_path = segs_path / subject / 'anat' / f'{segs_suffix}.nii.gz'
            if not seg_path.is_file():
                seg_path = segs_path / subject / 'anat' / f'{segs_suffix}.mgz'
            if not seg_path.is_file():
                if verbose: print(f'Warning: Segmentation file not found segmentation file for {image_path}')
                continue
            if verbose: print(f"Processing {image_path.name} and {seg_path.name}...")

            create_slice_seg_jpg(image_path, seg_path, output_jpg_path, orient, sliceloc)
            print(f"Saved combined JPG image to {output_jpg_path}")

def create_slice_seg_jpg(nifti_image_path, segmentation_path, jpg_output_path, orient, sliceloc):
    # Set the seed to ensure consistent colors for each label
    np.random.seed(42)

    # Load the NIfTI image file
    img = nib.load(nifti_image_path)

    # Load the segmentation file (NIfTI or MGZ)
    if segmentation_path.suffix == '.mgz':
        seg = freesurfer.load(segmentation_path)
        # Convert the MGHImage to a NIfTI image
        seg = nib.Nifti1Image(seg.get_fdata(), seg.affine)
    else:
        seg = nib.load(segmentation_path)

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

    # Save the Image object as a JPG file
    jpg_image.save(jpg_output_path)


class DirPath(object):
    """
    Get path parameter from argparse and return it as pathlib Path object.

    Args:
    create (bool): Indicate if the directorie should be created. Default: False.
    """

    def __init__(self, create:bool=False):
        self.create = create

    def __call__(self, dir):

        path = Path(dir)

        # Create dir if create was specified
        if self.create and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except: pass

        # Test if path exists
        if path.is_dir():
            return path
        else:
            raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')


if __name__ == '__main__':
    main()