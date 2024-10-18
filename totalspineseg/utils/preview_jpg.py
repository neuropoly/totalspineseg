import sys, argparse, textwrap, json
from PIL import Image, ImageDraw, ImageFont
import multiprocessing as mp
from functools import partial
from pathlib import Path
from nibabel import freesurfer
import numpy as np
import scipy.ndimage as ndi
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
            preview_jpg -i . -s derivatives/labels -o derivatives/preview --seg-suffix "_seg" -p "sub-*/anat/"
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
        '--levels', action="store_true", default=False,
        help='extend the segmentation ortogonal to the orien and dilate 3 voxels to ensure visibility of the levels. Usful for single point each label like vertebrae levels.'
    )
    parser.add_argument(
        '--label-text-right', '-ltr', type=str, nargs='+', default=[],
        help=' '.join('''
            JSON file or mapping from label integers to text labels to be placed on the right side.
            The format should be input_label:text_label without any spaces.
            For example, you can use a JSON file like right_labels.json containing {"1": "SC", "2": "Canal"},
            or provide mappings directly like 1:SC 2:Canal
        '''.split()),
    )
    parser.add_argument(
        '--label-text-left', '-ltl', type=str, nargs='+', default=[],
        help=' '.join('''
            JSON file or mapping from label integers to text labels to be placed on the left side.
            The format should be input_label:text_label without any spaces.
            For example, you can use a JSON file like left_labels.json containing {"1": "SC", "2": "Canal"},
            or provide mappings directly like 1:SC 2:Canal
        '''.split()),
    )
    parser.add_argument(
        '--overwrite', '-r', action="store_true", default=False,
        help='Overwrite existing output files, defaults to false (Do not overwrite).'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel process, defaults to multiprocessing.cpu_count().'
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
    prefix = args.prefix
    image_suffix = args.image_suffix
    output_suffix = args.output_suffix
    seg_suffix = args.seg_suffix
    orient = args.orient
    sliceloc = args.sliceloc
    levels = args.levels
    label_text_right_list = args.label_text_right
    label_text_left_list = args.label_text_left
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Load label_texts_right into a dict
    label_texts_right = load_label_texts(label_text_right_list, 'label-text-right')

    # Load label_texts_left into a dict
    label_texts_left = load_label_texts(label_text_left_list, 'label-text-left')

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            output_dir = "{output_path}"
            segs_path = "{segs_path}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            output_suffix = "{output_suffix}"
            seg_suffix = "{seg_suffix}"
            orient = "{orient}"
            sliceloc = {sliceloc}
            levels = {levels}
            label_texts_right = {label_texts_right}
            label_texts_left = {label_texts_left}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    preview_jpg_mp(
        images_path=images_path,
        output_path=output_path,
        segs_path=segs_path,
        prefix=prefix,
        image_suffix=image_suffix,
        output_suffix=output_suffix,
        seg_suffix=seg_suffix,
        orient=orient,
        sliceloc=sliceloc,
        levels=levels,
        label_texts_right=label_texts_right,
        label_texts_left=label_texts_left,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def load_label_texts(label_text_list, param_name):
    if len(label_text_list) == 1 and label_text_list[0][-5:] == '.json':
        # Load label mappings from JSON file
        with open(label_text_list[0], 'r', encoding='utf-8') as map_file:
            label_texts = json.load(map_file)
            # Ensure keys are ints
            label_texts = {int(k): v for k, v in label_texts.items()}
    else:
        try:
            label_texts = {int(l_in): l_out for l_in, l_out in map(lambda x: x.split(':'), label_text_list)}
        except:
            raise ValueError(f"Input param --{param_name} is not in the right structure. Make sure it is in the right format, e.g., 1:SC 2:Canal")
    return label_texts

def preview_jpg_mp(
        images_path,
        output_path,
        segs_path=None,
        prefix='',
        image_suffix='_0000',
        output_suffix='',
        seg_suffix='',
        orient='sag',
        sliceloc=0.5,
        levels=False,
        label_texts_right={},
        label_texts_left={},
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    images_path = Path(images_path)
    output_path = Path(output_path)
    segs_path = segs_path and Path(segs_path)

    glob_pattern = f'{prefix}*{image_suffix}'

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
            levels=levels,
            label_texts_right=label_texts_right,
            label_texts_left=label_texts_left,
            overwrite=overwrite,
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
        levels=False,
        label_texts_right={},
        label_texts_left={},
        overwrite=False,
    ):
    '''
    Combine the specified slice of the image and possibly the segmentation file and save the result as a JPG image.
    '''
    image_path = Path(image_path)
    output_path = Path(output_path)
    seg_path = seg_path and Path(seg_path)

    # If the output image already exists and we are not overriding it, return
    if not overwrite and output_path.exists():
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

    # Flip and rotate the image slice
    slice_img = np.flipud(slice_img)
    slice_img = np.rot90(slice_img, k=1)

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

        if levels:
            # Extend the segmentation ortogonal to orient
            seg_data = np.broadcast_to(np.max(seg_data, axis=axis, keepdims=True), seg_data.shape)
            # Dilate 3 voxels to ensure visibility of the levels
            seg_data = ndi.grey_dilation(seg_data, size=(3, 3, 3))

        slice_seg = seg_data.take(slice_index, axis=axis)

        # Flip and rotate the segmentation slice
        slice_seg = np.flipud(slice_seg)
        slice_seg = np.rot90(slice_seg, k=1)

        # Generate consistent random colors for each label
        unique_labels = np.unique(slice_seg).astype(int)
        colors = {}
        for label in unique_labels:
            rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(label * 10)))
            colors[label] = rs.randint(0, 255, 3)

        # Create a blank color image with the same dimensions as the input image
        output_data = np.zeros_like(slice_img, dtype=np.uint8)

        # Apply the segmentation mask to the image and assign colors
        for label, color in colors.items():
            if label != 0:  # Ignore the background (label 0)
                mask = slice_seg == label
                output_data[mask] = color

        output_data = np.where(output_data > 0, output_data, slice_img)
    else:
        output_data = slice_img
        unique_labels = []
        colors = {}

    # Create an Image object from the output data
    output_image = Image.fromarray(output_data, mode="RGB")

    # Draw text labels if label_texts are provided
    if (label_texts_right or label_texts_left) and seg_path and seg_path.is_file():
        draw = ImageDraw.Draw(output_image)
        # Use a bold TrueType font for better sharpness and boldness
        font = ImageFont.load_default(size=15)
        width, height = output_image.size
        used_positions = []
        for label in unique_labels:
            if label != 0:
                text = None
                side = None
                if label in label_texts_right:
                    text = label_texts_right[label]
                    side = 'right'
                elif label in label_texts_left:
                    text = label_texts_left[label]
                    side = 'left'

                if text and side:
                    mask = slice_seg == label
                    positions = np.argwhere(mask)
                    if positions.size > 0:
                        # Get the bounding box of the label
                        ys, xs = positions[:, 0], positions[:, 1]
                        x_min, x_max = xs.min(), xs.max()
                        y_min, y_max = ys.min(), ys.max()
                        # Start from just outside the label
                        if side == 'right':
                            x_new = x_max + 1
                        else:
                            x_new = x_min - 1
                        y_new = int((y_min + y_max) / 2)

                        # Ensure starting positions are within image bounds
                        x_new = min(max(0, x_new), width - 1)
                        y_new = min(max(0, y_new), height - 1)

                        # Search for a position outside the segmentation
                        found_position = False
                        if side == 'right':
                            for dx in range(0, width - x_new):
                                if slice_seg[y_new, min(x_new + dx, width - 1)] == 0:
                                    x_new = x_new + dx
                                    found_position = True
                                    break
                        else:
                            for dx in range(0, x_new + 1):
                                if slice_seg[y_new, max(x_new - dx, 0)] == 0:
                                    x_new = x_new - dx
                                    found_position = True
                                    break
                        if not found_position:
                            # Try moving down
                            for dy in range(1, height - y_new):
                                if slice_seg[min(y_new + dy, height - 1), x_new] == 0:
                                    y_new = y_new + dy
                                    found_position = True
                                    break
                        if not found_position:
                            # Try moving up
                            for dy in range(1, y_new + 1):
                                if slice_seg[max(y_new - dy, 0), x_new] == 0:
                                    y_new = y_new - dy
                                    found_position = True
                                    break

                        if found_position:
                            text_color = tuple(colors[label].tolist())

                            # Avoid overlapping labels
                            if (x_new, y_new) not in used_positions:
                                # Get text size
                                try:
                                    # For Pillow >= 8.0.0
                                    bbox = font.getbbox(text)
                                    text_width = bbox[2] - bbox[0]
                                    text_height = bbox[3] - bbox[1]
                                except AttributeError:
                                    try:
                                        # For older versions
                                        text_width, text_height = font.getsize(text)
                                    except AttributeError:
                                        # As a last resort, approximate text size
                                        text_width, text_height = draw.textsize(text, font=font)

                                # Adjust x_new for left side labels
                                if side == 'left':
                                    x_new = x_new - text_width

                                # Ensure text is within image bounds
                                x_new = min(max(0, x_new), width - text_width)
                                y_new = min(max(0, y_new), height - text_height)

                                # Draw outline by drawing text multiple times around the perimeter
                                outline_color = (255, 255, 255)  # White outline
                                outline_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                                for dx, dy in outline_offsets:
                                    draw.text((x_new + dx, y_new + dy), text, font=font, fill=outline_color)

                                # Draw the text over the outline multiple times to make it thicker
                                for _ in range(3):
                                    draw.text((x_new, y_new), text, fill=text_color, font=font)

                                used_positions.append((x_new, y_new))
                        else:
                            # No suitable position found, skip drawing the label
                            pass

    # Make sure output directory exists and save the image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path, quality=95)

if __name__ == '__main__':
    main()