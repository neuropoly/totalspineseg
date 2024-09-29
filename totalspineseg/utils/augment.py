import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import nibabel as nib
import numpy as np
import torchio as tio
import gryds
import scipy.ndimage as ndi
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

rs = np.random.RandomState()

def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
            It apply transformation on the image and the segmentation to make augmented image.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            augment -i images -s labels -o images -g labels --labels2image --seg-classes 1 2 11-50 63-100
            For BIDS:
            augment -i . -s derivatives/labels -o . -g derivatives/labels --image-suffix "" --output-image-suffix "" --seg-suffix "_seg" --output-seg-suffix "_seg" -p "sub-*/anat/" --labels2image --seg-classes 1 2 11-50 63-100
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
        '--output-segs-dir', '-g', type=Path, required=True,
        help='The folder where output augmented segmentation will be saved with _a1, _a2 etc. suffixes (required).'
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
        '--output-seg-suffix', type=str, default='',
        help='Segmentation suffix for output, defaults to "".'
    )
    parser.add_argument(
        '--augmentations-per-image', '-n', type=int, default=7,
        help='Number of augmentation images to generate. Default is 7.'
    )
    parser.add_argument(
        '--labels2image', action="store_true", default=False,
        help='Use Random Labels To Image augmentation, defaults to false.'
    )
    parser.add_argument(
        '--seg-classes', type=parse_class, nargs='+', default=None,
        help='Define classes of labels for per class augmentation. Example: 1 2 11-50 63-100 for Spinal Cord, Canal, vertebrae and IVDs (Default to use each label as a separate class ).'
    )
    parser.add_argument(
        '--overwrite', '-r', action="store_true", default=False,
        help='If provided, overwrite existing output files, defaults to false (Do not overwrite).'
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
    output_images_path = args.output_images_dir
    output_segs_path = args.output_segs_dir
    prefix = args.prefix
    image_suffix = args.image_suffix
    seg_suffix = args.seg_suffix
    output_image_suffix = args.output_image_suffix
    output_seg_suffix = args.output_seg_suffix
    augmentations_per_image = args.augmentations_per_image
    labels2image = args.labels2image
    seg_classes = args.seg_classes
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            segs_path = "{segs_path}"
            output_images_path = "{output_images_path}"
            output_segs_path = "{output_segs_path}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            seg_suffix = "{seg_suffix}"
            output_image_suffix = "{output_image_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            augmentations_per_image = {augmentations_per_image}
            labels2image = {labels2image}
            seg_classes = {seg_classes}
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    augment_mp(
        images_path=images_path,
        segs_path=segs_path,
        output_images_path=output_images_path,
        output_segs_path=output_segs_path,
        prefix=prefix,
        image_suffix=image_suffix,
        seg_suffix=seg_suffix,
        output_image_suffix=output_image_suffix,
        output_seg_suffix=output_seg_suffix,
        augmentations_per_image=augmentations_per_image,
        labels2image=labels2image,
        seg_classes=seg_classes,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def augment_mp(
        images_path,
        segs_path,
        output_images_path,
        output_segs_path,
        prefix='',
        image_suffix='_0000',
        seg_suffix='',
        output_image_suffix='_0000',
        output_seg_suffix='',
        augmentations_per_image=7,
        labels2image=False,
        seg_classes=None,
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    images_path = Path(images_path)
    segs_path = Path(segs_path)
    output_images_path = Path(output_images_path)
    output_segs_path = Path(output_segs_path)

    glob_pattern = f'{prefix}*{image_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    image_path_list = list(images_path.glob(glob_pattern))
    seg_path_list = [segs_path / _.relative_to(images_path).parent / _.name.replace(f'{image_suffix}.nii.gz', f'{seg_suffix}.nii.gz') for _ in image_path_list]
    output_image_path_pattern_list = [output_images_path / _.relative_to(images_path).parent / _.name.replace(f'{image_suffix}.nii.gz', f'_a*{output_image_suffix}.nii.gz') for _ in image_path_list]
    output_seg_path_pattern_list = [output_segs_path / _.relative_to(segs_path).parent / _.name.replace(f'{seg_suffix}.nii.gz', f'_a*{output_seg_suffix}.nii.gz') for _ in seg_path_list]

    process_map(
        partial(
            _augment,
            augmentations_per_image=augmentations_per_image,
            labels2image=labels2image,
            seg_classes=seg_classes,
            overwrite=overwrite,
        ),
        image_path_list,
        seg_path_list,
        output_image_path_pattern_list,
        output_seg_path_pattern_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _augment(
        image_path,
        seg_path,
        output_image_path_pattern,
        output_seg_path_pattern,
        augmentations_per_image=7,
        labels2image=False,
        seg_classes=None,
        overwrite=False,
    ):
    '''
    Wrapper function to handle IO.
    '''
    image_path = Path(image_path)
    seg_path = Path(seg_path)
    output_image_path_pattern = Path(output_image_path_pattern)
    output_seg_path_pattern = Path(output_seg_path_pattern)

    if overwrite:
        for f in output_image_path_pattern.parent.glob(output_image_path_pattern.name):
            f.unlink()
        for f in output_seg_path_pattern.parent.glob(output_seg_path_pattern.name):
            f.unlink()

    if not seg_path.is_file():
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

    for i in range(augmentations_per_image):
        output_image_path = output_image_path_pattern.with_name(output_image_path_pattern.name.replace('*', f'{i}'))
        output_seg_path = output_seg_path_pattern.with_name(output_seg_path_pattern.name.replace('*', f'{i}'))

        # If the output image already exists and we are not overriding it, continue
        if not overwrite and (output_image_path.exists() or output_seg_path.exists()):
            continue

        output_image, output_seg = augment(
            image,
            seg,
            labels2image=labels2image,
            seg_classes=seg_classes
        )

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

def augment(
        image,
        seg,
        labels2image = False,
        seg_classes = None,
    ):
    '''
    Augment the image and the segmentation using a random sequence of transformations. Augmentation is performed on the image and the segmentation simultaneously to ensure consistency.
    Augmentation includes:
    - Contrast augmentation (Laplace, Gamma, Histogram Equalization, Log, Sqrt, Exp, Sin, Sig, Inverse)
    - Image from segmentation augmentation
    - Redistribute segmentation values
    - Artifacts augmentation (Motion, Ghosting, Spike, Bias Field, Blur, Noise)
    - Spatial augmentation (Flip, BSpline, Affine, Elastic)
    - Anisotropy augmentation

    Parameters
    ----------
    image : nibabel.Nifti1Image
        The input image to augment.
    seg : nibabel.Nifti1Image
        The input segmentation to augment.
    labels2image : bool, optional
        If True, use Random Labels To Image augmentation, by default False.
    seg_classes : list, optional
        Define classes of labels for per class augmentation, by default None. E.g. [[202, 224], [18, 19, 92]].

    Returns
    -------
    tuple[nibabel.Nifti1Image, nibabel.Nifti1Image]
        The augmented image and the augmented segmentation.
    '''
    image_data = np.asanyarray(image.dataobj).astype(np.float64)
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    _aug_redistribute_seg = partial(aug_redistribute_seg, classes=seg_classes)
    _aug_labels2image = partial(aug_labels2image, classes=seg_classes)
    downsampling = max(1, 7 / max(image.header.get_zooms()))
    _aug_anisotropy = partial(aug_anisotropy, downsampling=(min(1.5, downsampling), downsampling))

    # loop until valid augmentation is found
    for cur_cycle in range(100):

        augs = []

        # Contrast augmentation

        if rs.rand() < 0.1:
            augs.append(aug_laplace)

        else:

            if rs.rand() < 0.2:
                augs.append(aug_clip_values)

            # Image form segmentation augmentation
            if labels2image and rs.rand() < 0.15:
                augs.append(_aug_labels2image)

            if rs.rand() < 0.8:
                augs.append(_aug_redistribute_seg)

            if rs.rand() < 0.3:
                augs.append(aug_gamma)

            if rs.rand() < 0.1:
                augs.append(aug_histogram_equalization)

            if rs.rand() < 0.05:
                augs.append(aug_log)

            if rs.rand() < 0.05:
                augs.append(aug_sqrt)

            if rs.rand() < 0.05:
                augs.append(aug_exp)

            if rs.rand() < 0.05:
                augs.append(aug_sin)

            if rs.rand() < 0.05:
                augs.append(aug_sig)

        # Inverse color augmentation
        if rs.rand() < 0.3:
            augs.append(aug_inverse)

        # Artifacts augmentation
        if rs.rand() < 0.7:
            augs.append(rs.choice([
                aug_motion,
                aug_ghosting,
                aug_spike,
                aug_bias_field,
                aug_blur,
                aug_noise,
            ]))

        # Spatial augmentation
        if rs.rand() < 0.3:
            augs.append(aug_flip)

        if rs.rand() < 0.7:
            augs.append(rs.choice([
                aug_bspline,
                aug_aff,
                aug_elastic,
            ]))

        # Anisotropy augmentation
        if rs.rand() < 0.7:
            augs.append(_aug_anisotropy)

        # Augment the images
        output_image_data, output_seg_data = image_data, seg_data
        for a in augs:
            output_image_data, output_seg_data = a(output_image_data, output_seg_data)

        # Return to original range
        output_image_data = np.interp(output_image_data, (output_image_data.min(), output_image_data.max()), (image_data.min(), image_data.max()))

        # Validate augmentation results and break the loop if valid
        output_image_min, output_image_max = output_image_data.min(), output_image_data.max()
        output_image_range = output_image_max - output_image_min
        output_image_iqr5 = np.percentile(output_image_data, 52.5) - np.percentile(output_image_data, 47.5)
        output_image_iqr95 = np.percentile(output_image_data, 97.5) - np.percentile(output_image_data, 2.5)
        if output_image_range > 0 and output_image_iqr5 < 0.99 * output_image_range and output_image_iqr95 > 0.01 * output_image_range:
            break
        # print("Invalid augmentation, retrying...")
    # print(f"\nAugmentations: {[a.func.__name__ if isinstance(a, partial) else a.__name__ for a in augs]}", end='')

    output_image = nib.Nifti1Image(output_image_data, image.affine, image.header)
    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_image, output_seg

def aug_clip_values(img, seg, clip_max_fraction=0.4):
    """
    Augment the image by clipping its values to a random range.
    """
    # Calculate the minimum and maximum values of the input image
    min_val, max_val = img.min(), img.max()

    # Compute the range of values in the image
    range_val = max_val - min_val

    # Set a random lower clipping threshold
    clip_min = rs.uniform(0, clip_max_fraction / 2) * range_val + min_val

    # Set a random upper clipping threshold
    clip_max = rs.uniform(1 - clip_max_fraction / 2, 1) * range_val + min_val

    # Clip the image to the new thresholds
    img_clipped = np.clip(img, clip_min, clip_max)

    return img_clipped, seg

def aug_redistribute_seg(img, seg, classes=None, in_seg=0.2):
    """
    Augment the image by redistributing the values of the image within the
    regions defined by the segmentation.
    """
    _seg = seg
    in_seg_bool = 1 - rs.rand() <= in_seg

    if classes:
        _seg = combine_classes(_seg, classes)

    # Compute original mean, std and min/max values
    original_mean, original_std = np.mean(img), np.std(img)
    original_min, original_max = np.min(img), np.max(img)

    # Normlize
    img = (img - original_mean) / original_std
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min)

    # Get the unique label values (excluding 0)
    labels = [_ for _ in np.unique(_seg) if _ != 0]

    to_add = np.zeros_like(img)
    # Loop over each label value
    for l in labels:
        # Get the mask for the current label
        l_mask = (_seg == l)
        # Get mean and std of the current label
        l_mean, l_std = np.mean(img[l_mask]), np.std(img[l_mask])
        # Dilate the mask
        l_mask_dilate = ndi.binary_dilation(l_mask, ndi.iterate_structure(ndi.generate_binary_structure(3, 1), 3))
        # Create mask of the dilated mask excluding the original mask
        l_mask_dilate_excl = l_mask_dilate & ~l_mask
        # Get the mean and std of the substraction of mask from the dilated mask
        l_mean_dilate, l_std_dilate = np.mean(img[l_mask_dilate_excl]), np.std(img[l_mask_dilate_excl])

        redist_std = max(rs.uniform(0.4, 0.6) * abs((l_mean - l_mean_dilate) * l_std / l_std_dilate), 0.01)

        redist = partial(norm.pdf, loc=l_mean, scale=redist_std)

        if in_seg_bool:
            to_add[l_mask] += redist(img[l_mask]) * rs.uniform(-1, 1)
        else:
            to_add += redist(img) * rs.uniform(-1, 1)

    img += 2 * (to_add - to_add.min()) / (to_add.max() - to_add.min())

    # Return to original range
    img = np.interp(img, (img.min(), img.max()), (original_min, original_max))

    return img, seg

def aug_histogram_equalization(img, seg):
    img_min, img_max = img.min(), img.max()

    # Flatten the image and compute the histogram
    img_flattened = img.flatten()
    hist, bins = np.histogram(img_flattened, bins=256, range=[img_flattened.min(), img_flattened.max()])

    # Compute the normalized cumulative distribution function (CDF) from the histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf * (hist.max() / cdf.max())

    # Perform histogram equalization using the normalized CDF
    img = np.interp(img_flattened, bins[:-1], cdf_normalized).reshape(img.shape)

    # Rescale the image to the original minimum and maximum values
    img = np.interp(img, (img.min(), img.max()), (img_min, img_max))

    return img, seg

def aug_transform(img, seg, transform):
    """
    Augment the image by applying a given transformation function.
    """
    # Compute original mean, std and min/max values
    img_min, img_max = img.min(), img.max()
    # Normlize
    img = (img - img.mean()) / img.std()
    img = np.interp(img, (img.min(), img.max()), (0, 1))

    # Transform
    img = transform(img)

    # Return to original range
    img = np.interp(img, (img.min(), img.max()), (img_min, img_max))

    return img, seg

def aug_log(img, seg):
    return aug_transform(img, seg, lambda x:np.log(1 + x))

def aug_sqrt(img, seg):
    return aug_transform(img, seg, np.sqrt)

def aug_sin(img, seg):
    return aug_transform(img, seg, np.sin)

def aug_exp(img, seg):
    return aug_transform(img, seg, np.exp)

def aug_sig(img, seg):
    return aug_transform(img, seg, lambda x:1/(1 + np.exp(-x)))

def aug_laplace(img, seg):
    return aug_transform(img, seg, lambda x:np.abs(ndi.laplace(x)))

def aug_inverse(img, seg):
    img = img.min() + img.max() - img
    return img, seg

def aug_bspline(img, seg):
    grid = rs.rand(3, 3, 3, 3)

    bspline = gryds.BSplineTransformation((grid-.5)/5)
    grid[:,0] += ((grid[:,0] > 0) * 2 - 1) * .9 # Increase the effect on the Y-axis
    return gryds.Interpolator(img).transform(bspline).astype(np.float64), gryds.Interpolator(seg, order=0).transform(bspline).astype(np.uint8)

def aug_flip(img, seg):
    subject = tio.RandomFlip(axes=('LR',))(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_aff(img, seg):
    subject = tio.RandomAffine()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_elastic(img, seg):
    subject = tio.RandomElasticDeformation(max_displacement=40)(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_anisotropy(img, seg, downsampling=7):
    subject = tio.RandomAnisotropy(downsampling=downsampling)(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_motion(img, seg):
    subject = tio.RandomMotion()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_ghosting(img, seg):
    subject = tio.RandomGhosting()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_spike(img, seg):
    subject = tio.RandomSpike(intensity=(1, 2))(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_bias_field(img, seg):
    subject = tio.RandomBiasField()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_blur(img, seg):
    subject = tio.RandomBlur()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_noise(img, seg):
    original_mean, original_std = np.mean(img), np.std(img)
    img = (img - original_mean) / original_std
    subject = tio.RandomNoise()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    img = img  * original_std + original_mean
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_swap(img, seg):
    subject = tio.RandomSwap()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_labels2image(img, seg, leave_background=0.5, classes=None):
    _seg = seg
    if classes:
        _seg = combine_classes(seg, classes)
    subject = tio.RandomLabelsToImage(label_key="seg", image_key="image")(tio.Subject(
        seg=tio.LabelMap(tensor=np.expand_dims(_seg, axis=0))
    ))
    new_img = subject.image.data.squeeze().numpy().astype(np.float64)

    if rs.rand() < leave_background:
        img_min, img_max = np.min(img), np.max(img)
        _img = (img - img_min) / (img_max - img_min)

        new_img_min, new_img_max = np.min(new_img), np.max(new_img)
        new_img = (new_img - new_img_min) / (new_img_max - new_img_min)
        new_img[_seg == 0] = _img[_seg == 0]
        # Return to original range
        new_img = np.interp(new_img, (new_img.min(), new_img.max()), (img_min, img_max))

    return new_img, seg

def aug_gamma(img, seg):
    subject = tio.RandomGamma()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy().astype(np.float64), subject.seg.data.squeeze().numpy().astype(np.uint8)

def parse_class(c):
    c = [_.split('-') for _ in c.split(',')]
    c = tuple(__ for _ in c for __ in list(range(int(_[0]), int(_[-1]) + 1)))
    return c

def combine_classes(seg, classes):
    _seg = np.zeros_like(seg)
    for i, c in enumerate(classes):
        _seg[np.isin(seg, c)] = i + 1
    return _seg

if __name__ == '__main__':
    main()