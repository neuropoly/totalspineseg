import sys, argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import nibabel as nib
import numpy as np
import torchio as tio
import gryds
import warnings

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
        It apply transformation on the image and the segmentation to make augmented image.'''
        ),
        epilog=textwrap.dedent('''
            Examples:
            generate_augmentations -i images -s labels -o images -g labels
            For BIDS:
            generate_augmentations -i . -s derivatives/labels -o . -g derivatives/labels --image-suffix "" --output-image-suffix "" --seg-suffix "_seg" --output-seg-suffix "_seg" -d "sub-" -u "anat"
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
        '--seg-classes', type=parse_class, nargs='+', default=[parse_class(_) for _ in ['202-224', '18-41,92', '200', '201', ]],
        help='Define classes of labels for per class augmentation. Default 202-224 18-41,92 200 201 (IVDs, vertebrae, disc, csf).'
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
    output_segs_path = args.output_segs_dir
    subject_dir = args.subject_dir
    subject_subdir = args.subject_subdir
    prefix = args.prefix
    image_suffix = args.image_suffix
    seg_suffix = args.seg_suffix
    output_image_suffix = args.output_image_suffix
    output_seg_suffix = args.output_seg_suffix
    augmentations_per_image = args.augmentations_per_image
    labels2image = args.labels2image
    seg_classes = args.seg_classes
    max_workers = args.max_workers
    verbose = args.verbose
    
    # Print the argument values if verbose is enabled
    if verbose:
        print(textwrap.dedent(f''' 
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            segs_path = "{segs_path}"
            output_images_path = "{output_images_path}"
            output_segs_path = "{output_segs_path}"
            subject_dir = "{subject_dir}"
            subject_subdir = "{subject_subdir}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            seg_suffix = "{seg_suffix}"
            output_image_suffix = "{output_image_suffix}"
            output_seg_suffix = "{output_seg_suffix}"
            augmentations_per_image = "{augmentations_per_image}"
            labels2image = "{labels2image}"
            seg_classes = "{seg_classes}"
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
    partial_generate_augmentations = partial(
        generate_augmentations,
        images_path=images_path,
        segs_path=segs_path,
        output_images_path=output_images_path,
        output_segs_path=output_segs_path,
        image_suffix=image_suffix,
        seg_suffix=seg_suffix,
        output_image_suffix=output_image_suffix,
        output_seg_suffix=output_seg_suffix,
        augmentations_per_image=augmentations_per_image,
        labels2image=labels2image,
        seg_classes=seg_classes,
    )

    with mp.Pool() as pool:
        process_map(partial_generate_augmentations, images_path_list, max_workers=max_workers)


def generate_augmentations(
        image_path,
        images_path,
        segs_path,
        output_images_path,
        output_segs_path,
        image_suffix,
        seg_suffix,
        output_image_suffix,
        output_seg_suffix,
        augmentations_per_image,
        labels2image,
        seg_classes,
    ):
    
    seg_path = segs_path / image_path.relative_to(images_path).parent /  image_path.name.replace(f'{image_suffix}.nii.gz', f'{seg_suffix}.nii.gz')
    
    if not seg_path.is_file():
        print(f'Segmentation file not found: {seg_path}')
        return
    
    image = nib.load(image_path)
    image_data = image.get_fdata()
    seg = nib.load(seg_path)
    seg_data = seg.get_fdata().astype(np.uint8)

    for i in range(augmentations_per_image):
        output_image_path = output_images_path / image_path.relative_to(images_path).parent / image_path.name.replace(f'{image_suffix}.nii.gz', f'_a{i}{output_image_suffix}.nii.gz')
        output_seg_path = output_segs_path / image_path.relative_to(images_path).parent / seg_path.name.replace(f'{seg_suffix}.nii.gz', f'_a{i}{output_seg_suffix}.nii.gz')

        augs = []

        # Contrast augmentation
        if np.random.rand() < 0.8:
                _aug_redistribute_seg = partial(aug_redistribute_seg, classes=seg_classes)
                augs.append(np.random.choice([
                    _aug_redistribute_seg,
                    aug_gamma,
                    np.random.choice([
                        aug_log,
                        aug_sqrt,
                        aug_exp,
                        aug_sin,
                        aug_sig,
                    ]),
                ], p=[0.4, 0.3, 0.3]))

        # Image form segmentation augmentation
        elif labels2image and np.random.rand() < 0.5:
            _aug_labels2image = partial(aug_labels2image, classes=seg_classes)
            augs.append(_aug_labels2image)

        # Inverse color augmentation
        if np.random.rand() < 0.3:
            augs.append(aug_inverse)

        # Artifacts augmentation
        if np.random.rand() < 0.7:
            augs.append(np.random.choice([
                aug_motion,
                aug_ghosting,
                aug_spike,
                aug_bias_field,
                aug_blur,
                aug_noise,
            ]))

        # Spatial augmentation
        if np.random.rand() < 0.3:
            augs.append(aug_flip)

        if np.random.rand() < 0.7:
            augs.append(np.random.choice([
                aug_bspline,
                aug_aff,
                aug_elastic,
            ]))

        # Anisotropy augmentation
        downsampling = max(1, 7 / max(image.header.get_zooms()))
        _aug_anisotropy = partial(aug_anisotropy, downsampling=(min(1.5, downsampling), downsampling))
        if np.random.rand() < 0.7:
            augs.append(_aug_anisotropy)

        # Augment the images
        output_image_data, output_seg_data = image_data, seg_data
        for a in augs:
            output_image_data, output_seg_data = a(output_image_data, output_seg_data)
            # output_seg_data = output_seg_data.astype(np.uint8)

        # Create result
        output_image = nib.Nifti1Image(output_image_data, image.affine, image.header)
        output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)
        output_seg.set_data_dtype(np.uint8)

        # Make sure output directory exists
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        output_seg_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save mapped segmentation
        nib.save(output_image, output_image_path)
        nib.save(output_seg, output_seg_path)
        # print(f"\n{output_seg_path.name.replace('.nii.gz', '')}: {[a.func.__name__ if isinstance(a, partial) else a.__name__ for a in augs]}", end='')

def aug_redistribute_seg(img, seg, factor=(-0.2, 0.7), classes=None):

    _seg = seg
    # Ensure factor is a list
    factor=np.array([factor]).flat

    if classes:
        _seg = combine_classes(_seg, classes)

    # Compute original mean, std and min/max values
    original_mean, original_std = np.mean(img), np.std(img)
    img_min, img_max = np.min(img), np.max(img)

    # Normlize
    img = (img - original_mean) / original_std
    img = (img - img.min()) / (img.max() - img.min())

    # Get the unique label values (excluding 0)
    labels = [_ for _ in np.unique(_seg) if _ != 0]
    
    to_add = np.zeros_like(img)
    # Loop over each label value
    for l in labels:
        # Get the mask for the current label
        l_mask = (_seg == l)
        l_mean = np.mean(img[l_mask])
        l_std = np.std(img[l_mask])

        # Change std
        l_std *= np.random.uniform(0.9, 1.1)

        # Calculate the normal distribution
        exponent = -((img - l_mean) ** 2) / (2 * l_std ** 2)
        coeff = 1 / (l_std * np.sqrt(2 * np.pi))
        to_add += coeff * np.exp(exponent) * np.random.uniform(-1, 1)

    img += (to_add - to_add.min()) / ((to_add.max() - to_add.min()) / np.random.uniform(factor[0], factor[-1]))

    # Return to original range
    img = np.interp(img, (img_min, img_max), (img_min, img_max))

    return img, seg

def aug_transform(img, seg, transform):
    # Compute original mean, std and min/max values
    original_mean, original_std = np.mean(img), np.std(img)
    img_min, img_max = np.min(img), np.max(img)

    # Generate random mean and std
    random_mean = np.random.rand() * (img_max - img_min) + img_min
    random_std = (np.random.rand() * 1.5 + 0.5) * original_std

    # Normlize
    img = (img - original_mean) / original_std
    img = (img - img.min()) / (img.max() - img.min())

    # Transform
    img = transform(img)

    # Redistribution
    img = img  * random_std + random_mean

    # Return to original range
    img = np.interp(img, (img_min, img_max), (img_min, img_max))

    return img, seg

def aug_log(img, seg):
    return aug_transform(img, seg, np.log)

def aug_sqrt(img, seg):
    return aug_transform(img, seg, np.sqrt)

def aug_sin(img, seg):
    return aug_transform(img, seg, np.sin)

def aug_exp(img, seg):
    return aug_transform(img, seg, np.exp)

def aug_sig(img, seg):
    return aug_transform(img, seg, lambda x:1/(1 + np.exp(-x)))

def aug_inverse(img, seg):
    img = img.min() + img.max() - img
    return img, seg

def aug_bspline(img, seg):
    grid = np.random.rand(3, 3, 3, 3)

    bspline = gryds.BSplineTransformation((grid-.5)/5)
    grid[:,0] += ((grid[:,0] > 0) * 2 - 1) * .9 # Increase the effect on the Y-axis
    return gryds.Interpolator(img).transform(bspline), gryds.Interpolator(seg, order=0).transform(bspline).astype(np.uint8)

def aug_flip(img, seg):
    subject = tio.RandomFlip(axes=('LR',))(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_aff(img, seg):
    subject = tio.RandomAffine()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_elastic(img, seg):
    subject = tio.RandomElasticDeformation(max_displacement=40)(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_anisotropy(img, seg, downsampling=7):
    subject = tio.RandomAnisotropy(downsampling=downsampling)(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_motion(img, seg):
    subject = tio.RandomMotion()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_ghosting(img, seg):
    subject = tio.RandomGhosting()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_spike(img, seg):
    subject = tio.RandomSpike()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_bias_field(img, seg):
    subject = tio.RandomBiasField()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))  
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_blur(img, seg):
    subject = tio.RandomBlur()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_noise(img, seg):
    original_mean, original_std = np.mean(img), np.std(img)
    img = (img - original_mean) / original_std
    subject = tio.RandomNoise()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    img = img  * original_std + original_mean
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_swap(img, seg):
    subject = tio.RandomSwap()(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def aug_labels2image(img, seg, classes=None):
    _seg = seg
    if classes:
        _seg = combine_classes(seg, classes)
    subject = tio.RandomLabelsToImage(label_key="seg", image_key="image")(tio.Subject(
        seg=tio.LabelMap(tensor=np.expand_dims(_seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy(), seg

def aug_gamma(img, seg):
    subject = tio.RandomGamma((-1, 1))(tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(img, axis=0)),
        seg=tio.LabelMap(tensor=np.expand_dims(seg, axis=0))
    ))
    return subject.image.data.squeeze().numpy(), subject.seg.data.squeeze().numpy().astype(np.uint8)

def parse_class(c):
    c = [_.split('-') for _ in c.split(',')]
    c = tuple(__ for _ in c for __ in list(range(int(_[0]), int(_[-1]) + 1)))
    return c

def combine_classes(seg, classes):
    seg = np.zeros_like(_seg:=seg)
    for i, c in enumerate(classes):
        seg[np.isin(_seg, c)] = i + 1
    return seg

if __name__ == '__main__':
    main()