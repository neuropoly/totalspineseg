
import os, sys, argparse, textwrap, json
from pathlib import Path
import numpy as np
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
        This script uses SynthSeg.brain_generator to generate image from calculated prior signal '''
        '''statistics (mean + std), and saves the rsults `image.nii.gz` and `labels.nii.gz` to output dir.
        '''),
        epilog=textwrap.dedent('''
        Example:
        generate_image -s segmentstions/sub-0287/anat/sub-0287_ct_seg.nii.gz -p priors -o output
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--seg', '-s', type= argparse.FileType('rb'), required=True,
        help='The folder where segmentations are located.'
    )
    parser.add_argument(
        '--prior-dir', '-p', type= DirPath(), required=True,
        help='The folder where calculated prior signal statistics (`prior_means.npy` ,`prior_stds.npy` '
        ' ,`labels.npy` and `classes.npy`) are located.'
    )
    parser.add_argument(
        '--output-dir', '-o', type= DirPath(True), required=True,
        help='The folder where output files (`image.nii.gz` and `labels.nii.gz`) will be saved to'
    )
    parser.add_argument(
        '--verbose', '-v', type= int, default=1, choices=[0, 1],
        help='verbose. 0: Display only errors/warnings, 1: Errors/warnings + info messages (default: 1)'
    )

    try:
        args = parser.parse_args()
    except BaseException as e:
        sys.exit()

    # Get args
    seg_file = args.seg.name
    prior_path = args.prior_dir
    output_path = args.output_dir
    verbose = args.verbose

    # Close files
    args.seg.close()

    if verbose:
        print(textwrap.dedent(f''' 
            Running with the following arguments:
            seg = "{seg_file}"
            prior_dir = "{prior_path}"
            output_dir = "{output_path}"
            verbose = {verbose}
        '''))

    # Get priors paths
    prior_means_path = prior_path / 'prior_means.npy'
    prior_stds_path = prior_path / 'prior_stds.npy'
    generation_labels_path = prior_path / 'labels.npy'
    generation_classes_path = prior_path / 'classes.npy'

    # Ensure priors exists
    if not all([prior_means_path.exists(), prior_stds_path.exists()]):
        print(f'`prior_means.npy` or `prior_stds.npy` was not found in {prior_path}')
        sys.exit()
    
    # Get priors values
    prior_means = f'{prior_means_path}'
    prior_stds = f'{prior_stds_path}'
    generation_labels = f'{generation_labels_path}' if generation_labels_path.exists() else None
    generation_classes = f'{generation_classes_path}' if generation_classes_path.exists() else None
    
    if verbose and not generation_labels_path.exists():
        print(f'{generation_labels} was not found')
    
    if verbose and not generation_classes_path.exists():
        print(f'{generation_classes} was not found')

    # Create synthetic image
    brain_generator = BrainGenerator(
        f'{seg_file}',
        # generation_labels=None,
        generation_labels=generation_labels,
        n_neutral_labels=None,
        output_labels=None,
        subjects_prob=None,
        batchsize=1,
        n_channels=1,
        # target_res=None,
        target_res=np.array([1., 1., 1.]),
        output_shape=160,
        output_div_by_n=None,
        prior_distributions='uniform',
        # generation_classes=None,
        generation_classes=generation_classes,
        # prior_means=None,
        prior_means = prior_means,
        # prior_stds=None,
        prior_stds = prior_stds,
        use_specific_stats_for_channel=False,
        mix_prior_and_random=False,
        # flipping=True,
        flipping=False,
        # scaling_bounds=.2,
        scaling_bounds=0.2,
        # rotation_bounds=15,
        rotation_bounds=20,
        # shearing_bounds=.012,
        shearing_bounds=0.05,
        translation_bounds=True,
        nonlin_std=3.,
        nonlin_scale=.04,
        randomise_res=True,
        #randomise_res=False,
        max_res_iso=4.,
        max_res_aniso=8.,
        # data_res=None,
        data_res=np.array([1., 1., 1.]),
        # thickness=None,
        thickness=np.array([1., 1., 1.]),
        bias_field_std=.7,
        # bias_scale=.025,
        bias_scale=0.05,
        return_gradients=False
    )

    im, lab = brain_generator.generate_brain()

    # save output image and label map

    utils.save_volume(
        im, brain_generator.aff, brain_generator.header,
        f"{output_path / 'image.nii.gz'}"
    )
    utils.save_volume(
        lab, brain_generator.aff, brain_generator.header,
        f"{output_path / 'labels.nii.gz'}"
    )


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

        # Create dir if create was specify
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
