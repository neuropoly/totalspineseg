from totalspineseg.utils.image import Image
import os, argparse
import numpy as np
from copy import deepcopy
from totalspineseg.inference import inference
from totalspineseg.init_inference import init_inference
from totalspineseg.utils.utils import ZIP_URLS, VERSION
from totalspineseg import models
import shutil
import importlib.resources
import torch

import multiprocessing as mp
mp.set_start_method('forkserver', force=True)

def main():
    parser = argparse.ArgumentParser(
        description='Test TotalSpineSeg on cropped images'
    )
    parser.add_argument(
        'input', type=str,
        help='The input folder containing the .nii.gz (or .nii) images.'
    )
    parser.add_argument(
        'output', type=str,
        help='The output folder where the model outputs will be stored.'
    )
    parser.add_argument(
        '--data-dir', '-d', type=str, default=None,
        help=' '.join(f'''
            The path to store the nnUNet data.
        '''.split())
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    input_path = args.input
    output_path = args.output

    # Use device
    device = torch.device('cuda')

    # Init data_path
    if not args.data_dir is None:
        data_path = args.data_dir
    elif 'TOTALSPINESEG_DATA' in os.environ:
        data_path = os.environ.get('TOTALSPINESEG_DATA', '')
    else:
        data_path = str(importlib.resources.files(models))

    # Default release to use
    default_release = list(ZIP_URLS.values())[0].split('/')[-2]

    # Install weights if not present
    init_inference(
        data_path=data_path,
        dict_urls=ZIP_URLS,
        quiet=False
        )

    # Create crop list
    crop_list = [{
        'xmin':0,
        'xmax':-1,
        'ymin':10,
        'ymax':160,
        'zmin':200,
        'zmax':400,
    }]
    crop_dir = os.path.join(output_path, 'crop')

    for file in os.listdir(input_path):
        if file.endswith('.nii.gz'):
            # Load input image
            img_path = os.path.join(input_path, file)
            img_full = Image(img_path).change_orientation('RPI')
            for crop in crop_list:
                # Create crop name
                crop_name = os.path.basename(img_path).replace(
                    '.nii.gz',
                    f'_xmin{crop["xmin"]}_xmax{crop["xmax"]}_ymin{crop["ymin"]}_ymax{crop["ymax"]}_zmin{crop["zmin"]}_zmax{crop["zmax"]}.nii.gz'
                )

                # Init path out for cropped image
                crop_path = os.path.join(crop_dir, crop_name)

                # Create folder if does not exists
                if not os.path.exists(crop_dir):
                    os.makedirs(crop_dir)

                # Crop image
                img_crop = crop_img(deepcopy(img_full), crop)
                img_crop.save(crop_path)

            # Output directory
            pred_dir = os.path.join(output_path, 'pred')

            # Call TotalSpineSeg
            inference(
                input_path=crop_dir,
                output_path=pred_dir,
                data_path=data_path,
                default_release=default_release,
                step1_only=True,
                keep_only=['step1_vert'],
                device=device
            )

            # Remove crop folder
            shutil.rmtree(crop_dir, ignore_errors=True)


def crop_img(img_in, crop_size):
    '''
    Based on https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/cropping.py
    Inputs:
        img_in: Image class
        crop_size: {
            xmin: 0,
            xmax: 0,
            ymin: 0,
            ymax: 0,
            zmin: 0,
            zmax: 0
        }
    '''
    data_crop = img_in.data[crop_size['xmin']:crop_size['xmax'],
                            crop_size['ymin']:crop_size['ymax'],
                            crop_size['zmin']:crop_size['zmax']]

    img_out = Image(param=data_crop, hdr=img_in.hdr)

    # adapt the origin in the qform matrix
    new_origin = np.dot(img_out.hdr.get_qform(), [crop_size['xmin'], crop_size['ymin'], crop_size['zmin'], 1])
    img_out.hdr.structarr['qoffset_x'] = new_origin[0]
    img_out.hdr.structarr['qoffset_y'] = new_origin[1]
    img_out.hdr.structarr['qoffset_z'] = new_origin[2]
    # adapt the origin in the sform matrix
    new_origin = np.dot(img_out.hdr.get_sform(), [crop_size['xmin'], crop_size['ymin'], crop_size['zmin'], 1])
    img_out.hdr.structarr['srow_x'][-1] = new_origin[0]
    img_out.hdr.structarr['srow_y'][-1] = new_origin[1]
    img_out.hdr.structarr['srow_z'][-1] = new_origin[2]

    return img_out

if __name__=='__main__':
    main()