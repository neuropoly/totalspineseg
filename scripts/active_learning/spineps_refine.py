"""
The aim of this script is to generate ground truth segmentation for TotalSpineSeg using Spineps' predictions (https://github.com/Hendrik-code/spineps). 
While the segmentation quality is good, quality control revealed the necessity for correction of the labeling. Also the C1 vertebrae is not segmented.
This script uses both TotalSpineSeg and Spineps prediction to generate better ground truths.
"""

import argparse, os, glob, json
from totalspineseg.utils.image import Image, zeros_like
import time 
import numpy as np

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Refine segmentation using nnInteractive.')
    parser.add_argument('-spineps', required=True, help='Path to the predictions of spineps (Required)')
    parser.add_argument('-totalspineseg', required=True, help='Path to the predictions of totalspineseg (Required)')
    parser.add_argument('-canal', required=True, help='Path to the canal segmentations (Required)')
    parser.add_argument('-ofolder', required=True, help='Path to the output folder (Required)')
    return parser

def create_json_file(path_json_out):
    """
    Create a json sidecar file
    :param path_file_out: path to the output file
    """
    
    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "totalspineseg",
                "Link": "https://github.com/neuropoly/totalspineseg",
                "Version": "r20250224",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                "Name": "spineps",
                "Link": "https://github.com/Hendrik-code/spineps",
                "Version": "v1.3.1",
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
    }
    with open(path_json_out, 'w') as f:
        json.dump(data_json, f, indent=4)
        print(f'Created: {path_json_out}')

def main():
    # Load parser
    parser = get_parser()
    args = parser.parse_args()

    # Define paths
    spineps_folder = args.spineps
    tss_folder = args.totalspineseg
    canal_folder = args.canal
    output_folder = args.ofolder

    # List paths
    spineps_files = [file for file in glob.glob(spineps_folder + "/*" + "_seg-vert_msk.nii.gz", recursive=True)]

    err_file = []
    for spineps_file in spineps_files:
        sub = os.path.basename(spineps_file).split('_')[0]

        # Fetch totalspineseg segmentation
        gl = glob.glob(tss_folder + "/" + sub + "_T2w.nii.gz", recursive=True) 
        if len(gl) > 1:
            raise ValueError(f'Multiple files detected for {sub}: {"\n".join(gl)}')
        
        tss_file = gl[0]

        # Fetch canal segmentation
        gl = glob.glob(canal_folder + "/" + sub + "_T2w_label-canal_seg.nii.gz", recursive=True) 
        if len(gl) > 1:
            raise ValueError(f'Multiple files detected for {sub}: {"\n".join(gl)}')
        
        canal_file = gl[0]

        # Load images
        tss = Image(tss_file).change_orientation('RSP')
        spineps = Image(spineps_file).change_orientation('RSP')
        canal = Image(canal_file).change_orientation('RSP')
        output = zeros_like(tss)

        # Fetch unique values from segmentation
        unique_tss = [val for val in np.unique(tss.data) if val not in [0, 1, 2, 11]]
        unique_spineps = [val for val in np.unique(spineps.data) if val != 0 and val < 200]

        # Add C1 segmentation to output from tss segmentation
        output.data[np.where(tss.data == 11)] = 11

        # Loop over all the structures and compute the dice
        i = 0
        val_output_list = []
        for val in unique_spineps:
            dice = 0
            c = 0
            dice_dict = {}
            while dice < 0.5 and c < len(unique_tss)+1:
                c+=1
                val_tss = unique_tss[i]
                dice = compute_dsc(np.where(tss.data == val_tss, 1, 0), np.where(spineps.data == val, 1, 0))
                i+=1
                dice_dict[val_tss] = dice
                if i == len(unique_tss):
                    i = 0            
            
            # Avoid infinite loop
            if c == len(unique_tss) + 1:
                max_dice = 0
                if val < 100: # Spineps vertebrae
                    for test_val in dice_dict.keys():
                        if dice_dict[test_val] > max_dice and 10 < test_val < 60: # TSS vertebrae
                            max_dice = dice_dict[test_val]
                            val_tss = test_val
                else: # Spineps discs
                    for test_val in dice_dict.keys():
                        if dice_dict[test_val] > max_dice and 60 < test_val < 101: # TSS discs
                            max_dice = dice_dict[test_val]
                            val_tss = test_val
                if max_dice == 0:
                    print(f"Structure {val} from SPINEPS does not match any structure for TotalSpineSeg")
                else:
                    val_output_list.append(val_tss)
                    output.data[np.where(spineps.data == val)] = val_tss
                    print(f'Low dice for structure {val} asociated with {val_tss} with DSC={max_dice}')
            else:
                # Add spineps segmentation to output with tss label value
                val_output_list.append(val_tss)
                output.data[np.where(spineps.data == val)] = val_tss
        
        # Remove segmentation in canal region
        output.data[canal.data.astype(bool)] = 0

        # Create output directory
        output_path = os.path.join(output_folder, sub, 'anat', sub + "_T2w_label-spine_dseg.nii.gz")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
    
        # Save output file
        output.change_orientation('RPI').save(output_path)
        create_json_file(output_path.replace('.nii.gz', '.json'))
        
        # Check if same labels in output and totalspineseg
        if sorted(val_output_list) != sorted(unique_tss):
            err_file.append("\n" + output_path + " - missing: " + str(np.array(unique_tss)[~np.in1d(unique_tss, val_output_list)].tolist()))
        
        with open(os.path.join(output_folder, 'err.txt'), 'w') as f:
            f.writelines(err_file)


def compute_dsc(gt_mask, pred_mask):
    """
    :param gt_mask: Ground truth mask used as the reference
    :param pred_mask: Prediction mask

    :return: dsc=2*intersection/(number of non zero pixels)
    """
    numerator = 2 * np.sum(gt_mask*pred_mask)
    denominator = np.sum(gt_mask) + np.sum(pred_mask)
    if denominator == 0:
        # Both ground truth and prediction are empty
        return 0
    else:
        return numerator / denominator 


if __name__ == '__main__':
    main()