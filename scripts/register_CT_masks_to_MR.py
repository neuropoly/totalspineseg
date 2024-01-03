import os
import argparse

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Register and duplicate segmentations from CT to MRI if not available.')
    parser.add_argument('--path-img', required=True, help='Path to the BIDS-compliant folder where CT and MRI images are stored (Required)')
    parser.add_argument('--path-label', default='', help='Path to the folder with labels where CT and MRI segmentations are stored. If not specified "--path-img + derivatives/labels"  will be used')
    parser.add_argument('--path-out', default='', help='Path to output directory where registered segmentations will be stored. If not specified "--path-img + derivatives/labels"  will be used')
    parser.add_argument('--seg-suffix', default='_seg', help='Suffix used for segmentations. This suffix will be added to the raw filename to name the segmentation file. Default="_seg"')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Fetch input variables
    path_img = args.path_img
    path_label = os.path.join(args.path_img, 'derivatives/labels') if not args.path_label else args.path_label
    path_out = os.path.join(args.path_img, 'derivatives/labels') if not args.path_out else args.path_out

    # Loop inside BIDS raw folder
    for sub in os.listdir(path_img):
        if sub.startswith('sub'):
            # Fetch subject files
            path_sub_anat = os.path.join(path_img, sub, 'anat')
            path_der_sub_anat = os.path.join(path_label, sub, 'anat')
            raw_files = os.listdir(path_sub_anat)
            der_files = os.listdir(path_der_sub_anat)

            # Check if segmentations are available for a given contrast



if __name__ == '__main__':
    main()