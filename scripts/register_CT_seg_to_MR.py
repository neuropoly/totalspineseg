import os
import argparse
import subprocess

from image import Image


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Register and duplicate segmentations from CT to MRI if not available.')
    parser.add_argument('--path-img', required=True, type=str, help='Path to the BIDS-compliant folder where CT and MRI images are stored (Required)')
    parser.add_argument('--path-label', type=str, default='', help='Path to the folder with labels where CT and MRI segmentations are stored. If not specified "--path-img + derivatives/labels"  will be used')
    parser.add_argument('--path-out', type=str, default='', help='Path to output directory where registered segmentations will be stored. If not specified "--path-img + derivatives/reg-labels"  will be used')
    parser.add_argument('--seg-suffix', type=str, default='_seg', help='Suffix used for segmentations. This suffix will be added to the raw filename to name the segmentation file. Default="_seg"')
    parser.add_argument('--sacrum-idx', type=int, default=92, help='Index used to map the sacrum area. Default=92')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Fetch input variables
    path_img = args.path_img
    path_label = os.path.join(args.path_img, 'derivatives/labels') if not args.path_label else args.path_label
    path_out = os.path.join(args.path_img, 'derivatives/reg-labels') if not args.path_out else args.path_out
    suffix_seg = args.seg_suffix

    # Loop inside BIDS raw folder
    for sub in os.listdir(path_img):
        if sub.startswith('sub'):
            # Fetch subject files
            path_sub_anat = os.path.join(path_img, sub, 'anat')
            path_der_sub_anat = os.path.join(path_label, sub, 'anat')
            path_out_sub_anat = os.path.join(path_out, sub, 'anat')
            raw_files = os.listdir(path_sub_anat)
            der_files = os.listdir(path_der_sub_anat)

            # Check if segmentations are available for a given contrast
            for raw_file in raw_files:
                path_raw = os.path.join(path_sub_anat, raw_file)
                ext = '.' + raw_file.split('_')[-1].split('.', 1)[-1]
                if ext == '.nii.gz':
                    cont = raw_file.split('_')[-1].split(ext)[0]
                    out_seg = raw_file.split('.')[0] + suffix_seg + ext
                    file_seg = raw_file.split('.')[0] + suffix_seg + ext
                    # If no segmentation is available, the CT segmentation is used instead
                    if not file_seg in der_files:
                        file_seg = '_'.join(raw_file.split('_')[:-1]) + '_CT' + suffix_seg + ext
                        seg_from_CT = True

                        # Check if CT segmentation does not exists
                        if not file_seg in der_files:
                            raise ValueError(f'Subject {sub} has no segmentations')
                    else:
                        seg_from_CT = False
                    path_seg = os.path.join(path_der_sub_anat, file_seg)
                    path_out_seg = os.path.join(path_out_sub_anat, out_seg)

                    
                    # Keep only sacrum segmentation
                    seg = Image(path_seg)
                    if not args.sacrum_idx in seg.data:
                        raise ValueError(f'The value {args.sacrum_idx} was not detected in the mask')
                    
                    seg.data[seg.data != args.sacrum_idx] = 0
                    seg.data[seg.data == args.sacrum_idx] = 1

                    # Save the new segmentation in the output folder
                    os.makedirs(os.path.dirname(path_out_seg), exist_ok=True)
                    seg.save(path=path_out_seg, dtype='float32')

                    # If the original segmentation was created using the CT image, register the segmentation to the MRI image
                    if seg_from_CT:
                        subprocess.check_call([
                            'sct_register_multimodal',
                            '-i', path_out_seg,
                            '-d', path_raw,
                            '-o', path_out_seg,
                            '-identity', '1',
                            '-x', 'nn'
                        ])
                    




if __name__ == '__main__':
    main()