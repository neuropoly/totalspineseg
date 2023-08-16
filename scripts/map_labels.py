import sys, argparse, textwrap, json
from pathlib import Path

import numpy as np
import nibabel as nib

# Set default compression level 
nib.openers.Opener.default_compresslevel = 9


def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(f'''
            Map segmentation labels to other labels using json mapping file.
            
            The data is assumed to follow the BIDS structure (or similar):

            labels
            ├── sub-1
            │   └── anat
            │       ├── sub-1_T1w_seg.nii.gz
            │       └── sub-1_T2w_seg.nii.gz
            └── sub-2
                └── anat
                    ├── sub-2_T1w_seg.nii.gz
                    └── sub-2_T2w_seg.nii.gz
            

        '''),
        epilog=textwrap.dedent('''
            Example:
            map_labels -d derivatives/labels_orig -o derivatives/labels
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--seg-dir', '-d', type=DirPath(), required=True,
        help='Folder containing input segmentations for each subject.'
    )
    parser.add_argument(
        '--output-dir', '-o', type=DirPath(True), required=True, 
        help='Folder to save output segmentations for each subject.'
    )
    parser.add_argument(
        '--map', '-m', type=argparse.FileType('r', encoding='utf-8'), required=True,
        help='JSON file mapping each mask to a unique number, e.g. {"1": 2, "2": 15}'
    )
    parser.add_argument(
        '--subject-prefix', type=str, default='sub-',
        help='Subject prefix in filenames, defaults to "sub-" (BIDS standard).'
    )
    parser.add_argument(
        '--subject-subdir', type=str, default='anat', 
        help='Subfolder inside subject folder containing masks, defaults to "anat" (BIDS standard).'
    )
    parser.add_argument(
        '--seg-suffix', type=str, default='_seg',
        help='Suffix for output segmentation, defaults to "_seg".'
    )
    parser.add_argument(
        '--verbose', '-v', type=int, default=1, choices=[0, 1],
        help='Verbosity level. 0: Errors/warnings only, 1: Errors/warnings + info (default: 1)'
    )

    try:
        args = parser.parse_args()
    except BaseException as e:
        sys.exit()

    # Get arguments
    seg_path = args.seg_dir
    output_path = args.output_dir
    map_file = args.map
    subject_prefix = args.subject_prefix
    subject_subdir = args.subject_subdir
    seg_suffix = args.seg_suffix
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f'''
            Running with arguments:
            seg_dir = "{seg_path}"
            output_dir = "{output_path}"
            map = "{map_file.name}"
            subject_prefix = "{subject_prefix}"
            subject_subdir = "{subject_subdir}"
            seg_suffix = "{seg_suffix}"
            verbose = {verbose}
        '''))

    # Load label mappings from JSON file
    map_dict = json.load(map_file)
    map_file.close()

    # Process NIfTI segmentation files
    subject_dirs = list(seg_path.glob(f'{subject_prefix}*'))
    
    for i, subject_dir in enumerate(subject_dirs):
        
        subject = subject_dir.name
        
        # Check that subject subdir exists
        if not (subject_dir / subject_subdir).is_dir():
            if verbose: 
                print(f'Warning: "{subject_subdir}" not found for {subject}')
            continue
            
        if verbose:
            print(f'Processing {subject} ({i+1}/{len(subject_dirs)})')
            
        # Process each segmentation file
        for seg_path in (subject_dir / subject_subdir).glob(f'*{seg_suffix}.nii.gz'):
            
            output_seg_path = output_path / subject / subject_subdir / seg_path.name

            # Load segmentation
            seg = nib.load(seg_path)
            seg_data = seg.get_fdata()

            # Convert data to uint8 to avoid issues with segmentation IDs
            seg_data = seg_data.astype(np.uint8) 

            # Apply label mapping
            mapped_seg_data = np.zeros_like(seg_data)
            
            # Apply label mapping for all labels that are not mapped to 0
            for orig, new in map_dict.items():
                if new != 0:
                    mapped_seg_data[seg_data==int(orig)] = int(new)

            # Restores the original segmentation values for any voxels that were not mapped, to avoid losing data.
            mapped_seg_data = np.where(mapped_seg_data == 0, seg_data, mapped_seg_data)
                    
            # Handle mapping labels to 0  
            for orig, new in map_dict.items():
                if new == 0:
                    mapped_seg_data[seg_data==int(orig)] = int(new)

            # Create result segmentation 
            mapped_seg = nib.Nifti1Image(mapped_seg_data, seg.affine, seg.header)
            mapped_seg.set_data_dtype(np.uint8)

            # Make sure output directory exists
            output_seg_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save mapped segmentation
            nib.save(mapped_seg, output_seg_path)
            
            if verbose:
                print(f"Saved mapped {seg_path} to {output_seg_path}")
                

class DirPath:
    """
    Get path from argparse and return as Path object.
    
    Args:
        create: Create directory if it doesn't exist
        
    """

    def __init__(self, create=False):
        self.create = create

    def __call__(self, dir_path):
        path = Path(dir_path)
        
        if self.create and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True) 
            except:
                pass
                
        if path.is_dir():
            return path
        else:
            raise argparse.ArgumentTypeError(
                f"readble_dir:{path} is not a valid path")


if __name__ == '__main__':
    main()