import SimpleITK as sitk
import argparse
from pathlib import Path

def convert_mha_to_nii(input_file, output_file, verbose):
    if not input_file.lower().endswith('.mha'):
        raise argparse.ArgumentTypeError(f'{input_file} must end with .mha')
    if not output_file.lower().endswith('.nii.gz'):
        raise argparse.ArgumentTypeError(f'{output_file} must end with .nii.gz')
    sitk.WriteImage(sitk.ReadImage(input_file), output_file)
    if verbose:
        print(f'Converted {input_file} to {output_file}')

def main():
    parser = argparse.ArgumentParser(description='Convert .mha to .nii.gz')
    parser.add_argument('--input_file', '-i', type=valid_file, required=True, help='Input .mha file path')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Output .nii.gz file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode. Enable additional print statements.', default=True)

    args = parser.parse_args()

    convert_mha_to_nii(args.input_file, args.output_file, args.verbose)

def valid_file(file_path):
    path = Path(file_path)
    if path.exists():
        return str(path)
    else:
        raise argparse.ArgumentTypeError(f'{file_path} does not exist')

if __name__ == '__main__':
    main()
