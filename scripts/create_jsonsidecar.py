import json
import time
import argparse

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment an image using nnUNetV2 model.')
    parser.add_argument('-path-json', help='Input image to segment. Example: derivatives/labels/sub-001/anat/sub-001_T2w_label-sacrum_seg.json', required=True)
    parser.add_argument('-process', help='Process used to generate the data. Example: nnUNet3D', required=True, type=str)
    parser.add_argument('-author', help='Author who quality controlled the data', required=True, type=str)
    return parser

def create_json_file():
    """
    Create a json sidecar file
    :param path_file_out: path to the output file
    """
    parser = get_parser()
    args = parser.parse_args()

    path_json_out = args.path_json
    process=args.process
    author=args.author
    data_json = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": process,
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                "Name": "Quality Control",
                "Author": author,
                "Date": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
    }
    with open(path_json_out, 'w') as f:
        json.dump(data_json, f, indent=4)
        print(f'Created: {path_json_out}')

if __name__=='__main__':
    create_json_file()