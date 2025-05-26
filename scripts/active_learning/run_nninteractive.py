"""
Refine segmentations using nnInteractive. Script created by Yehuda Warszawer and edited by Nathan Molinier
"""

from pathlib import Path
import numpy as np
import torch
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
import argparse, os

from vrac.data_management.image import Image, zeros_like
from skimage.morphology import dilation, ball

def refine_segmentation_single(session, seg_data, seg_label, seg_label_neg, iterations, lasso=True):

    # Extract labels from segmentation mask
    scribble = np.isin(seg_data, seg_label).astype(np.uint8)

    if not np.any(scribble):
        return None
    
    # Dilate
    size = 2
    footprint = ball(size)
    scribble = dilation(scribble, footprint)

    # Add scribble interaction
    print("Adding scribble interaction...")
    for i in range(iterations):
        session.set_target_buffer(torch.zeros(seg_data.shape, dtype=torch.uint8))
        session.reset_interactions()
        if lasso:
            # Add positive lasso interaction
            print(f"Adding positive lasso interaction...")
            session.add_lasso_interaction(scribble, include_interaction=True)
        else:
            # Add positive scribble interaction
            print(f"Adding positive scribble interaction...")
            session.add_scribble_interaction(scribble, include_interaction=True)

        if seg_label_neg > 0:
            # Add negative scribble interaction
            print("Adding negative scribble interaction...")
            session.add_scribble_interaction((seg_data == seg_label_neg).astype(np.uint8), include_interaction=False)

        results = session.target_buffer.clone().numpy().astype(np.uint8)

        # If negative labels are present, set them to zero in the results
        if seg_label_neg > 0:
            results[seg_data == seg_label_neg] = 0
        scribble = results

    return results


def refine_segmentation(session, img_path, seg_path, output_path):
    ###############################################
    # Load input data
    ###############################################

    # Load input image
    print(f"Loading input image from {img_path}")
    img_nib = Image(img_path).change_orientation('RPI')
    img_data = img_nib.data
    img_data = np.expand_dims(img_data, axis=0)  # Add channel dimension (1, x, y, z)

    # Load segmentation mask
    print(f"Loading segmentation mask from {seg_path}")
    seg_nib = Image(seg_path).change_orientation('RPI')
    seg_data = seg_nib.data.astype(np.uint8)

    # Init output files
    canal_img = zeros_like(seg_nib)
    spine_img = zeros_like(seg_nib)

    #####################################
    # Set image to the session
    #####################################

    # Set image to the session
    print("Setting image to session...")
    session.set_image(img_data)

    # for spinal canal segmentation spinal cord is negative label with 1 iteration and usin scribbler not lasso
    print("Refining spinal canal segmentation...")
    cur_redults = refine_segmentation_single(session, seg_data, [1,2], 0, 1)
    if cur_redults is not None:
        canal_img.data[cur_redults == 1] = 1

    # for C1 segmentation - !!!not working well!!!
    print("Refining C1 segmentation...")
    cur_redults = refine_segmentation_single(session, seg_data, 11, 0, 1)
    if cur_redults is not None:
        spine_img.data[cur_redults == 1] = 11

    # for each vertebral label
    print("Refining vertebral segmentation...")
    for s in [12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 41, 42, 43, 44, 45, 50]:
        print(f"Refining segmentation for label {s}...")
        # Set target buffer to zero
        cur_redults = refine_segmentation_single(session, seg_data, s, 0, 5)
        if cur_redults is not None:
            spine_img.data[cur_redults == 1] = s

    # for each IVD label
    print("Refining IVD segmentation...")
    for s in [63, 64, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 91, 92, 93, 94, 95, 100]:
        print(f"Refining segmentation for label {s}...")
        # Set target buffer to zero
        cur_redults = refine_segmentation_single(session, seg_data, s, 0, 7)
        if cur_redults is not None:
            spine_img.data[cur_redults == 1] = s

    ##########################################
    # Save results
    ##########################################

    # Save results
    canal_path = os.path.join(output_path, os.path.basename(img_path).replace('.nii.gz', '_label-canal_seg.nii.gz'))
    spine_path = os.path.join(output_path, os.path.basename(img_path).replace('.nii.gz', '_label-spine_dseg.nii.gz'))

    print(f"Saving canal to {canal_path}")
    canal_img.data = canal_img.data.astype(np.uint8)
    canal_img.save(canal_path)

    print(f"Saving spine to {spine_path}")
    spine_img.data = spine_img.data.astype(np.uint8)
    spine_img.save(spine_path)

    print("Segmentation refinement complete!")


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Refine segmentation using nnInteractive.')
    parser.add_argument('--image', '-i', required=True, help='Path to the input image (Required)')
    parser.add_argument('--seg', '-s', required=True, help='Path to the input segmentation (Required)')
    parser.add_argument('--ofolder', '-o', required=True, help='Path to the output folder (Required)')
    return parser


def main():
    # Load parser
    parser = get_parser()
    args = parser.parse_args()

    # Define paths
    img_path = args.image
    seg_path = args.seg
    output_path = args.ofolder

    # Define constants
    REPO_ID = "nnInteractive/nnInteractive"
    MODEL_NAME = "nnInteractive_v1.0"
    DOWNLOAD_DIR = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/nnInteractive/weights/"

    ################################################
    # Download model
    ################################################

    if not Path(os.path.join(DOWNLOAD_DIR, MODEL_NAME)).exists():
        print("Downloading model...")
        Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
        download_path = snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{MODEL_NAME}/*"],
            local_dir=DOWNLOAD_DIR
        )
        print(f"Model downloaded to {download_path}")

    ###########################################
    # Initialize inference session
    ###########################################

    print("Initializing inference session...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    session = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=False,
        verbose=True,
        torch_n_threads=8,  # Adjust based on your CPU
        do_autozoom=True,
        use_pinned_memory=True,
    )

    # Load the model
    model_path = Path(DOWNLOAD_DIR) / MODEL_NAME
    print(f"Loading model from {model_path}")
    session.initialize_from_trained_model_folder(str(model_path))

    refine_segmentation(session, img_path, seg_path, output_path)

if __name__=='__main__':
    main()