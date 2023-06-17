# Paste this part in 3D Slicer Python window:
####################################################################################################

from pathlib import Path

# import nibabel as nib
# nib.openers.Opener.default_compresslevel = 9

data_path = r'D:/reg2pam50/whole-spine/data'

def load_files(base_file_name):
    
    # Separate the base_file_name into subject_id and scan_type
    subject_id, scan_type = base_file_name.rsplit('_', 1)

    # Define the paths to the volume and segmentation files
    volume_path = f"{data_path}/{subject_id}/anat/{base_file_name}.nii.gz"
    segmentation_path = f"{data_path}/derivatives/labels/{subject_id}/anat/{base_file_name}_seg.nii.gz"

    # Close the current scene
    slicer.mrmlScene.Clear(0)

    # Load the volume
    volume_node = slicer.util.loadVolume(volume_path)
    if not volume_node:
        print(f"Error loading volume from {volume_path}")
        return

    # Load the segmentation
    segmentation_node = slicer.util.loadSegmentation(segmentation_path)
    if not segmentation_node:
        print(f"Error loading segmentation from {segmentation_path}")
        return

    # Check if the voxel size and FOV of the label map match the volume
    volume_image_data = volume_node.GetImageData()
    volume_spacing = volume_image_data.GetSpacing()

    segmentation_first_segment_id = segmentation_node.GetSegmentation().GetNthSegmentID(0)
    segmentation_image_data = segmentation_node.GetBinaryLabelmapInternalRepresentation(segmentation_first_segment_id)
    segmentation_spacing = segmentation_image_data.GetSpacing()

    if volume_spacing != segmentation_spacing:

        # Resample the label map to match the volume using nearest neighbor interpolation
        parameters = {
            "interpolationMode": "NearestNeighbor",
            "InputVolume":segmentation_node.GetID(),
            "referenceVolume": volume_node.GetID(),
            "OutputVolume":segmentation_node.GetID()}
        slicer.cli.runSync(slicer.modules.resamplescalarvolume, None, parameters)

    #Set segmentation geometry to match volume
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

    # Set segmentation_node name
    segmentation_node.SetName(f"{base_file_name}_seg-manual")
    # Set segmentation_node file name
    segmentation_node.AddDefaultStorageNode()
    segmentation_node.GetStorageNode().SetFileName(segmentation_path.replace('seg.nii.gz', 'seg-manual.nii.gz'))



def save_segmentation_with_original_dtype():

    # Get the currently active segmentation
    segmentation_node = slicer.mrmlScene.GetNodeByID('vtkMRMLSegmentationNode1')

    # Get the currently active volume
    volume_node = slicer.mrmlScene.GetNodeByID('vtkMRMLScalarVolumeNode1')
    
    # Save the label map to the file
    segmentation_path = Path(segmentation_node.GetStorageNode().GetFileName())
    slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsBinaryLabelmapRepresentationToFiles(str(segmentation_path.parent), segmentation_node, None, "nii.gz", True, volume_node)

    # Fix filename if '-' removed
    if not segmentation_path.is_file():
        (segmentation_path.parent / segmentation_path.name.replace('-', '')).rename(segmentation_path)

    # # Load the original and temporary files using nibabel
    # seg_src = nib.load(original_file_name)
    # seg = nib.load(temp_file_name)

    # # Create a new Nifti1Image with the same data type as the original file
    # new_seg = nib.Nifti1Image(seg.get_fdata().astype(seg_src.get_data_dtype()), seg.affine, seg.header)

    # # Update the 'datatype' and 'bitpix' fields in the header
    # new_seg.header.set_data_dtype(seg_src.get_data_dtype())

    # # Save the new segmentation to the temporary file
    # nib.save(new_seg, temp_file_name)


# Run this functions to use the code:
####################################################################################################

# Call the function to close scene and load new volume and segmmentation
load_files(input("Please enter the base file name (e.g. sub-amuPAM50001_T1w): "))

# Call the function to save seg-manual
save_segmentation_with_original_dtype()
