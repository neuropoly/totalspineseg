import numpy as np
import nibabel as nib
import scipy.ndimage as ndi


def remove_if_component_is_not_big_enough(
    label,
    seg_data,
    labeled_components,
    components_of_interest,
    min_component_size,
    fallback_label,
):
    count = np.sum(labeled_components == label)
    is_big_enough = count >= min_component_size
    if is_big_enough:
        print(f"{label}th component has {count} voxels. ")
    if not is_big_enough:
        print(f"{label}th component has {count} voxels. Removing because it has fewer than {min_component_size} voxels...")
        seg_data[labeled_components == label] = fallback_label
        components_of_interest.remove(label)


def remove_if_component_is_not_inferior_enough(
    label,
    seg_data,
    labeled_components,
    components_of_interest,
    ivd_label,
    max_inferior_ivd_voxels,
    fallback_label,
):
    component_mask = labeled_components == label
    center_of_gravity = ndi.center_of_mass(component_mask)
    inferior_voxels = np.argwhere((seg_data == ivd_label) & (np.indices(seg_data.shape)[2] < center_of_gravity[2]))  # Assume LPI orientation, so inferior voxels have a smaller z-coordinate than the CoG
    n_inferior_voxels = len(inferior_voxels)
    is_inferior_enough = n_inferior_voxels <= max_inferior_ivd_voxels
    if is_inferior_enough:
        print(f"{label}th component has {n_inferior_voxels} inferior IVD voxels.")
    if not is_inferior_enough:
        print(f"{label}th component has {n_inferior_voxels} inferior IVD voxels. Removing because it has more than {max_inferior_ivd_voxels} inferior IVD voxels...")
        seg_data[labeled_components == label] = fallback_label
        components_of_interest.remove(label)


def filter_bad_sacrum(
    input_dir, 
    output_dir, 
    ivd_label=1,
    vertebra_extra_label=6,
    l5_s1_label=5,
    sacrum_label=9,
    min_component_size=100,
    max_inferior_ivd_voxels=1000,
    overwrite=False
):

    if not overwrite:
        raise NotImplementedError()

    for seg_file in (input_dir).glob('*.nii.gz'):
        seg = nib.load(seg_file)
        seg_data = seg.get_fdata()

        # Find all connected components of the (raw) sacrum segment
        if sacrum_label in np.unique(seg_data):
            mask = seg_data == sacrum_label
            labeled_components, num_labels = ndi.label(mask, structure=np.ones((3, 3, 3)))

            print(f"Found {num_labels} connected components for the sacrum segment (label {sacrum_label}).")
            components_of_interest = list(np.unique(labeled_components)[1:])  # Skip the background (label 0)

            for component_of_interest in components_of_interest.copy():
                remove_if_component_is_not_big_enough(
                    label=component_of_interest,
                    seg_data=seg_data,
                    labeled_components=labeled_components,
                    components_of_interest=components_of_interest,
                    min_component_size=min_component_size,
                    fallback_label=vertebra_extra_label
                )

            for component_of_interest in components_of_interest.copy():
                remove_if_component_is_not_inferior_enough(
                    label=component_of_interest,
                    seg_data=seg_data,
                    labeled_components=labeled_components,
                    components_of_interest=components_of_interest,
                    ivd_label=ivd_label,
                    max_inferior_ivd_voxels=0,  # Sacrum should not have any inferior IVD voxels
                    fallback_label=vertebra_extra_label
                )

            if len(components_of_interest) == 0:
                print(f"Removed all sacrum components.")

        # Find all connected components of the (raw) L5/S1-IVD segment prediction
        if l5_s1_label in np.unique(seg_data):
            mask = seg_data == l5_s1_label
            labeled_components, num_labels = ndi.label(mask, structure=np.ones((3, 3, 3)))

            print(f"Found {num_labels} connected components for the L5/S1-IVD segment (label {l5_s1_label}).")
            components_of_interest = list(np.unique(labeled_components)[1:])  # Skip the background (label 0)

            for component_of_interest in components_of_interest.copy():
                remove_if_component_is_not_big_enough(
                    label=component_of_interest,
                    seg_data=seg_data,
                    labeled_components=labeled_components,
                    components_of_interest=components_of_interest,
                    min_component_size=min_component_size,
                    fallback_label=ivd_label
                )

            for component_of_interest in components_of_interest.copy():
                remove_if_component_is_not_inferior_enough(
                    label=component_of_interest,
                    seg_data=seg_data,
                    labeled_components=labeled_components,
                    components_of_interest=components_of_interest,
                    ivd_label=ivd_label,
                    max_inferior_ivd_voxels=max_inferior_ivd_voxels,
                    fallback_label=ivd_label
                )
            
            print("Removed all L5/S1-IVD components.")

        # Save the modified segmentation
        modified_seg = nib.Nifti1Image(seg_data, seg.affine, seg.header)
        nib.save(modified_seg, output_dir / seg_file.name)
