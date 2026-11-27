import nibabel as nib
import numpy as np

from totalspineseg.utils.iterative_label import _iterative_label, iterative_label


def _make_spine_seg(discs):
    """Build a small synthetic LPI NIfTI segmentation with a canal and discs.

    Parameters
    ----------
    discs : list of tuple[int, int, int]
        Each tuple is (z_start, z_end, label_value). Higher z is more superior.
    """
    data = np.zeros((24, 24, 60), dtype=np.uint8)
    # Canal along the posterior side (small y) spanning the full z range
    data[10:14, 4:8, 2:58] = 8
    # Discs anterior to the canal (larger y)
    for z0, z1, label in discs:
        data[8:16, 12:20, z0:z1] = label
    return nib.Nifti1Image(data, np.eye(4))


COMMON_KWARGS = dict(
    disc_labels=[1, 2, 3, 4, 5],
    disc_landmark_labels=[2, 3, 4, 5],
    disc_landmark_output_labels=[63, 71, 91, 100],
    canal_labels=[8],
    canal_output_label=2,
)


def test_iterative_label_preserves_selected_disc_landmarks_c2c3():
    lumbar_seg = _make_spine_seg([(40, 44, 1), (10, 14, 5)])
    cervical_seg = _make_spine_seg([(40, 44, 2), (10, 14, 1)])
    landmarks = [2, 5, 3, 4]

    lumbar_out = iterative_label(
        lumbar_seg, selected_disc_landmarks=landmarks, **COMMON_KWARGS
    )
    cervical_out = iterative_label(
        cervical_seg, selected_disc_landmarks=landmarks, **COMMON_KWARGS
    )

    assert landmarks == [2, 5, 3, 4]
    assert {95, 100}.issubset(set(np.unique(lumbar_out.get_fdata()).astype(int)))
    assert {63, 64}.issubset(set(np.unique(cervical_out.get_fdata()).astype(int)))


def test_iterative_label_preserves_selected_disc_landmarks_l5s1():
    # L5-S1 (5) followed inferiorly by a larger disc (1) discards 5 for this image
    # while C2-C3 (2) at the top still anchors the labeling.
    spurious_l5s1_seg = _make_spine_seg([(44, 48, 2), (28, 32, 5), (8, 16, 1)])
    lumbar_seg = _make_spine_seg([(40, 44, 1), (10, 14, 5)])
    landmarks = [2, 5, 3, 4]

    iterative_label(
        spurious_l5s1_seg, selected_disc_landmarks=landmarks, **COMMON_KWARGS
    )
    lumbar_out = iterative_label(
        lumbar_seg, selected_disc_landmarks=landmarks, **COMMON_KWARGS
    )

    assert landmarks == [2, 5, 3, 4]
    assert {95, 100}.issubset(set(np.unique(lumbar_out.get_fdata()).astype(int)))


def test_iterative_label_io_wrapper_preserves_landmarks(tmp_path):
    in_lumbar = tmp_path / "lumbar.nii.gz"
    in_cervical = tmp_path / "cervical.nii.gz"
    out_lumbar = tmp_path / "out_lumbar.nii.gz"
    out_cervical = tmp_path / "out_cervical.nii.gz"

    nib.save(_make_spine_seg([(40, 44, 1), (10, 14, 5)]), in_lumbar)
    nib.save(_make_spine_seg([(40, 44, 2), (10, 14, 1)]), in_cervical)

    landmarks = [2, 5, 3, 4]
    _iterative_label(
        in_lumbar,
        out_lumbar,
        selected_disc_landmarks=landmarks,
        overwrite=True,
        **COMMON_KWARGS,
    )
    _iterative_label(
        in_cervical,
        out_cervical,
        selected_disc_landmarks=landmarks,
        overwrite=True,
        **COMMON_KWARGS,
    )

    assert landmarks == [2, 5, 3, 4]
    assert out_lumbar.is_file()
    assert out_cervical.is_file()
    cervical_labels = set(np.unique(nib.load(out_cervical).get_fdata()).astype(int))
    assert {63, 64}.issubset(cervical_labels)
