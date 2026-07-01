from pathlib import Path

import pytest

from clbfield.data.domains import Contrast
from clbfield.data.sources import nifti_image_loader, records_from_directory


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_records_from_directory_parses_official_filenames_and_caps_count(tmp_path: Path) -> None:
    _touch(tmp_path / "Training_retrospective" / "T1W" / "3T" / "R_T1W_3T_0001.nii.gz")
    _touch(tmp_path / "Training_retrospective" / "T2W" / "0.1T" / "R_T2W_0.1T_0002.nii.gz")
    _touch(tmp_path / "Training_retrospective" / "T2W" / "0.1T" / "R_T2W_0.1T_0002_seg.nii.gz")
    _touch(tmp_path / "Validating_prospective" / "T2FLAIR" / "7T" / "P_T2FLAIR_7T_0001.nii.gz")

    records = records_from_directory(tmp_path, max_records=2)

    assert len(records) == 2
    # Validating_prospective is prioritized over Training_retrospective.
    assert records[0].split == "Validating_prospective"
    assert records[0].domain.field_strength_t == 7.0
    assert records[0].domain.contrast == Contrast.T2_FLAIR
    assert records[0].subject_id == "0001"


def test_records_from_directory_ignores_segmentation_files(tmp_path: Path) -> None:
    _touch(tmp_path / "R_T1W_3T_0001.nii.gz")
    _touch(tmp_path / "R_T1W_3T_0001_seg.nii.gz")

    records = records_from_directory(tmp_path, max_records=None)

    assert len(records) == 1
    assert records[0].image_path.name == "R_T1W_3T_0001.nii.gz"


def test_nifti_image_loader_shape_and_finite(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    import numpy as np

    volume = np.random.default_rng(0).random((4, 5, 6)).astype("float32")
    path = tmp_path / "R_T1W_3T_0001.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), str(path))

    [record] = records_from_directory(tmp_path)
    tensor = nifti_image_loader(path, record)

    assert tensor.shape == (1, 4, 5, 6)
    assert tensor.isfinite().all()
