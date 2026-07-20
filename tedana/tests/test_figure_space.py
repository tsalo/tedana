"""Tests for report-only transformation of spatial figures."""

import importlib.util

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from tedana.reporting import figure_space as figure_space_module
from tedana.reporting.figure_space import (
    FigureSpace,
    load_dseg_labels,
    validate_figure_options,
)
from tedana.workflows import tedana as tedana_cli


def _write_img(path, data, affine=None):
    """Write a small NIfTI image and return its path."""
    affine = np.eye(4) if affine is None else affine
    nb.save(nb.Nifti1Image(np.asarray(data), affine), path)
    return str(path)


def test_validate_figure_option_relationships(tmp_path):
    """Paired transform options and dseg labels should be enforced."""
    xfm = tmp_path / "identity.mat"
    xfm.touch()
    reference = tmp_path / "reference.nii.gz"
    reference.touch()
    dseg_tsv = tmp_path / "dseg.tsv"
    dseg_tsv.touch()

    with pytest.raises(ValueError, match="xfms.*reference"):
        validate_figure_options(xfms=[str(xfm)], check_dependency=False)
    with pytest.raises(ValueError, match="dseg_tsv.*dseg"):
        validate_figure_options(dseg_tsv=str(dseg_tsv), check_dependency=False)

    validate_figure_options(
        xfms=[str(xfm)],
        reference=str(reference),
        check_dependency=False,
    )
    validate_figure_options(
        xfms=str(xfm),
        reference=str(reference),
        check_dependency=False,
    )


def test_missing_antspyx_has_actionable_error(tmp_path, monkeypatch):
    """Requesting transforms without ANTsPyX should fail before the workflow runs."""
    xfm = tmp_path / "identity.mat"
    xfm.touch()
    reference = tmp_path / "reference.nii.gz"
    reference.touch()
    original_import_module = figure_space_module.importlib.import_module

    def _missing_ants(name):
        if name == "ants":
            raise ImportError("not installed")
        return original_import_module(name)

    monkeypatch.setattr(
        figure_space_module.importlib,
        "import_module",
        _missing_ants,
    )
    with pytest.raises(ImportError, match=r"tedana\[transforms\]"):
        validate_figure_options(
            xfms=[str(xfm)],
            reference=str(reference),
        )


def test_parser_accepts_figure_space_options(tmp_path):
    """The tedana parser should preserve ordered transforms and segmentation inputs."""
    paths = {}
    for name in ("data", "xfm1", "xfm2", "reference", "dseg", "dseg_tsv"):
        paths[name] = tmp_path / name
        paths[name].touch()

    args = tedana_cli._get_parser().parse_args(
        [
            "-d",
            str(paths["data"]),
            "-e",
            "0.01",
            "--xfms",
            str(paths["xfm1"]),
            str(paths["xfm2"]),
            "--reference",
            str(paths["reference"]),
            "--dseg",
            str(paths["dseg"]),
            "--dseg-tsv",
            str(paths["dseg_tsv"]),
        ]
    )

    assert args.xfms == [str(paths["xfm1"]), str(paths["xfm2"])]
    assert args.reference == str(paths["reference"])
    assert args.dseg == str(paths["dseg"])
    assert args.dseg_tsv == str(paths["dseg_tsv"])


def test_load_dseg_labels(tmp_path):
    """BIDS index/name columns should become Nilearn's name-to-value mapping."""
    dseg_tsv = tmp_path / "dseg.tsv"
    pd.DataFrame({"index": [0, 1, 2], "name": ["Background", "GM", "WM"]}).to_csv(
        dseg_tsv,
        sep="\t",
        index=False,
    )

    assert load_dseg_labels(dseg_tsv) == {"Background": 0, "GM": 1, "WM": 2}


def test_figure_space_validates_dseg_fov_and_values(tmp_path):
    """A dseg must be discrete and match the active figure grid."""
    native = _write_img(tmp_path / "native.nii.gz", np.zeros((3, 3, 3)))
    mismatched = _write_img(tmp_path / "mismatch.nii.gz", np.ones((2, 2, 2)))
    continuous = _write_img(
        tmp_path / "continuous.nii.gz",
        np.full((3, 3, 3), 1.5),
    )

    with pytest.raises(ValueError, match="field of view"):
        FigureSpace(native_reference=native, dseg=mismatched)
    with pytest.raises(ValueError, match="discrete integer"):
        FigureSpace(native_reference=native, dseg=continuous)


@pytest.mark.skipif(importlib.util.find_spec("ants") is None, reason="ANTsPyX is not installed")
def test_identity_transform_preserves_data_and_uses_reference_grid(tmp_path):
    """A real ANTs identity transform should support four-dimensional figure data."""
    import ants

    moving_data = np.arange(4 * 4 * 4 * 3, dtype=np.float32).reshape((4, 4, 4, 3))
    moving = _write_img(tmp_path / "moving.nii.gz", moving_data)
    reference = _write_img(tmp_path / "reference.nii.gz", np.zeros((4, 4, 4)))
    transform_file = tmp_path / "identity.mat"
    transform = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    ants.write_transform(transform, str(transform_file))

    figure_space = FigureSpace(
        native_reference=moving,
        xfms=[str(transform_file)],
        reference=reference,
    )
    transformed = figure_space.transform(moving, imagetype=3)

    assert transformed.shape == moving_data.shape
    assert np.allclose(transformed.affine, nb.load(reference).affine)
    assert np.allclose(transformed.get_fdata(), moving_data, atol=1e-5)

    label_data = np.zeros((4, 4, 4), dtype=np.int16)
    label_data[:2, ...] = 1
    label_data[2:, ...] = 2
    label_img = _write_img(tmp_path / "labels.nii.gz", label_data)
    transformed_labels = figure_space.transform(
        label_img,
        interpolation="genericLabel",
    )
    assert np.array_equal(transformed_labels.get_fdata(), label_data)
