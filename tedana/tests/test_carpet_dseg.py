"""Tests for segmentation-aware carpet plots."""

from types import SimpleNamespace

import nibabel as nb
import numpy as np
import pandas as pd

from tedana.reporting import static_figures
from tedana.reporting.figure_space import FigureSpace


def _write_img(path, data):
    """Write a small NIfTI image and return its path."""
    nb.save(nb.Nifti1Image(np.asarray(data), np.eye(4)), path)
    return str(path)


def test_carpet_plot_uses_dseg_labels_and_native_tr(tmp_path, monkeypatch):
    """Carpet plots should use the dseg atlas, labels, and original repetition time."""
    (tmp_path / "figures").mkdir()
    native_img = nb.Nifti1Image(np.zeros((2, 2, 2, 4)), np.eye(4))
    native_img.header.set_zooms((1.0, 1.0, 1.0, 2.5))
    native = tmp_path / "native.nii.gz"
    nb.save(native_img, native)
    mask = _write_img(tmp_path / "mask.nii.gz", np.ones((2, 2, 2)))
    dseg = _write_img(
        tmp_path / "dseg.nii.gz",
        np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]]], dtype=np.int16),
    )
    dseg_tsv = tmp_path / "dseg.tsv"
    pd.DataFrame({"index": [1, 2], "name": ["Anterior", "Posterior"]}).to_csv(
        dseg_tsv,
        sep="\t",
        index=False,
    )
    figure_space = FigureSpace(
        native_reference=str(native),
        dseg=dseg,
        dseg_tsv=dseg_tsv,
    )

    io_generator = SimpleNamespace(
        out_dir=str(tmp_path),
        prefix="",
        verbose=False,
        reference_img=nb.load(native),
        mask=nb.load(mask),
    )

    calls = []

    def _capture_plot_carpet(img, mask_img, **kwargs):
        calls.append((img, mask_img, kwargs))

    monkeypatch.setattr(static_figures.plotting, "plot_carpet", _capture_plot_carpet)
    data = np.arange(8 * 4, dtype=float).reshape((8, 4))
    static_figures.carpet_plot(
        data,
        data,
        data,
        data,
        nb.load(mask),
        io_generator,
        figure_space=figure_space,
    )

    assert len(calls) == 4
    for _, mask_img, kwargs in calls:
        assert mask_img is figure_space.dseg
        assert kwargs["mask_labels"] == {"Anterior": 1, "Posterior": 2}
        assert kwargs["t_r"] == 2.5
