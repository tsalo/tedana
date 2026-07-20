"""Prepare spatial images for report figures without changing workflow outputs."""

import importlib
import os.path as op
from os import PathLike
from tempfile import TemporaryDirectory

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn import image


def _as_xfm_list(xfms):
    """Normalize one or more transform paths to a list."""
    if isinstance(xfms, (str, PathLike)):
        return [xfms]
    return list(xfms or [])


def validate_figure_options(
    *,
    xfms=None,
    reference=None,
    dseg=None,
    dseg_tsv=None,
    check_dependency=True,
):
    """Validate relationships and file existence for figure-space options."""
    has_xfms = bool(xfms)
    has_reference = reference is not None
    if has_xfms != has_reference:
        raise ValueError("'xfms' and 'reference' must be provided together.")

    if dseg_tsv is not None and dseg is None:
        raise ValueError("'dseg_tsv' requires 'dseg'.")

    paths = _as_xfm_list(xfms)
    paths.extend(path for path in (reference, dseg, dseg_tsv) if path is not None)
    for path in paths:
        if not op.isfile(path):
            raise FileNotFoundError(f"Figure-space input does not exist: {path}")

    if has_xfms and check_dependency:
        try:
            importlib.import_module("ants")
        except ImportError as exc:
            raise ImportError(
                "Transforming report figures requires ANTsPyX. "
                'Install it with `pip install "tedana[transforms]"`.'
            ) from exc


def _same_fov(img1, img2):
    """Return whether two images have the same three-dimensional field of view."""
    return img1.shape[:3] == img2.shape[:3] and np.allclose(img1.affine, img2.affine)


def load_dseg_labels(dseg_tsv):
    """Load a BIDS segmentation table as a Nilearn mask-label dictionary."""
    if dseg_tsv is None:
        return None

    labels_df = pd.read_table(dseg_tsv)
    required_columns = {"index", "name"}
    missing_columns = required_columns.difference(labels_df.columns)
    if missing_columns:
        raise ValueError(f"dseg TSV is missing required column(s): {sorted(missing_columns)}")
    if labels_df["index"].isna().any() or labels_df["name"].isna().any():
        raise ValueError("dseg TSV columns 'index' and 'name' cannot contain missing values.")

    indices = pd.to_numeric(labels_df["index"], errors="raise")
    if not np.allclose(indices, np.round(indices)):
        raise ValueError("dseg TSV column 'index' must contain integers.")
    indices = indices.astype(int)
    names = labels_df["name"].astype(str)
    if indices.duplicated().any() or names.duplicated().any():
        raise ValueError("dseg TSV values in 'index' and 'name' must be unique.")

    # Nilearn expects {label_name: atlas_value}.
    return dict(zip(names, indices))


class FigureSpace:
    """Transform images into a common space for figures only."""

    def __init__(
        self,
        *,
        native_reference,
        xfms=None,
        reference=None,
        dseg=None,
        dseg_tsv=None,
    ):
        """Initialize and validate the figure target space."""
        validate_figure_options(
            xfms=xfms,
            reference=reference,
            dseg=dseg,
            dseg_tsv=dseg_tsv,
        )

        self.xfms = [op.abspath(path) for path in _as_xfm_list(xfms)]
        self.enabled = bool(self.xfms)
        self.native_reference = image.load_img(native_reference)
        self.reference = image.load_img(reference) if self.enabled else self.native_reference
        if self.enabled and self.reference.ndim != 3:
            raise ValueError("Figure-space reference must be a three-dimensional image.")

        self.dseg = image.load_img(dseg) if dseg is not None else None
        if self.dseg is not None:
            if self.dseg.ndim != 3:
                raise ValueError("dseg must be a three-dimensional image.")
            if not _same_fov(self.dseg, self.reference):
                expected_space = "reference" if self.enabled else "input data"
                raise ValueError(f"dseg must have the same field of view as the {expected_space}.")
            dseg_data = np.asanyarray(self.dseg.dataobj)
            if not np.all(np.isfinite(dseg_data)) or not np.allclose(
                dseg_data, np.round(dseg_data)
            ):
                raise ValueError("dseg must contain finite, discrete integer values.")

        self.mask_labels = load_dseg_labels(dseg_tsv)
        self._ants = importlib.import_module("ants") if self.enabled else None
        self.ants_version = getattr(self._ants, "__version__", None)
        self._fixed = (
            self._ants.image_read(op.abspath(reference)).clone("float") if self.enabled else None
        )

    def transform(self, img, *, interpolation="linear", imagetype=None):
        """Return an image in figure space, or the original-space image when disabled."""
        moving_img = image.load_img(img)
        if not self.enabled:
            return moving_img

        if imagetype is None:
            imagetype = 3 if moving_img.ndim == 4 else 0

        with TemporaryDirectory(prefix="tedana-figure-space-") as tmpdir:
            moving_file = op.join(tmpdir, "moving.nii.gz")
            warped_file = op.join(tmpdir, "warped.nii.gz")
            nb.save(moving_img, moving_file)
            moving = self._ants.image_read(moving_file)
            warped = self._ants.apply_transforms(
                fixed=self._fixed,
                moving=moving,
                transformlist=self.xfms,
                interpolator=interpolation,
                imagetype=imagetype,
                singleprecision=True,
            )
            self._ants.image_write(warped, warped_file)
            warped_img = nb.load(warped_file)
            # Copy eagerly so the returned image does not depend on the temporary file.
            return nb.Nifti1Image(
                np.asanyarray(warped_img.dataobj).copy(),
                warped_img.affine.copy(),
                warped_img.header.copy(),
            )
