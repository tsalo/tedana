"""Spatial noise metrics.

These metrics are independent of the TE-dependence model.
They identify acquisition/motion artifacts that can mimic mixed TE-dependence:

- ``compute_grappa_artifact`` counts local-autocorrelation-violation voxels
  (in-plane parallel-imaging / GRAPPA aliasing) in a component's spatial map.
"""

import logging

import nibabel as nb
import numpy as np
from nilearn import masking
from scipy import ndimage

LGR = logging.getLogger("GENERAL")


def _spherical_footprint() -> np.ndarray:
    """3x3x3 boolean footprint of the 19-voxel sphere (center + 6 face + 12 edge).

    Matches AFNI ``SPHERE(-1.42)``: includes every offset whose squared distance
    from the center is <= 2 (excludes the 8 corner voxels at distance sqrt(3)).
    """
    footprint = np.zeros((3, 3, 3), dtype=bool)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if i * i + j * j + k * k <= 2:
                    footprint[i + 1, j + 1, k + 1] = True
    return footprint


def _local_masked_variance(
    field: np.ndarray, valid: np.ndarray, footprint: np.ndarray
) -> np.ndarray:
    """Per-voxel variance of ``field`` over in-``valid`` voxels in each neighborhood.

    Returns ``np.nan`` where fewer than two valid voxels fall in the neighborhood.
    """
    weights = footprint.astype(float)
    valid_f = valid.astype(float)
    n = ndimage.convolve(valid_f, weights, mode="constant", cval=0.0)
    s1 = ndimage.convolve(np.where(valid, field, 0.0), weights, mode="constant", cval=0.0)
    s2 = ndimage.convolve(np.where(valid, field * field, 0.0), weights, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = s2 / n - mean * mean
    var[n < 2] = np.nan
    return var


def compute_grappa_artifact(*, psc_maps: np.ndarray, mask_img: nb.Nifti1Image) -> np.ndarray:
    """Count local-autocorrelation-violation voxels (GRAPPA aliasing) per component.

    For each component percent-signal-change map and each spatial axis ``d``, a
    local first-difference autocorrelation is estimated over a 19-voxel spherical
    neighbourhood, ``r_d = 1 - Var(dI_d) / (2 Var(I))``. Where ``r_d <= 0`` (i.e.
    ``Var(dI_d) >= 2 Var(I)``) adjacent voxels are anti-correlated and the implied
    smoothness/FWHM is indeterminate -- the signature of in-plane parallel-imaging
    (GRAPPA) reconstruction error. The metric is the number of ``(voxel, axis)``
    pairs flagged, summed over the three axes, faithful to me-ica's
    ``score_fourier_artifact_count``. Scale- and sign-invariant.

    Parameters
    ----------
    psc_maps : (M_s x C) array_like
        Component percent-signal-change maps in strict-mask voxel space.
    mask_img : img_like
        Strict mask used to unmask ``psc_maps`` to the image grid.

    Returns
    -------
    (C,) :obj:`numpy.ndarray`
        Count of local-autocorrelation-violation voxel-axis pairs per component.
    """
    mask_bool = np.asanyarray(mask_img.dataobj).astype(bool)
    img = masking.unmask(psc_maps.T, mask_img)
    vols = np.asanyarray(img.dataobj)  # (X, Y, Z, C)
    n_comp = vols.shape[-1]
    footprint = _spherical_footprint()
    out = np.zeros(n_comp, dtype=int)
    for c in range(n_comp):
        vol = vols[..., c].astype(np.float64)
        var_i = _local_masked_variance(vol, mask_bool, footprint)
        flagged = 0
        for axis in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(0, -1)
            hi[axis] = slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            diff = np.zeros_like(vol)
            valid = np.zeros(vol.shape, dtype=bool)
            diff[lo] = vol[hi] - vol[lo]
            valid[lo] = mask_bool[hi] & mask_bool[lo]
            var_d = _local_masked_variance(diff, valid, footprint)
            with np.errstate(invalid="ignore"):
                indeterminate = mask_bool & (var_i > 0) & (var_d >= 2.0 * var_i)
            flagged += int(np.count_nonzero(indeterminate))
        out[c] = flagged
    return out
