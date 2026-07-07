"""Tests for tedana.metrics.spatial."""

import os.path as op

import nibabel as nb
import numpy as np

from tedana import io, utils
from tedana.metrics import spatial
from tedana.metrics._utils import dependency_resolver


def _grid_mask(shape=(10, 10, 12)):
    aff = np.eye(4)
    mask = np.ones(shape, dtype=np.int16)
    return nb.Nifti1Image(mask, aff), shape


def test_grappa_flags_checkerboard_and_spares_smooth():
    mask_img, shape = _grid_mask((12, 12, 12))
    ix, iy, iz = np.indices(shape)
    # Component 0: Nyquist checkerboard (alternating sign every voxel) -> GRAPPA-like.
    checker = ((-1.0) ** (ix + iy + iz)).astype(float)
    # Component 1: smooth Gaussian blob (high local autocorrelation) -> clean.
    center = (np.array(shape) - 1) / 2.0
    r2 = (ix - center[0]) ** 2 + (iy - center[1]) ** 2 + (iz - center[2]) ** 2
    smooth = np.exp(-r2 / (2 * 4.0**2))
    maps = np.stack([checker.reshape(-1), smooth.reshape(-1)], axis=1)
    out = spatial.compute_grappa_artifact(psc_maps=maps, mask_img=mask_img)
    assert out.shape == (2,)
    assert out[0] > 0  # checkerboard flagged
    assert out[1] == 0  # smooth spared
    assert out[0] > out[1]


def test_grappa_spares_constant_map():
    mask_img, shape = _grid_mask()
    maps = np.ones((int(np.prod(shape)), 1))
    out = spatial.compute_grappa_artifact(psc_maps=maps, mask_img=mask_img)
    assert out[0] == 0


def test_grappa_flags_single_axis_alternation():
    mask_img, shape = _grid_mask((10, 10, 12))
    _ix, _iy, iz = np.indices(shape)
    alt_z = ((-1.0) ** iz).astype(float)  # alternates along z only
    maps = alt_z.reshape(-1)[:, None]
    out = spatial.compute_grappa_artifact(psc_maps=maps, mask_img=mask_img)
    assert out[0] > 0


def test_grappa_is_scale_invariant():
    mask_img, shape = _grid_mask((10, 10, 10))
    rng = np.random.default_rng(0)
    ix, iy, iz = np.indices(shape)
    vol = ((-1.0) ** (ix + iy + iz)) + 0.1 * rng.standard_normal(shape)
    maps = vol.reshape(-1)[:, None]
    a = spatial.compute_grappa_artifact(psc_maps=maps, mask_img=mask_img)
    b = spatial.compute_grappa_artifact(psc_maps=maps * 7.0, mask_img=mask_img)
    assert np.array_equal(a, b)


def test_grappa_artifact_is_registered():
    """grappa_artifact and its fraction must be in metrics.json with correct deps."""
    cfg = io.load_json(op.join(utils.get_resource_path(), "config", "metrics.json"))
    assert "grappa_artifact" in cfg["dependencies"]
    assert "grappa_artifact_fraction" in cfg["dependencies"]
    required = dependency_resolver(
        cfg["dependencies"], ["grappa_artifact_fraction"], cfg["inputs"]
    )
    assert "grappa_artifact" in required  # fraction pulls in the raw count
    assert "map percent signal change" in required  # count's dependency
    assert "mask" in required
