"""Tests for tensor-ICA decomposition."""

import importlib.util
import shutil

import numpy as np
import pytest


def _make_data(n_voxels=200, n_echoes=3, n_timepoints=80, seed=42):
    """Return (data_cat, mask, echo_times_ms)."""
    rng = np.random.default_rng(seed)
    data_cat = rng.standard_normal((n_voxels, n_echoes, n_timepoints))
    mask = np.ones(n_voxels, dtype=bool)
    echo_times = np.array([13.0, 28.0, 43.0])
    return data_cat, mask, echo_times


# ---------------------------------------------------------------------------
# tensor_ica public interface
# ---------------------------------------------------------------------------


def test_tensor_ica_raises_on_unknown_method():
    from tedana.decomposition.tensor_ica import tensor_ica

    data_cat, mask, echo_times = _make_data()
    with pytest.raises(ValueError, match="Unknown tensor-ICA method"):
        tensor_ica(data_cat, mask, echo_times, method="invalid")


# ---------------------------------------------------------------------------
# tensorly backend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("tensorly") is None,
    reason="tensorly not installed",
)
class TestTensorlyBackend:
    def test_output_shapes(self):
        from tedana.decomposition.tensor_ica import _tensorly_tica

        n_voxels, n_echoes, n_timepoints = 200, 3, 80
        n_components = 5
        data_cat, mask, echo_times = _make_data(n_voxels, n_echoes, n_timepoints)

        mixing, s_modes, spatial_maps = _tensorly_tica(
            data_cat, mask, echo_times, n_components=n_components, seed=42
        )

        assert mixing.shape == (n_timepoints, n_components)
        assert s_modes.shape == (n_echoes, n_components)
        assert spatial_maps.shape == (n_voxels, n_components)

    def test_mixing_is_zscored(self):
        from tedana.decomposition.tensor_ica import _tensorly_tica

        data_cat, mask, echo_times = _make_data()
        mixing, _, _ = _tensorly_tica(data_cat, mask, echo_times, n_components=5, seed=42)

        np.testing.assert_allclose(mixing.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(mixing.std(axis=0), 1, atol=1e-10)

    def test_partial_mask(self):
        from tedana.decomposition.tensor_ica import _tensorly_tica

        data_cat, _, echo_times = _make_data(n_voxels=200)
        mask = np.zeros(200, dtype=bool)
        mask[:150] = True  # only 150 of 200 voxels in mask

        mixing, s_modes, spatial_maps = _tensorly_tica(
            data_cat, mask, echo_times, n_components=4, seed=0
        )

        assert spatial_maps.shape[0] == 200
        assert spatial_maps[~mask].sum() == 0  # unmasked voxels stay zero

    def test_n_components_default(self):
        from tedana.decomposition.tensor_ica import _tensorly_tica

        data_cat, mask, echo_times = _make_data(n_timepoints=80)
        mixing, _, _ = _tensorly_tica(data_cat, mask, echo_times, n_components=None)
        # default = min(50, 80 // 4) = 20
        assert mixing.shape[1] == 20

    def test_tensor_ica_dispatches_to_tensorly(self):
        from tedana.decomposition.tensor_ica import tensor_ica

        data_cat, mask, echo_times = _make_data()
        mixing, s_modes, spatial_maps = tensor_ica(
            data_cat, mask, echo_times, method="tensorly", n_components=5
        )
        assert mixing.ndim == 2
        assert s_modes.ndim == 2
        assert spatial_maps.ndim == 2


# ---------------------------------------------------------------------------
# FSL backend
# ---------------------------------------------------------------------------


def test_fsl_rejects_incompatible_tedpca():
    from tedana.decomposition.tensor_ica import tensor_ica

    data_cat, mask, echo_times = _make_data()
    with pytest.raises(ValueError, match="not compatible with the FSL backend"):
        tensor_ica(data_cat, mask, echo_times, method="fsl", tedpca="kundu", t_r=2.0)


def test_fsl_raises_when_melodic_missing(monkeypatch):
    from tedana.decomposition import tensor_ica as tica_mod

    monkeypatch.setattr(tica_mod.shutil, "which", lambda _: None)
    from tedana.decomposition.tensor_ica import tensor_ica

    data_cat, mask, echo_times = _make_data()
    with pytest.raises(RuntimeError, match="FSL MELODIC is not available"):
        tensor_ica(data_cat, mask, echo_times, method="fsl", t_r=2.0)


@pytest.mark.skipif(
    shutil.which("melodic") is None,
    reason="FSL MELODIC not on PATH",
)
def test_fsl_output_shapes(tmp_path):
    from tedana.decomposition.tensor_ica import tensor_ica

    data_cat, mask, echo_times = _make_data()
    n_voxels, n_echoes, n_timepoints = data_cat.shape

    mixing, s_modes, spatial_maps = tensor_ica(
        data_cat,
        mask,
        echo_times,
        method="fsl",
        tedpca="mdl",
        t_r=2.0,
        out_dir=str(tmp_path),
    )

    assert mixing.ndim == 2
    assert mixing.shape[0] == n_timepoints
    assert s_modes.shape[0] == n_echoes
    assert s_modes.shape[1] == mixing.shape[1]
    assert spatial_maps.shape == (n_voxels, mixing.shape[1])
