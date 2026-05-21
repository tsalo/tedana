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

        # n_components_actual = min(spatial_rank, temporal_rank); echo rank does
        # NOT limit ICA components — s_modes are computed post-hoc via projection.
        n_components_actual = min(
            min(n_components, mask.sum()),  # spatial_rank
            min(n_components, n_timepoints),  # temporal_rank
        )
        assert mixing.shape == (n_timepoints, n_components_actual)
        assert s_modes.shape == (n_echoes, n_components_actual)
        assert spatial_maps.shape == (n_voxels, n_components_actual)

    def test_mixing_is_zscored(self):
        from tedana.decomposition.tensor_ica import _tensorly_tica

        data_cat, mask, echo_times = _make_data()
        mixing, _, _ = _tensorly_tica(data_cat, mask, echo_times, n_components=5, seed=42)

        np.testing.assert_allclose(mixing.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(mixing.std(axis=0), 1, atol=1e-6)

    def test_partial_mask(self):
        from tedana.decomposition.tensor_ica import _tensorly_tica

        data_cat, _, echo_times = _make_data(n_voxels=200)
        mask = np.zeros(200, dtype=bool)
        mask[:150] = True  # only 150 of 200 voxels in mask

        mixing, s_modes, spatial_maps = _tensorly_tica(
            data_cat, mask, echo_times, n_components=4, seed=0
        )

        assert spatial_maps.shape[0] == 200
        assert np.all(spatial_maps[~mask] == 0)  # unmasked voxels stay zero

    def test_n_components_default(self):
        from tedana.decomposition.tensor_ica import _tensorly_tica

        n_voxels, n_echoes, n_timepoints = 200, 3, 80
        data_cat, mask, echo_times = _make_data(n_voxels, n_echoes, n_timepoints)
        mixing, _, _ = _tensorly_tica(data_cat, mask, echo_times, n_components=None)
        # default n_components = min(50, 80 // 4) = 20
        # n_components_actual = min(spatial_rank, temporal_rank); echo rank does NOT limit
        n_components_default = max(1, min(50, n_timepoints // 4))
        n_components_actual = min(
            min(n_components_default, mask.sum()),
            min(n_components_default, n_timepoints),
        )
        assert mixing.shape[1] == n_components_actual

    def test_tensor_ica_dispatches_to_tensorly(self):
        from tedana.decomposition.tensor_ica import tensor_ica

        data_cat, mask, echo_times = _make_data()
        mixing, s_modes, spatial_maps = tensor_ica(
            data_cat, mask, echo_times, method="tensorly", n_components=5
        )
        assert mixing.ndim == 2
        assert s_modes.ndim == 2
        assert spatial_maps.ndim == 2

    def test_few_masked_voxels_fewer_than_components(self):
        """spatial_rank = n_masked < n_components should not crash."""
        from tedana.decomposition.tensor_ica import _tensorly_tica

        n_voxels, n_echoes, n_timepoints, n_components = 200, 3, 80, 5
        data_cat, _, echo_times = _make_data(n_voxels, n_echoes, n_timepoints)
        # Only 3 voxels in mask — fewer than n_components=5
        mask = np.zeros(n_voxels, dtype=bool)
        mask[:3] = True

        mixing, s_modes, spatial_maps = _tensorly_tica(
            data_cat, mask, echo_times, n_components=n_components, seed=0
        )

        # All outputs should have the same number of components
        n_comp_out = mixing.shape[1]
        assert s_modes.shape[1] == n_comp_out
        assert spatial_maps.shape[1] == n_comp_out
        assert spatial_maps.shape[0] == n_voxels

    def test_few_timepoints_fewer_than_components(self):
        """temporal_rank = n_timepoints < n_components should not crash."""
        from tedana.decomposition.tensor_ica import _tensorly_tica

        n_voxels, n_echoes, n_timepoints, n_components = 200, 3, 4, 10
        data_cat, mask, echo_times = _make_data(n_voxels, n_echoes, n_timepoints)

        mixing, s_modes, spatial_maps = _tensorly_tica(
            data_cat, mask, echo_times, n_components=n_components, seed=0
        )

        n_comp_out = mixing.shape[1]
        assert s_modes.shape[1] == n_comp_out
        assert spatial_maps.shape[1] == n_comp_out

    def test_many_echoes_spatial_bottleneck(self):
        """When spatial_rank is the bottleneck (n_masked < n_echoes < n_components)."""
        from tedana.decomposition.tensor_ica import _tensorly_tica

        n_voxels, n_echoes, n_timepoints, n_components = 200, 5, 80, 8
        data_cat, _, echo_times = _make_data(n_voxels, n_echoes, n_timepoints, seed=1)
        echo_times = np.array([13.0, 21.0, 29.0, 37.0, 45.0])  # 5 echoes
        # 3 masked voxels — spatial_rank=3 is bottleneck below n_echoes=5
        mask = np.zeros(n_voxels, dtype=bool)
        mask[:3] = True

        mixing, s_modes, spatial_maps = _tensorly_tica(
            data_cat, mask, echo_times, n_components=n_components, seed=0
        )

        n_comp_out = mixing.shape[1]
        assert s_modes.shape == (n_echoes, n_comp_out)
        assert spatial_maps.shape == (n_voxels, n_comp_out)


# ---------------------------------------------------------------------------
# FSL backend
# ---------------------------------------------------------------------------


def test_fsl_rejects_incompatible_tedpca():
    from tedana.decomposition.tensor_ica import tensor_ica

    data_cat, mask, echo_times = _make_data()
    with pytest.raises(ValueError, match="not compatible with the FSL backend"):
        tensor_ica(data_cat, mask, echo_times, method="fsl", tedpca="kundu", t_r=2.0)


def test_fsl_raises_when_melodic_missing(monkeypatch):
    import importlib

    import tedana.decomposition.tensor_ica  # ensure module is in sys.modules

    tica_mod = importlib.import_module("tedana.decomposition.tensor_ica")
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _make_metric_inputs(n_echoes=3, n_components=5, n_timepoints=100, n_voxels=200, seed=0):
    rng = np.random.default_rng(seed)
    s_modes = rng.standard_normal((n_echoes, n_components))
    mixing = rng.standard_normal((n_timepoints, n_components))
    spatial_maps = rng.standard_normal((n_voxels, n_components))
    echo_times = np.array([13.0, 28.0, 43.0])
    tr = 2.0
    return s_modes, mixing, spatial_maps, echo_times, tr, n_timepoints


def test_generate_tensor_metrics_shape_and_columns():
    from tedana.metrics.tensor_ica import generate_tensor_metrics

    s_modes, mixing, spatial_maps, echo_times, tr, n_vols = _make_metric_inputs()
    n_components = mixing.shape[1]

    ct = generate_tensor_metrics(s_modes, mixing, spatial_maps, echo_times, tr, n_vols)

    assert len(ct) == n_components
    for col in ("te_peak", "freq_ratio", "variance explained", "classification"):
        assert col in ct.columns, f"Missing column: {col}"


def test_freq_ratio_between_zero_and_one():
    from tedana.metrics.tensor_ica import generate_tensor_metrics

    s_modes, mixing, spatial_maps, echo_times, tr, n_vols = _make_metric_inputs()
    ct = generate_tensor_metrics(s_modes, mixing, spatial_maps, echo_times, tr, n_vols)

    assert ct["freq_ratio"].between(0.0, 1.0).all(), "freq_ratio must be in [0, 1]"


def test_variance_explained_sums_to_one():
    from tedana.metrics.tensor_ica import generate_tensor_metrics

    s_modes, mixing, spatial_maps, echo_times, tr, n_vols = _make_metric_inputs()
    ct = generate_tensor_metrics(s_modes, mixing, spatial_maps, echo_times, tr, n_vols)

    np.testing.assert_allclose(ct["variance explained"].sum(), 1.0, atol=1e-6)


def test_te_peak_is_finite():
    from tedana.metrics.tensor_ica import generate_tensor_metrics

    s_modes, mixing, spatial_maps, echo_times, tr, n_vols = _make_metric_inputs()
    ct = generate_tensor_metrics(s_modes, mixing, spatial_maps, echo_times, tr, n_vols)

    assert ct["te_peak"].notna().all()
    assert np.isfinite(ct["te_peak"].values).all()


# ---------------------------------------------------------------------------
# Selection nodes
# ---------------------------------------------------------------------------
import math

import pandas as pd


def _make_selector(te_peaks, freq_ratios=None):
    """Return a minimal ComponentSelector-like object with te_peak and freq_ratio."""
    from unittest.mock import MagicMock

    n = len(te_peaks)
    data = {
        "Component": [f"ICA_{i:02d}" for i in range(n)],
        "classification": ["unclassified"] * n,
        "rationale": [""] * n,
        "classification_tags": [""] * n,
        "te_peak": te_peaks,
    }
    if freq_ratios is not None:
        data["freq_ratio"] = freq_ratios

    ct = pd.DataFrame(data)

    selector = MagicMock()
    selector.component_table_ = ct
    selector.n_comps_ = n
    selector.current_node_idx_ = 0
    selector.tree = {"nodes": [{"outputs": {}}]}
    selector.cross_component_metrics_ = {}
    return selector


def test_dec_te_peak_range_rejects_out_of_range():
    from tedana.selection.selection_nodes import dec_te_peak_range

    # peaks: 5 (too low), 25 (ok), 45 (ok), 65 (too high)
    selector = _make_selector([5.0, 25.0, 45.0, 65.0])
    selector = dec_te_peak_range(selector, "rejected", "nochange", "all")

    ct = selector.component_table_
    # if_true="rejected": out-of-range → rejected
    # if_false="nochange": in-range → classification unchanged (stays "unclassified")
    assert ct.loc[0, "classification"] == "rejected"
    assert ct.loc[1, "classification"] == "unclassified"
    assert ct.loc[2, "classification"] == "unclassified"
    assert ct.loc[3, "classification"] == "rejected"


def test_dec_te_peak_range_custom_bounds():
    from tedana.selection.selection_nodes import dec_te_peak_range

    selector = _make_selector([10.0, 30.0, 50.0])
    selector = dec_te_peak_range(
        selector, "rejected", "nochange", "all", te_peak_min=20, te_peak_max=40
    )
    ct = selector.component_table_
    assert ct.loc[0, "classification"] == "rejected"       # 10 < 20
    assert ct.loc[1, "classification"] == "unclassified"   # 30, in range; nochange keeps original
    assert ct.loc[2, "classification"] == "rejected"       # 50 > 40


def test_dec_freq_ratio_accepts_above_threshold():
    from tedana.selection.selection_nodes import dec_freq_ratio

    selector = _make_selector([25.0, 25.0, 25.0], freq_ratios=[0.8, 0.5, 0.72])
    selector = dec_freq_ratio(selector, "accepted", "rejected", "all")

    ct = selector.component_table_
    assert ct.loc[0, "classification"] == "accepted"   # 0.8 > 0.7
    assert ct.loc[1, "classification"] == "rejected"   # 0.5 <= 0.7
    assert ct.loc[2, "classification"] == "accepted"   # 0.72 > 0.7


def test_dec_keep_top_n_keeps_correct_count():
    from tedana.selection.selection_nodes import dec_keep_top_n

    # 10 components total, keep_ratio=0.3 → keep ceil(10*0.3)=3
    # min_keep_fraction=0.7 → keep ceil(10*0.7)=7; max(3,7)=7 kept
    te_peaks = list(range(10, 110, 10))  # [10, 20, ..., 100]
    selector = _make_selector(te_peaks)
    # if_true="nochange": top-N kept → classification unchanged (stays "unclassified")
    # if_false="rejected": bottom components → rejected
    selector = dec_keep_top_n(selector, "nochange", "rejected", "all", keep_ratio=0.3)

    ct = selector.component_table_
    # kept = not rejected (classification stayed "unclassified")
    kept = (ct["classification"] == "unclassified").sum()
    rejected = (ct["classification"] == "rejected").sum()
    expected_keep = max(math.ceil(10 * 0.3), math.ceil(10 * 0.7))
    assert kept == expected_keep
    assert kept + rejected == 10


def test_dec_keep_top_n_only_used_metrics():
    from tedana.selection.selection_nodes import dec_keep_top_n

    selector = _make_selector([25.0])
    result = dec_keep_top_n(selector, "nochange", "rejected", "all", only_used_metrics=True)
    assert "te_peak" in result


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("tensorly") is None,
    reason="tensorly not installed",
)
def test_tensorly_workflow_end_to_end(tmp_path):
    """Smoke test: tensorly backend runs workflow and produces key output files."""
    import nibabel as nib
    from tedana.workflows.tedana import tedana_workflow

    # Build minimal 3-echo synthetic data files
    rng = np.random.default_rng(0)
    n_x, n_y, n_z, n_t = 5, 5, 3, 60
    affine = np.eye(4)
    affine[0, 0] = 2.0
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0
    affine[3, 3] = 2.0  # TR = 2 seconds
    echo_times_s = [0.013, 0.028, 0.043]  # seconds (tedana convention)
    data_files = []

    for i, te in enumerate(echo_times_s):
        # Simple T2*-weighted signal
        signal = rng.standard_normal((n_x, n_y, n_z, n_t)) * np.exp(-te / 0.030) + 100
        img = nib.Nifti1Image(signal.astype(np.float32), affine)
        p = tmp_path / f"echo{i+1}.nii.gz"
        nib.save(img, str(p))
        data_files.append(str(p))

    # Provide an explicit all-ones mask so compute_epi_mask is not called on
    # pure-noise synthetic data (which would produce an empty mask).
    mask_data = np.ones((n_x, n_y, n_z), dtype=np.uint8)
    mask_img = nib.Nifti1Image(mask_data, affine)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(mask_img, str(mask_path))

    tedana_workflow(
        data=data_files,
        tes=echo_times_s,
        out_dir=str(tmp_path),
        mask=str(mask_path),
        ica_method="tensorly",
        fittype="loglin",
        fixed_seed=42,
        verbose=False,
        no_reports=True,
    )

    # Check key output files exist
    assert (tmp_path / "desc-denoised_bold.nii.gz").exists()
    assert (tmp_path / "desc-ICA_mixing.tsv").exists()
    assert (tmp_path / "desc-ICA_smodes.tsv").exists()
    # Component table is saved as desc-tedana_metrics.tsv
    assert (tmp_path / "desc-tedana_metrics.tsv").exists()

    # Check component table has expected columns
    ct = pd.read_csv(tmp_path / "desc-tedana_metrics.tsv", sep="\t")
    for col in ("te_peak", "freq_ratio", "variance explained", "classification"):
        assert col in ct.columns, f"Missing column in component table: {col}"

    # All components classified
    assert ct["classification"].isin(["accepted", "rejected"]).all()
