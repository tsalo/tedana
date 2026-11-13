"""Tests for tedana.combine."""

import numpy as np
import pytest

from tedana import combine


def test__combine_t2s():
    """Test tedana.combine._combine_t2s."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([[0.010, 0.020, 0.030]])  # 1 x E

    # Voxel- and volume-wise T2* estimates
    t2s = np.random.random((n_voxels, n_trs, 1))  # Mb x T x 1
    comb = combine._combine_t2s(data, tes, t2s)
    assert comb.shape == (n_voxels, n_trs)

    # Voxel-wise T2* estimates
    t2s = np.random.random((n_voxels, 1))  # M x 1
    comb = combine._combine_t2s(data, tes, t2s)
    assert comb.shape == (n_voxels, n_trs)


def test__combine_paid():
    """Test tedana.combine._combine_paid."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([[0.010, 0.020, 0.030]])  # 1 x E
    comb = combine._combine_paid(data, tes)
    assert comb.shape == (n_voxels, n_trs)

    comb2 = combine._combine_paid(data, tes, report=False)
    assert np.array_equal(comb, comb2)


def test_make_optcom():
    """Test tedana.combine.make_optcom."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    n_mask = 5
    data = np.random.random((n_voxels, n_echos, n_trs))
    adaptive_mask = np.zeros(n_voxels, dtype=int)
    adaptive_mask[:n_mask] = 1
    adaptive_mask[0] = 2
    tes = np.array([0.010, 0.020, 0.030])  # E

    # Voxel- and volume-wise T2* estimates
    t2s = np.random.random((n_voxels, n_trs))
    comb = combine.make_optcom(data, tes, adaptive_mask, t2s=t2s, combmode="t2s")
    assert comb.shape == (n_voxels, n_trs)

    # Voxel-wise T2* estimates
    t2s = np.random.random(n_voxels)
    comb = combine.make_optcom(data, tes, adaptive_mask, t2s=t2s, combmode="t2s")
    assert comb.shape == (n_voxels, n_trs)

    # STE with erroneously included T2* argument
    comb = combine.make_optcom(data, tes, adaptive_mask, t2s=t2s, combmode="paid")
    assert comb.shape == (n_voxels, n_trs)

    # Normal STE call
    comb = combine.make_optcom(data, tes, adaptive_mask, t2s=None, combmode="paid")
    assert comb.shape == (n_voxels, n_trs)

    # Test with invalid shapes
    bad_data = np.random.random((n_voxels, n_echos))
    with pytest.raises(ValueError, match="Input data must be 3D"):
        combine.make_optcom(bad_data, tes, adaptive_mask, t2s=t2s, combmode="t2s")

    bad_tes = np.array([0.010, 0.020])
    with pytest.raises(ValueError, match="Number of echos provided does not match second"):
        combine.make_optcom(data, bad_tes, adaptive_mask, t2s=t2s, combmode="t2s")

    bad_adaptive_mask = np.ones((n_voxels, 2), dtype=int)
    with pytest.raises(ValueError, match="Mask is not 1D"):
        combine.make_optcom(data, tes, bad_adaptive_mask, t2s=t2s, combmode="t2s")

    bad_adaptive_mask2 = np.ones(n_voxels - 1, dtype=int)
    with pytest.raises(ValueError, match="Mask and data do not have same number of voxels"):
        combine.make_optcom(data, tes, bad_adaptive_mask2, t2s=t2s, combmode="t2s")

    with pytest.raises(ValueError, match="Argument 'combmode' must be either 't2s' or 'paid'"):
        combine.make_optcom(data, tes, adaptive_mask, t2s=t2s, combmode="bad")

    with pytest.raises(
        ValueError,
        match="Argument 't2s' must be supplied if 'combmode' is set to 't2s'",
    ):
        combine.make_optcom(data, tes, adaptive_mask, t2s=None, combmode="t2s")

    bad_t2s = np.random.random((n_voxels - 1))
    with pytest.raises(
        ValueError,
        match="estimates and data do not have same number of voxels",
    ):
        combine.make_optcom(data, tes, adaptive_mask, t2s=bad_t2s, combmode="t2s")


def _complex_data(n_voxels=4, n_echos=3, n_trs=50, seed=0):
    """Simulate magnitude data and echo-specific constant phase offsets."""
    rng = np.random.default_rng(seed)
    magnitude = 100 + rng.normal(0, 5, size=(n_voxels, n_echos, n_trs))
    theta = rng.uniform(-np.pi, np.pi, size=(n_voxels, n_echos))
    return magnitude, theta


def test_align_complex_echoes_known_phases():
    """Alignment removes constant phase offsets, so temporal means are real and positive."""
    magnitude, theta = _complex_data()
    data = magnitude * np.exp(1j * theta[..., np.newaxis])
    aligned, ref_phase, coherence, failed = combine.align_complex_echoes(data)
    mean = aligned.mean(axis=-1)
    assert np.allclose(mean.imag, 0, atol=1e-10)
    assert np.all(mean.real > 0)
    assert np.allclose(np.angle(np.exp(1j * (ref_phase - theta))), 0)
    assert np.allclose(coherence, 1)
    assert not failed.any()
    # Magnitudes are unchanged.
    assert np.allclose(np.abs(aligned), magnitude)


def test_align_complex_echoes_preserves_temporal_phase():
    """Alignment removes echo offsets but keeps temporal phase fluctuations."""
    magnitude, theta = _complex_data()
    rng = np.random.default_rng(1)
    delta = rng.normal(0, 0.2, size=magnitude.shape[-1])
    data = magnitude * np.exp(1j * (theta[..., np.newaxis] + delta))
    aligned, _, _, _ = combine.align_complex_echoes(data)
    # Volume-to-volume phase changes are unchanged.
    phase_diff = np.angle(aligned[:, :, 1:] * np.conj(aligned[:, :, :-1]))
    assert np.allclose(phase_diff, np.diff(delta)[np.newaxis, np.newaxis, :], atol=1e-10)
    # The echo-specific offsets are removed, so all echoes share one phase time course.
    assert np.allclose(np.angle(aligned[:, 0, :]), np.angle(aligned[:, 1, :]), atol=0.05)


def test_align_complex_echoes_phase_wrapping():
    """Offsets near +/-pi with fluctuations that wrap are aligned without unwrapping."""
    n_trs = 40
    rng = np.random.default_rng(2)
    wobble = rng.normal(0, 0.3, size=n_trs)
    theta = np.array([np.pi - 0.01, -np.pi + 0.01])
    phase = np.angle(np.exp(1j * (theta[:, np.newaxis] + wobble)))  # wrapped
    data = (100 * np.exp(1j * phase))[np.newaxis]
    aligned, _, _, _ = combine.align_complex_echoes(data)
    mean = aligned.mean(axis=-1)
    assert np.allclose(mean.imag, 0, atol=1e-10)
    assert np.all(mean.real > 90)
    # Aligned phases stay near zero instead of jumping by 2 pi.
    assert np.max(np.abs(np.angle(aligned))) < np.pi / 2


def test_align_complex_echoes_incoherent_reference():
    """A zero temporal mean gives finite output, zero rotation and a flag."""
    n_trs = 8
    phase = np.linspace(0, 2 * np.pi, n_trs, endpoint=False)
    data = np.stack([np.exp(1j * phase), np.zeros(n_trs, dtype=complex)])[np.newaxis]
    aligned, ref_phase, coherence, failed = combine.align_complex_echoes(data)
    assert np.all(np.isfinite(aligned))
    assert np.array_equal(failed, [[True, True]])
    assert np.array_equal(ref_phase, np.zeros((1, 2)))
    assert np.allclose(aligned, data)
    assert np.all(coherence < 1e-6)


def test_align_complex_echoes_excluded_volumes():
    """Excluded volumes do not affect the reference but are kept in the output."""
    magnitude, theta = _complex_data(n_trs=30)
    data = magnitude * np.exp(1j * theta[..., np.newaxis])
    use_volumes = np.ones(30, dtype=bool)
    use_volumes[[3, 10]] = False
    corrupted = data.copy()
    corrupted[:, :, [3, 10]] = 1e6 * np.exp(1j * 2.5)
    _, ref_clean, _, _ = combine.align_complex_echoes(data[..., use_volumes])
    aligned, ref_corrupt, _, _ = combine.align_complex_echoes(corrupted, use_volumes=use_volumes)
    assert np.allclose(ref_clean, ref_corrupt)
    assert aligned.shape == data.shape
    assert np.allclose(np.abs(aligned[:, :, [3, 10]]), 1e6)


def test_align_complex_echoes_rejects_real_data():
    """Real-valued input raises an error."""
    with pytest.raises(ValueError, match="complex"):
        combine.align_complex_echoes(np.ones((2, 3, 4)))


@pytest.mark.parametrize("combmode", ["t2s-complex", "paid-complex"])
def test_make_optcom_complex_weights(combmode):
    """Weights are real, nonnegative, zero for unused echoes, and sum to one."""
    magnitude, theta = _complex_data(n_voxels=5, n_echos=4)
    phase = np.broadcast_to(theta[..., np.newaxis], magnitude.shape)
    tes = np.array([0.012, 0.028, 0.044, 0.060])
    adaptive_mask = np.array([0, 1, 2, 3, 4])
    combined, diag = combine.make_optcom_complex(
        magnitude,
        phase,
        tes,
        adaptive_mask,
        t2s=np.full(5, 0.035) if combmode == "t2s-complex" else None,
        combmode=combmode,
    )
    weights = diag["weights"]
    assert np.isrealobj(weights)
    assert np.all(weights >= 0)
    assert np.allclose(weights[1:].sum(axis=1), 1)
    assert np.all(weights[0] == 0)
    assert np.all(combined[0] == 0)
    # Voxels with one or two good echoes use two echoes, as in make_optcom.
    assert np.array_equal((weights > 0).sum(axis=1), [0, 2, 2, 3, 4])


@pytest.mark.parametrize("combmode", ["t2s", "paid"])
def test_make_optcom_complex_matches_real(combmode):
    """With zero phase, complex combination equals magnitude combination."""
    rng = np.random.default_rng(3)
    n_voxels, n_echos, n_trs = 30, 4, 20
    magnitude = rng.uniform(50, 150, size=(n_voxels, n_echos, n_trs))
    tes = np.array([0.012, 0.028, 0.044, 0.060])
    adaptive_mask = rng.integers(0, n_echos + 1, size=n_voxels)
    t2s = rng.uniform(0.02, 0.06, size=n_voxels) if combmode == "t2s" else None
    expected = combine.make_optcom(magnitude, tes, adaptive_mask, t2s=t2s, combmode=combmode)
    combined, _ = combine.make_optcom_complex(
        magnitude,
        np.zeros_like(magnitude),
        tes,
        adaptive_mask,
        t2s=t2s,
        combmode=f"{combmode}-complex",
    )
    assert np.allclose(combined.imag, 0)
    assert np.allclose(combined.real, expected)


def test_make_optcom_complex_prevents_cancellation():
    """Echoes with opposite phase cancel when averaged naively but not after alignment."""
    n_trs = 10
    magnitude = np.ones((1, 2, n_trs))
    phase = np.zeros((1, 2, n_trs))
    phase[:, 1, :] = np.pi
    tes = np.array([0.03, 0.03])  # equal TEs give equal weights
    naive = (magnitude * np.exp(1j * phase)).mean(axis=1)
    assert np.allclose(naive, 0)
    combined, _ = combine.make_optcom_complex(
        magnitude,
        phase,
        tes,
        np.array([2]),
        t2s=np.array([0.03]),
        combmode="t2s-complex",
    )
    assert np.allclose(combined, 1)


@pytest.mark.parametrize("combmode", ["t2s-complex", "paid-complex"])
def test_make_optcom_complex_global_rotation(combmode):
    """A common phase rotation of all echoes leaves the combined data unchanged."""
    magnitude, theta = _complex_data()
    rng = np.random.default_rng(4)
    phase = theta[..., np.newaxis] + rng.normal(0, 0.1, size=magnitude.shape)
    tes = np.array([0.015, 0.035, 0.055])
    adaptive_mask = np.full(4, 3)
    t2s = np.full(4, 0.04) if combmode == "t2s-complex" else None
    combined, _ = combine.make_optcom_complex(
        magnitude,
        phase,
        tes,
        adaptive_mask,
        t2s=t2s,
        combmode=combmode,
    )
    rotated, _ = combine.make_optcom_complex(
        magnitude,
        phase + 1.3,
        tes,
        adaptive_mask,
        t2s=t2s,
        combmode=combmode,
    )
    assert np.allclose(combined, rotated)


def test_make_optcom_complex_paid_weights_use_magnitude():
    """PAID weights do not depend on the phase."""
    magnitude, theta = _complex_data()
    tes = np.array([0.015, 0.035, 0.055])
    adaptive_mask = np.full(4, 3)
    _, diag_zero = combine.make_optcom_complex(
        magnitude,
        np.zeros_like(magnitude),
        tes,
        adaptive_mask,
        combmode="paid-complex",
    )
    phase = np.broadcast_to(theta[..., np.newaxis], magnitude.shape)
    _, diag_phase = combine.make_optcom_complex(
        magnitude,
        phase,
        tes,
        adaptive_mask,
        combmode="paid-complex",
    )
    assert np.allclose(diag_zero["weights"], diag_phase["weights"])


def test_make_optcom_complex_paid_excluded_volumes():
    """Excluded volumes do not inform PAID weights but are retained in the output."""
    magnitude, theta = _complex_data(n_trs=40)
    phase = np.broadcast_to(theta[..., np.newaxis], magnitude.shape).copy()
    tes = np.array([0.015, 0.035, 0.055])
    adaptive_mask = np.full(4, 3)
    use_volumes = np.ones(40, dtype=bool)
    use_volumes[[5, 6]] = False
    _, diag_clean = combine.make_optcom_complex(
        magnitude[..., use_volumes],
        phase[..., use_volumes],
        tes,
        adaptive_mask,
        combmode="paid-complex",
    )
    corrupted_mag = magnitude.copy()
    corrupted_mag[:, :, [5, 6]] = 1e5
    phase[:, :, [5, 6]] = 0.7
    combined, diag = combine.make_optcom_complex(
        corrupted_mag,
        phase,
        tes,
        adaptive_mask,
        combmode="paid-complex",
        use_volumes=use_volumes,
    )
    assert np.allclose(diag["weights"], diag_clean["weights"])
    assert np.allclose(diag["reference_phase"], diag_clean["reference_phase"])
    assert combined.shape == (4, 40)
    # Excluded volumes are rotated and weighted like the others.
    expected = np.sum(
        diag["weights"] * 1e5 * np.exp(1j * (0.7 - diag["reference_phase"])),
        axis=1,
    )
    assert np.allclose(combined[:, 5], expected)
    assert np.allclose(combined[:, 6], expected)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"combmode": "t2s"}, "must be one of"),
        ({"combmode": "t2s-complex"}, "'t2s' must be supplied"),
        ({"combmode": "t2s-complex", "t2s": np.ones((4, 10))}, "volume-wise"),
        ({"combmode": "t2s-complex", "t2s": np.ones(3)}, "same number"),
        ({"combmode": "paid-complex", "use_volumes": np.zeros(10, dtype=bool)}, "At least"),
        ({"combmode": "paid-complex", "use_volumes": np.ones(9, dtype=bool)}, "one value"),
        ({"combmode": "paid-complex", "tes": [0.01, np.nan, 0.03]}, "finite"),
        ({"combmode": "paid-complex", "phase": np.zeros((4, 3, 9))}, "does not match"),
    ],
)
def test_make_optcom_complex_errors(kwargs, match):
    """Invalid inputs raise informative errors."""
    inputs = {
        "magnitude": np.ones((4, 3, 10)),
        "phase": np.zeros((4, 3, 10)),
        "tes": [0.01, 0.02, 0.03],
        "adaptive_mask": np.full(4, 3),
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        combine.make_optcom_complex(**inputs)
