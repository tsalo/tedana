"""Tests for tedana.tv."""

import numpy as np
import pytest

from tedana import tv


class TestTVL2Denoising:
    def test_constant_series_unchanged(self):
        """TV-L2 denoising should leave constant time series unchanged."""
        data = np.ones((2, 3, 8), dtype=np.float32)
        denoised = tv.denoise_tv_l2(data, chunk_size=2)

        assert denoised.dtype == data.dtype
        assert denoised.shape == data.shape
        assert np.allclose(denoised, data)

    def test_spike_is_attenuated(self):
        """TV-L2 denoising should attenuate isolated temporal spikes."""
        data = np.ones((1, 7), dtype=np.float32)
        data[0, 3] = 10

        denoised = tv.denoise_tv_l2(data, chunk_size=1)

        assert np.all(np.isfinite(denoised))
        assert denoised[0, 3] < data[0, 3]
        assert denoised[0, 3] > data[0, 0]

    def test_chunking_matches_full_batch(self):
        """Chunked TV-L2 denoising should match full-batch denoising."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(5, 2, 12)).astype(np.float32)

        full = tv.denoise_tv_l2(data, chunk_size=100)
        chunked = tv.denoise_tv_l2(data, chunk_size=3)

        assert np.allclose(full, chunked)

    def test_invalid_parameters(self):
        """TV-L2 denoising should reject invalid parameters."""
        data = np.ones((1, 5), dtype=np.float32)

        with pytest.raises(ValueError, match="tv_l2_mu"):
            tv.denoise_tv_l2(data, mu=0)
        with pytest.raises(ValueError, match="tv_l2_beta"):
            tv.denoise_tv_l2(data, beta=0)
        with pytest.raises(ValueError, match="tv_l2_chunk_size"):
            tv.denoise_tv_l2(data, chunk_size=0)
