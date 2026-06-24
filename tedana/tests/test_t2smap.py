"""Tests for t2smap."""

import json
import os.path as op
from shutil import rmtree

import nibabel as nb
import numpy as np
import pytest

from tedana import workflows
from tedana.tests.utils import get_test_data_path


class TestTVL2Denoising:
    def test_constant_series_unchanged(self):
        """TV-L2 denoising should leave constant time series unchanged."""
        data = np.ones((2, 3, 8), dtype=np.float32)
        denoised = workflows.t2smap.denoise_tv_l2(data, chunk_size=2)

        assert denoised.dtype == data.dtype
        assert denoised.shape == data.shape
        assert np.allclose(denoised, data)

    def test_spike_is_attenuated(self):
        """TV-L2 denoising should attenuate isolated temporal spikes."""
        data = np.ones((1, 7), dtype=np.float32)
        data[0, 3] = 10

        denoised = workflows.t2smap.denoise_tv_l2(data, chunk_size=1)

        assert np.all(np.isfinite(denoised))
        assert denoised[0, 3] < data[0, 3]
        assert denoised[0, 3] > data[0, 0]

    def test_chunking_matches_full_batch(self):
        """Chunked TV-L2 denoising should match full-batch denoising."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(5, 2, 12)).astype(np.float32)

        full = workflows.t2smap.denoise_tv_l2(data, chunk_size=100)
        chunked = workflows.t2smap.denoise_tv_l2(data, chunk_size=3)

        assert np.allclose(full, chunked)

    def test_invalid_parameters(self):
        """TV-L2 denoising should reject invalid parameters."""
        data = np.ones((1, 5), dtype=np.float32)

        with pytest.raises(ValueError, match="tv_l2_mu"):
            workflows.t2smap.denoise_tv_l2(data, mu=0)
        with pytest.raises(ValueError, match="tv_l2_beta"):
            workflows.t2smap.denoise_tv_l2(data, beta=0)
        with pytest.raises(ValueError, match="tv_l2_chunk_size"):
            workflows.t2smap.denoise_tv_l2(data, chunk_size=0)


class TestT2smap:
    def test_basic_t2smap1(self):
        """
        A very simple test, to confirm that t2smap creates output.

        files.
        """
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        workflows.t2smap_workflow(
            data, [14.5, 38.5, 62.5], combmode="t2s", fitmode="all", out_dir=out_dir
        )

        # Check outputs
        assert op.isfile(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4

    def test_basic_t2smap2(self):
        """
        A very simple test, to confirm that t2smap creates output.

        files when fitmode is set to ts.
        """
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        workflows.t2smap_workflow(
            data, [14.5, 38.5, 62.5], combmode="t2s", fitmode="ts", out_dir=out_dir
        )

        # Check outputs
        assert op.isfile(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4

    def test_basic_t2smap3(self):
        """
        A very simple test, to confirm that t2smap creates output.

        files when combmode is set to 'paid'.
        """
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        workflows.t2smap_workflow(
            data, [14.5, 38.5, 62.5], combmode="paid", fitmode="all", out_dir=out_dir
        )

        # Check outputs
        assert op.isfile(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4

    def test_basic_t2smap4(self):
        """
        A very simple test, to confirm that t2smap creates output.

        files when combmode is set to 'paid' and fitmode is set to 'ts'.
        """
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        workflows.t2smap_workflow(
            data, [14.5, 38.5, 62.5], combmode="paid", fitmode="ts", out_dir=out_dir
        )

        # Check outputs
        assert op.isfile(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4

    def test_t2smap_cli(self):
        """Run test_basic_t2smap1, but use the CLI method."""
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        args = (
            ["-d"]
            + data
            + [
                "-e",
                "14.5",
                "38.5",
                "62.5",
                "--dummy-scans",
                "1",
                "--exclude",
                "0:2",  # exclude one volume beyond the dummy scan
                "--combmode",
                "t2s",
                "--fitmode",
                "all",
                "--out-dir",
                out_dir,
            ]
        )
        workflows.t2smap._main(args)

        # Check outputs
        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4
        in_img = nb.load(data[0])
        target_shape = list(in_img.shape)
        target_shape[3] = target_shape[3] - 1  # account for dummy scans, but not exclude; #1401
        output_shape = list(img.shape)
        assert output_shape == target_shape

    def test_basic_t2smap_tv_l2(self):
        """A simple test to confirm that t2smap can use TV-L2 denoising."""
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        workflows.t2smap_workflow(
            data,
            [14.5, 38.5, 62.5],
            combmode="t2s",
            fitmode="ts",
            out_dir=out_dir,
            tv_l2_denoise=True,
            tv_l2_n_outer_iterations=2,
            tv_l2_n_inner_iterations=1,
            tv_l2_chunk_size=5,
            tv_l2_save_denoised_echos=True,
        )

        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 4
        assert op.isfile(op.join(out_dir, "echo-1_desc-TVL2Denoised_bold.nii.gz"))
        assert op.isfile(op.join(out_dir, "echo-2_desc-TVL2Denoised_bold.nii.gz"))
        assert op.isfile(op.join(out_dir, "echo-3_desc-TVL2Denoised_bold.nii.gz"))

        with open(op.join(out_dir, "dataset_description.json")) as fo:
            dataset_description = json.load(fo)
        tv_metadata = dataset_description["GeneratedBy"][0]["TVL2Denoising"]
        assert tv_metadata["Mu"] == 2**-10
        assert tv_metadata["OuterIterations"] == 2

    def test_failing_t2smap_01(self):
        """A simple failing configuration for t2smap."""
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        with pytest.raises(ValueError, match="Excluding volumes is not supported for fitmode"):
            workflows.t2smap_workflow(
                data,
                [14.5, 38.5, 62.5],
                combmode="t2s",
                fitmode="ts",
                out_dir=out_dir,
                exclude="0,1,2,3",
            )

    def test_failing_t2smap_02(self):
        """A simple failing configuration for t2smap."""
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        with pytest.raises(ValueError, match="The maximum exclude index"):
            workflows.t2smap_workflow(
                data,
                [14.5, 38.5, 62.5],
                combmode="t2s",
                fitmode="all",
                out_dir=out_dir,
                exclude="1000",
            )

    def teardown_method(self):
        # Clean up folders
        if op.isdir("TED.echo1.t2smap"):
            rmtree("TED.echo1.t2smap")
