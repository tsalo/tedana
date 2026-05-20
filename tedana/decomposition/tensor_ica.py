"""Tensor-ICA decomposition methods for multi-echo fMRI."""

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def tensor_ica(
    data_cat,
    mask,
    echo_times,
    method="tensorly",
    n_components=None,
    tedpca=None,
    t_r=None,
    out_dir=".",
    seed=42,
):
    """Perform tensor-ICA on multi-echo fMRI data.

    Parameters
    ----------
    data_cat : (n_voxels, n_echoes, n_timepoints) :obj:`numpy.ndarray`
        Concatenated multi-echo data in tedana's native axis order.
    mask : (n_voxels,) :obj:`numpy.ndarray` of bool
        Brain mask.
    echo_times : (n_echoes,) :obj:`numpy.ndarray`
        Echo times in milliseconds.
    method : {"tensorly", "fsl"}, optional
        Decomposition backend. Default is "tensorly".
    n_components : :obj:`int` or None, optional
        Number of components. If None, defaults to ``min(50, n_timepoints // 4)``
        for the tensorly backend, or MDL estimation for the FSL backend.
    tedpca : :obj:`str` or None, optional
        Dimension estimation label forwarded to FSL as ``--dimest``. Only
        ``"mdl"`` and ``"aic"`` are compatible. Ignored for tensorly backend.
    t_r : :obj:`float` or None, optional
        Repetition time in seconds. Required by the FSL backend.
    out_dir : :obj:`str`, optional
        Working directory for FSL intermediate files. Default is ``"."``.
    seed : :obj:`int`, optional
        Random seed (tensorly backend only). Default is 42.

    Returns
    -------
    mixing : (n_timepoints, n_components) :obj:`numpy.ndarray`
        Z-scored temporal component time courses.
    s_modes : (n_echoes, n_components) :obj:`numpy.ndarray`
        TE-mode loadings per component.
    spatial_maps : (n_voxels, n_components) :obj:`numpy.ndarray`
        Spatial component maps (full brain space, zeros outside mask).
    """
    if method == "tensorly":
        return _tensorly_tica(
            data_cat, mask, echo_times, n_components=n_components, seed=seed
        )
    elif method == "fsl":
        return _fsl_melodic_tica(
            data_cat,
            mask,
            echo_times,
            n_components=n_components,
            tedpca=tedpca,
            t_r=t_r,
            out_dir=out_dir,
        )
    else:
        raise ValueError(
            f"Unknown tensor-ICA method: {method!r}. Choose 'tensorly' or 'fsl'."
        )


def _tensorly_tica(data_cat, mask, echo_times, n_components=None, seed=42):
    """Tensor-ICA via Tucker decomposition + FastICA (tensorly backend).

    Parameters
    ----------
    data_cat : (n_voxels, n_echoes, n_timepoints) :obj:`numpy.ndarray`
    mask : (n_voxels,) :obj:`numpy.ndarray` of bool
    echo_times : (n_echoes,) :obj:`numpy.ndarray`
    n_components : :obj:`int` or None
    seed : :obj:`int`

    Returns
    -------
    mixing : (n_timepoints, n_components) :obj:`numpy.ndarray`
    s_modes : (n_echoes, n_components) :obj:`numpy.ndarray`
    spatial_maps : (n_voxels, n_components) :obj:`numpy.ndarray`
    """
    try:
        import tensorly as tl
        from tensorly.decomposition import tucker
    except ImportError as exc:
        raise ImportError(
            "tensorly is required for --ica-method tensorly. "
            "Install it with: pip install tedana[tensor-ica]"
        ) from exc

    n_voxels, n_echoes, n_timepoints = data_cat.shape

    if n_components is None:
        n_components = max(1, min(50, n_timepoints // 4))

    LGR.info(
        f"Running tensorly Tucker decomposition with {n_components} components "
        f"on {mask.sum()} masked voxels."
    )
    RepLGR.info(
        "Tensor-ICA via Tucker decomposition (tensorly) followed by FastICA "
        "was used to decompose the multi-echo data."
    )

    # data_cat is (n_voxels, n_echoes, n_timepoints); reshape masked subset to
    # (n_masked, n_timepoints, n_echoes) for Tucker decomposition
    masked_data = data_cat[mask]  # (n_masked, n_echoes, n_timepoints)
    tensor = tl.tensor(masked_data.transpose(0, 2, 1))  # (n_masked, n_timepoints, n_echoes)

    # Tucker rank for the echo mode is at most n_echoes; for spatial/temporal
    # modes we clamp to available dimensions.
    echo_rank = min(n_echoes, n_components)
    n_masked = mask.sum()
    spatial_rank = min(n_components, n_masked)
    temporal_rank = min(n_components, n_timepoints)
    ranks = [spatial_rank, temporal_rank, echo_rank]
    _, factors = tucker(tensor, rank=ranks, random_state=seed)
    # factors[0]: (n_masked, spatial_rank)    — spatial
    # factors[1]: (n_timepoints, temporal_rank) — temporal
    # factors[2]: (n_echoes, echo_rank)       — echo / TE-mode

    # Clamp n_components to the smallest rank so all outputs are consistent
    n_components_actual = min(spatial_rank, temporal_rank, echo_rank)
    ica = FastICA(n_components=n_components_actual, random_state=seed, max_iter=500)
    mixing_raw = ica.fit_transform(factors[1][:, :n_components_actual])

    mixing = stats.zscore(mixing_raw, axis=0)

    # Slice echo factors to (n_echoes, n_components_actual) for consistent output shape
    s_modes = factors[2][:, :n_components_actual]

    spatial_maps = np.zeros((n_voxels, n_components_actual))
    spatial_maps[mask] = factors[0][:, :n_components_actual]

    return mixing, s_modes, spatial_maps


_FSL_COMPATIBLE_TEDPCA = {"mdl", "aic", None}


def _fsl_melodic_tica(
    data_cat, mask, echo_times, n_components=None, tedpca=None, t_r=None, out_dir="."
):
    """Tensor-ICA via FSL MELODIC (fsl backend).

    Parameters
    ----------
    data_cat : (n_voxels, n_echoes, n_timepoints) :obj:`numpy.ndarray`
    mask : (n_voxels,) :obj:`numpy.ndarray` of bool
    echo_times : (n_echoes,) :obj:`numpy.ndarray`
    n_components : :obj:`int` or None
    tedpca : :obj:`str` or None
        Must be one of ``{"mdl", "aic", None}``. ``None`` defaults to ``"mdl"``.
    t_r : :obj:`float` or None
        Repetition time in seconds. Required.
    out_dir : :obj:`str`

    Returns
    -------
    mixing : (n_timepoints, n_components) :obj:`numpy.ndarray`
    s_modes : (n_echoes, n_components) :obj:`numpy.ndarray`
    spatial_maps : (n_voxels, n_components) :obj:`numpy.ndarray`
    """
    import nibabel as nib

    if tedpca not in _FSL_COMPATIBLE_TEDPCA:
        raise ValueError(
            f"tedpca={tedpca!r} is not compatible with the FSL backend. "
            "Only 'mdl' or 'aic' are supported; other --tedpca values "
            "(fixed integer, 'kundu', 'kundu-stabilize') cannot be mapped to "
            "FSL's --dimest parameter."
        )

    if shutil.which("melodic") is None:
        raise RuntimeError(
            "FSL MELODIC is not available. Ensure FSL is installed and 'melodic' is on PATH."
        )

    if t_r is None:
        raise ValueError("t_r (repetition time in seconds) is required for the FSL backend.")

    dimest = tedpca if tedpca is not None else "mdl"
    n_voxels, n_echoes, n_timepoints = data_cat.shape

    melodic_dir = Path(out_dir) / "melodic_tica"
    melodic_dir.mkdir(parents=True, exist_ok=True)

    melodic_out = melodic_dir / "melodic_out"
    melodic_out.mkdir(exist_ok=True)

    # Concatenate masked data into a 4D NIfTI: (n_masked, 1, n_echoes, n_timepoints)
    concat_4d = data_cat[mask].transpose(2, 0, 1)  # (n_timepoints, n_masked, n_echoes)
    concat_4d = concat_4d.reshape(n_timepoints, mask.sum(), 1, n_echoes)
    concat_4d = concat_4d.transpose(1, 2, 3, 0)  # (n_masked, 1, n_echoes, n_timepoints)

    affine = np.eye(4)
    in_nii = nib.Nifti1Image(concat_4d.astype(np.float32), affine)
    in_path = melodic_dir / "data_concat.nii.gz"
    nib.save(in_nii, str(in_path))

    cmd = [
        "melodic",
        "-i", str(in_path),
        "-o", str(melodic_out),
        "--tica",
        f"--tr={t_r}",
        f"--dimest={dimest}",
        "--Oall",
        "--vn",
    ]
    if n_components is not None:
        cmd += ["-d", str(n_components)]

    LGR.info(f"Running FSL MELODIC tensor-ICA: {' '.join(cmd)}")
    RepLGR.info(
        "Tensor-ICA was performed using FSL MELODIC with the --tica flag."
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FSL MELODIC failed (exit code {result.returncode}):\n{result.stderr}"
        )

    # Parse MELODIC outputs
    ic_img = nib.load(str(melodic_out / "melodic_IC.nii.gz"))
    spatial_maps_masked = ic_img.get_fdata().reshape(mask.sum(), -1)  # (n_masked, n_comp)

    mixing = np.loadtxt(str(melodic_out / "melodic_Tmodes"))  # (n_timepoints, n_comp)
    s_modes = np.loadtxt(str(melodic_out / "melodic_Smodes"))  # (n_echoes, n_comp)

    mixing = stats.zscore(mixing, axis=0)

    spatial_maps = np.zeros((n_voxels, spatial_maps_masked.shape[1]))
    spatial_maps[mask] = spatial_maps_masked

    return mixing, s_modes, spatial_maps
