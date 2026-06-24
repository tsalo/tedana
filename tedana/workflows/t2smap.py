"""Estimate T2 and S0, and optimally combine data across TEs."""

import argparse
import logging
import os
import os.path as op
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from tedana import __version__, combine, decay, io, utils
from tedana.utils import parse_volume_indices
from tedana.workflows.parser_utils import is_valid_file

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def _forward_difference(data):
    """Calculate non-circular forward differences along the time axis."""
    return np.diff(data, axis=-1)


def _difference_transpose(differences, n_vols):
    """Apply the transpose of the non-circular forward-difference operator."""
    out = np.zeros((*differences.shape[:-1], n_vols), dtype=differences.dtype)
    out[..., 0] = -differences[..., 0]
    out[..., 1:-1] = differences[..., :-1] - differences[..., 1:]
    out[..., -1] = differences[..., -1]
    return out


def _soft_threshold(data, threshold):
    """Apply scalar soft-thresholding."""
    return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)


def _validate_tv_l2_options(mu, beta, n_outer_iterations, n_inner_iterations):
    """Validate TV-L2 denoising parameters."""
    if mu <= 0:
        raise ValueError("Argument 'tv_l2_mu' must be positive.")
    if beta <= 0:
        raise ValueError("Argument 'tv_l2_beta' must be positive.")
    if n_outer_iterations <= 0:
        raise ValueError("Argument 'tv_l2_n_outer_iterations' must be a positive integer.")
    if n_inner_iterations <= 0:
        raise ValueError("Argument 'tv_l2_n_inner_iterations' must be a positive integer.")


def _denoise_tv_l2_chunk(data, mu, beta, n_outer_iterations, n_inner_iterations):
    """Denoise a two-dimensional batch of time series with TV-L2 ADMM."""
    n_series, n_vols = data.shape
    if n_series == 0 or n_vols < 2:
        return data.copy()

    # Work in float64 internally for stable line-search denominators.
    b = np.asarray(data, dtype=np.float64)
    u = b.copy()
    w = _forward_difference(u)
    # Standard ADMM multiplier sign convention for the Du = w constraint.
    lagrange = np.zeros_like(w)
    threshold = 1.0 / beta

    for _ in range(n_outer_iterations):
        for _ in range(n_inner_iterations):
            du = _forward_difference(u)
            residual = du - w + (lagrange / beta)
            gradient = beta * _difference_transpose(residual, n_vols) + mu * (u - b)
            a_gradient = beta * _difference_transpose(
                _forward_difference(gradient), n_vols
            ) + mu * gradient

            numerator = np.sum(gradient * gradient, axis=-1)
            denominator = np.sum(gradient * a_gradient, axis=-1)
            step_size = np.zeros_like(numerator)
            np.divide(
                numerator,
                denominator,
                out=step_size,
                where=np.abs(denominator) > np.finfo(denominator.dtype).eps,
            )
            u = u - (step_size[:, np.newaxis] * gradient)

        du = _forward_difference(u)
        w = _soft_threshold(du + (lagrange / beta), threshold)
        lagrange = lagrange + beta * (du - w)

    return u.astype(data.dtype, copy=False)


def denoise_tv_l2(
    data,
    mu=2**-10,
    beta=2**-4,
    n_outer_iterations=11,
    n_inner_iterations=5,
    chunk_size=10000,
):
    """Denoise echo time series with the Michalek and Mikl TV-L2 method.

    Parameters
    ----------
    data : (S x E x T) or (S x T) array_like
        Data to denoise. Denoising is applied independently to every time series along
        the last axis.
    mu : :obj:`float`, optional
        Weight on the L2 data-fidelity term. Default is ``2**-10``, matching
        Michalek and Mikl (2025).
    beta : :obj:`float`, optional
        ADMM penalty parameter. Default is ``2**-4``, matching Michalek and Mikl (2025).
    n_outer_iterations : :obj:`int`, optional
        Number of outer ADMM iterations. Default is 11, corresponding to
        ``k = 0, 1, ..., 10`` in Michalek and Mikl (2025).
    n_inner_iterations : :obj:`int`, optional
        Number of inner gradient steps for the inexact ``u`` update. Default is 5.
    chunk_size : :obj:`int`, optional
        Number of flattened time series to process per chunk.

    Returns
    -------
    denoised : :obj:`numpy.ndarray`
        TV-L2 denoised data with the same shape and dtype as ``data``.
    """
    _validate_tv_l2_options(mu, beta, n_outer_iterations, n_inner_iterations)
    if chunk_size <= 0:
        raise ValueError("Argument 'tv_l2_chunk_size' must be a positive integer.")

    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError("Argument 'data' must have at least two dimensions.")
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    original_shape = data.shape
    data_2d = data.reshape(-1, original_shape[-1])
    denoised = np.empty_like(data_2d)

    for start in range(0, data_2d.shape[0], chunk_size):
        stop = min(start + chunk_size, data_2d.shape[0])
        denoised[start:stop] = _denoise_tv_l2_chunk(
            data_2d[start:stop],
            mu=mu,
            beta=beta,
            n_outer_iterations=n_outer_iterations,
            n_inner_iterations=n_inner_iterations,
        )

    return denoised.reshape(original_shape)

def _get_parser():
    """Parse command line inputs for t2smap.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    required_args = parser.add_argument_group("Required Arguments")
    required_args.add_argument(
        "-d",
        dest="data",
        nargs="+",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Multi-echo dataset for analysis. "
            "A set of echo-specific files in ascending order. "
            "The TEs of the data should match the TEs listed in the -e argument."
        ),
        required=True,
    )
    required_args.add_argument(
        "-e",
        dest="tes",
        nargs="+",
        metavar="TE",
        type=float,
        help=(
            "Ascending echo times in seconds (per BIDS convention). E.g., 0.015 0.039 0.063. "
            "Millisecond values (e.g., 15.0 39.0 63.0) are still accepted but deprecated."
        ),
        required=True,
    )

    output_args = parser.add_argument_group("Output Control")
    output_args.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        metavar="PATH",
        help="Output directory.",
        default=".",
    )
    output_args.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        help="Prefix for filenames generated.",
        default="",
    )
    output_args.add_argument(
        "--convention",
        dest="convention",
        choices=["orig", "bids"],
        help='Filenaming convention. "bids" will use the latest BIDS derivatives version.',
        default="bids",
    )
    output_args.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Generate intermediate and additional files.",
        default=False,
    )
    output_args.add_argument(
        "--overwrite",
        "-f",
        dest="overwrite",
        action="store_true",
        help="Force overwriting of files.",
        default=False,
    )

    masking_args = parser.add_argument_group("Temporal and Spatial Masking")
    masking_args.add_argument(
        "--mask",
        dest="mask",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Binary mask of voxels to include in TE Dependent ANAlysis. "
            "Must be in the same space as `data`. "
            "If an explicit mask is not provided, then Nilearn's compute_epi_mask "
            "function will be used to derive a mask from the first echo's data. "
            "Providing a mask is recommended."
        ),
        default=None,
    )
    masking_args.add_argument(
        "--masktype",
        dest="masktype",
        nargs="+",
        help=(
            "Method(s) by which to define the adaptive mask. "
            "The adaptive mask starts with the mask from '--mask', when provided. "
            "It identifies voxels that have good data in all vs a subset of echoes. "
            '"dropout" removes voxels with much lower voxels than other voxels within each echo. '
            '"decay" removes voxels where the raw signal does not decay across echoes. '
            "Users can specify one, both, or neither of the models."
        ),
        choices=["dropout", "decay", "none"],
        default=["dropout"],
    )
    masking_args.add_argument(
        "--dummy-scans",
        dest="dummy_scans",
        metavar="INT",
        type=int,
        help="Number of dummy scans to remove from the beginning of the data.",
        default=0,
    )
    masking_args.add_argument(
        "--exclude",
        dest="exclude",
        type=str,
        help=(
            "Volume indices to exclude from adaptive mask generation and T2* and S0 estimation, "
            "but which will be retained in the optimally combined data. "
            "Can be individual indices (e.g., '0,5,10'), ranges (e.g., '5:10'), "
            "or a mix of the two (e.g., '0,5:10,15'). "
            "Indices are 0-based. "
            "As in Python lists, ranges are start-inclusive and end-exclusive "
            "(for example, '0:5' includes the first [0] through fifth [4] timepoints). "
            "Default is to not exclude any volumes."
        ),
        default=None,
    )

    decay_args = parser.add_argument_group("Decay Model Fitting and Optimal Combination")
    decay_args.add_argument(
        "--fittype",
        dest="fittype",
        choices=["loglin", "curvefit"],
        help=(
            "Desired T2*/S0 fitting method. "
            '"loglin" means that a linear model is fit to the log of the data. '
            '"curvefit" means that a more computationally demanding monoexponential model is fit '
            "to the raw data. "
        ),
        default="loglin",
    )
    decay_args.add_argument(
        "--fitmode",
        dest="fitmode",
        choices=["all", "ts"],
        help=(
            "Monoexponential model fitting scheme. "
            '"all" means that the model is fit, per voxel, across all timepoints. '
            '"ts" means that the model is fit, per voxel and per timepoint.'
        ),
        default="all",
    )
    decay_args.add_argument(
        "--combmode",
        dest="combmode",
        choices=["t2s", "paid"],
        help='Combination scheme for TEs: "t2s" (Posse 1999), "paid" (Poser)',
        default="t2s",
    )

    tv_args = parser.add_argument_group("TV-L2 Echo Denoising")
    tv_args.add_argument(
        "--tv-denoise",
        dest="tv_l2_denoise",
        action="store_true",
        help=(
            "Denoise each voxelwise echo time series with TV-L2 denoising before "
            "T2*/S0 fitting."
        ),
        default=False,
    )
    tv_args.add_argument(
        "--tv-mu",
        dest="tv_l2_mu",
        metavar="FLOAT",
        type=float,
        help="Weight on the TV-L2 data-fidelity term.",
        default=2**-10,
    )
    tv_args.add_argument(
        "--tv-beta",
        dest="tv_l2_beta",
        metavar="FLOAT",
        type=float,
        help="ADMM penalty parameter for TV-L2 denoising.",
        default=2**-4,
    )
    tv_args.add_argument(
        "--tv-n-outer-iterations",
        dest="tv_l2_n_outer_iterations",
        metavar="INT",
        type=int,
        help="Number of outer ADMM iterations for TV-L2 denoising.",
        default=11,
    )
    tv_args.add_argument(
        "--tv-n-inner-iterations",
        dest="tv_l2_n_inner_iterations",
        metavar="INT",
        type=int,
        help="Number of inner gradient steps for each TV-L2 ADMM iteration.",
        default=5,
    )
    tv_args.add_argument(
        "--tv-chunk-size",
        dest="tv_l2_chunk_size",
        metavar="INT",
        type=int,
        help="Number of voxel/echo time series to denoise per processing chunk.",
        default=10000,
    )
    tv_args.add_argument(
        "--tv-save-denoised-echos",
        dest="tv_l2_save_denoised_echos",
        action="store_true",
        help="Save TV-L2-denoised echo time series for each echo.",
        default=False,
    )
    decomposition_args = parser.add_argument_group("Component Selection")
    decomposition_args.add_argument(
        "--n-independent-echos",
        dest="n_independent_echos",
        metavar="INT",
        type=int,
        help=(
            "Number of independent echoes to use in goodness of fit metrics (fstat). "
            "Primarily used for EPTI acquisitions, which have dependency across echoes. "
            "If not provided, number of echoes will be used."
        ),
        default=None,
    )

    performance_args = parser.add_argument_group("Performance Control")
    performance_args.add_argument(
        "--n-threads",
        dest="n_threads",
        metavar="INT",
        type=int,
        help=(
            "Number of threads to use. "
            "Used by threadpoolctl to set the parameter outside of the workflow function. "
            "Higher numbers of threads tend to slow down performance on typical datasets."
        ),
        default=1,
    )
    performance_args.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=(
            "Logs in the terminal will have increased verbosity, "
            "and will also be written into a TSV file in the output directory."
        ),
        default=False,
    )

    # Hidden arguments
    parser.add_argument(
        "--quiet",
        dest="quiet",
        help=argparse.SUPPRESS,
        action="store_true",
        default=False,
    )

    # Version argument
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"t2smap v{__version__}",
    )
    return parser


def t2smap_workflow(
    data,
    tes,
    n_independent_echos=None,
    out_dir=".",
    mask=None,
    prefix="",
    convention="bids",
    dummy_scans=0,
    exclude=None,
    masktype=["dropout"],
    fittype="loglin",
    fitmode="all",
    combmode="t2s",
    tv_l2_denoise=False,
    tv_l2_mu=2**-10,
    tv_l2_beta=2**-4,
    tv_l2_n_outer_iterations=11,
    tv_l2_n_inner_iterations=5,
    tv_l2_chunk_size=10000,
    tv_l2_save_denoised_echos=False,
    debug=False,
    verbose=False,
    quiet=False,
    overwrite=False,
    n_threads=1,
    t2smap_command=None,
):
    """
    Estimate T2 and S0, and optimally combine data across TEs.

    Please remember to cite :footcite:t:`dupre2021te`.

    Parameters
    ----------
    data : :obj:`str` or :obj:`list` of :obj:`str`
        Either a single z-concatenated file (single-entry list or str) or a
        list of echo-specific files, in ascending order.
    tes : :obj:`list`
        List of echo times associated with data. Values should be in seconds
        per BIDS convention. Millisecond values are still accepted but deprecated.
    n_independent_echos : :obj:`int`, optional
        Number of independent echoes to use in goodness of fit metrics (fstat).
        Primarily used for EPTI acquisitions.
        If None, number of echoes will be used. Default is None.
    out_dir : :obj:`str`, optional
        Output directory.
    mask : :obj:`str`, optional
        Binary mask of voxels to include in TE Dependent ANAlysis. Must be spatially
        aligned with `data`.
    dummy_scans : :obj:`int`, optional
        Number of dummy scans to remove from the beginning of the data. Default is 0.
        dummy_scans are excluded from the optimally combined data.
    exclude : :obj:`str`, optional
        Volume indices to exclude from adaptive mask generation and T2* and S0 estimation,
        but which will be retained in the optimally combined data.
        Can be individual indices (e.g., '0,5,10'), ranges (e.g., '5:10'),
        or a mix of the two (e.g., '0,5:10,15').
        Indices are 0-based.
        Default is to not exclude any volumes.
    masktype : :obj:`list` with 'dropout' and/or 'decay' or None, optional
        Method(s) by which to define the adaptive mask. Default is ["dropout"].
    fittype : {'loglin', 'curvefit'}, optional
        Monoexponential fitting method.
        'loglin' means to use the the default linear fit to the log of
        the data.
        'curvefit' means to use a monoexponential fit to the raw data,
        which is slightly slower but may be more accurate.
    fitmode : {'all', 'ts'}, optional
        Monoexponential model fitting scheme.
        'all' means that the model is fit, per voxel, across all timepoints.
        'ts' means that the model is fit, per voxel and per timepoint.
        Default is 'all'.
    combmode : {'t2s', 'paid'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default), 'paid' (Poser).
    tv_l2_denoise : :obj:`bool`, optional
        Denoise each voxelwise echo time series with TV-L2 denoising before T2*/S0
        fitting. Default is False.
    tv_l2_mu : :obj:`float`, optional
        Weight on the TV-L2 data-fidelity term. Default is ``2**-10``.
    tv_l2_beta : :obj:`float`, optional
        ADMM penalty parameter for TV-L2 denoising. Default is ``2**-4``.
    tv_l2_n_outer_iterations : :obj:`int`, optional
        Number of TV-L2 outer ADMM iterations. Default is 11.
    tv_l2_n_inner_iterations : :obj:`int`, optional
        Number of inner gradient steps per TV-L2 ADMM iteration. Default is 5.
    tv_l2_chunk_size : :obj:`int`, optional
        Number of voxel/echo time series to denoise per processing chunk.
        Default is 10000.
    tv_l2_save_denoised_echos : :obj:`bool`, optional
        Save TV-L2-denoised echo time series for each echo. Default is False.
    verbose : :obj:`bool`, optional
        Generate intermediate and additional files. Default is False.
    overwrite : :obj:`bool`, optional
        If True, force overwriting of files. Default is False.
    n_threads : :obj:`int`, optional
        Number of threads to use. Used by threadpoolctl to set the parameter
        outside of the workflow function, as well as the number of threads to use
        for the decay model fitting. Default is 1.
    t2smap_command : :obj:`str`, optional
        The command used to run t2smap. Default is None.

    Other Parameters
    ----------------
    debug : :obj:`bool`, optional
        Whether to run in debugging mode or not. Default is False.
    quiet : :obj:`bool`, optional
        If True, suppress logging/printing of messages. Default is False.

    Notes
    -----
    This workflow writes out several files, which are described below:

    ============================= =================================================
    Filename                      Content
    ============================= =================================================
    T2starmap.nii.gz              Estimated T2* 3D map or 4D timeseries.
                                  Will be a 3D map if ``fitmode`` is 'all' and a
                                  4D timeseries if it is 'ts'.
    S0map.nii.gz                  S0 3D map or 4D timeseries.
    desc-limited_T2starmap.nii.gz Limited T2* map/timeseries. The difference between
                                  the limited and full maps is that, for voxels
                                  affected by dropout where only one echo contains
                                  good data, the full map uses the T2* estimate
                                  from the first two echos, while the limited map
                                  will have a NaN.
    desc-limited_S0map.nii.gz     Limited S0 map/timeseries. The difference between
                                  the limited and full maps is that, for voxels
                                  affected by dropout where only one echo contains
                                  good data, the full map uses the S0 estimate
                                  from the first two echos, while the limited map
                                  will have a NaN.
    desc-optcom_bold.nii.gz       Optimally combined timeseries.
    ============================= =================================================

    References
    ----------
    .. footbibliography::
    """
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    # Parse exclude parameter
    exclude_idx = parse_volume_indices(exclude)
    n_exclude = len(exclude_idx)

    utils.setup_loggers(quiet=quiet, debug=debug)

    LGR.info(f"Using output directory: {out_dir}")

    # Save command into sh file, if the command-line interface was used
    if t2smap_command is not None:
        command_file = open(os.path.join(out_dir, "t2smap_call.sh"), "w")
        command_file.write(t2smap_command)
        command_file.close()
    else:
        # Get variables passed to function if the tedana command is None
        variables = ", ".join(f"{name}={value}" for name, value in locals().items())
        # From variables, remove everything after ", tedana_command"
        variables = variables.split(", t2smap_command")[0]
        t2smap_command = f"t2smap_workflow({variables})"

    # Save system info to json
    info_dict = utils.get_system_version_info()
    info_dict["Command"] = t2smap_command

    if fitmode == "ts" and n_exclude > 0:
        raise ValueError(
            "Excluding volumes is not supported for fitmode='ts'. "
            "Please set fitmode='all' or remove the exclude argument."
        )

    if tv_l2_denoise:
        _validate_tv_l2_options(
            tv_l2_mu,
            tv_l2_beta,
            tv_l2_n_outer_iterations,
            tv_l2_n_inner_iterations,
        )
        if tv_l2_chunk_size <= 0:
            raise ValueError("Argument 'tv_l2_chunk_size' must be a positive integer.")
        if fitmode == "all":
            LGR.warning(
                "TV-L2 denoising was proposed for dynamic T2* mapping. "
                "Applying it before fitmode='all' static T2*/S0 estimation."
            )

    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    tes = utils.check_te_values(tes)
    n_echos = len(tes)

    # coerce data to samples x echos x time array
    if isinstance(data, str):
        data = [data]

    # Initialize OutputGenerator with reference image
    # XXX: This doesn't support AFNI data yet.
    ref_img = io.load_ref_img(data=data, n_echos=n_echos)
    io_generator = io.OutputGenerator(
        ref_img,
        convention=convention,
        out_dir=out_dir,
        prefix=prefix,
        config="auto",
        overwrite=overwrite,
        verbose=verbose,
    )

    mask_img, _ = utils.load_mask(ref_img, mask=mask, t2smap=None)
    io_generator.register_mask(mask_img)

    LGR.info(f"Loading input data: {[f for f in data]}")
    data_cat = io.load_data_nilearn(data, mask_img=mask_img, n_echos=n_echos)

    if dummy_scans > 0:
        LGR.warning(f"Removing the first {dummy_scans} volumes as dummy scans.")
        data_cat = data_cat[..., dummy_scans:]

    n_samp, n_echos, n_vols = data_cat.shape
    LGR.debug(f"Resulting data shape: {data_cat.shape}")

    # Create mask for volumes to use based on exclude indices
    use_volumes = None
    if n_exclude > 0:
        LGR.info(f"Excluding volumes: {exclude_idx}")
        # Adjust exclude indices for dummy scans that are already removed
        exclude_idx = np.setdiff1d(exclude_idx, np.arange(dummy_scans))
        # Offset exclude indices by the number of dummy scans so they index into loaded data_cat
        exclude_idx = exclude_idx - dummy_scans
        n_exclude = len(exclude_idx)
        if n_exclude == 0:
            LGR.warning(f"All exclude indices overlap with dummy scans ({dummy_scans}).")

    if n_exclude > 0 and np.max(exclude_idx) > n_vols:
        raise ValueError(
            f"The maximum exclude index ({np.max(exclude_idx)}) is greater than the number of "
            f"timepoints in the data ({n_vols})."
        )
    elif n_exclude > 0:
        LGR.info(f"Excluding {n_exclude} volumes from adaptive mask and T2*/S0 estimation")
        use_volumes = np.ones(n_vols, dtype=bool)
        use_volumes[exclude_idx] = False
        data_without_excluded_vols = data_cat[:, :, use_volumes]
    else:
        data_without_excluded_vols = data_cat.copy()

    # Create an adaptive mask with at least 1 good echo, for denoising
    mask_denoise, masksum_denoise = utils.make_adaptive_mask(
        data_without_excluded_vols,
        n_independent_echos=n_independent_echos,
        threshold=1,
        methods=masktype,
    )
    LGR.debug(f"Retaining {mask_denoise.sum()}/{n_samp} samples for denoising")
    io_generator.save_file(masksum_denoise, "adaptive mask img")

    if tv_l2_denoise:
        n_series = int(mask_denoise.sum() * n_echos)
        LGR.info(
            "Denoising %d voxel/echo time series with TV-L2 denoising "
            "(mu=%s, beta=%s, outer iterations=%d, inner iterations=%d)",
            n_series,
            tv_l2_mu,
            tv_l2_beta,
            tv_l2_n_outer_iterations,
            tv_l2_n_inner_iterations,
        )
        data_denoised_masked = denoise_tv_l2(
            data_cat[mask_denoise, ...],
            mu=tv_l2_mu,
            beta=tv_l2_beta,
            n_outer_iterations=tv_l2_n_outer_iterations,
            n_inner_iterations=tv_l2_n_inner_iterations,
            chunk_size=tv_l2_chunk_size,
        )

        if tv_l2_save_denoised_echos:
            tv_echo_files = []
            for i_echo in range(n_echos):
                fout = io_generator.save_file(
                    data_denoised_masked[:, i_echo, :],
                    "tv denoised ts split img",
                    echo=i_echo + 1,
                    mask=mask_denoise,
                )
                tv_echo_files.append(op.basename(fout))
                LGR.info(
                    f"Writing TV-L2-denoised echo #{i_echo + 1:01d} timeseries: {fout}"
                )
            io_generator.registry["tv denoised ts split img"] = tv_echo_files

        if n_exclude > 0:
            data_without_excluded_vols = data_denoised_masked[:, :, use_volumes]
        else:
            data_without_excluded_vols = data_denoised_masked

    else:
        data_without_excluded_vols = data_without_excluded_vols[mask_denoise, ...]

    LGR.info("Computing T2* map")
    masksum_masked = masksum_denoise[mask_denoise]
    decay_function = decay.fit_decay if fitmode == "all" else decay.fit_decay_ts
    t2s_full, s0_full, failures, t2s_var, s0_var, t2s_s0_covar = decay_function(
        data=data_without_excluded_vols,
        tes=tes,
        adaptive_mask=masksum_masked,
        fittype=fittype,
        n_threads=n_threads,
    )
    del data_without_excluded_vols

    if fittype == "curvefit":
        io_generator.save_file(
            failures.astype(np.uint8),
            "fit failures img",
            mask=mask_denoise,
        )
        if verbose:
            io_generator.save_file(t2s_var, "t2star variance img", mask=mask_denoise)
            io_generator.save_file(s0_var, "s0 variance img", mask=mask_denoise)
            io_generator.save_file(
                t2s_s0_covar,
                "t2star-s0 covariance img",
                mask=mask_denoise,
            )

    # Delete unused variables
    del failures, t2s_var, s0_var, t2s_s0_covar

    t2s_full, s0_full, t2s_limited, s0_limited = decay.modify_t2s_s0_maps(
        t2s=t2s_full,
        s0=s0_full,
        adaptive_mask=masksum_masked,
        tes=tes,
    )
    del masksum_masked

    t2s_full = utils.unmask(t2s_full, mask_denoise)
    s0_full = utils.unmask(s0_full, mask_denoise)
    t2s_limited = utils.unmask(t2s_limited, mask_denoise)
    s0_limited = utils.unmask(s0_limited, mask_denoise)

    io_generator.save_file(s0_full, "s0 img")
    del s0_full

    LGR.info("Calculating model fit quality metrics")
    rmse_map, rmse_df = decay.rmse_of_fit_decay_ts(
        data=data_cat,
        tes=tes,
        adaptive_mask=masksum_denoise,
        t2s=t2s_limited,
        s0=s0_limited,
        fitmode=fitmode,
    )
    io_generator.save_file(rmse_map, "rmse img")
    io_generator.save_file(rmse_df, "confounds tsv")
    io_generator.save_file(s0_limited, "limited s0 img")
    del s0_limited
    io_generator.save_file(t2s_limited, "limited t2star img")
    del t2s_limited

    LGR.info("Computing optimal combination")
    # optimally combine data, including the ignored volumes
    data_optcom = combine.make_optcom(
        data_cat,
        tes,
        masksum_denoise,
        t2s=t2s_full,
        combmode=combmode,
    )

    io_generator.save_file(t2s_full, "t2star img")
    io_generator.save_file(data_optcom, "combined img")

    # Write out BIDS-compatible description file
    derivative_metadata = {
        "Name": "t2smap Outputs",
        "BIDSVersion": "1.5.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "t2smap",
                "Version": __version__,
                "Description": (
                    "A pipeline estimating T2* from multi-echo fMRI data and "
                    "combining data across echoes."
                ),
                "CodeURL": "https://github.com/ME-ICA/tedana",
                "Node": {
                    "Name": info_dict["Node"],
                    "System": info_dict["System"],
                    "Machine": info_dict["Machine"],
                    "Processor": info_dict["Processor"],
                    "Release": info_dict["Release"],
                    "Version": info_dict["Version"],
                },
                "Python": info_dict["Python"],
                "Python_Libraries": info_dict["Python_Libraries"],
                "Command": info_dict["Command"],
            }
        ],
    }

    if tv_l2_denoise:
        derivative_metadata["GeneratedBy"][0]["TVL2Denoising"] = {
            "Description": (
                "One-dimensional temporal TV-L2 echo denoising before T2*/S0 fitting."
            ),
            "Reference": "Michalek and Mikl 2025, doi:10.3389/fnins.2025.1544748",
            "DifferenceOperator": "Non-circular forward differences along time",
            "Mu": tv_l2_mu,
            "Beta": tv_l2_beta,
            "OuterIterations": tv_l2_n_outer_iterations,
            "InnerIterations": tv_l2_n_inner_iterations,
            "ChunkSize": tv_l2_chunk_size,
        }

    io_generator.save_file(derivative_metadata, "data description json")
    io_generator.save_self()

    LGR.info("Workflow completed")

    # Add newsletter info to the log
    utils.log_newsletter_info()

    utils.teardown_loggers()


def _main(argv=None):
    """Run the t2smap workflow."""
    if argv:
        # relevant for tests when CLI called with t2smap_cli._main(args)
        t2smap_command = "t2smap " + " ".join(argv)
    else:
        t2smap_command = "t2smap " + " ".join(sys.argv[1:])
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    n_threads = kwargs.get("n_threads", 1)
    n_threads = None if n_threads == -1 else n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        t2smap_workflow(**kwargs, t2smap_command=t2smap_command)


if __name__ == "__main__":
    _main()
