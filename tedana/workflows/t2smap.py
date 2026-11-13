"""Estimate T2 and S0, and optimally combine data across TEs."""

import argparse
import datetime
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
            "A set of echo-specific 3D or 4D files in ascending order, "
            "or one 3D or 4D z-concatenated file. "
            "The TEs of the data should match the TEs listed in the -e argument."
        ),
        required=True,
    )
    required_args.add_argument(
        "--phase",
        dest="phase",
        nargs="+",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Phase images for each echo, in radians, in ascending echo order "
            "matching -d. Required when --fittype is 'nlls' or --combmode is "
            "'t2s-complex' or 'paid-complex'."
        ),
        default=None,
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
        choices=["loglin", "curvefit", "nlls"],
        help=(
            "Desired T2*/S0 fitting method. "
            '"loglin" means that a linear model is fit to the log of the data. '
            '"curvefit" means that a more computationally demanding monoexponential model is fit '
            "to the raw data. "
            '"nlls" fits a complex monoexponential model to magnitude+phase data and requires '
            "--phase. "
        ),
        default="loglin",
    )
    decay_args.add_argument(
        "--fitmode",
        dest="fitmode",
        choices=["all", "ts", "varys0"],
        help=(
            "Monoexponential model fitting scheme. "
            '"all" means that the model is fit, per voxel, across all timepoints. '
            '"ts" means that the model is fit, per voxel and per timepoint. '
            '"varys0" (nlls only) shares R2*/frequency across timepoints while '
            "allowing complex S0 to vary per timepoint. "
        ),
        default="all",
    )
    decay_args.add_argument(
        "--combmode",
        dest="combmode",
        choices=["t2s", "paid", "t2s-complex", "paid-complex"],
        help=(
            'Combination scheme for TEs: "t2s" (Posse 1999), "paid" (Poser). '
            '"t2s-complex" and "paid-complex" combine complex-valued data built from the '
            "magnitude (-d) and phase (--phase) images, after rotating each voxel's echoes "
            "so that their temporal complex means are real (Mikl 2024), using the same "
            "T2*-based or PAID weights. These modes require --phase and write the real, "
            "imaginary, magnitude and phase parts of the combined data instead of the "
            "magnitude-only optimal combination."
        ),
        default="t2s",
    )
    decay_args.add_argument(
        "--interpolate-failing-voxels",
        dest="interpolate_failing_voxels",
        action="store_true",
        help=(
            "If fittype='curvefit', replace T2*/S0 values for voxels where the "
            "monoexponential fit failed with nearest-neighbor interpolated values "
            "from non-failing neighbors. Ignored if fittype='loglin'."
        ),
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
        "--backend",
        choices=["cpu", "cuda", "auto"],
        help=(
            "Compute backend. CUDA accelerates curvefit and nlls T2*/S0 estimation. "
            "'auto' uses CUDA when available and otherwise uses CPU."
        ),
        default="cpu",
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
    phase=None,
    fittype="loglin",
    interpolate_failing_voxels=False,
    fitmode="all",
    combmode="t2s",
    debug=False,
    verbose=False,
    quiet=False,
    overwrite=False,
    n_threads=1,
    backend="cpu",
    t2smap_command=None,
):
    """
    Estimate T2 and S0, and optimally combine data across TEs.

    Please remember to cite :footcite:t:`dupre2021te`.

    Parameters
    ----------
    data : :obj:`str` or :obj:`list` of :obj:`str`
        Either a single z-concatenated file (single-entry list or str) or a
        list of echo-specific files, in ascending order. Files may be 3D
        single-volume images or 4D time series.
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
    phase : :obj:`list` of :obj:`str`, optional
        Phase images for each echo, in radians, in ascending echo order matching ``data``.
        Required when ``fittype`` is ``'nlls'`` or ``combmode`` is ``'t2s-complex'``
        or ``'paid-complex'``. Default is None.
    fittype : {'loglin', 'curvefit', 'nlls'}, optional
        Monoexponential fitting method.
        'loglin' means to use the the default linear fit to the log of
        the data.
        'curvefit' means to use a monoexponential fit to the raw data,
        which is slightly slower but may be more accurate.
        'nlls' fits a complex monoexponential model to magnitude+phase data
        and requires ``phase``.
    interpolate_failing_voxels : :obj:`bool`, optional
        If ``True`` and ``fittype='curvefit'``, replace T2*/S0 values for
        voxels where the monoexponential fit failed with nearest-neighbor
        interpolated values from non-failing neighbors.
        Ignored if ``fittype='loglin'``. Default is ``False``.
    fitmode : {'all', 'ts', 'varys0'}, optional
        Monoexponential model fitting scheme.
        'all' means that the model is fit, per voxel, across all timepoints.
        'ts' means that the model is fit, per voxel and per timepoint.
        'varys0' (nlls only) shares R2*/frequency across timepoints while
        allowing complex S0 to vary per timepoint.
        Default is 'all'.
    combmode : {'t2s', 'paid', 't2s-complex', 'paid-complex'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default), 'paid' (Poser).
        't2s-complex' and 'paid-complex' phase-align the complex-valued data built from
        ``data`` and ``phase`` and combine them with the same weights
        (see :func:`tedana.combine.make_optcom_complex`).
        They require ``phase`` and voxel-wise T2* (``fitmode`` 'all' or 'varys0').
        With ``fittype='nlls'``, 't2s-complex' uses the T2* from the complex fit;
        otherwise it uses the T2* fit to the magnitude data.
        The combined phase is relative to each echo's temporal-mean phase and has no
        single echo time, so it should not replace echo-wise phase for field mapping
        or susceptibility mapping.
    verbose : :obj:`bool`, optional
        Generate intermediate and additional files. Default is False.
    overwrite : :obj:`bool`, optional
        If True, force overwriting of files. Default is False.
    n_threads : :obj:`int`, optional
        Number of threads to use. Used by threadpoolctl to set the parameter
        outside of the workflow function, as well as the number of threads to use
        for the decay model fitting. Default is 1.
    backend : {'cpu', 'cuda', 'auto'}, optional
        Compute backend. CUDA accelerates curvefit and nlls T2*/S0 estimation.
        Default is 'cpu'.
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
    S0map.nii.gz                  S0 3D map or 4D timeseries. With ``fittype``
                                  'nlls', S0 is complex and is written instead
                                  as part-mag_S0map.nii.gz and
                                  part-phase_S0map.nii.gz.
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
    desc-optcom_bold.nii.gz       Optimally combined timeseries. Not written for
                                  complex ``combmode`` options.
    part-<real|imag|mag|phase>_   Real, imaginary, magnitude and phase (radians)
    desc-optcom_bold.nii.gz       parts of the complex optimally combined
                                  timeseries. Only written for complex
                                  ``combmode`` options.
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

    # create logfile name
    basename = "t2smap_"
    extension = "tsv"
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = op.join(out_dir, (basename + start_time + "." + extension))
    utils.setup_loggers(logname, repname=None, quiet=quiet, debug=debug)

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

    complex_combination = combmode in combine.COMPLEX_COMBMODES
    if fittype == "nlls" and phase is None:
        raise ValueError("fittype='nlls' requires phase images (--phase).")
    if complex_combination and phase is None:
        raise ValueError(f"combmode='{combmode}' requires phase images (--phase).")
    if phase is not None and fittype != "nlls" and not complex_combination:
        raise ValueError(
            "phase images are only used with fittype='nlls' or "
            "combmode='t2s-complex'/'paid-complex'."
        )
    if complex_combination and fitmode == "ts":
        raise ValueError(
            f"combmode='{combmode}' requires voxel-wise T2* estimates and does not support "
            "fitmode='ts'. Please use fitmode='all' or 'varys0'."
        )
    if fitmode == "varys0" and fittype != "nlls":
        raise ValueError("fitmode='varys0' is only valid with fittype='nlls'.")

    if isinstance(phase, str):
        phase = [phase]

    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    tes = utils.check_te_values(tes)
    n_echos = len(tes)

    # coerce data to samples x echos x time array
    if isinstance(data, str):
        data = [data]

    if phase is not None and len(phase) != len(data):
        raise ValueError("--phase must have the same number of files as -d.")

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

    phase_cat = None
    if phase is not None:
        LGR.info(f"Loading phase data: {[f for f in phase]}")
        phase_cat = io.load_data_nilearn(phase, mask_img=mask_img, n_echos=n_echos)
        if dummy_scans > 0:
            phase_cat = phase_cat[..., dummy_scans:]
        if phase_cat.shape != data_cat.shape:
            raise ValueError(
                f"Phase shape {phase_cat.shape} does not match data shape {data_cat.shape}."
            )
        _check_phase_radians(phase_cat)

    n_samp, n_echos, n_vols = data_cat.shape
    LGR.debug(f"Resulting data shape: {data_cat.shape}")
    if combmode in ("paid", "paid-complex") and n_vols == 1:
        raise ValueError("PAID combination requires more than one volume.")

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

    if interpolate_failing_voxels and fittype not in ("curvefit", "nlls"):
        LGR.warning(
            "interpolate_failing_voxels is set but fittype is not 'curvefit' or 'nlls'; "
            "interpolation will be skipped."
        )

    LGR.info("Computing T2* map")
    t2s_full = decay.t2smap_subworkflow(
        data_cat=data_cat,
        tes=tes,
        fittype=fittype,
        fitmode=fitmode,
        mask_denoise=mask_denoise,
        masksum_denoise=masksum_denoise,
        interpolate_failing_voxels=interpolate_failing_voxels,
        io_generator=io_generator,
        data_without_excluded_vols=data_without_excluded_vols,
        n_threads=n_threads,
        backend=backend,
        phase_cat=phase_cat,
        use_volumes=use_volumes,
    )

    LGR.info("Computing optimal combination")
    # optimally combine data, including the ignored volumes
    if complex_combination:
        _combine_complex(
            data_cat=data_cat,
            phase_cat=phase_cat,
            tes=tes,
            masksum_denoise=masksum_denoise,
            t2s_full=t2s_full,
            combmode=combmode,
            fittype=fittype,
            use_volumes=use_volumes,
            io_generator=io_generator,
        )
    else:
        data_optcom = combine.make_optcom(
            data_cat,
            tes,
            masksum_denoise,
            t2s=t2s_full,
            combmode=combmode,
        )
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

    io_generator.save_file(derivative_metadata, "data description json")
    io_generator.save_self()

    LGR.info("Workflow completed")

    # Add newsletter info to the log
    utils.log_newsletter_info()

    utils.teardown_loggers()


def _check_phase_radians(phase_cat, tolerance=0.01):
    r"""Check that phase data appear to be in radians.

    Wrapped phase in radians spans at most :math:`2\pi`.
    Phase spanning more than :math:`2\pi` is rejected if every value is an integer,
    which indicates unscaled scanner phase (for example, integers from -4096 to 4095).
    Otherwise, it is accepted as unwrapped or shifted radians, with a warning, since only
    :math:`\exp(i\phi)` is used.

    Parameters
    ----------
    phase_cat : (S x E x T) :obj:`numpy.ndarray`
        Phase data.
    tolerance : float, optional
        Allowed excess beyond a :math:`2\pi` range. Default is 0.01.

    Raises
    ------
    ValueError
        If the phase data are not finite or appear to be unscaled scanner phase.
    """
    if not np.all(np.isfinite(phase_cat)):
        raise ValueError("Phase data contain non-finite values.")

    phase_min, phase_max = float(phase_cat.min()), float(phase_cat.max())
    if phase_max - phase_min <= 2 * np.pi + tolerance:
        return

    if np.array_equal(phase_cat, np.round(phase_cat)):
        raise ValueError(
            f"Phase data are integers ranging from {phase_min:.4g} to {phase_max:.4g}, "
            "which looks like unscaled scanner phase rather than radians. Please convert "
            "the phase images to radians (e.g., [-pi, pi)) before running t2smap."
        )

    LGR.warning(
        f"Phase data range from {phase_min:.4g} to {phase_max:.4g}, which spans more than "
        "2*pi. They are assumed to be unwrapped phase in radians. If they are in other "
        "units (e.g., degrees), please convert them to radians."
    )


def _combine_complex(
    data_cat,
    phase_cat,
    tes,
    masksum_denoise,
    t2s_full,
    combmode,
    fittype,
    use_volumes,
    io_generator,
):
    """Optimally combine complex-valued data and write its outputs.

    Parameters
    ----------
    data_cat, phase_cat : (S x E x T) :obj:`numpy.ndarray`
        Magnitude and phase (radians) data.
    tes : (E,) :obj:`list`
        Echo times in seconds.
    masksum_denoise : (S,) :obj:`numpy.ndarray`
        Adaptive mask.
    t2s_full : (S,) :obj:`numpy.ndarray`
        Voxel-wise T2* estimates.
    combmode : {"t2s-complex", "paid-complex"}
        Complex combination method.
    fittype : str
        T2* fitting method, used to describe the T2* source.
    use_volumes : (T,) :obj:`numpy.ndarray` of bool or None
        Volumes used to estimate the phase reference and PAID weights.
    io_generator : :obj:`tedana.io.OutputGenerator`
        Output generator.
    """
    weighting = combine.COMPLEX_COMBMODES[combmode]
    if weighting == "t2s":
        if fittype == "nlls":
            t2s_source = "complex monoexponential (nlls) fit to the magnitude and phase data"
        else:
            t2s_source = f"monoexponential ({fittype}) fit to the magnitude data"

        LGR.info(f"Using T2* from the {t2s_source} for complex optimal combination")
    else:
        t2s_source = None

    combined, diagnostics = combine.make_optcom_complex(
        data_cat,
        phase_cat,
        tes,
        masksum_denoise,
        t2s=t2s_full if weighting == "t2s" else None,
        combmode=combmode,
        use_volumes=use_volumes,
    )

    n_vols = data_cat.shape[-1]
    if use_volumes is None:
        excluded = []
    else:
        excluded = np.where(~use_volumes)[0].tolist()
        LGR.info(
            f"Estimated phase references{' and PAID weights' if weighting == 'paid' else ''} "
            f"from {int(use_volumes.sum())} of {n_vols} volumes; "
            "excluded volumes are retained in the combined data."
        )

    common_metadata = {
        "CombinationMethod": combmode,
        "EchoWeighting": (
            "T2*-based (Posse et al., 1999)"
            if weighting == "t2s"
            else "PAID, from magnitude tSNR (Poser et al., 2006)"
        ),
        "PhaseAlignment": (
            "Each voxel's echo time series was rotated by the negative phase of its "
            "temporal complex mean, so the temporal mean of each echo is real and "
            "nonnegative. Undetermined (near-zero) temporal means were not rotated."
        ),
        "ExcludedVolumesFromEstimation": excluded,
    }
    if t2s_source is not None:
        common_metadata["T2starSource"] = t2s_source

    phase_note = (
        "The phase is relative to each echo's temporal-mean phase, not the scanner's "
        "absolute phase, and does not correspond to a single echo time. It should not "
        "be used in place of echo-wise phase for field mapping or susceptibility mapping."
    )
    parts = {
        "real": (combined.real, "Real part", {}),
        "imag": (combined.imag, "Imaginary part", {}),
        "mag": (np.abs(combined), "Magnitude", {}),
        "phase": (np.angle(combined), "Phase", {"Units": "rad"}),
    }
    for part, (part_data, part_label, extra_metadata) in parts.items():
        io_generator.save_file(part_data, f"complex combined {part} img")
        metadata = {
            "Description": (
                f"{part_label} of the phase-aligned complex-valued optimally combined "
                f"time series. {phase_note}"
            ),
            **extra_metadata,
            **common_metadata,
        }
        io_generator.save_file(metadata, f"complex combined {part} json")

    if io_generator.verbose:
        io_generator.save_file(diagnostics["weights"], "optcom weights img")
        io_generator.save_file(diagnostics["reference_phase"], "phase alignment img")
        io_generator.save_file(
            {
                "Units": "rad",
                "Description": (
                    "Echo-wise phase of the temporal complex mean removed from each echo "
                    "before complex optimal combination. Volumes correspond to echoes."
                ),
            },
            "phase alignment json",
        )
        io_generator.save_file(diagnostics["coherence"], "phase coherence img")
        io_generator.save_file(
            diagnostics["failed"].astype(np.uint8),
            "phase alignment failures img",
        )


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
