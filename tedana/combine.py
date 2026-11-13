"""Functions to optimally combine data across echoes."""

import logging

import numpy as np

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")

# Complex-valued combination methods and the real-valued weighting each one uses.
COMPLEX_COMBMODES = {"t2s-complex": "t2s", "paid-complex": "paid"}

# Phase references whose temporal-mean magnitude is at most this fraction of the
# mean magnitude are treated as undetermined and replaced with zero rotation.
PHASE_REFERENCE_RTOL = 1e-6


def _combine_t2s(data, tes, ft2s, report=True):
    """Combine data across echoes using weighted averaging according to estimates of T2*.

    The T2* estimates may be voxel- or voxel- and volume-wise.

    This method was proposed in :footcite:t:`posse1999enhancement`.

    Parameters
    ----------
    data : (Mb x E x T) array_like
        Masked data.
    tes : (1 x E) array_like
        Echo times in seconds.
    ft2s : (Mb [x T] X 1) array_like
        Either voxel-wise or voxel- and volume-wise estimates of T2*.
    report : bool, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    combined : (Mb x T) :obj:`numpy.ndarray`
        Data combined across echoes according to T2* estimates.

    References
    ----------
    .. footbibliography::
    """
    if report:
        RepLGR.info(
            "Multi-echo data were then optimally combined using the "
            "T2* combination method \\citep{posse1999enhancement}."
        )

    n_vols = data.shape[-1]
    alpha = tes * np.exp(-tes / ft2s)
    if alpha.ndim == 2:
        # Voxel-wise T2 estimates
        alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    elif alpha.ndim == 3:
        # Voxel- and volume-wise T2 estimates
        # alpha is currently (S, T, E) but should be (S, E, T) like mdata
        alpha = np.swapaxes(alpha, 1, 2)

        # If all values across echos are 0, set to 1 to avoid
        # divide-by-zero errors
        ax0_idx, ax2_idx = np.where(np.all(alpha == 0, axis=1))
        alpha[ax0_idx, :, ax2_idx] = 1.0

    combined = np.average(data, axis=1, weights=alpha)
    return combined


def _combine_paid(data, tes, report=True):
    """Combine data across echoes using the PAID combination method.

    This method uses SNR/signal and TE via the parallel-acquired inhomogeneity desensitized (PAID)
    ME-fMRI combination method :footcite:t:`poser2006bold`.

    Parameters
    ----------
    data : (Mb x E x T) array_like
        Masked data, where `Mb` is samples in base mask, `E` is echos, and `T` is time.
    tes : (1 x E) array_like
        Echo times in seconds.
    report : bool, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    combined : (Mb x T) :obj:`numpy.ndarray`
        Data combined across echoes according to SNR/signal.

    References
    ----------
    .. footbibliography::
    """
    if report:
        RepLGR.info(
            "Multi-echo data were then optimally combined using the "
            "parallel-acquired inhomogeneity desensitized (PAID) "
            "combination method \\citep{poser2006bold}."
        )

    n_vols = data.shape[-1]
    mean_signal = data.mean(axis=-1)
    signal_std = data.std(axis=-1)
    snr = np.zeros_like(mean_signal, dtype=float)
    np.divide(mean_signal, signal_std, out=snr, where=signal_std != 0)
    alpha = snr * tes
    alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    combined = np.average(data, axis=1, weights=alpha)
    return combined


def align_complex_echoes(data, use_volumes=None):
    r"""Rotate each voxel's echo time series so its temporal complex mean is real.

    For each sample :math:`v` and echo :math:`e`, the reference phasor is the
    unit-magnitude temporal complex mean over the volumes used for estimation,

    .. math::
        q_{v,e} = \bar z_{v,e} / |\bar z_{v,e}|, \quad
        \bar z_{v,e} = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} z_{v,e,t},

    and the aligned data are :math:`\tilde z_{v,e,t} = z_{v,e,t} q_{v,e}^*`.
    This is the constant rotation that maximizes the real part of the temporal
    mean, and it does not require phase unwrapping.
    The same rotation is applied to every volume, so temporal phase fluctuations,
    including those in excluded volumes, are preserved.

    Parameters
    ----------
    data : (S x E x T) array_like
        Complex-valued multi-echo data.
    use_volumes : (T,) array_like of bool or None, optional
        Volumes used to estimate the reference phase. The rotation is applied to
        all volumes. None means all volumes are used. Default is None.

    Returns
    -------
    aligned : (S x E x T) :obj:`numpy.ndarray`
        Phase-aligned complex data.
    reference_phase : (S x E) :obj:`numpy.ndarray`
        Phase of the reference phasor, :math:`\arg(q_{v,e})`, in radians.
        Zero where the reference is undetermined.
    coherence : (S x E) :obj:`numpy.ndarray`
        :math:`|\bar z_{v,e}|` divided by the temporal mean magnitude,
        between 0 (incoherent phase) and 1 (constant phase).
        A measure of confidence in the phase reference.
    failed : (S x E) :obj:`numpy.ndarray` of bool
        True where the temporal mean was (near) zero, so the reference phase is
        undetermined and no rotation was applied.
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError("Input data must be 3D (S x E x T)")

    if not np.iscomplexobj(data):
        raise ValueError("Input data must be complex-valued.")

    used = data if use_volumes is None else data[..., np.asarray(use_volumes, dtype=bool)]
    if used.shape[-1] == 0:
        raise ValueError("At least one volume must be used to estimate the phase reference.")

    mean_phasor = used.mean(axis=-1, dtype=np.complex128)
    mean_magnitude = np.abs(used).mean(axis=-1, dtype=np.float64)
    mean_phasor_magnitude = np.abs(mean_phasor)

    coherence = np.zeros_like(mean_magnitude)
    np.divide(mean_phasor_magnitude, mean_magnitude, out=coherence, where=mean_magnitude > 0)
    failed = (mean_magnitude == 0) | (coherence <= PHASE_REFERENCE_RTOL)

    reference_phase = np.where(failed, 0.0, np.angle(mean_phasor))
    rotation = np.exp(-1j * reference_phase).astype(data.dtype)
    aligned = data * rotation[..., np.newaxis]
    return aligned, reference_phase, coherence, failed


def _used_echoes(adaptive_mask, n_echos):
    """Identify the echoes combined in each sample, following :func:`make_optcom`.

    Samples with one good echo use the first two echoes, and samples without
    good signal use none.

    Parameters
    ----------
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Number of echoes with good signal per sample.
    n_echos : int
        Number of echoes.

    Returns
    -------
    used_echoes : (S x E) :obj:`numpy.ndarray` of bool
        True for echoes that are combined in each sample.
    """
    n_used = np.where(adaptive_mask > 0, np.maximum(adaptive_mask, 2), 0)
    return np.arange(n_echos)[np.newaxis, :] < n_used[:, np.newaxis]


def _complex_combination_weights(magnitude, tes, adaptive_mask, weighting, t2s, use_volumes):
    """Calculate real-valued, normalized echo weights for complex combination.

    Parameters
    ----------
    magnitude : (S x E x T) :obj:`numpy.ndarray`
        Magnitude data, used for the PAID (tSNR) weights.
    tes : (E,) :obj:`numpy.ndarray`
        Echo times in seconds.
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Number of echoes with good signal per sample.
    weighting : {"t2s", "paid"}
        Weighting scheme.
    t2s : (S,) :obj:`numpy.ndarray` or None
        Voxel-wise T2* estimates. Required if ``weighting`` is "t2s".
    use_volumes : (T,) :obj:`numpy.ndarray` of bool or None
        Volumes used to estimate the tSNR for PAID weights.

    Returns
    -------
    weights : (S x E) :obj:`numpy.ndarray`
        Weights that are nonnegative, zero for unused echoes, and sum to one across
        the echoes used in each sample. All weights are zero in samples with no
        usable echoes or whose unnormalized weights sum to zero.
    """
    n_samp, n_echos, _ = magnitude.shape
    tes = np.asarray(tes, dtype=float)

    used_echoes = _used_echoes(adaptive_mask, n_echos)

    if weighting == "t2s":
        alpha = tes[np.newaxis, :] * np.exp(-tes[np.newaxis, :] / t2s[:, np.newaxis])
    else:
        mag = magnitude if use_volumes is None else magnitude[..., use_volumes]
        mean_signal = mag.mean(axis=-1, dtype=np.float64)
        signal_std = mag.std(axis=-1, dtype=np.float64)
        tsnr = np.zeros_like(mean_signal)
        np.divide(mean_signal, signal_std, out=tsnr, where=signal_std > 0)
        alpha = tsnr * tes[np.newaxis, :]

    alpha = np.where(used_echoes & np.isfinite(alpha), alpha, 0.0)
    alpha = np.maximum(alpha, 0.0)
    alpha_sum = alpha.sum(axis=1, keepdims=True)
    weights = np.zeros_like(alpha)
    np.divide(alpha, alpha_sum, out=weights, where=alpha_sum > 0)

    n_zero = int(np.sum((alpha_sum[:, 0] <= 0) & used_echoes.any(axis=1)))
    if n_zero:
        LGR.warning(
            f"{n_zero} sample(s) with good signal have weights that sum to zero "
            "across echoes; their complex optimally combined values are set to zero."
        )

    return weights


def make_optcom_complex(
    magnitude,
    phase,
    tes,
    adaptive_mask,
    t2s=None,
    combmode="t2s-complex",
    use_volumes=None,
):
    r"""Optimally combine complex-valued multi-echo data after phase alignment.

    The complex signal :math:`z_{v,e,t} = M_{v,e,t} \exp(i \phi_{v,e,t})` is phase-aligned
    per voxel and echo with :func:`align_complex_echoes`, and the aligned echoes are
    averaged with real-valued, nonnegative weights :footcite:p:`mikl2024processing`.

    Parameters
    ----------
    magnitude : (S x E x T) :obj:`numpy.ndarray`
        Magnitude data.
    phase : (S x E x T) :obj:`numpy.ndarray`
        Phase data in radians.
    tes : (E,) array_like
        Echo times in seconds.
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Number of echoes with good signal per sample, determined from the magnitude data.
    t2s : (S,) :obj:`numpy.ndarray` or None, optional
        Voxel-wise T2* estimates. Required if ``combmode`` is "t2s-complex".
        Volume-wise T2* is not supported.
    combmode : {"t2s-complex", "paid-complex"}, optional
        Weighting scheme. "t2s-complex" uses the :math:`T_2^*`-based weights of
        :footcite:t:`posse1999enhancement`, and "paid-complex" uses the
        PAID/CNR weights of :footcite:t:`poser2006bold`, with tSNR from the magnitude data.
        Default is "t2s-complex".
    use_volumes : (T,) array_like of bool or None, optional
        Volumes used to estimate the phase reference and the tSNR for PAID weights.
        All volumes are combined. None means all volumes are used. Default is None.

    Returns
    -------
    combined : (S x T) :obj:`numpy.ndarray`
        Complex-valued optimally combined data. Zero in samples without good signal.
    diagnostics : dict
        ``(S x E)`` arrays ``weights``, ``reference_phase`` (radians), ``coherence``,
        and boolean ``failed`` (undetermined phase references in the echoes used).

    Notes
    -----
    The weights are

    .. math::
        w_{v,e} = \frac{a_{v,e}}{\sum_{j \in \mathcal{E}_v} a_{v,j}}, \quad
        a^{T_2^*}_{v,e} = TE_e \exp(-TE_e / T^*_{2,v}), \quad
        a^{\mathrm{PAID}}_{v,e} = TE_e \, tSNR_{v,e},

    where :math:`\mathcal{E}_v` are the echoes used for voxel :math:`v`, as in
    :func:`make_optcom`. The combined data are
    :math:`z_{v,t} = \sum_{e \in \mathcal{E}_v} w_{v,e} \tilde z_{v,e,t}`.

    The combined phase is relative to each echo's temporal-mean phase, not the scanner's
    absolute phase. It does not correspond to a single echo time and should not be
    used in place of echo-wise phase for field mapping or susceptibility mapping.

    References
    ----------
    .. footbibliography::
    """
    magnitude = np.asarray(magnitude)
    phase = np.asarray(phase)
    if magnitude.ndim != 3:
        raise ValueError("Input data must be 3D (S x E x T)")

    if phase.shape != magnitude.shape:
        raise ValueError(
            f"Phase shape {phase.shape} does not match magnitude shape {magnitude.shape}."
        )

    n_samp, n_echos, n_vols = magnitude.shape
    if len(tes) != n_echos:
        raise ValueError(
            "Number of echos provided does not match second "
            f"dimension of input data: {len(tes)} != {n_echos}"
        )

    if not np.all(np.isfinite(tes)):
        raise ValueError("Echo times must be finite.")

    if adaptive_mask.ndim != 1:
        raise ValueError("Mask is not 1D")
    elif adaptive_mask.shape[0] != n_samp:
        raise ValueError(
            "Mask and data do not have same number of "
            f"voxels/samples: {adaptive_mask.shape[0]} != {n_samp}"
        )

    if combmode not in COMPLEX_COMBMODES:
        raise ValueError(
            f"Argument 'combmode' must be one of {list(COMPLEX_COMBMODES)}, not '{combmode}'."
        )

    weighting = COMPLEX_COMBMODES[combmode]
    if weighting == "t2s":
        if t2s is None:
            raise ValueError("Argument 't2s' must be supplied if 'combmode' is 't2s-complex'.")

        t2s = np.asarray(t2s)
        if t2s.ndim != 1:
            raise ValueError(
                "Complex optimal combination requires voxel-wise (static) T2* estimates; "
                "volume-wise T2* (fitmode='ts') is not supported."
            )

        if t2s.shape[0] != n_samp:
            raise ValueError(
                "T2* estimates and data do not have same number of "
                f"voxels/samples: {t2s.shape[0]} != {n_samp}"
            )
    elif t2s is not None:
        LGR.warning(
            "Argument 't2s' is not required if 'combmode' is 'paid-complex'. "
            "'t2s' array will not be used."
        )

    if use_volumes is not None:
        use_volumes = np.asarray(use_volumes, dtype=bool)
        if use_volumes.shape != (n_vols,):
            raise ValueError(
                f"use_volumes must have one value per volume ({n_vols}), "
                f"not shape {use_volumes.shape}."
            )

        if not use_volumes.any():
            raise ValueError("At least one volume must be used for complex optimal combination.")

    if weighting == "t2s":
        LGR.info("Optimally combining complex data with voxel-wise T2* estimates")
        RepLGR.info(
            "Complex-valued multi-echo data were optimally combined by phase-aligning each "
            "echo to its temporal complex mean and averaging the aligned echoes with the "
            "T2*-based weights of \\citep{posse1999enhancement}, following "
            "\\citep{mikl2024processing}."
        )
    else:
        LGR.info("Optimally combining complex data with PAID (magnitude tSNR) weights")
        RepLGR.info(
            "Complex-valued multi-echo data were optimally combined by phase-aligning each "
            "echo to its temporal complex mean and averaging the aligned echoes with the "
            "PAID weights of \\citep{poser2006bold}, calculated from the magnitude tSNR, "
            "following \\citep{mikl2024processing}."
        )

    weights = _complex_combination_weights(
        magnitude,
        np.asarray(tes, dtype=float),
        adaptive_mask,
        weighting,
        t2s,
        use_volumes,
    )

    complex_dtype = np.result_type(magnitude.dtype, phase.dtype, np.complex64)
    combined = np.zeros((n_samp, n_vols), dtype=complex_dtype)
    reference_phase = np.zeros((n_samp, n_echos))
    coherence = np.zeros((n_samp, n_echos))
    failed = np.zeros((n_samp, n_echos), dtype=bool)
    # Build and align one echo at a time to limit memory use.
    for i_echo in range(n_echos):
        echo = (magnitude[:, i_echo, :] * np.exp(1j * phase[:, i_echo, :])).astype(complex_dtype)
        aligned, ref_phase, coh, fail = align_complex_echoes(
            echo[:, np.newaxis, :],
            use_volumes=use_volumes,
        )
        reference_phase[:, i_echo] = ref_phase[:, 0]
        coherence[:, i_echo] = coh[:, 0]
        failed[:, i_echo] = fail[:, 0]
        combined += weights[:, i_echo, np.newaxis] * aligned[:, 0, :]

    # Only report failed references for echoes that contribute to the combination.
    failed &= _used_echoes(adaptive_mask, n_echos)
    n_failed = int(failed.any(axis=1).sum())
    if n_failed:
        LGR.warning(
            f"The phase reference was undetermined (near-zero temporal mean) for at least one "
            f"echo in {n_failed} sample(s); no phase rotation was applied to those echoes."
        )

    diagnostics = {
        "weights": weights,
        "reference_phase": reference_phase,
        "coherence": coherence,
        "failed": failed,
    }
    return combined, diagnostics


def make_optcom(data, tes, adaptive_mask, t2s=None, combmode="t2s"):
    r"""Optimally combine BOLD data across TEs.

    Optimally combine BOLD data across TEs, using only those echos with reliable signal
    across at least three echos. If the number of echos providing reliable signal is greater
    than three but less than the total number of collected echos, we assume that later
    echos do not provided meaningful signal.

    Parameters
    ----------
    data : (Mb x E x T) :obj:`numpy.ndarray`
        Concatenated BOLD data, where `Mb` is samples in base mask, `E` is echos, and `T` is time.
    tes : (E,) :obj:`numpy.ndarray`
        Array of TEs, in seconds.
    adaptive_mask : (Mb,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    t2s : (Mb [x T]) :obj:`numpy.ndarray` or None, optional
        Estimated T2* values. Only required if combmode = 't2s'.
        Default is None.
    combmode : {'t2s', 'paid'}, optional
        How to combine data. Either 'paid' or 't2s'. If 'paid', argument 't2s'
        is not required. Default is 't2s'.

    Returns
    -------
    combined : (Mb x T) :obj:`numpy.ndarray`
        Optimally combined data.

    Notes
    -----
    This function supports both the ``'t2s'`` method :footcite:p:`posse1999enhancement`
    and the ``'paid'`` method :footcite:p:`poser2006bold`.
    The ``'t2s'`` method operates according to the following logic:

    1.  Estimate voxel- and TE-specific weights based on estimated :math:`T_2^*`:

            .. math::
                w(T_2^*)_n = \frac{TE_n * exp(\frac{-TE}\
                {T_{2(est)}^*})}{\sum TE_n * exp(\frac{-TE}{T_{2(est)}^*})}
    2.  Perform weighted average per voxel and TR across TEs based on weights
        estimated in the previous step.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
                                              parameter.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be 3D (S x E x T)")

    if len(tes) != data.shape[1]:
        raise ValueError(
            "Number of echos provided does not match second "
            f"dimension of input data: {len(tes)} != {data.shape[1]}"
        )

    if adaptive_mask.ndim != 1:
        raise ValueError("Mask is not 1D")
    elif adaptive_mask.shape[0] != data.shape[0]:
        raise ValueError(
            "Mask and data do not have same number of "
            f"voxels/samples: {adaptive_mask.shape[0]} != {data.shape[0]}"
        )

    if combmode not in ["t2s", "paid"]:
        raise ValueError("Argument 'combmode' must be either 't2s' or 'paid'")
    elif combmode == "t2s" and t2s is None:
        raise ValueError("Argument 't2s' must be supplied if 'combmode' is set to 't2s'.")
    elif combmode == "paid" and t2s is not None:
        LGR.warning(
            "Argument 't2s' is not required if 'combmode' is 'paid'. "
            "'t2s' array will not be used."
        )

    if combmode == "paid":
        LGR.info(
            "Optimally combining data with parallel-acquired "
            "inhomogeneity desensitized (PAID) method"
        )
    else:
        if t2s.shape[0] != data.shape[0]:
            raise ValueError(
                "T2* estimates and data do not have same number of "
                f"voxels/samples: {t2s.shape[0]} != {data.shape[0]}"
            )

        if t2s.ndim == 1:
            msg = "Optimally combining data with voxel-wise T2* estimates"
        else:
            msg = "Optimally combining data with voxel- and volume-wise T2* estimates"
        LGR.info(msg)

    echos_to_run = np.unique(adaptive_mask)
    # When there is one good echo, use two
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    tes = np.array(tes)[np.newaxis, ...]  # (1 x E) array_like
    combined = np.zeros((data.shape[0], data.shape[2]))
    report = True
    for i_echo, echo_num in enumerate(echos_to_run):
        if echo_num == 2:
            # Use the first two echoes for cases where there are
            # either one or two good echoes
            voxel_idx = np.where(np.logical_and(adaptive_mask > 0, adaptive_mask <= echo_num))[0]
        else:
            voxel_idx = np.where(adaptive_mask == echo_num)[0]

        if combmode == "paid":
            combined[voxel_idx, :] = _combine_paid(
                data[voxel_idx, :echo_num, :], tes[:, :echo_num]
            )
        else:
            t2s_ = t2s[..., np.newaxis]  # add singleton

            combined[voxel_idx, :] = _combine_t2s(
                data[voxel_idx, :echo_num, :],
                tes[:, :echo_num],
                t2s_[voxel_idx, ...],
                report=report,
            )
        report = False

    return combined
