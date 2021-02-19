"""
Functions to estimate S0 and T2* from multi-echo data.
"""
import logging
import scipy
import numpy as np
from tedana import utils

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


def _apply_t2s_floor(t2s, echo_times):
    """
    Apply a floor to T2* values to prevent zero division errors during
    optimal combination.

    Parameters
    ----------
    t2s : (S,) array_like
        T2* estimates.
    echo_times : (E,) array_like
        Echo times in milliseconds.

    Returns
    -------
    t2s_corrected : (S,) array_like
        T2* estimates with very small, positive values replaced with a floor value.
    """
    t2s_corrected = t2s.copy()
    echo_times = np.asarray(echo_times)
    if echo_times.ndim == 1:
        echo_times = echo_times[:, None]

    eps = np.finfo(dtype=t2s.dtype).eps  # smallest value for datatype
    temp_arr = np.exp(-echo_times / t2s)  # (E x V) array
    bad_voxel_idx = np.any(temp_arr == 0, axis=0) & (t2s != 0)
    n_bad_voxels = np.sum(bad_voxel_idx)
    if n_bad_voxels > 0:
        n_voxels = temp_arr.size
        floor_percent = 100 * n_bad_voxels / n_voxels
        LGR.debug(
            "T2* values for {0}/{1} voxels ({2:.2f}%) have been "
            "identified as close to zero and have been "
            "adjusted".format(n_bad_voxels, n_voxels, floor_percent)
        )
    t2s_corrected[bad_voxel_idx] = np.min(-echo_times) / np.log(eps)
    return t2s_corrected


def monoexponential(tes, s0, t2star):
    """
    Specifies a monoexponential model for use with scipy curve fitting

    Parameters
    ----------
    tes : (E,) :obj:`list`
        Echo times
    s0 : :obj:`float`
        Initial signal parameter
    t2star : :obj:`float`
        T2* parameter

    Returns
    -------
    :obj:`float`
        Predicted signal
    """
    return s0 * np.exp(-tes / t2star)


def fit_monoexponential(data_cat, echo_times, adaptive_mask, report=True):
    """
    Fit monoexponential decay model with nonlinear curve-fitting.

    Parameters
    ----------
    data_cat : (S x E x T) :obj:`numpy.ndarray`
        Multi-echo data.
    echo_times : (E,) array_like
        Echo times in milliseconds.
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    report : bool, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    t2s_limited, s0_limited, t2s_full, s0_full : (S,) :obj:`numpy.ndarray`
        T2* and S0 estimate maps.

    Notes
    -----
    This method is slower, but more accurate, than the log-linear approach.

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
                                              parameter.
    """
    if report:
        RepLGR.info("A monoexponential model was fit to the data at each voxel "
                    "using nonlinear model fitting in order to estimate T2* and S0 "
                    "maps, using T2*/S0 estimates from a log-linear fit as "
                    "initial values. For each voxel, the value from the adaptive "
                    "mask was used to determine which echoes would be used to "
                    "estimate T2* and S0. In cases of model fit failure, T2*/S0 "
                    "estimates from the log-linear fit were retained instead.")
    n_samp, n_echos, n_vols = data_cat.shape

    # Currently unused
    # fit_data = np.mean(data_cat, axis=2)
    # fit_sigma = np.std(data_cat, axis=2)

    t2s_full, s0_full = fit_loglinear(
        data_cat, echo_times, adaptive_mask, report=False)

    unique_patterns = np.unique(adaptive_mask, axis=0)
    unique_patterns = [p for p in unique_patterns if np.sum(p) >= 2]

    for pattern in unique_patterns:
        echo_idx = np.where(pattern)[0]
        pattern_idx = np.where((adaptive_mask == pattern).all(axis=1))[0]
        selected_data = data_cat[pattern_idx, pattern, :]
        selected_echo_times = echo_times[pattern]
        data_2d = selected_data.reshape(selected_data.shape[0], -1).T
        echo_times_1d = np.repeat(selected_echo_times, n_vols)

        # perform a monoexponential fit of echo times against MR signal
        # using loglin estimates as initial starting points for fit
        fail_count = 0
        for i_voxel, voxel_idx in enumerate(pattern_idx):
            try:
                popt, cov = scipy.optimize.curve_fit(
                    monoexponential, echo_times_1d, data_2d[:, i_voxel],
                    p0=(s0_full[voxel_idx], t2s_full[voxel_idx]),
                    bounds=((np.min(data_2d[:, i_voxel]), 0),
                            (np.inf, np.inf)))
                s0_full[voxel_idx] = popt[0]
                t2s_full[voxel_idx] = popt[1]
            except (RuntimeError, ValueError):
                # If curve_fit fails to converge, fall back to loglinear estimate
                fail_count += 1

        if fail_count:
            fail_percent = 100 * fail_count / selected_data.shape[0]
            LGR.debug(
                "With echoes {0}, monoexponential fit failed on {1}/{2} ({3:.2f}%) voxel(s), "
                "used log linear estimate instead".format(
                        ", ".join(echo_idx),
                        fail_count,
                        selected_data.shape[0],
                        fail_percent
                )
            )

    return t2s_full, s0_full


def fit_loglinear(data_cat, echo_times, adaptive_mask, report=True):
    """Fit monoexponential decay model with log-linear regression.

    The monoexponential decay function is fitted to all values for a given
    voxel across TRs, per TE, to estimate voxel-wise :math:`S_0` and :math:`T_2^*`.
    At a given voxel, only those echoes with "good signal", as indicated by the
    value of the voxel in the adaptive mask, are used.
    Therefore, for a voxel with an adaptive mask value of five, the first five
    echoes would be used to estimate T2* and S0.

    Parameters
    ----------
    data_cat : (S x E x T) :obj:`numpy.ndarray`
        Multi-echo data. S is samples, E is echoes, and T is timepoints.
    echo_times : (E,) array_like
        Echo times in milliseconds.
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    report : :obj:`bool`, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    t2s_limited, s0_limited, t2s_full, s0_full: (S,) :obj:`numpy.ndarray`
        T2* and S0 estimate maps.

    Notes
    -----
    The approach used in this function involves transforming the raw signal values
    (:math:`log(|data| + 1)`) and then fitting a line to the transformed data using
    ordinary least squares.
    This results in two parameter estimates: one for the slope  and one for the intercept.
    The slope estimate is inverted (i.e., 1 / slope) to get  :math:`T_2^*`,
    while the intercept estimate is exponentiated (i.e., e^intercept) to get :math:`S_0`.

    This method is faster, but less accurate, than the nonlinear approach.
    """
    if report:
        RepLGR.info("A monoexponential model was fit to the data at each voxel "
                    "using log-linear regression in order to estimate T2* and S0 "
                    "maps. For each voxel, the value from the adaptive mask was "
                    "used to determine which echoes would be used to estimate T2* "
                    "and S0.")
    n_samp, n_echos, n_vols = data_cat.shape

    s0_full = np.empty(n_samp)
    t2s_full = np.empty(n_samp)

    unique_patterns = np.unique(adaptive_mask, axis=0)
    unique_patterns = [p for p in unique_patterns if np.sum(p) >= 2]

    for pattern in unique_patterns:
        LGR.warning("Fitting pattern {}".format(pattern))
        echo_idx = np.where(pattern)[0]
        pattern_idx = np.where((adaptive_mask == pattern).all(axis=1))[0]
        LGR.warning(pattern_idx.shape)
        LGR.warning(echo_idx.shape)
        LGR.warning(data_cat.shape)
        selected_data = data_cat[pattern_idx, ...][:, echo_idx, :]
        data_2d = selected_data.reshape(selected_data.shape[0], -1).T
        log_data = np.log(np.abs(data_2d) + 1)

        # make IV matrix: intercept/TEs x (time series * echos)
        x = np.column_stack([np.ones(int(np.sum(pattern))), [-te for te in echo_times[echo_idx]]])
        X = np.repeat(x, n_vols, axis=0)

        # Log-linear fit
        betas = np.linalg.lstsq(X, log_data, rcond=None)[0]
        s0_full[pattern_idx] = 1. / betas[1, :].T
        t2s_full[pattern_idx] = np.exp(betas[0, :]).T

    return t2s_full, s0_full


def fit_decay(data, tes, mask, adaptive_mask, fittype, report=True):
    """
    Fit voxel-wise monoexponential decay models to `data`

    Parameters
    ----------
    data : (S x E [x T]) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    tes : (E,) :obj:`list`
        Echo times
    mask : (S,) array_like
        Boolean array indicating samples that are consistently (i.e., across
        time AND echoes) non-zero
    adaptive_mask : (S,) array_like
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    fittype : {loglin, curvefit}
        The type of model fit to use
    report : bool, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    t2s_limited : (S,) :obj:`numpy.ndarray`
        Limited T2* map. The limited map only keeps the T2* values for data
        where there are at least two echos with good signal.
    s0_limited : (S,) :obj:`numpy.ndarray`
        Limited S0 map.  The limited map only keeps the S0 values for data
        where there are at least two echos with good signal.
    t2s_full : (S,) :obj:`numpy.ndarray`
        Full T2* map. For voxels affected by dropout, with good signal from
        only one echo, the full map uses the T2* estimate from the first two
        echoes.
    s0_full : (S,) :obj:`numpy.ndarray`
        Full S0 map. For voxels affected by dropout, with good signal from
        only one echo, the full map uses the S0 estimate from the first two
        echoes.

    Notes
    -----
    This function replaces infinite values in the :math:`T_2^*` map with 500 and
    :math:`T_2^*` values less than or equal to zero with 1.
    Additionally, very small :math:`T_2^*` values above zero are replaced with a floor
    value to prevent zero-division errors later on in the workflow.
    It also replaces NaN values in the :math:`S_0` map with 0.

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
                                              parameter.
    """
    if data.shape[1] != len(tes):
        raise ValueError('Second dimension of data ({0}) does not match number '
                         'of echoes provided (tes; {1})'.format(data.shape[1], len(tes)))
    elif not (data.shape[0] == mask.shape[0] == adaptive_mask.shape[0]):
        raise ValueError('First dimensions (number of samples) of data ({0}), '
                         'mask ({1}), and adaptive_mask ({2}) do not '
                         'match'.format(data.shape[0], mask.shape[0], adaptive_mask.shape[0]))

    data = data.copy()
    if data.ndim == 2:
        data = data[:, :, None]

    # Mask the inputs
    data_masked = data[mask, :, :]
    adaptive_mask_masked = adaptive_mask[mask]

    if fittype == 'loglin':
        t2s_full, s0_full = fit_loglinear(
            data_masked, tes, adaptive_mask_masked, report=report)
    elif fittype == 'curvefit':
        t2s_full, s0_full = fit_monoexponential(
            data_masked, tes, adaptive_mask_masked, report=report)
    else:
        raise ValueError('Unknown fittype option: {}'.format(fittype))

    t2s_full[np.isinf(t2s_full)] = 500.  # why 500?
    t2s_full[t2s_full <= 0] = 1.  # let's get rid of negative values!
    t2s_full = _apply_t2s_floor(t2s_full, tes)
    s0_full[np.isnan(s0_full)] = 0.  # why 0?

    t2s_full = utils.unmask(t2s_full, mask)
    s0_full = utils.unmask(s0_full, mask)

    return t2s_full, s0_full


def fit_decay_ts(data, tes, mask, adaptive_mask, fittype):
    """
    Fit voxel- and timepoint-wise monoexponential decay models to `data`

    Parameters
    ----------
    data : (S x E x T) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    tes : (E,) :obj:`list`
        Echo times
    mask : (S,) array_like
        Boolean array indicating samples that are consistently (i.e., across
        time AND echoes) non-zero
    adaptive_mask : (S,) array_like
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    fittype : :obj: `str`
        The type of model fit to use

    Returns
    -------
    t2s_limited_ts : (S x T) :obj:`numpy.ndarray`
        Limited T2* map. The limited map only keeps the T2* values for data
        where there are at least two echos with good signal.
    s0_limited_ts : (S x T) :obj:`numpy.ndarray`
        Limited S0 map.  The limited map only keeps the S0 values for data
        where there are at least two echos with good signal.
    t2s_full_ts : (S x T) :obj:`numpy.ndarray`
        Full T2* timeseries.  For voxels affected by dropout, with good signal
        from only one echo, the full timeseries uses the single echo's value
        at that voxel/volume.
    s0_full_ts : (S x T) :obj:`numpy.ndarray`
        Full S0 timeseries. For voxels affected by dropout, with good signal
        from only one echo, the full timeseries uses the single echo's value
        at that voxel/volume.

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
                                              parameter.
    """
    n_samples, _, n_vols = data.shape
    tes = np.array(tes)

    t2s_limited_ts = np.zeros([n_samples, n_vols])
    s0_limited_ts = np.copy(t2s_limited_ts)
    t2s_full_ts = np.copy(t2s_limited_ts)
    s0_full_ts = np.copy(t2s_limited_ts)

    report = True
    for vol in range(n_vols):
        t2s_limited, s0_limited, t2s_full, s0_full = fit_decay(
            data[:, :, vol][:, :, None], tes, mask, adaptive_mask, fittype, report=report)
        t2s_limited_ts[:, vol] = t2s_limited
        s0_limited_ts[:, vol] = s0_limited
        t2s_full_ts[:, vol] = t2s_full
        s0_full_ts[:, vol] = s0_full
        report = False

    return t2s_limited_ts, s0_limited_ts, t2s_full_ts, s0_full_ts
