"""Total variation (TV-L2) denoising of echo time series.

Implements the one-dimensional temporal TV-L2 echo denoising method of
Michalek and Mikl (2025, doi:10.3389/fnins.2025.1544748), solved with the
partially inexact ADMM algorithm of Michalek (2015).
"""

import logging

import numpy as np

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
