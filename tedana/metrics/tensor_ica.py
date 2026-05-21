"""TE-mode metrics for tensor-ICA component classification."""

import logging

import numpy as np
import pandas as pd

from tedana import utils

LGR = logging.getLogger("GENERAL")


def generate_tensor_metrics(s_modes, mixing, spatial_maps, echo_times, tr, n_vols):
    """Compute per-component metrics for tensor-ICA classification.

    Parameters
    ----------
    s_modes : (n_echoes, n_components) :obj:`numpy.ndarray`
        TE-mode loadings from tensor decomposition.
    mixing : (n_timepoints, n_components) :obj:`numpy.ndarray`
        Z-scored temporal component time courses.
    spatial_maps : (n_masked_voxels, n_components) :obj:`numpy.ndarray`
        Spatial component maps (masked).
    echo_times : (n_echoes,) :obj:`numpy.ndarray`
        Echo times in milliseconds.
    tr : :obj:`float`
        Repetition time in seconds.
    n_vols : :obj:`int`
        Number of volumes (must equal ``mixing.shape[0]``).

    Returns
    -------
    component_table : :obj:`pandas.DataFrame`
        One row per component with columns: ``te_peak``, ``freq_ratio``,
        ``variance_explained``, ``classification``, ``rationale``,
        ``classification_tags``, ``Component``.
    """
    n_echoes, n_components = s_modes.shape

    te_peaks = []
    freq_ratios = []

    for i in range(n_components):
        # 2nd-order polynomial fit; np.polyfit returns [a, b, c] for ax^2 + bx + c
        coeff = np.polyfit(echo_times, s_modes[:, i], 2)
        a, b = coeff[0], coeff[1]
        peak = -b / (2 * a) if a != 0 else float(echo_times.mean())
        te_peaks.append(peak)

        spectrum, freqs = utils.get_spectrum(mixing[:, i], tr)
        neural = spectrum[(freqs > 0.01) & (freqs < 0.1)].sum()
        total = spectrum[freqs > 0.01].sum()
        freq_ratios.append(neural / total if total > 0 else 0.0)

    # Variance explained: proportional to product of spatial and temporal norms
    spatial_norms = np.linalg.norm(spatial_maps, axis=0)
    temporal_norms = np.linalg.norm(mixing, axis=0)
    weights = spatial_norms * temporal_norms
    total_weight = weights.sum()
    variance_explained = (
        weights / total_weight
        if total_weight > 0
        else np.ones(n_components) / n_components
    )

    component_table = pd.DataFrame(
        {
            "Component": [f"ICA_{i:02d}" for i in range(n_components)],
            "te_peak": te_peaks,
            "freq_ratio": freq_ratios,
            "variance explained": variance_explained,
            "normalized variance explained": variance_explained,
            "classification": "unclassified",
            "rationale": "",
            "classification_tags": "",
        }
    )

    return component_table
