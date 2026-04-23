"""Spatial metrics that do not directly relate to TE-dependence or -independence."""

import nibabel as nb
import numpy as np
from nilearn import masking


def calculate_slice_artifact(
    *,
    betas: np.ndarray,
    mask_img: nb.Nifti1Image,
    slice_axis: int,
) -> np.ndarray:
    """Calculate slice artifact metrics.

    Parameters
    ----------
    betas : (M x C) array_like
        Unstandardized parameter estimates from regression of optimally combined data against
        component time series.
    slice_axis : int
        Axis of the image corresponding to the slice dimension. Must be 0, 1, or 2.

    Returns
    -------
    slice_artifact : (C) array_like
        Slice artifact metrics.

    Notes
    -----
    The slice artifact metric quantifies the degree to which component weights align with slices
    in the image.
    """
    if betas.ndim != 2:
        raise ValueError(f"betas must be two-dimensional. Received shape {betas.shape}.")

    if slice_axis not in (0, 1, 2):
        raise ValueError(f"slice_axis must be 0, 1, or 2. Received {slice_axis}.")

    mask_arr = np.asanyarray(mask_img.dataobj).astype(bool)
    if mask_arr.ndim != 3:
        raise ValueError(f"mask_img must be three-dimensional. Received shape {mask_arr.shape}.")

    n_comps = betas.shape[1]
    betas_img = masking.unmask(betas, mask_img)
    betas_arr = np.asanyarray(betas_img.dataobj)
    slice_artifact = np.zeros(n_comps)
    for i_comp in range(n_comps):
        comp_map = betas_arr[..., i_comp]
        slice_means = []
        within_slice_vars = []
        for i_slice in range(comp_map.shape[slice_axis]):
            slice_mask = np.take(mask_arr, i_slice, axis=slice_axis)
            if not np.any(slice_mask):
                continue

            slice_vals = np.take(comp_map, i_slice, axis=slice_axis)[slice_mask]
            slice_means.append(slice_vals.mean())
            if slice_vals.size > 1:
                within_slice_vars.append(slice_vals.var())

        # No informative slices or no adjacent-slice comparison can be made.
        if len(slice_means) < 2:
            continue

        slice_means = np.asarray(slice_means)
        between_adjacent = np.mean(np.diff(slice_means) ** 2)
        within_slice = np.mean(within_slice_vars) if within_slice_vars else 0.0

        # Higher values indicate stronger striping by slice:
        # adjacent-slice jumps dominate over in-slice variability.
        denom = within_slice + np.finfo(float).eps
        slice_artifact[i_comp] = between_adjacent / denom

    return slice_artifact
