"""A command-line interface to denoise data using ICA components and classifications."""

import argparse
import sys

import nibabel as nb
from nilearn import masking

from tedana.workflows.parser_utils import is_valid_file


def _get_parser():
    """Get parser object for tedana.workflows.ica_denoise."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "in_file",
        help="Input file to denoise",
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        metavar="PATH",
        help="Output directory.",
        default=".",
    )
    parser.add_argument(
        "--prefix",
        dest="prefix",
        type=str,
        help="Prefix for filenames generated.",
        default="",
    )
    parser.add_argument(
        "--mix",
        dest="mix",
        type=lambda x: is_valid_file(parser, x),
        help="Path to the mixing matrix file.",
        required=True,
    )
    parser.add_argument(
        "--comp",
        dest="comp",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Path to the component table, which specifies which components are accepted or "
            "rejected."
        ),
        required=True,
    )
    parser.add_argument(
        "--mask",
        dest="mask",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Binary mask of voxels to include in TE "
            "Dependent ANAlysis. Must be in the same "
            "space as `data`. If an explicit mask is not "
            "provided, then Nilearn's compute_epi_mask "
            "function will be used to derive a mask "
            "from the first echo's data. "
            "Providing a mask is recommended."
        ),
        default=None,
    )
    parser.add_argument(
        "--denoising-method",
        dest="denoising_method",
        type=str,
        nargs="+",
        help="Denoising method(s) to use.",
        default=["nonaggr"],
        choices=["aggr", "nonaggr", "orthaggr"],
    )
    parser.add_argument(
        "--dummy-scans",
        dest="dummy_scans",
        type=int,
        help="Number of dummy scans to remove from the beginning of the data.",
        default=0,
    )
    return parser


def denoise_workflow(
    in_file,
    out_dir,
    prefix,
    mix,
    comp,
    mask,
    denoising_method,
    dummy_scans,
):
    """Denoise data using ICA components and classifications.

    Parameters
    ----------
    in_file : str
        Path to the input file to denoise.
    out_dir : str
        Path to the output directory.
    prefix : str
        Prefix for the output files.
    mix : str
        Path to the mixing matrix file.
    comp : str
        Path to the component table file.
    mask : str or None
        Path to the mask file.
    denoising_method : list of str
        List of denoising methods to use.
    dummy_scans : int
        Number of dummy scans to remove from the beginning of the data.
    """
    import os

    import numpy as np
    import pandas as pd

    # Load the mixing matrix
    mixing_df = pd.read_table(mix)  # Shape is time-by-components

    # Load the component table
    metrics_df = pd.read_table(comp)
    rejected_columns = metrics_df.loc[metrics_df["classification"] == "rejected", "Component"]
    accepted_columns = metrics_df.loc[metrics_df["classification"] == "accepted", "Component"]

    # Select "bad" components from the mixing matrix
    rejected_components = mixing_df[rejected_columns].to_numpy()
    accepted_components = mixing_df[accepted_columns].to_numpy()

    if prefix is None:
        prefix = ""
    elif not prefix.endswith("_"):
        prefix = f"{prefix}_"

    if in_file.endswith((".nii", ".nii.gz")):
        out_ext = "nii.gz"
        img = nb.load(in_file)
        if mask is None:
            # Create a mask of all voxels
            mask = nb.Nifti1Image(
                np.ones(img.shape[:3]),
                img.affine,
                img.header,
            )

        data = masking.apply_mask(img, mask)
    else:
        out_ext = ".".join(in_file.split(".")[-2:])
        data = nb.load(in_file).get_fdata()

    if "aggr" in denoising_method:
        # Fit GLM to rejected components and intercept
        regressors = np.hstack(
            (
                rejected_components,
                np.ones((mixing_df.shape[0], 1)),
            ),
        )
        betas = np.linalg.lstsq(regressors, data, rcond=None)[0][:-1]

        # Denoise the data using the betas from just the bad components
        confounds_idx = np.arange(rejected_components.shape[1])
        pred_data = np.dot(rejected_components, betas[confounds_idx, :])
        data_denoised = data - pred_data

        # Save to file
        denoised_img = _to_niimg(data_denoised, img.affine, img.header, out_ext, mask)
        denoised_img.to_filename(os.path.join(out_dir, f"{prefix}desc-aggr_bold.{out_ext}"))

    if "nonaggr" in denoising_method:
        # Fit GLM to accepted components, rejected components and intercept
        regressors = np.hstack(
            (
                rejected_components,
                accepted_components,
                np.ones((mixing_df.shape[0], 1)),
            ),
        )
        betas = np.linalg.lstsq(regressors, data, rcond=None)[0][:-1]

        # Denoise the data using the betas from just the bad components
        confounds_idx = np.arange(rejected_components.shape[1])
        pred_data = np.dot(rejected_components, betas[confounds_idx, :])
        data_denoised = data - pred_data

        # Save to file
        denoised_img = _to_niimg(data_denoised, img.affine, img.header, out_ext, mask)
        denoised_img.to_filename(os.path.join(out_dir, f"{prefix}desc-nonaggr_bold{out_ext}"))

    if "orthaggr" in denoising_method:
        # Regress the good components out of the bad time series to get "pure evil" regressors
        betas = np.linalg.lstsq(accepted_components, rejected_components, rcond=None)[0]
        pred_bad_timeseries = np.dot(accepted_components, betas)
        orth_bad_timeseries = rejected_components - pred_bad_timeseries

        # Fit GLM to rejected components and intercept
        regressors = np.hstack(
            (
                orth_bad_timeseries,
                np.ones((mixing_df.shape[0], 1)),
            ),
        )
        betas = np.linalg.lstsq(regressors, data, rcond=None)[0][:-1]

        # Denoise the data using the betas from just the bad components
        confounds_idx = np.arange(rejected_components.shape[1])
        pred_data = np.dot(rejected_components, betas[confounds_idx, :])
        data_denoised = data - pred_data

        # Save to file
        denoised_img = _to_niimg(data_denoised, img.affine, img.header, out_ext, mask)
        denoised_img.to_filename(os.path.join(out_dir, f"{prefix}desc-orthaggr_bold.{out_ext}"))


def _to_niimg(data, affine, header, out_ext, mask):
    if out_ext == "nii.gz":
        img = masking.unmask(data, mask)
    else:
        img = nb.Cifti2Image(data, header)

    return img


def _main(argv=None):
    """Main function for tedana.workflows.ica_denoise."""
    parser = _get_parser()
    args = parser.parse_args(argv)
    denoise_workflow(**vars(args))


if __name__ == "__main__":
    sys.exit(_main())