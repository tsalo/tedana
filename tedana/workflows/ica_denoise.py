"""A command-line interface to denoise data using ICA components and classifications."""

import argparse
import sys

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
    """Denoise data using ICA components and classifications."""
    import os

    import pandas as pd
    from nilearn.maskers import NiftiMasker

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

    if "aggr" in denoising_method:
        masker = NiftiMasker(
            mask_img=mask,
            standardize_confounds=True,
            standardize=False,
            smoothing_fwhm=None,
            detrend=False,
            low_pass=None,
            high_pass=None,
            t_r=None,  # This shouldn't be necessary since we aren't bandpass filtering
            reports=False,
        )

        # Denoise the data by fitting and transforming the data file using the masker
        denoised_img_2d = masker.fit_transform(in_file, confounds=rejected_components)

        # Transform denoised data back into 4D space
        denoised_img_4d = masker.inverse_transform(denoised_img_2d)

        # Save to file
        denoised_img_4d.to_filename(os.path.join(out_dir, f"{prefix}desc-aggr_bold.nii.gz"))

    if "nonaggr" in denoising_method:
        import numpy as np
        from nilearn.masking import apply_mask, unmask  # Functions for (un)masking fMRI data

        # Apply the mask to the data image to get a 2d array
        data = apply_mask(in_file, mask)

        # Fit GLM to accepted components, rejected components and nuisance regressors
        # (after adding a constant term)
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
        denoised_img = unmask(data_denoised, mask)
        denoised_img.to_filename(os.path.join(out_dir, f"{prefix}desc-nonaggr_bold.nii.gz"))

    if "orthaggr" in denoising_method:
        # Regress the good components out of the bad time series to get "pure evil" regressors
        betas = np.linalg.lstsq(accepted_components, rejected_components, rcond=None)[0]
        pred_bad_timeseries = np.dot(accepted_components, betas)
        orth_bad_timeseries = rejected_components - pred_bad_timeseries

        # Once you have these "pure evil" components, you can denoise the data
        masker = NiftiMasker(
            mask_img=mask,
            standardize_confounds=True,
            standardize=False,
            smoothing_fwhm=None,
            detrend=False,
            low_pass=None,
            high_pass=None,
            t_r=None,  # This shouldn't be necessary since we aren't bandpass filtering
            reports=False,
        )

        # Denoise the data by fitting and transforming the data file using the masker
        denoised_img_2d = masker.fit_transform(in_file, confounds=orth_bad_timeseries)

        # Transform denoised data back into 4D space
        denoised_img_4d = masker.inverse_transform(denoised_img_2d)

        # Save to file
        denoised_img_4d.to_filename(os.path.join(out_dir, f"{prefix}desc-orthaggr_bold.nii.gz"))


def _main(argv=None):
    """Main function for tedana.workflows.ica_denoise."""
    parser = _get_parser()
    args = parser.parse_args(argv)
    denoise_workflow(**vars(args))


if __name__ == "__main__":
    sys.exit(_main())