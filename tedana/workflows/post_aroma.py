"""
Apply AROMA to tedana output folder.
"""


def aroma_workflow(tedana_dir, motpars_file, t_r, xforms=None):
    betas_file = op.join(tedana_dir, 'betas_OC.nii')
    mix_file = op.join(tedana_dir, 'meica_mix.1D')
    betas_file2 = op.join(tedana_dir, 'betas_OC_std.nii')
    apply_xforms(betas_file, betas_file2, xforms)
    betas_data = nib.load(betas_file2)
    # variance normalize betas to make pseudo-z-values
    z_data = betas_data / np.std(betas_data, axis=-1)
    z_thresh_idx = np.abs(z_data) > np.median(np.abs(z_data))
    z_thresh_data = z_data * z_thresh_idx
    clf_df = aroma.run_aroma(mix_file, z_thresh_data, motpars_file, t_r)
