# GRAPPA / in-plane-acceleration artifact metric

A component-level metric that flags in-plane parallel-imaging (GRAPPA)
reconstruction artifacts, which are **independent of the TE-dependence model**.
Standard tedana metrics (kappa, rho, and the F-statistics behind them) ask
whether a component's signal scales with TE like BOLD (T2\*) or like a non-BOLD
S0 effect. They say nothing about the *spatial* voxel-scale texture left behind
by parallel-imaging reconstruction error, so a component can carry an obvious
aliasing artifact while still looking "mixed" or even mildly BOLD-like to the
dependence metrics. `grappa_artifact` targets that blind spot directly.

The metric lives in `tedana/metrics/spatial.py`, is registered in
`tedana/resources/config/metrics.json`, and is computed in
`tedana.metrics.collect.generate_metrics` when requested by a decision tree or
the metric set.

---

## What it computes

For each component percent-signal-change (PSC) map `I`, and each spatial array
axis `d`, over a 19-voxel spherical neighborhood (center + 6 face + 12 edge
voxels; AFNI `SPHERE(-1.42)`):

1. Local image variance `Var(I)` over in-mask neighborhood voxels.
2. Local variance of the first difference `dI_d(v) = I(v + e_d) - I(v)`.
3. Local lag-1 autocorrelation `r_d = 1 - Var(dI_d) / (2 Var(I))`.
4. Flag the voxel/axis where `r_d <= 0` (equivalently `Var(dI_d) >= 2 Var(I)`):
   adjacent voxels are anti-correlated, the implied FWHM is indeterminate.

The metric is the **count of flagged (voxel, axis) pairs** over the map.
`grappa_artifact_fraction` normalizes this by `3 * n_mask_voxels` for
cross-acquisition comparability.

## Logic and rationale

In-plane parallel imaging (GRAPPA) leaves high-spatial-frequency, sign-alternating,
often non-stationary reconstruction errors. These evade kappa/rho and even global
Fourier-energy filters. Following MEICA4, the artifact is detected as a *local
autocorrelation violation*: where the first-difference-based FWHM estimator becomes
indeterminate (`r_d <= 0`), the voxel sits in voxel-scale alternating texture. The
estimator's "failure" is the detection feature.

## Strengths

- Mechanism-specific to in-plane aliasing (anti-correlated neighbors).
- Local, so it catches spatially confined / non-stationary aliasing a global
  spectral measure averages away.
- Scale- and sign-invariant (a variance ratio of squared first differences).

## Weaknesses and caveats

- **Resolution / Nyquist sensitivity.** me-ica slightly upsamples so the artifact
  is "not at Nyquist"; tedana does not resample, so very-near-Nyquist artifacts may
  be under-registered.
- **Raw count is not acquisition-portable** (scales with mask size/resolution);
  prefer `grappa_artifact_fraction`, or a per-acquisition threshold.
- **No universal threshold** (me-ica learns it per dataset; not ported here).
- **Low-SNR false positives**: noise-dominated component maps show voxel-scale
  anti-correlation that is not GRAPPA.

## Failure cases

- Smoothed or spatially-normalized data -> aliasing blurred or rotated off the
  voxel axes -> missed. (Atypical for tedana, which runs on native, unsmoothed
  data.)
- Strong susceptibility-distortion-correction interpolation along the
  phase-encode direction -> attenuated aliasing -> reduced sensitivity.
- One fixed threshold across heterogeneous acquisitions -> mis-calibrated.

## Potential alternatives / extensions

- Global high-frequency Fourier fraction (simpler; misses non-stationary aliasing).
- Phase-encode-direction-aware detection from BIDS metadata.
- Port me-ica's dataset-adaptive threshold for a self-calibrating decision-tree node.

---

The metric is a deliberately simple, TE-independent statistic meant to
complement — not replace — the dependence metrics. Its threshold is the
weakest part of the story and should be calibrated against labelled data per
acquisition before being trusted in an automated decision tree.
