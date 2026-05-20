.. _tensor_ica:

Tensor-ICA
==========

Tensor-ICA is an alternative decomposition method for multi-echo fMRI that
exploits the echo-time (TE) dimension directly, without first performing
optimal combination. It decomposes data into spatial, temporal, and TE-mode
components simultaneously, then classifies components based on their
TE-dependency profile and frequency content.

When to Use Tensor-ICA
-----------------------

Tensor-ICA is appropriate when you want to:

* Compare decomposition results against the standard ME-PCA + ME-ICA pipeline.
* Classify components using TE-peak polynomial fitting rather than kappa/rho.
* Use FSL MELODIC as the decomposition engine while retaining tedana's
  classification and reporting infrastructure.

For most datasets, the standard ``tedana`` workflow (``--ica-method fastica``
or ``robustica``) is recommended. Tensor-ICA is experimental.

Backends
--------

Two backends are available:

**tensorly** (pure Python, no additional system dependencies)
  Requires the ``tensorly`` package::

      pip install tedana[tensor-ica]

  Performs Tucker decomposition on the 3-mode data tensor
  (voxels × timepoints × echoes) followed by FastICA on the temporal factor.

**fsl** (requires FSL on PATH)
  Calls ``melodic --tica`` via subprocess. Compatible with ``--tedpca mdl``
  and ``--tedpca aic`` only; other ``--tedpca`` values raise an error.

CLI Usage
---------

.. code-block:: bash

   # Tensorly backend (default for tensor-ICA)
   tedana -d echo1.nii.gz echo2.nii.gz echo3.nii.gz \
          -e 0.013 0.028 0.043 \
          --ica-method tensorly

   # FSL backend
   tedana -d echo1.nii.gz echo2.nii.gz echo3.nii.gz \
          -e 0.013 0.028 0.043 \
          --ica-method fsl \
          --tedpca mdl

   # Adjust keep-ratio threshold (default 0.3)
   tedana -d echo1.nii.gz echo2.nii.gz echo3.nii.gz \
          -e 0.013 0.028 0.043 \
          --ica-method tensorly \
          --keep-ratio 0.4

Component Classification
------------------------

Tensor-ICA uses the ``tensor_ica`` decision tree, which implements two-level
filtering:

1. **TE-peak range check**: Fit a 2nd-order polynomial to the TE-mode
   loadings (S-modes) for each component. Reject if the peak falls outside
   15-55 ms (outside the physiological BOLD T2* range).

2. **Top-N selection**: Of the remaining components, keep only
   ``max(ceil(n_total × keep_ratio), ceil(n_remaining × 0.7))`` by TE-peak
   height, to limit computation in the next step.

3. **Frequency ratio check**: Accept if the ratio of power in 0.01-0.1 Hz
   to total power above 0.01 Hz exceeds 0.7; reject otherwise.

Outputs
-------

Tensor-ICA produces all standard tedana outputs plus:

.. csv-table::
   :header: "File", "Description"
   :widths: 40, 60

   "``desc-ICA_smodes.tsv``", "TE-mode loadings matrix (n_echoes × n_components)"

See :ref:`outputs` for the full list of standard output files.
