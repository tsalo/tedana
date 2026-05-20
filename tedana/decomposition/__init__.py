# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""Functions for decomposing BOLD signals."""

from tedana.decomposition.ica import tedica
from tedana.decomposition.pca import tedpca
from tedana.decomposition.tensor_ica import tensor_ica

__all__ = ["tedpca", "tedica", "tensor_ica"]
