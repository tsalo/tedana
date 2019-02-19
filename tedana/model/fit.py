"""
Fit models.
"""
import logging
import os.path as op

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import lpmv
import nilearn.image as niimg
from nilearn._utils import check_niimg
from nilearn.regions import connected_regions

from tedana import (combine, io, utils)

LGR = logging.getLogger(__name__)

F_MAX = 500
Z_MAX = 8


def fitmodels_direct(catd, mmix, mask, t2s, t2s_full, tes, combmode, ref_img,
                     reindex=False, mmixN=None, full_sel=True, label=None,
                     out_dir='.', verbose=False):
    """
    Fit TE-dependence and -independence models to components.

    Parameters
    ----------
    catd : (S x E x T) array_like
        Input data, where `S` is samples, `E` is echos, and `T` is time
    mmix : (T x C) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `catd`
    mask : (S [x E]) array_like
        Boolean mask array
    t2s : (S [x T]) array_like
        Limited T2* map or timeseries.
    t2s_full : (S [x T]) array_like
        Full T2* map or timeseries. For voxels with good signal in only one
        echo, which are zeros in the limited T2* map, this map uses the T2*
        estimate using the first two echoes.
    tes : list
        List of echo times associated with `catd`, in milliseconds
    combmode : {'t2s', 'ste'} str
        How optimal combination of echos should be made, where 't2s' indicates
        using the method of Posse 1999 and 'ste' indicates using the method of
        Poser 2006
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    reindex : bool, optional
        Default: False
    mmixN : array_like, optional
        Default: None
    full_sel : bool, optional
        Whether to perform selection of components based on Rho/Kappa scores.
        Default: True

    Returns
    -------
    seldict : dict
    comptab : (N x 5) :obj:`pandas.DataFrame`
        Array with columns denoting (1) index of component, (2) Kappa score of
        component, (3) Rho score of component, (4) variance explained by
        component, and (5) normalized variance explained by component
    betas : :obj:`numpy.ndarray`
    mmix_new : :obj:`numpy.ndarray`
    """
    if not (catd.shape[0] == t2s.shape[0] == t2s_full.shape[0] == mask.shape[0]):
        raise ValueError('First dimensions (number of samples) of catd ({0}), '
                         't2s ({1}), and mask ({2}) do not '
                         'match'.format(catd.shape[0], t2s.shape[0],
                                        mask.shape[0]))
    elif catd.shape[1] != len(tes):
        raise ValueError('Second dimension of catd ({0}) does not match '
                         'number of echoes provided (tes; '
                         '{1})'.format(catd.shape[1], len(tes)))
    elif catd.shape[2] != mmix.shape[0]:
        raise ValueError('Third dimension (number of volumes) of catd ({0}) '
                         'does not match first dimension of '
                         'mmix ({1})'.format(catd.shape[2], mmix.shape[0]))
    elif t2s.shape != t2s_full.shape:
        raise ValueError('Shape of t2s array {0} does not match shape of '
                         't2s_full array {1}'.format(t2s.shape,
                                                     t2s_full.shape))
    elif t2s.ndim == 2:
        if catd.shape[2] != t2s.shape[1]:
            raise ValueError('Third dimension (number of volumes) of catd '
                             '({0}) does not match second dimension of '
                             't2s ({1})'.format(catd.shape[2], t2s.shape[1]))

    mask = t2s != 0  # Override mask because problems

    # compute optimal combination of raw data
    tsoc = combine.make_optcom(catd, tes, mask, t2s=t2s_full, combmode=combmode,
                               verbose=False).astype(float)[mask]

    # demean optimal combination
    tsoc_dm = tsoc - tsoc.mean(axis=-1, keepdims=True)

    # compute un-normalized weight dataset (features)
    if mmixN is None:
        mmixN = mmix

    betas_, WTS = get_coeffs_and_zstats(utils.unmask(tsoc, mask), mmixN, mask, normalize=False)

    # compute PSC dataset - shouldn't have to refit data
    tsoc_B = get_coeffs(tsoc_dm, mmix, mask=None)
    tsoc_Babs = np.abs(tsoc_B)
    PSC = tsoc_B / tsoc.mean(axis=-1, keepdims=True) * 100

    # compute skews to determine signs based on unnormalized weights,
    # correct mmix & WTS signs based on spatial distribution tails
    signs = stats.skew(WTS, axis=0)
    signs /= np.abs(signs)
    mmix = mmix.copy()
    mmix *= signs
    WTS *= signs
    PSC *= signs
    totvar = (tsoc_B**2).sum()
    totvar_norm = (WTS**2).sum()

    # compute Betas and means over TEs for TE-dependence analysis
    betas = get_coeffs(catd, mmix, np.repeat(mask[:, np.newaxis], len(tes),
                                             axis=1))
    n_samp, n_echos, n_components = betas.shape
    n_voxels = mask.sum()
    n_data_voxels = (t2s != 0).sum()
    mu = catd.mean(axis=-1, dtype=float)
    tes = np.reshape(tes, (n_echos, 1))
    fmin, _, _ = utils.getfbounds(n_echos)

    # mask arrays
    mumask = mu[t2s != 0]
    t2smask = t2s[t2s != 0]
    betamask = betas[t2s != 0]

    # set up Xmats
    X1 = mumask.T  # Model 1
    X2 = np.tile(tes, (1, n_data_voxels)) * mumask.T / t2smask.T  # Model 2

    # tables for component selection
    kappas = np.zeros([n_components])
    rhos = np.zeros([n_components])
    varex = np.zeros([n_components])
    varex_norm = np.zeros([n_components])
    Z_maps = np.zeros([n_voxels, n_components])
    F_R2_maps = np.zeros([n_data_voxels, n_components])
    F_S0_maps = np.zeros([n_data_voxels, n_components])
    Z_clmaps = np.zeros([n_voxels, n_components])
    F_R2_clmaps = np.zeros([n_data_voxels, n_components])
    F_S0_clmaps = np.zeros([n_data_voxels, n_components])
    Br_R2_clmaps = np.zeros([n_voxels, n_components])
    Br_S0_clmaps = np.zeros([n_voxels, n_components])
    pred_R2_maps = np.zeros([n_data_voxels, n_echos, n_components])
    pred_S0_maps = np.zeros([n_data_voxels, n_echos, n_components])

    LGR.info('Fitting TE- and S0-dependent models to components')
    for i_comp in range(n_components):
        # size of B is (n_echoes, n_samples)
        B = np.atleast_3d(betamask)[:, :, i_comp].T
        alpha = (np.abs(B)**2).sum(axis=0)
        varex[i_comp] = (tsoc_B[:, i_comp]**2).sum() / totvar * 100.
        varex_norm[i_comp] = (utils.unmask(WTS, mask)[t2s != 0][:, i_comp]**2).sum() /\
            totvar_norm * 100.

        # S0 Model
        # (S,) model coefficient map
        coeffs_S0 = (B * X1).sum(axis=0) / (X1**2).sum(axis=0)
        pred_S0 = X1 * np.tile(coeffs_S0, (n_echos, 1))
        pred_S0_maps[:, :, i_comp] = pred_S0.T
        SSE_S0 = (B - pred_S0)**2
        SSE_S0 = SSE_S0.sum(axis=0)  # (S,) prediction error map
        F_S0 = (alpha - SSE_S0) * (n_echos - 1) / (SSE_S0)
        F_S0_maps[:, i_comp] = F_S0

        # R2 Model
        coeffs_R2 = (B * X2).sum(axis=0) / (X2**2).sum(axis=0)
        pred_R2 = X2 * np.tile(coeffs_R2, (n_echos, 1))
        pred_R2_maps[:, :, i_comp] = pred_R2.T
        SSE_R2 = (B - pred_R2)**2
        SSE_R2 = SSE_R2.sum(axis=0)
        F_R2 = (alpha - SSE_R2) * (n_echos - 1) / (SSE_R2)
        F_R2_maps[:, i_comp] = F_R2

        # compute weights as Z-values
        wtsZ = stats.zscore(WTS[:, i_comp])
        # cap weights at Z_MAX
        wtsZ[np.abs(wtsZ) > Z_MAX] = (Z_MAX * np.sign(wtsZ))[np.abs(wtsZ) > Z_MAX]
        Z_maps[:, i_comp] = wtsZ
        norm_weights = wtsZ ** 2.

        # compute Kappa and Rho
        F_S0[F_S0 > F_MAX] = F_MAX
        F_R2[F_R2 > F_MAX] = F_MAX
        kappas[i_comp] = np.average(F_R2, weights=norm_weights)
        rhos[i_comp] = np.average(F_S0, weights=norm_weights)

    # tabulate component values
    comptab = np.vstack([kappas, rhos, varex, varex_norm]).T
    if reindex:
        # re-index all components in Kappa order
        sort_idx = comptab[:, 0].argsort()[::-1]
        comptab = comptab[sort_idx, :]
        mmix_new = mmix[:, sort_idx]
        betas = betas[..., sort_idx]
        pred_R2_maps = pred_R2_maps[:, :, sort_idx]
        pred_S0_maps = pred_S0_maps[:, :, sort_idx]
        F_S0_maps = F_S0_maps[:, sort_idx]
        F_R2_maps = F_R2_maps[:, sort_idx]
        Z_maps = Z_maps[:, sort_idx]
        WTS = WTS[:, sort_idx]
        PSC = PSC[:, sort_idx]
        tsoc_B = tsoc_B[:, sort_idx]
        tsoc_Babs = tsoc_Babs[:, sort_idx]
    else:
        mmix_new = mmix

    if verbose:
        # Echo-specific weight maps for each of the ICA components.
        io.filewrite(betas, op.join(out_dir, '{0}betas_catd.nii'.format(label)),
                     ref_img)
        # Echo-specific maps of predicted values for R2 and S0 models for each
        # component.
        io.filewrite(utils.unmask(pred_R2_maps, mask),
                     op.join(out_dir, '{0}R2_pred.nii'.format(label)), ref_img)
        io.filewrite(utils.unmask(pred_S0_maps, mask),
                     op.join(out_dir, '{0}S0_pred.nii'.format(label)), ref_img)
        # Weight maps used to average metrics across voxels
        io.filewrite(utils.unmask(Z_maps ** 2., mask),
                     op.join(out_dir, '{0}metric_weights.nii'.format(label)),
                     ref_img)

    comptab = pd.DataFrame(comptab,
                           columns=['kappa', 'rho',
                                    'variance explained',
                                    'normalized variance explained'])
    comptab.index.name = 'component'

    # full selection including clustering criteria
    seldict = None
    if full_sel:
        LGR.info('Performing spatial clustering of components')
        csize = np.max([int(n_voxels * 0.0005) + 5, 20])
        LGR.debug('Using minimum cluster size: {}'.format(csize))
        for i_comp in range(n_components):
            # save out files
            out = np.zeros((n_samp, 4))
            out[:, 0] = np.squeeze(utils.unmask(PSC[:, i_comp], mask))
            out[:, 1] = np.squeeze(utils.unmask(F_R2_maps[:, i_comp],
                                                t2s != 0))
            out[:, 2] = np.squeeze(utils.unmask(F_S0_maps[:, i_comp],
                                                t2s != 0))
            out[:, 3] = np.squeeze(utils.unmask(Z_maps[:, i_comp], mask))

            ccimg = io.new_nii_like(ref_img, out)

            # Do simple clustering on F
            sel = spatclust(ccimg, min_cluster_size=csize, threshold=fmin,
                            index=[1, 2], mask=(t2s != 0))
            F_R2_clmaps[:, i_comp] = sel[:, 0]
            F_S0_clmaps[:, i_comp] = sel[:, 1]
            countsigFR2 = F_R2_clmaps[:, i_comp].sum()
            countsigFS0 = F_S0_clmaps[:, i_comp].sum()

            # Do simple clustering on Z at p<0.05
            sel = spatclust(ccimg, min_cluster_size=csize, threshold=1.95,
                            index=3, mask=mask)
            Z_clmaps[:, i_comp] = sel

            # Do simple clustering on ranked signal-change map
            spclust_input = utils.unmask(stats.rankdata(tsoc_Babs[:, i_comp]),
                                         mask)
            spclust_input = io.new_nii_like(ref_img, spclust_input)
            Br_R2_clmaps[:, i_comp] = spatclust(
                spclust_input, min_cluster_size=csize,
                threshold=(max(tsoc_Babs.shape) - countsigFR2), mask=mask)
            Br_S0_clmaps[:, i_comp] = spatclust(
                spclust_input, min_cluster_size=csize,
                threshold=(max(tsoc_Babs.shape) - countsigFS0), mask=mask)

        seldict = {}
        selvars = ['WTS', 'tsoc_B', 'PSC',
                   'Z_maps', 'F_R2_maps', 'F_S0_maps',
                   'Z_clmaps', 'F_R2_clmaps', 'F_S0_clmaps',
                   'Br_R2_clmaps', 'Br_S0_clmaps']
        for vv in selvars:
            seldict[vv] = eval(vv)

    return seldict, comptab, betas, mmix_new


def get_coeffs(data, X, mask=None, add_const=False):
    """
    Performs least-squares fit of `X` against `data`

    Parameters
    ----------
    data : (S [x E] x T) array_like
        Array where `S` is samples, `E` is echoes, and `T` is time
    X : (T [x C]) array_like
        Array where `T` is time and `C` is predictor variables
    mask : (S [x E]) array_like
        Boolean mask array
    add_const : bool, optional
        Add intercept column to `X` before fitting. Default: False

    Returns
    -------
    betas : (S [x E] x C) :obj:`numpy.ndarray`
        Array of `S` sample betas for `C` predictors
    """
    if data.ndim not in [2, 3]:
        raise ValueError('Parameter data should be 2d or 3d, not {0}d'.format(data.ndim))
    elif X.ndim not in [2]:
        raise ValueError('Parameter X should be 2d, not {0}d'.format(X.ndim))
    elif data.shape[-1] != X.shape[0]:
        raise ValueError('Last dimension (dimension {0}) of data ({1}) does not '
                         'match first dimension of '
                         'X ({2})'.format(data.ndim, data.shape[-1], X.shape[0]))

    # mask data and flip (time x samples)
    if mask is not None:
        if mask.ndim not in [1, 2]:
            raise ValueError('Parameter data should be 1d or 2d, not {0}d'.format(mask.ndim))
        elif data.shape[0] != mask.shape[0]:
            raise ValueError('First dimensions of data ({0}) and mask ({1}) do not '
                             'match'.format(data.shape[0], mask.shape[0]))
        mdata = data[mask, :].T
    else:
        mdata = data.T

    # coerce X to >=2d
    X = np.atleast_2d(X)

    if len(X) == 1:
        X = X.T

    if add_const:  # add intercept, if specified
        X = np.column_stack([X, np.ones((len(X), 1))])

    betas = np.linalg.lstsq(X, mdata, rcond=None)[0].T
    if add_const:  # drop beta for intercept, if specified
        betas = betas[:, :-1]

    if mask is not None:
        betas = utils.unmask(betas, mask)

    return betas


def t_to_z(t_values, dof):
    """
    From Vanessa Sochat's TtoZ package.
    """
    # Select just the nonzero voxels
    nonzero = t_values[t_values != 0]

    # We will store our results here
    z_values = np.zeros(len(nonzero))

    # Select values less than or == 0, and greater than zero
    c = np.zeros(len(nonzero))
    k1 = (nonzero <= c)
    k2 = (nonzero > c)

    # Subset the data into two sets
    t1 = nonzero[k1]
    t2 = nonzero[k2]

    # Calculate p values for <=0
    p_values_t1 = stats.t.cdf(t1, df=dof)
    z_values_t1 = stats.norm.ppf(p_values_t1)
    z_values_t1[np.isinf(z_values_t1)] = stats.norm.ppf(1e-16)

    # Calculate p values for > 0
    p_values_t2 = stats.t.cdf(-t2, df=dof)
    z_values_t2 = -stats.norm.ppf(p_values_t2)
    z_values_t2[np.isinf(z_values_t2)] = -stats.norm.ppf(1e-16)
    z_values[k1] = z_values_t1
    z_values[k2] = z_values_t2

    # Write new image to file
    out = np.zeros(t_values.shape)
    out[t_values != 0] = z_values
    return out


def get_coeffs_and_zstats(data, mmix, mask, normalize=True):
    """
    Converts `data` to component space using `mmix`.

    Parameters
    ----------
    data : (S x T) array_like
        Input data
    mmix : (T [x C]) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`.
        1D mmix not currently supported.
    mask : (S,) array_like
        Boolean mask array
    normalize : bool, optional
        Whether to variance-normalize output z-value maps. Default: True

    Notes
    -----
    Calculation of t-statistics from https://stackoverflow.com/a/42677750/2589328
    """
    if data.ndim != 2:
        raise ValueError('Parameter data should be 2d, not {0}d'.format(data.ndim))
    elif mmix.ndim not in [2]:
        raise ValueError('Parameter mmix should be 2d, not {0}d'.format(mmix.ndim))
    elif mask.ndim != 1:
        raise ValueError('Parameter mask should be 1d, not {0}d'.format(mask.ndim))
    elif data.shape[0] != mask.shape[0]:
        raise ValueError('First dimensions (number of samples) of data ({0}) '
                         'and mask ({1}) do not match.'.format(data.shape[0],
                                                               mask.shape[0]))
    elif data.shape[1] != mmix.shape[0]:
        raise ValueError('Second dimensions (number of volumes) of data ({0}) '
                         'and mmix ({1}) do not match.'.format(data.shape[0],
                                                               mmix.shape[0]))
    assert mmix.ndim == 2
    if data.ndim != 2:
        data = data[None, :]

    assert mmix.shape[0] == data.shape[-1]
    data = data[mask].T
    mmix -= np.mean(mmix, axis=0)
    data -= np.mean(data, axis=0)

    slopes, _, _, _ = np.linalg.lstsq(mmix, data, rcond=None)
    df = float(mmix.shape[0] - (mmix.shape[1] + 1))

    # Override degrees of freedom if they're too low or negative
    if df < 1:
        df = 1
    pred = np.dot(mmix, slopes)

    # (N DVs, 1)
    MSE = np.sum((data - pred) ** 2, axis=0) / df
    MSE = np.atleast_2d(MSE).T

    # (N DVs, N IVs)
    var_b = np.dot(MSE,
                   np.atleast_2d(np.linalg.inv(np.dot(mmix.T, mmix)).diagonal()))
    sd_b = np.sqrt(var_b)
    slopes = slopes.T
    tstats = slopes / sd_b
    zstats = t_to_z(tstats, df)
    zstats[np.isnan(zstats)] = 0.
    return slopes, zstats


def gscontrol_raw(catd, optcom, n_echos, ref_img, dtrank=4):
    """
    Removes global signal from individual echo `catd` and `optcom` time series

    This function uses the spatial global signal estimation approach to
    to removal global signal out of individual echo time series datasets. The
    spatial global signal is estimated from the optimally combined data after
    detrending with a Legendre polynomial basis of `order = 0` and
    `degree = dtrank`.

    Parameters
    ----------
    catd : (S x E x T) array_like
        Input functional data
    optcom : (S x T) array_like
        Optimally combined functional data (i.e., the output of `make_optcom`)
    n_echos : :obj:`int`
        Number of echos in data. Should be the same as `E` dimension of `catd`
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk
    dtrank : :obj:`int`, optional
        Specifies degree of Legendre polynomial basis function for estimating
        spatial global signal. Default: 4

    Returns
    -------
    dm_catd : (S x E x T) array_like
        Input `catd` with global signal removed from time series
    dm_optcom : (S x T) array_like
        Input `optcom` with global signal removed from time series
    """
    LGR.info('Applying amplitude-based T1 equilibration correction')
    if catd.shape[0] != optcom.shape[0]:
        raise ValueError('First dimensions of catd ({0}) and optcom ({1}) do not '
                         'match'.format(catd.shape[0], optcom.shape[0]))
    elif catd.shape[1] != n_echos:
        raise ValueError('Second dimension of catd ({0}) does not match '
                         'n_echos ({1})'.format(catd.shape[1], n_echos))
    elif catd.shape[2] != optcom.shape[1]:
        raise ValueError('Third dimension of catd ({0}) does not match '
                         'second dimension of optcom '
                         '({1})'.format(catd.shape[2], optcom.shape[1]))

    # Legendre polynomial basis for denoising
    bounds = np.linspace(-1, 1, optcom.shape[-1])
    Lmix = np.column_stack([lpmv(0, vv, bounds) for vv in range(dtrank)])

    # compute mean, std, mask local to this function
    # inefficient, but makes this function a bit more modular
    Gmu = optcom.mean(axis=-1)  # temporal mean
    Gmask = Gmu != 0

    # find spatial global signal
    dat = optcom[Gmask] - Gmu[Gmask][:, np.newaxis]
    sol = np.linalg.lstsq(Lmix, dat.T, rcond=None)[0]  # Legendre basis for detrending
    detr = dat - np.dot(sol.T, Lmix.T)[0]
    sphis = (detr).min(axis=1)
    sphis -= sphis.mean()
    io.filewrite(utils.unmask(sphis, Gmask), 'T1gs', ref_img)

    # find time course ofc the spatial global signal
    # make basis with the Legendre basis
    glsig = np.linalg.lstsq(np.atleast_2d(sphis).T, dat, rcond=None)[0]
    glsig = stats.zscore(glsig, axis=None)
    np.savetxt('glsig.1D', glsig)
    glbase = np.hstack([Lmix, glsig.T])

    # Project global signal out of optimally combined data
    sol = np.linalg.lstsq(np.atleast_2d(glbase), dat.T, rcond=None)[0]
    tsoc_nogs = dat - np.dot(np.atleast_2d(sol[dtrank]).T,
                             np.atleast_2d(glbase.T[dtrank])) + Gmu[Gmask][:, np.newaxis]

    io.filewrite(optcom, 'tsoc_orig', ref_img)
    dm_optcom = utils.unmask(tsoc_nogs, Gmask)
    io.filewrite(dm_optcom, 'tsoc_nogs', ref_img)

    # Project glbase out of each echo
    dm_catd = catd.copy()  # don't overwrite catd
    for echo in range(n_echos):
        dat = dm_catd[:, echo, :][Gmask]
        sol = np.linalg.lstsq(np.atleast_2d(glbase), dat.T, rcond=None)[0]
        e_nogs = dat - np.dot(np.atleast_2d(sol[dtrank]).T,
                              np.atleast_2d(glbase.T[dtrank]))
        dm_catd[:, echo, :] = utils.unmask(e_nogs, Gmask)

    return dm_catd, dm_optcom


def spatclust(img, min_cluster_size, threshold=None, index=None, mask=None):
    """
    Spatially clusters `img`

    Parameters
    ----------
    img : str or img_like
        Image file or object to be clustered
    min_cluster_size : int
        Minimum cluster size (in voxels)
    threshold : float, optional
        Whether to threshold `img` before clustering
    index : array_like, optional
        Whether to extract volumes from `img` for clustering
    mask : (S,) array_like, optional
        Boolean array for masking resultant data array

    Returns
    -------
    clustered : :obj:`numpy.ndarray`
        Binarized array (values are 0 or 1) of clustered (and thresholded)
        `img` data
    """

    # we need a 4D image for `niimg.iter_img`, below
    img = niimg.copy_img(check_niimg(img, atleast_4d=True))

    # temporarily set voxel sizes to 1mm isotropic so that `min_cluster_size`
    # represents the minimum number of voxels we want to be in a cluster,
    # rather than the minimum size of the desired clusters in mm^3
    if not np.all(np.abs(np.diag(img.affine)) == 1):
        img.set_sform(np.sign(img.affine))

    # grab desired volumes from provided image
    if index is not None:
        if not isinstance(index, list):
            index = [index]
        img = niimg.index_img(img, index)

    # threshold image
    if threshold is not None:
        img = niimg.threshold_img(img, float(threshold))

    clout = []
    for subbrick in niimg.iter_img(img):
        # `min_region_size` is not inclusive (as in AFNI's `3dmerge`)
        # subtract one voxel to ensure we aren't hitting this thresholding issue
        try:
            clsts = connected_regions(subbrick,
                                      min_region_size=int(min_cluster_size) - 1,
                                      smoothing_fwhm=None,
                                      extract_type='connected_components')[0]
        # if no clusters are detected we get a TypeError; create a blank 4D
        # image object as a placeholder instead
        except TypeError:
            clsts = niimg.new_img_like(subbrick,
                                       np.zeros(subbrick.shape + (1,)))
        # if multiple clusters detected, collapse into one volume
        clout += [niimg.math_img('np.sum(a, axis=-1)', a=clsts)]

    # convert back to data array and make boolean
    clustered = utils.load_image(niimg.concat_imgs(clout).get_data()) != 0

    # if mask provided, mask output
    if mask is not None:
        clustered = clustered[mask]

    return clustered
