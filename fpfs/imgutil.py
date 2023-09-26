# FPFS shear estimator
# Copyright 20210905 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# python lib

import jax
import math
import numpy as np
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=["ny", "nx", "klim", "return_grid"])
def _gauss_kernel_fft(ny, nx, sigma, klim, return_grid=False):
    """Generates a Gaussian kernel on grids for np.fft.fft transform
    (we always shift k=0 to (ngird//2, ngird//2)). The kernel is truncated at
    radius klim.

    Args:
        ny (int):    		    grid size in y-direction
        nx (int):    		    grid size in x-direction
        sigma (float):		    scale of Gaussian in Fourier space
        klim (float):           upper limit of k
        return_grid (bool):     return grids [True] or not [Flase]
                                [default: False]
    Returns:
        out (ndarray):          Gaussian on grids
        xgrid,ygrid (typle):    grids for [y, x] axes if return_grid
    """
    # mask
    x = jnp.fft.fftshift(jnp.fft.fftfreq(nx, 1 / np.pi / 2.0))
    y = jnp.fft.fftshift(jnp.fft.fftfreq(ny, 1 / np.pi / 2.0))
    ygrid, xgrid = jnp.meshgrid(y, x, indexing="ij")
    r2 = xgrid**2.0 + ygrid**2.0
    mask = (r2 <= klim**2).astype(jnp.float64)
    out = jnp.exp(-r2 / 2.0 / sigma**2.0) * mask
    if not return_grid:
        return out
    else:
        return out, (ygrid, xgrid)


@partial(jax.jit, static_argnames=["ny", "nx", "klim", "return_grid"])
def _gauss_kernel_rfft(ny, nx, sigma, klim, return_grid=False):
    """Generates a Gaussian kernel on grids for np.fft.rfft transform
    The kernel is truncated at radius klim.

    Args:
        ny (int):    		    grid size in y-direction
        nx (int):    		    grid size in x-direction
        sigma (float):		    scale of Gaussian in Fourier space
        klim (float):           upper limit of k
        return_grid (bool):     return grids or not
    Returns:
        out (ndarray):          Gaussian on grids
        ygrid, xgrid (typle):   grids for [y, x] axes, if return_grid
    """
    x = jnp.fft.rfftfreq(nx, 1 / np.pi / 2.0)
    y = jnp.fft.fftfreq(ny, 1 / np.pi / 2.0)
    ygrid, xgrid = jnp.meshgrid(y, x, indexing="ij")
    r2 = xgrid**2.0 + ygrid**2.0
    mask = (r2 <= klim**2).astype(jnp.float64)
    out = jnp.exp(-r2 / 2.0 / sigma**2.0) * mask
    if not return_grid:
        return out
    else:
        return out, (ygrid, xgrid)


@jax.jit
def get_fourier_pow_fft(input_data):
    """Gets Fourier power function

    Args:
        input_data (ndarray):  image array, centroid does not matter.
    Returns:
        out (ndarray):      Fourier Power
    """
    out = (jnp.abs(jnp.fft.fft2(input_data)) ** 2.0).astype(jnp.float64)
    out = jnp.fft.fftshift(out)
    return out


@jax.jit
def get_fourier_pow_rfft(input_data):
    """Gets Fourier power function

    Args:
        input_data (ndarray):  image array. The centroid does not matter.
    Returns:
        galpow (ndarray):   Fourier Power
    """

    out = (jnp.abs(jnp.fft.rfft2(input_data)) ** 2.0).astype(jnp.float64)
    return out


def detlets2d(ngrid, sigma, klim):
    """Generates shapelets function in Fourier space, chi00 are normalized to 1.
    This function only supports square stamps: ny=nx=ngrid.

    Args:
        ngrid (int):    number of pixels in x and y direction
        sigma (float):  scale of shapelets in Fourier space
        klim (float):   upper limit of |k|
    Returns:
        psi (ndarray):  2d detlets basis in shape of [8,3,ngrid,ngrid]
    """
    # Gaussian Kernel
    gauss_ker, (k2grid, k1grid) = _gauss_kernel_fft(
        ngrid, ngrid, sigma, klim, return_grid=True
    )
    # for inverse Fourier transform
    gauss_ker = gauss_ker / ngrid**2.0
    # for shear response
    q1_ker = (k1grid**2.0 - k2grid**2.0) / sigma**2.0 * gauss_ker
    q2_ker = (2.0 * k1grid * k2grid) / sigma**2.0 * gauss_ker
    # quantities for neighbouring pixels
    d1_ker = (-1j * k1grid) * gauss_ker
    d2_ker = (-1j * k2grid) * gauss_ker
    # initial output psi function
    ny, nx = gauss_ker.shape
    psi = np.zeros((8, 3, ny, nx), dtype=np.complex64)
    for _ in range(8):
        x = np.cos(np.pi / 4.0 * _)
        y = np.sin(np.pi / 4.0 * _)
        foub = np.exp(1j * (k1grid * x + k2grid * y))
        psi[_, 0] = gauss_ker - gauss_ker * foub
        psi[_, 1] = q1_ker - (q1_ker + x * d1_ker - y * d2_ker) * foub
        psi[_, 2] = q2_ker - (q2_ker + y * d1_ker + x * d2_ker) * foub
    return psi


def shapelets2d(ngrid, nord, sigma, klim):
    """Generates complex shapelets function in Fourier space, chi00 are
    normalized to 1
    [only support square stamps: ny=nx=ngrid]

    Args:
        ngrid (int):    number of pixels in x and y direction
        nord (int):     radial order of the shaplets
        sigma (float):  scale of shapelets in Fourier space
        klim (float):   upper limit of |k|
    Returns:
        chi (ndarray):  2d shapelet basis
    """

    mord = nord
    gaufunc, (yfunc, xfunc) = _gauss_kernel_fft(
        ngrid, ngrid, sigma, klim, return_grid=True
    )
    rfunc = np.sqrt(xfunc**2.0 + yfunc**2.0)  # radius
    r2_over_sigma2 = (rfunc / sigma) ** 2.0
    ny, nx = gaufunc.shape

    rmask = rfunc != 0.0
    xtfunc = np.zeros((ny, nx), dtype=np.float64)
    ytfunc = np.zeros((ny, nx), dtype=np.float64)
    np.divide(xfunc, rfunc, where=rmask, out=xtfunc)  # cos(phi)
    np.divide(yfunc, rfunc, where=rmask, out=ytfunc)  # sin(phi)
    eulfunc = xtfunc + 1j * ytfunc  # e^{jphi}
    # Set up Laguerre function
    lfunc = np.zeros((nord + 1, mord + 1, ny, nx), dtype=np.float64)
    lfunc[0, :, :, :] = 1.0
    lfunc[1, :, :, :] = 1.0 - r2_over_sigma2 + np.arange(mord + 1)[None, :, None, None]
    #
    chi = np.zeros((nord + 1, mord + 1, ny, nx), dtype=np.complex64)
    for n in range(2, nord + 1):
        for m in range(mord + 1):
            lfunc[n, m, :, :] = (2.0 + (m - 1.0 - r2_over_sigma2) / n) * lfunc[
                n - 1, m, :, :
            ] - (1.0 + (m - 1.0) / n) * lfunc[n - 2, m, :, :]
    for nn in range(nord + 1):
        for mm in range(nn, -1, -2):
            c1 = (nn - abs(mm)) // 2
            d1 = (nn + abs(mm)) // 2
            cc = math.factorial(c1) + 0.0
            dd = math.factorial(d1) + 0.0
            cc = cc / dd
            chi[nn, mm, :, :] = (
                pow(-1.0, d1)
                * pow(cc, 0.5)
                * lfunc[c1, abs(mm), :, :]
                * pow(r2_over_sigma2, abs(mm) / 2)
                * gaufunc
                * eulfunc**mm
                * (1j) ** nn
            )
    chi = chi.reshape(((nord + 1) ** 2, ny, nx)) / ngrid**2.0
    return chi


def shapelets2d_real(ngrid, nord, sigma, klim):
    """Generates real shapelets function in Fourier space, chi00 are
    normalized to 1
    [only support square stamps: ny=nx=ngrid]

    Args:
        ngrid (int):    number of pixels in x and y direction
        nord (int):     radial order of the shaplets
        sigma (float):  scale of shapelets in Fourier space
        klim (float):   upper limit of |k|
    Returns:
        chi_2 (ndarray): 2d shapelet basis w/ shape [n,ngrid,ngrid]
        name_s (list):   A list of shaplet names w/ shape [n]

    """
    # nm = m*(nnord+1)+n
    if nord == 4:
        # This setup is for shear response only
        # Only uses M00, M20, M22 (real and img) and M40, M42
        indm = np.array([0, 10, 12, 20, 22])[:, None, None]
        name_s = ["m00", "m20", "m22c", "m22s", "m40", "m42c", "m42s"]
        ind_s = [
            [0, False],
            [1, False],
            [2, False],
            [2, True],
            [3, False],
            [4, False],
            [4, True],
        ]
    elif nord == 6:
        # This setup is able to derive kappa response and shear response
        # Only uses M00, M20, M22 (real and img), M40, M42(real and img), M60
        indm = np.array([0, 14, 16, 28, 30, 42])[:, None, None]
        name_s = ["m00", "m20", "m22c", "m22s", "m40", "m42c", "m42s", "m60"]
        ind_s = [
            [0, False],
            [1, False],
            [2, False],
            [2, True],
            [3, False],
            [4, False],
            [4, True],
            [5, False],
        ]
    else:
        raise ValueError(
            "only support for nnord= 4 or nnord=6, but your input\
                is nnord=%d"
            % nord
        )
    # generate the complex shaplet functions
    chi = shapelets2d(ngrid, nord, sigma, klim)[indm]
    # transform to real shapelet functions
    chi_2 = np.zeros((len(name_s), ngrid, ngrid), dtype=np.float64)
    for i, ind in enumerate(ind_s):
        if ind[1]:
            chi_2[i] = np.float64(chi[ind[0]].imag)
        else:
            chi_2[i] = np.float64(chi[ind[0]].real)
    del chi
    return chi_2, name_s


def fpfs_bases(ngrid, nord, sigma, sigma_det=None, klim=3.15):
    """Returns the FPFS bases (shapelets and detectlets)

    Args:
        ngrid (int):            stamp size
        nnord (int):            the highest order of Shapelets radial
                                components [default: 4]
        sigma (float):          shapelet kernel scale in Fourier space
        sigma_det (float):      detectlet kernel scale in Fourier space
        klim (float):           upper limit of |k| [default 3.15]
    """
    if sigma_det is None:
        sigma_det = sigma
    bfunc, bnames = shapelets2d_real(
        ngrid,
        nord,
        sigma,
        klim,
    )
    psi = detlets2d(
        ngrid,
        sigma_det,
        klim,
    )
    bnames = bnames + [
        "v0",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v0_g1",
        "v1_g1",
        "v2_g1",
        "v3_g1",
        "v4_g1",
        "v5_g1",
        "v6_g1",
        "v7_g1",
        "v0_g2",
        "v1_g2",
        "v2_g2",
        "v3_g2",
        "v4_g2",
        "v5_g2",
        "v6_g2",
        "v7_g2",
    ]
    bfunc = np.vstack([bfunc, np.vstack(np.swapaxes(psi, 0, 1))])
    return bfunc, bnames


def fit_noise_pf(ngrid, gal_pow, noise_mod, rlim):
    """
    Fit the noise power from observed galaxy power

    Args:
        ngrid (int):      number of pixels in x and y direction
        gal_pow (ndarray): galaxy Fourier power function
    Returns:
        out (ndarray): noise power to be subtracted
    """

    rlim2 = int(max(ngrid * 0.4, rlim))
    indx = np.arange(ngrid // 2 - rlim2, ngrid // 2 + rlim2 + 1)
    indy = indx[:, None]
    mask = np.ones((ngrid, ngrid), dtype=bool)
    mask[indy, indx] = False
    vl = gal_pow[mask]
    nl = noise_mod[:, mask]
    par = np.linalg.lstsq(nl.T, vl, rcond=None)[0]
    out = np.sum(par[:, None, None] * noise_mod, axis=0)
    return out


def pcaimages(xdata, nmodes):
    """Estimates the principal components of array list xdata

    Args:
        xdata (ndarray):        input data array
        nmodes (int):           number of pcs to keep
    Returns:
        out (ndarray):          pc images,
        stds (ndarray):         stds on the axis
        coeffs (ndarray):       projection coefficient
    """

    assert len(xdata.shape) == 3
    # vectorize
    nobj, nn2, nn1 = xdata.shape
    dim = nn1 * nn2
    # xdata is (x1,x2,x3..,xnobj).T [x_i is column vectors of data]
    xdata = xdata.reshape((nobj, dim))
    # x_ave  = xdata.mean(axis=0)
    # xdata     = xdata-x_ave
    # x_ave  = x_ave.reshape((1,nn2,nn1))

    # Get covariance matrix
    data_mat = np.dot(xdata, xdata.T) / (nobj - 1)
    # Solve the Eigen function of the covariance matrix
    # e is eigen value and eig_vec is eigen vector
    # eig_vec: (p1, p2, .., pnobj) [p_i is column vectors of parameters]
    eig_val, eig_vec = np.linalg.eigh(data_mat)
    # The eigen vector tells the combination of ndata
    tmp = np.dot(eig_vec.T, xdata)
    # rank from maximum eigen value to minimum
    # and only keep the first nmodes
    pcs = tmp[::-1][:nmodes]
    eig_val = eig_val[::-1][: nmodes + 10]
    stds = np.sqrt(eig_val)
    out = pcs.reshape((nmodes, nn2, nn1))
    coeffs = eig_vec.T[:nmodes]
    return out, stds, coeffs


def cut_img(img, rcut):
    """Cuts img into postage stamp with width=2rcut

    Args:
        img (ndarray):  input image
        rcut (int):     cutout radius
    Returns:
        out (ndarray):  image in a stamp
    """
    ngrid = img.shape[0]
    beg = ngrid // 2 - rcut
    end = beg + 2 * rcut
    out = img[beg:end, beg:end]
    return out


@jax.jit
def get_pixel_detect_mask(sel, img, thres2):
    for ax in [-1, -2]:
        for shift in [-1, 1]:
            filtered = img - jnp.roll(img, shift=shift, axis=ax)
            sel = jnp.logical_and(sel, (filtered > thres2))
    return sel


def find_peaks(img_conv, img_conv_det, thres, thres2=0.0, bound=20.0):
    """Detects peaks and returns the coordinates (y,x)
    This function does the pre-selection in Li & Mandelbaum (2023)

    Args:
        img_conv (ndarray):         convolved image
        img_conv_det (ndarray):     convolved image
        thres (float):              detection threshold
        thres2 (float):             peak identification difference threshold
        bound (float):              minimum distance to the image boundary
    Returns:
        coord_array (ndarray):      ndarray of coordinates [y,x]
    """
    sel = img_conv > thres
    sel = get_pixel_detect_mask(sel, img_conv_det, thres2)
    data = jnp.array(jnp.int_(jnp.asarray(jnp.where(sel))))
    del sel
    ny, nx = img_conv.shape
    y = data[0]
    x = data[1]
    msk = (y > bound) & (y < ny - bound) & (x > bound) & (x < nx - bound)
    data = data[:, msk]
    return data


@jax.jit
def convolve2gausspsf(img_data, psf_data, sigmaf, klim):
    """This function convolves an image to transform the PSF to a Gaussian

    Args:
        img_data (ndarray):     image data
        psf_data (ndarray):     psf data
        sigmaf (float):         sigma of Gaussian
        klim (float):           radius for masking in Fourier space

    Returns:
        img_conv (ndarray):     the reconvolved image
    """

    ny, nx = psf_data.shape
    # Fourier transform
    psf_fourier = jnp.fft.rfft2(jnp.fft.ifftshift(psf_data))
    # Gaussian kernel
    gauss_kernel = _gauss_kernel_rfft(ny, nx, sigmaf, klim, return_grid=False)
    # convolved images
    img_fourier = jnp.fft.rfft2(img_data) / psf_fourier * gauss_kernel
    img_conv = jnp.fft.irfft2(img_fourier, (ny, nx))
    return img_conv


@jax.jit
def get_klim(psf_array, sigma, thres=1e-20):
    """Gets klim, the region outside klim is supressed by the shaplet Gaussian
    kernel in FPFS shear estimation method; therefore we set values in this
    region to zeros

    Args:
        psf_array (ndarray):    PSF's Fourier power or Fourier transform
        sigma (float):          one sigma of Gaussian Fourier power
        thres (float):          the threshold for a tuncation on Gaussian
                                [default: 1e-20]
    Returns:
        klim (float):           the limit radius
    """
    ngrid = psf_array.shape[0]

    def cond_fun(dist):
        v1 = abs(
            jnp.exp(-(dist**2.0) / 2.0 / sigma**2.0)
            / psf_array[ngrid // 2 + dist, ngrid // 2]
        )
        v2 = abs(
            jnp.exp(-(dist**2.0) / 2.0 / sigma**2.0)
            / psf_array[ngrid // 2, ngrid // 2 + dist]
        )
        return jax.lax.cond(
            v1 < v2,
            v1,
            lambda x: x > thres,
            v2,
            lambda x: x > thres,
        )

    def body_fun(dist):
        return dist + 1

    klim = jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=ngrid // 5,
    )
    return klim


def truncate_square(arr, rcut):
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input array must be a 2D square array")

    ngrid = arr.shape[0]
    arr[: ngrid // 2 - rcut, :] = 0
    arr[ngrid // 2 + rcut :, :] = 0
    arr[:, : ngrid // 2 - rcut] = 0
    arr[:, ngrid // 2 + rcut :] = 0
    return


def truncate_circle(arr, rcut):
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input array must be a 2D square array")
    ngrid = arr.shape[0]
    y, x = np.ogrid[0:ngrid, 0:ngrid]
    center_x, center_y = ngrid // 2, ngrid // 2
    # Compute the squared distance to the center
    distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2
    # Mask values outside the circle
    arr[distance_squared > rcut**2] = 0.0
    return
