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

import numpy as np
import jax.numpy as jnp

from . import util


def get_det_col_names():
    detect_nrot = 8
    name_d = []
    for irot in range(detect_nrot):
        name_d.append("v%d" % irot)
    for irot in range(detect_nrot):
        name_d.append("v%dr1" % irot)
    for irot in range(detect_nrot):
        name_d.append("v%dr2" % irot)
    return name_d


def detlets2d(ngrid, sigma, klim):
    """Generates shapelets function in Fourier space, chi00 are normalized to 1.
    This function only supports square stamps: ny=nx=ngrid.

    Args:
    ngrid (int):    number of pixels in x and y direction
    sigma (float):  radius of shapelets in Fourier space
    klim (float):   upper limit of |k|

    Returns:
    psi (ndarray):  2d detlets basis in shape of [8,3,ngrid,ngrid]
    """
    # Gaussian Kernel
    gauss_ker, (k2grid, k1grid) = util.gauss_kernel_fft(
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
    psi = np.zeros((3, 8, ny, nx), dtype=np.complex64)
    for irot in range(8):
        x = np.cos(2.0 * np.pi / 8 * irot)
        y = np.sin(2.0 * np.pi / 8 * irot)
        foub = np.exp(1j * (k1grid * x + k2grid * y))
        psi[0, irot] = gauss_ker - gauss_ker * foub
        psi[1, irot] = q1_ker - (q1_ker + x * d1_ker - y * d2_ker) * foub
        psi[2, irot] = q2_ker - (q2_ker + y * d1_ker + x * d2_ker) * foub
    psi = jnp.vstack(psi)
    name_d = get_det_col_names()
    return psi, name_d


def detect_thres(imgf_use, thres, ny, nx, sigmaf, klim):
    # Gaussian kernel for shapelets
    gauss_kernel, (kygrids, kxgrids) = util.gauss_kernel_rfft(
        ny,
        nx,
        sigmaf,
        klim,
        return_grid=True,
    )
    # convolved images
    img_conv = jnp.fft.irfft2(imgf_use * gauss_kernel, (ny, nx))
    img_conv2 = jnp.fft.irfft2(
        imgf_use
        * gauss_kernel
        * (1.0 - (kxgrids**2.0 + kygrids**2.0) / sigmaf**2.0),
        (ny, nx),
    )
    sel = jnp.logical_and((img_conv > thres), ((img_conv + img_conv2) > 0.0))
    return sel


def detect_max(imgf_use, thres2, ny, nx, sigmaf_det, klim):
    gauss_kernel, (kygrids, kxgrids) = util.gauss_kernel_rfft(
        ny,
        nx,
        sigmaf_det,
        klim,
        return_grid=True,
    )
    sel = jnp.ones((ny, nx), dtype=bool)
    for irot in range(8):
        x = jnp.cos(2.0 * jnp.pi / 8 * irot)
        y = jnp.sin(2.0 * jnp.pi / 8 * irot)
        bb = (1.0 - jnp.exp(1j * (kxgrids * x + kygrids * y))) * gauss_kernel
        img_r = jnp.fft.irfft2(imgf_use * bb, (ny, nx))
        sel = jnp.logical_and(sel, (img_r > thres2))
    return sel