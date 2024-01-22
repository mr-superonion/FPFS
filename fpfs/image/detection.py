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
from . import util


def detlets2d(ngrid, detect_nrot, sigma, klim):
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
    psi = np.zeros((3, detect_nrot, ny, nx), dtype=np.complex64)
    for irot in range(detect_nrot):
        x = np.cos(2.0 * np.pi / detect_nrot * irot)
        y = np.sin(2.0 * np.pi / detect_nrot * irot)
        foub = np.exp(1j * (k1grid * x + k2grid * y))
        psi[0, irot] = gauss_ker - gauss_ker * foub
        psi[1, irot] = q1_ker - (q1_ker + x * d1_ker - y * d2_ker) * foub
        psi[2, irot] = q2_ker - (q2_ker + y * d1_ker + x * d2_ker) * foub

    name_d = []
    for irot in range(detect_nrot):
        name_d.append("v%d" % irot)
    for irot in range(detect_nrot):
        name_d.append("v%dr1" % irot)
    for irot in range(detect_nrot):
        name_d.append("v%dr2" % irot)
    return psi, name_d
