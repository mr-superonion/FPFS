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

import math
import numpy as np
from . import util


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
    gaufunc, (yfunc, xfunc) = util.gauss_kernel_fft(
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
    # Set up Laguerre polynomials
    lfunc = np.zeros((nord + 1, mord + 1, ny, nx), dtype=np.float32)
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
    name_s (list):   A list of shaplet names
    """
    # M_{nm}
    # nm = n*(nnord+1)+m
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
