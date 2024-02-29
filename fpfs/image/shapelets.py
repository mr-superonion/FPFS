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
import jax.numpy as jnp
import numpy as np
from . import util


def get_shapelets_col_names(nord):
    # M_{nm}
    # nm = n*(nord+1)+m
    if nord == 4:
        # This setup is for shear response only
        # Only uses M00, M20, M22 (real and img) and M40, M42 (real and img)
        name_s = [
            "m00",
            "m20",
            "m22c",
            "m22s",
            "m40",
            "m42c",
            "m42s",
            "m44c",
            "m44s",
        ]
        ind_s = [
            [0, False],
            [10, False],
            [12, False],
            [12, True],
            [20, False],
            [22, False],
            [22, True],
            [24, False],
            [24, True],
        ]
    elif nord == 6:
        # This setup is able to derive kappa response and shear response
        # Only uses M00, M20, M22 (real and img), M40, M42(real and img), M60
        name_s = [
            "m00",
            "m20",
            "m22c",
            "m22s",
            "m40",
            "m42c",
            "m42s",
            "m44c",
            "m44s",
            "m60",
        ]
        ind_s = [
            [0, False],
            [14, False],
            [16, False],
            [16, True],
            [28, False],
            [30, False],
            [30, True],
            [32, False],
            [32, True],
            [42, False],
        ]
    else:
        raise ValueError(
            "only support for nord= 4 or nord=6, but your input\
                is nord=%d"
            % nord
        )
    return name_s, ind_s


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
    xtfunc = np.zeros((ny, nx))
    ytfunc = np.zeros((ny, nx))
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
    name_s, ind_s = get_shapelets_col_names(nord)
    # generate the complex shaplet functions
    chi = shapelets2d(ngrid, nord, sigma, klim)
    # transform to real shapelet functions
    chi_2 = np.zeros((len(name_s), ngrid, ngrid))
    for i, ind in enumerate(ind_s):
        if ind[1]:
            chi_2[i] = chi[ind[0]].imag
        else:
            chi_2[i] = chi[ind[0]].real
    del chi
    return jnp.array(chi_2), name_s
