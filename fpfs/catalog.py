# FPFS shear estimator
# Copyright 20210805 Xiangchong Li.
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
import numpy as np
import jax.numpy as jnp

import fitsio
import numpy.lib.recfunctions as rfn

from .image.image import fpfs_base


def read_catalog(fname):
    x = fitsio.read(fname)
    if x.dtype.names is not None:
        x = rfn.structured_to_unstructured(
            x,
            dtype=np.float64,
            copy=True,
        )
    return jnp.array(x)


def ssfunc1(x, mu, sigma):
    """Returns the C2 smooth step weight funciton

    Args:
    x (ndarray):    input data vector
    mu (float):     center of the cut
    sigma (float):  half width of the selection function

    Returns:
    out (ndarray):  the weight funciton
    """

    def _func(t):
        return -2.0 * t**3.0 + 3 * t**2.0

    t = (x - mu) / sigma / 2.0 + 0.5
    return jnp.piecewise(t, [t < 0, (t >= 0) & (t <= 1), t > 1], [0.0, _func, 1.0])


def ssfunc2(x, mu, sigma):
    """Returns the C2 smooth step weight funciton

    Args:
    x (ndarray):    input data vector
    mu (float):     center of the cut
    sigma (float):  half width of the selection function

    Returns:
    out (ndarray):  the weight funciton
    """

    def _func(t):
        return 6 * t**5.0 - 15 * t**4.0 + 10 * t**3.0

    t = (x - mu) / sigma / 2.0 + 0.5
    return jnp.piecewise(t, [t < 0, (t >= 0) & (t <= 1), t > 1], [0.0, _func, 1.0])


def ssfunc3(x, mu, sigma):
    """Returns the C2 smooth step weight funciton

    Args:
    x (ndarray):    input data vector
    mu (float):     center of the cut
    sigma (float):  half width of the selection function

    Returns:
    out (ndarray):  the weight funciton
    """

    def _func(t):
        return -20 * t**7.0 + 70 * t**6.0 - 84 * t**5.0 + 35 * t**4.0

    t = (x - mu) / sigma / 2.0 + 0.5
    return jnp.piecewise(t, [t < 0, (t >= 0) & (t <= 1), t > 1], [0.0, _func, 1.0])


def sbfunc1(x, mu, sigma):
    """Returns the C2 smooth bump weight funciton

    Args:
    x (ndarray):    input data vector
    mu (float):     center of the cut
    sigma (float):  half width of the selection function

    Returns:
    out (ndarray):  the weight funciton
    """

    def _func(t):
        return (1 - t**2.0) ** 2.0

    t = (x - mu) / sigma
    return jnp.piecewise(t, [t < -1, (t >= -1) & (t <= 1), t > 1], [0.0, _func, 0.0])


def sigmoid(x, mu, sigma):
    """Returns the sigmoid step weight funciton

    Args:
    x (ndarray):    input data vector
    mu (float):     center shift
    sigma (float):  half width of the selection function

    Returns:
    out (ndarray):  the weight funciton
    """
    t = x / sigma - mu
    # mu is not divided by sigma so that f(0) is independent of sigma
    return jax.nn.sigmoid(t)


def gaussian(x, mu, sigma):
    """Returns the normalized Gaussian function value at x.

    Args:
    x (array):      Points at which to evaluate the Gaussian function.
    mu (float):     Mean of the Gaussian distribution.
    sigma (float):  Standard deviation of the Gaussian distribution.

    Returns:
    array:          The values of the Gaussian function at each point in x.
    """
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return jnp.exp(exponent)


def cosh_inv(x, mu, sigma):
    """Returns the 1/cosh function value at x.

    Args:
    x (array):      Points at which to evaluate the Gaussian function.
    mu (float):     Mean of the Gaussian distribution.
    sigma (float):  Standard deviation of the Gaussian distribution.

    Returns:
    array:          The values of the 1/cosh function at each point in x.
    """
    t = x / sigma - mu
    return 1.0 / jnp.cosh(t)


class catalog_base(fpfs_base):
    def __init__(
        self,
        ratio: float = 1.81,
        snr_min: float = 12.0,
        r2_min: float = 0.05,
        r2_max: float = 2.0,
        c0: float = 2.55,
        c2: float = 25.6,
        alpha: float = 0.27,
        beta: float = 0.83,
        pthres: float = 0.0,
        pratio: float = 0.02,
        cov_mat=None,
        sigma_m00: float | None = None,
        sigma_r2: float | None = None,
        sigma_v: float | None = None,
        nord: int = 4,
        det_nrot: int = 8,
    ):
        super().__init__(
            nord=nord,
            det_nrot=det_nrot,
        )
        if cov_mat is None:
            # cov_mat = jnp.eye(self.ncol)
            cov_mat = jnp.zeros((self.ncol, self.ncol))
        self.cov_mat = cov_mat
        std_modes = np.sqrt(np.diagonal(cov_mat))
        std_m00 = std_modes[self.di["m00"]]
        std_m20 = np.sqrt(
            cov_mat[self.di["m00"], self.di["m00"]]
            + cov_mat[self.di["m20"], self.di["m20"]]
            + cov_mat[self.di["m00"], self.di["m20"]]
            + cov_mat[self.di["m20"], self.di["m00"]]
        )
        std_v = jnp.average(
            jnp.array([std_modes[self.di["v%d" % _]] for _ in range(det_nrot)])
        )

        # control steepness
        if sigma_m00 is None:
            self.sigma_m00 = ratio * std_m00
        else:
            self.sigma_m00 = sigma_m00
        if sigma_r2 is None:
            self.sigma_r2 = ratio * std_m20
        else:
            self.sigma_r2 = sigma_r2
        if sigma_v is None:
            self.sigma_v = ratio * std_v
        else:
            self.sigma_v = sigma_v

        # selection thresholds
        self.m00_min = snr_min * std_m00
        self.r2_min = r2_min
        self.r2_max = r2_max

        # shape parameters
        self.C0 = c0 * std_m00
        self.C2 = c2 * std_m20
        self.alpha = alpha
        self.beta = beta

        # detection threshold
        self.pcut = pthres * std_v
        self.pratio = pratio
        return

    def _wsel(self, x):
        # selection on flux
        w0l = ssfunc2(x[self.di["m00"]], self.m00_min, self.sigma_m00)

        # selection on size (lower limit)
        # (M00 + M20) / M00 > r2_min
        r2l = x[self.di["m00"]] * (1.0 - self.r2_min) + x[self.di["m20"]]
        w2l = ssfunc2(r2l, self.sigma_r2, self.sigma_r2)
        # r2_2 = (x[self.di["m00"]] + x[self.di["m20"]]) / (x[self.di["m00"]] + self.C0)
        # w2l = w2l * jnp.exp(r2_2)

        # selection on size (upper limit)
        # (M00 + M20) / M00 < r2_max
        # M00 (1 - r2_max) + M20 < 0
        # M00 (r2_max - 1) - M20 > 0
        # r2u = x[self.di["m00"]] * (self.r2_max - 1.0) - x[self.di["m20"]]
        # w2u = ssfunc2(r2u, self.sigma_r2, self.sigma_r2)

        # # wlap
        # # (M00 - M20) / M00 > 0.3
        # lap = x[self.di["m00"]] * (1.0 - 0.3) - x[self.di["m20"]]
        # wlap =  ssfunc2(lap, self.sigma_r2, self.sigma_r2)
        # e_sq = self._e1(x) ** 2.0 + self._e2(x) ** 2.0
        # well = jnp.exp(-e_sq / 2.0 / 0.5 ** 2.0)

        wsel = w0l * w2l
        return wsel

    def _wdet(self, x):
        # detection
        out = 1.0
        for i in range(self.det_nrot):
            out = out * ssfunc2(
                x[self.di["v%d" % i]],
                self.sigma_v - self.pratio * x[self.di["m00"]] - self.pcut,
                self.sigma_v,
            )
        return out

    def _denom(self, x):
        denom = (x[self.di["m00"]] + self.C0) ** self.alpha * (
            x[self.di["m00"]] + x[self.di["m20"]] + self.C2
        ) ** self.beta
        return denom

    def _e1(self, x):
        return 0.0

    def _e2(self, x):
        return 0.0

    def _we1(self, x):
        return self._wsel(x) * self._e1(x) * self._wdet(x)

    def _we2(self, x):
        return self._wsel(x) * self._e2(x) * self._wdet(x)

    def _we1_sharp(self, x):
        e1 = self._wsel(x) * self._e1(x)
        w = self._wdet(x)
        return ssfunc2(w, 0.12, 0.04) * e1

    def _we2_sharp(self, x):
        e2 = self._wsel(x) * self._e2(x)
        w = self._wdet(x)
        return ssfunc2(w, 0.12, 0.04) * e2

    def _measure_g1(self, x):
        e1, linear_func = jax.linearize(
            self._we1,
            x,
        )
        dmm_dg1 = self._dg1(x)
        de1_dg1 = linear_func(dmm_dg1)
        return jnp.hstack([e1, de1_dg1])

    def _measure_g2(self, x):
        e2, linear_func = jax.linearize(
            self._we2,
            x,
        )
        dmm_dg2 = self._dg2(x)
        de2_dg2 = linear_func(dmm_dg2)
        return jnp.hstack([e2, de2_dg2])

    def _measure_g1_renoise(self, x, y=0.0):
        e1, linear_func = jax.linearize(
            self._we1_sharp,
            x,
        )
        dmm_dg1 = self._dg1(x - y * 2.0)
        de1_dg1 = linear_func(dmm_dg1)
        return jnp.hstack([e1, de1_dg1])

    def _measure_g2_renoise(self, x, y=0.0):
        e2, linear_func = jax.linearize(
            self._we2_sharp,
            x,
        )
        dmm_dg2 = self._dg2(x - y * 2.0)
        de2_dg2 = linear_func(dmm_dg2)
        return jnp.hstack([e2, de2_dg2])

    def _noisebias(self, func, x):
        hessian = jax.jacfwd(jax.jacrev(func))(x)
        out = (
            jnp.tensordot(
                hessian,
                self.cov_mat,
                axes=[[-2, -1], [-2, -1]],
            )
            / 2.0
        )
        return out

    def measure_g1_renoise(self, src, noise=None):
        """This function meausres the first component of shear using
        renoise method to revise noise bias.

        Args:
        src (ndarray):      source catalog
        noise (ndarray):    noise catalog

        Returns:
        result (ndarray):   ellipticity and shear response (first component)
        """
        src = jnp.atleast_2d(src)
        if noise is None:
            func = jax.vmap(
                self._measure_g1_renoise,
                in_axes=0,
                out_axes=0,
            )
            result = func(src)
        else:
            assert noise.shape == src.shape, "input shapes not matched"
            func = jax.vmap(
                self._measure_g1_renoise,
                in_axes=(0, 0),
                out_axes=0,
            )
            result = func(src, noise)
        return result

    def measure_g2_renoise(self, src, noise=None):
        """This function meausres the second component of shear using
        renoise method to revise noise bias.

        Args:
        src (ndarray):      source catalog
        noise (ndarray):    noise catalog

        Returns:
        result (ndarray):   ellipticity and shear response (second component)
        """
        src = jnp.atleast_2d(src)
        if noise is None:
            func = jax.vmap(
                self._measure_g2_renoise,
                in_axes=0,
                out_axes=0,
            )
            result = func(src)
        else:
            assert noise.shape == src.shape, "input shapes not matched"
            func = jax.vmap(
                self._measure_g2_renoise,
                in_axes=(0, 0),
                out_axes=0,
            )
            result = func(src, noise)
        return result

    def measure_g1_denoise(self, src, noise_rev=False):
        """This function meausres the first component of shear using
        denoise method to revise noise bias.

        Args:
        src (ndarray):      source catalog
        noise_rev (bool):   whether to noise bias correction

        Returns:
        result (ndarray):   ellipticity and shear response (first component)
        """
        src = jnp.atleast_2d(src)
        func = self._measure_g1
        if not noise_rev:
            func2 = lambda x: func(x) - self._noisebias(func, x)
        else:
            func2 = func
        result = jax.lax.map(func2, src)
        return result

    def measure_g2_denoise(self, src, noise_rev=False):
        """This function meausres the second component of shear using
        denoise method to revise noise bias.

        Args:
        src (ndarray):      source catalog
        noise_rev (bool):   whether to noise bias correction

        Returns:
        result (ndarray):   ellipticity and shear response (second component)
        """
        src = jnp.atleast_2d(src)
        func = self._measure_g2
        if not noise_rev:
            func2 = lambda x: func(x) - self._noisebias(func, x)
        else:
            func2 = func
        result = jax.lax.map(func2, src)
        return result


class fpfs_catalog(catalog_base):
    def __init__(
        self,
        ratio: float = 1.81,
        snr_min: float = 12.0,
        r2_min: float = 0.05,
        r2_max: float = 2.0,
        c0: float = 2.55,
        c2: float = 25.6,
        alpha: float = 0.27,
        beta: float = 0.83,
        pthres: float = 0.0,
        pratio: float = 0.02,
        cov_mat=None,
        sigma_m00: float | None = None,
        sigma_r2: float | None = None,
        sigma_v: float | None = None,
        det_nrot: int = 8,
    ):
        nord = 4
        super().__init__(
            ratio=ratio,
            snr_min=snr_min,
            r2_min=r2_min,
            r2_max=r2_max,
            c0=c0,
            c2=c2,
            alpha=alpha,
            beta=beta,
            pthres=pthres,
            pratio=pratio,
            cov_mat=cov_mat,
            sigma_m00=sigma_m00,
            sigma_r2=sigma_r2,
            sigma_v=sigma_v,
            nord=nord,
            det_nrot=det_nrot,
        )
        return

    def _e1(self, x):
        # ellipticity1
        e1 = x[self.di["m22c"]] / self._denom(x)
        return e1

    def _e2(self, x):
        # ellipticity2
        e2 = x[self.di["m22s"]] / self._denom(x)
        return e2


class fpfs4_catalog(catalog_base):
    def __init__(
        self,
        ratio=1.81,
        snr_min=12.0,
        r2_min=0.05,
        r2_max=2.0,
        c0=2.55,
        c2=25.6,
        alpha=0.27,
        beta=0.83,
        pthres=0.0,
        pratio=0.02,
        cov_mat=None,
        sigma_m00=None,
        sigma_r2=None,
        sigma_v=None,
    ):
        nord = 6
        super().__init__(
            ratio=ratio,
            snr_min=snr_min,
            r2_min=r2_min,
            r2_max=r2_max,
            c0=c0,
            c2=c2,
            alpha=alpha,
            beta=beta,
            pthres=pthres,
            pratio=pratio,
            cov_mat=cov_mat,
            sigma_m00=sigma_m00,
            sigma_r2=sigma_r2,
            sigma_v=sigma_v,
            nord=nord,
        )
        return

    def _e1(self, x):
        # ellipticity1
        e1 = x[self.di["m42c"]] / self._denom(x)
        return e1

    def _e2(self, x):
        # ellipticity2
        e2 = x[self.di["m42s"]] / self._denom(x)
        return e2


def m2e(mm, const=1.0, nn=None):
    """Estimates FPFS ellipticities from fpfs moments

    Args:
    mm (ndarray):
        FPFS moments
    const (float):
        the weight constant [default:1]
    nn (ndarray):
        noise covaraince elements [default: None]

    Returns:
    out (ndarray):
        an array of [FPFS ellipticities, FPFS ellipticity response, FPFS
        flux, size and FPFS selection response]
    """

    # ellipticity, q-ellipticity, sizes, e^2, eq
    types = [
        ("fpfs_e1", "<f8"),
        ("fpfs_e2", "<f8"),
        ("fpfs_R1E", "<f8"),
        ("fpfs_R2E", "<f8"),
    ]
    # make the output ndarray
    out = np.array(np.zeros(mm.size), dtype=types)

    # FPFS shape weight's inverse
    _w = mm["fpfs_m00"] + const
    # FPFS ellipticity
    e1 = mm["fpfs_m22c"] / _w
    e2 = mm["fpfs_m22s"] / _w
    # FPFS spin-0 observables
    s0 = mm["fpfs_m00"] / _w
    s4 = mm["fpfs_m40"] / _w
    # intrinsic ellipticity
    e1e1 = e1 * e1
    e2e2 = e2 * e2

    # spin-2 properties
    out["fpfs_e1"] = e1  # ellipticity
    out["fpfs_e2"] = e2
    # response for ellipticity
    out["fpfs_R1E"] = (s0 - s4 + 2.0 * e1e1) / np.sqrt(2.0)
    out["fpfs_R2E"] = (s0 - s4 + 2.0 * e2e2) / np.sqrt(2.0)
    return out
