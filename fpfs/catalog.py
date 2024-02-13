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

from .image.image import fpfs_base, det_ratio


def read_catalog(fname):
    x = fitsio.read(fname)
    if x.dtype.names is not None:
        x = rfn.structured_to_unstructured(
            x,
            dtype=np.float64,
            copy=True,
        )
    return jnp.array(x)


def _ssfunc2(t):
    return 6 * t**5.0 - 15 * t**4.0 + 10 * t**3.0


def ssfunc2(x, mu, sigma):
    """Returns the C2 smooth step weight funciton

    Args:
    x (ndarray):    input data vector
    mu (float):     center of the cut
    sigma (float):  half width of the selection function

    Returns:
    out (ndarray):  the weight funciton
    """
    t = (x - mu) / sigma / 2.0 + 0.5
    return jnp.piecewise(t, [t < 0, (t >= 0) & (t <= 1), t > 1], [0.0, _ssfunc2, 1.0])


class FpfsCatalog(fpfs_base):
    def __init__(
        self,
        funcnm="ss2",
        ratio=1.81,
        snr_min=12,
        r2_min=0.05,
        r2_max=2.0,
        c0=2.55,
        c2=25.6,
        alpha=0.27,
        beta=0.83,
        cov_mat=None,
        sigma_m00=None,
        sigma_r2=None,
        sigma_v=None,
        m00_min=None,
        C0=None,
        C2=None,
        nord=4,
        detect_nrot=8,
    ):
        super().__init__(
            nord=nord,
            detect_nrot=detect_nrot,
        )
        if cov_mat is None:
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
            jnp.array([std_modes[self.di["v%d" % _]] for _ in range(detect_nrot)])
        )

        # selection and detection function
        self.wfunc = ssfunc2
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

        # thresholds
        if m00_min is None:
            self.m00_min = snr_min * std_m00
        else:
            self.m00_min = m00_min
        self.r2_min = r2_min
        self.r2_max = r2_max

        # shape parameters
        if C0 is None:
            self.C0 = c0 * std_m00
        else:
            self.C0 = C0
        if C2 is None:
            self.C2 = c2 * std_m20
        else:
            self.C2 = C2
        self.alpha = alpha
        self.beta = beta
        return

    def _dg1(self, x):
        """Returns shear response array [first component] of shapelet pytree"""
        # shear response for shapelet modes
        dm00 = -jnp.sqrt(2.0) * x[self.di["m22c"]]
        dm20 = -jnp.sqrt(6.0) * x[self.di["m42c"]]
        dm22c = (
            (x[self.di["m00"]] - x[self.di["m40"]])
            / jnp.sqrt(2.0)
            # - jnp.sqrt(3.0) * x[self.di["m44c"]]
        )
        dm22s = 0.0  # - jnp.sqrt(3.0) * x[self.di["m44s"]]
        dm40 = 0.0
        dm42c = 0.0
        dm42s = 0.0
        # dm44c = 0.0
        # dm44s = 0.0
        out = jnp.stack(
            [
                dm00,
                dm20,
                dm22c,
                dm22s,
                dm40,
                dm42c,
                dm42s,
                # dm44c,
                # dm44s,
            ]
            + [x[self.di["v%dr1" % _]] for _ in range(self.detect_nrot)]
            + [0] * self.detect_nrot * 2
        )
        return out

    def _dg2(self, x):
        """Returns shear response array [second component] of shapelet pytree"""
        dm00 = -jnp.sqrt(2.0) * x[self.di["m22s"]]
        dm20 = -jnp.sqrt(6.0) * x[self.di["m42s"]]
        dm22c = 0.0  # - jnp.sqrt(3.0) * x[self.di["m44s"]]
        dm22s = (
            (x[self.di["m00"]] - x[self.di["m40"]])
            / jnp.sqrt(2.0)
            # + jnp.sqrt(3.0) * x[self.di["m44c"]]
        )
        dm40 = 0.0
        dm42c = 0.0
        dm42s = 0.0
        # dm44c = 0.0
        # dm44s = 0.0
        out = jnp.stack(
            [
                dm00,
                dm20,
                dm22c,
                dm22s,
                dm40,
                dm42c,
                dm42s,
                # dm44c,
                # dm44s,
            ]
            + [x[self.di["v%dr2" % _]] for _ in range(self.detect_nrot)]
            + [0] * self.detect_nrot * 2
        )
        return out

    def _wsel(self, x):
        # selection on flux
        w0l = self.wfunc(x[self.di["m00"]], self.m00_min, self.sigma_m00)

        # selection on size (lower limit)
        # (M00 + M20) / M00 > r2_min
        r2l = x[self.di["m00"]] * (1.0 - self.r2_min) + x[self.di["m20"]]
        w2l = self.wfunc(r2l, self.sigma_r2, self.sigma_r2)

        # selection on size (upper limit)
        # (M00 + M20) / M00 < r2_max
        # M00 (1 - r2_max) + M20 < 0
        # M00 (r2_max - 1) - M20 > 0
        r2u = x[self.di["m00"]] * (self.r2_max - 1.0) - x[self.di["m20"]]
        w2u = self.wfunc(r2u, self.sigma_r2, self.sigma_r2)
        wsel = w0l * w2l * w2u
        return wsel

    def _wdet(self, x):
        # detection
        wdet = 1.0
        for i in range(self.detect_nrot):
            wdet = wdet * self.wfunc(
                x[self.di["v%d" % i]] + det_ratio * x[self.di["m00"]],
                self.sigma_v,
                self.sigma_v,
            )
        return wdet

    def _e1(self, x):
        # ellipticity1
        denom = (x[self.di["m00"]] + self.C0) ** self.alpha * (
            x[self.di["m00"]] + x[self.di["m20"]] + self.C2
        ) ** self.beta
        e1 = x[self.di["m22c"]] / denom
        return e1

    def _e2(self, x):
        # ellipticity2
        denom = (x[self.di["m00"]] + self.C0) ** self.alpha * (
            x[self.di["m00"]] + x[self.di["m20"]] + self.C2
        ) ** self.beta
        e2 = x[self.di["m22s"]] / denom
        return e2

    def _we1(self, x):
        return self._wsel(x) * self._wdet(x) * self._e1(x)

    def _we2(self, x):
        return self._wsel(x) * self._wdet(x) * self._e2(x)

    def measure_g1(self, x):
        e1, linear_func = jax.linearize(
            self._we1,
            x,
        )
        dmm_dg1 = self._dg1(x)
        de1_dg1 = linear_func(dmm_dg1)
        return jnp.hstack([e1, de1_dg1])

    def measure_g2(self, x):
        e2, linear_func = jax.linearize(
            self._we2,
            x,
        )
        dmm_dg2 = self._dg2(x)
        de2_dg2 = linear_func(dmm_dg2)
        return jnp.hstack([e2, de2_dg2])

    def _noisebias1(self, x):
        hessian = jax.jacfwd(jax.jacrev(self.measure_g1))(x)
        out = jnp.tensordot(hessian, self.cov_mat, axes=[[-2, -1], [-2, -1]]) / 2.0
        return out

    def _noisebias2(self, x):
        hessian = jax.jacfwd(jax.jacrev(self.measure_g2))(x)
        out = jnp.tensordot(hessian, self.cov_mat, axes=[[-2, -1], [-2, -1]]) / 2.0
        return out

    def measure_g1_noise_correct(self, x):
        return self.measure_g1(x) - self._noisebias1(x)

    def measure_g2_noise_correct(self, x):
        return self.measure_g2(x) - self._noisebias2(x)


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
