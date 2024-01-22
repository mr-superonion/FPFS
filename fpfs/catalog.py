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
from copy import deepcopy

from fitsio import read as fitsread
import numpy.lib.recfunctions as rfn


# This file tells the default structure of the data
indexes = {
    "m00": 0,
    "m20": 1,
    "m22c": 2,
    "m22s": 3,
    "m40": 4,
    "m42c": 5,
    "m42s": 6,
    "v0": 7,
    "v1": 8,
    "v2": 9,
    "v3": 10,
    "v4": 11,
    "v5": 12,
    "v6": 13,
    "v7": 14,
    "v0_g1": 15,
    "v1_g1": 16,
    "v2_g1": 17,
    "v3_g1": 18,
    "v4_g1": 19,
    "v5_g1": 20,
    "v6_g1": 21,
    "v7_g1": 22,
    "v0_g2": 23,
    "v1_g2": 24,
    "v2_g2": 25,
    "v3_g2": 26,
    "v4_g2": 27,
    "v5_g2": 28,
    "v6_g2": 29,
    "v7_g2": 30,
}

col_names = [
    "fpfs_M00",
    "fpfs_M20",
    "fpfs_M22c",
    "fpfs_M22s",
    "fpfs_M40",
    "fpfs_M42c",
    "fpfs_M42s",
    "fpfs_v0",
    "fpfs_v1",
    "fpfs_v2",
    "fpfs_v3",
    "fpfs_v4",
    "fpfs_v5",
    "fpfs_v6",
    "fpfs_v7",
    "fpfs_v0r1",
    "fpfs_v1r1",
    "fpfs_v2r1",
    "fpfs_v3r1",
    "fpfs_v4r1",
    "fpfs_v5r1",
    "fpfs_v6r1",
    "fpfs_v7r1",
    "fpfs_v0r2",
    "fpfs_v1r2",
    "fpfs_v2r2",
    "fpfs_v3r2",
    "fpfs_v4r2",
    "fpfs_v5r2",
    "fpfs_v6r2",
    "fpfs_v7r2",
]

cov_names = [
    "fpfs_N00N00",
    "fpfs_N20N20",
    "fpfs_N22cN22c",
    "fpfs_N22sN22s",
    "fpfs_N40N40",
    "fpfs_N00N20",
    "fpfs_N00N22c",
    "fpfs_N00N22s",
    "fpfs_N00N40",
    "fpfs_N00N42c",
    "fpfs_N00N42s",
    "fpfs_N20N22c",
    "fpfs_N20N22s",
    "fpfs_N20N40",
    "fpfs_N22cN42c",
    "fpfs_N22sN42s",
    "fpfs_N00V0",
    "fpfs_N00V0r1",
    "fpfs_N00V0r2",
    "fpfs_N22cV0",
    "fpfs_N22sV0",
    "fpfs_N22cV0r1",
    "fpfs_N22sV0r2",
    "fpfs_N40V0",
    "fpfs_N00V1",
    "fpfs_N00V1r1",
    "fpfs_N00V1r2",
    "fpfs_N22cV1",
    "fpfs_N22sV1",
    "fpfs_N22cV1r1",
    "fpfs_N22sV1r2",
    "fpfs_N40V1",
    "fpfs_N00V2",
    "fpfs_N00V2r1",
    "fpfs_N00V2r2",
    "fpfs_N22cV2",
    "fpfs_N22sV2",
    "fpfs_N22cV2r1",
    "fpfs_N22sV2r2",
    "fpfs_N40V2",
    "fpfs_N00V3",
    "fpfs_N00V3r1",
    "fpfs_N00V3r2",
    "fpfs_N22cV3",
    "fpfs_N22sV3",
    "fpfs_N22cV3r1",
    "fpfs_N22sV3r2",
    "fpfs_N40V3",
    "fpfs_N00V4",
    "fpfs_N00V4r1",
    "fpfs_N00V4r2",
    "fpfs_N22cV4",
    "fpfs_N22sV4",
    "fpfs_N22cV4r1",
    "fpfs_N22sV4r2",
    "fpfs_N40V4",
    "fpfs_N00V5",
    "fpfs_N00V5r1",
    "fpfs_N00V5r2",
    "fpfs_N22cV5",
    "fpfs_N22sV5",
    "fpfs_N22cV5r1",
    "fpfs_N22sV5r2",
    "fpfs_N40V5",
    "fpfs_N00V6",
    "fpfs_N00V6r1",
    "fpfs_N00V6r2",
    "fpfs_N22cV6",
    "fpfs_N22sV6",
    "fpfs_N22cV6r1",
    "fpfs_N22sV6r2",
    "fpfs_N40V6",
    "fpfs_N00V7",
    "fpfs_N00V7r1",
    "fpfs_N00V7r2",
    "fpfs_N22cV7",
    "fpfs_N22sV7",
    "fpfs_N22cV7r1",
    "fpfs_N22sV7r2",
    "fpfs_N40V7",
]

ncol = 31


def read_catalog(fname):
    x = fitsread(fname)
    if x.dtype.names is not None:
        x = x[col_names]
        x = rfn.structured_to_unstructured(
            x,
            dtype=np.float64,
            copy=True,
        )
    return jnp.array(x)


di = deepcopy(indexes)


def _ssfunc2(t):
    return 6 * t**5.0 - 15 * t**4.0 + 10 * t**3.0


def ssfunc2(x, mu, sigma):
    """Returns the C2 smooth step weight funciton

    Args:
    x (ndarray):    input data vector
    mu (float):     center of the cut
    sigma (float):  width of the selection function

    Returns:
    out (ndarray):  the weight funciton
    """
    t = (x - mu) / sigma / 2.0 + 0.5
    return jnp.piecewise(t, [t < 0, (t >= 0) & (t <= 1), t > 1], [0.0, _ssfunc2, 1.0])


class FpfsCatalog(object):
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
        v_min=None,
        m00_min=None,
        C0=None,
        C2=None,
    ):
        if cov_mat is None:
            cov_mat = jnp.zeros((ncol, ncol))
        self.cov_mat = cov_mat
        std_modes = np.sqrt(np.diagonal(cov_mat))
        std_m00 = std_modes[di["m00"]]
        std_m20 = np.sqrt(
            cov_mat[di["m00"], di["m00"]]
            + cov_mat[di["m20"], di["m20"]]
            + cov_mat[di["m00"], di["m20"]]
            + cov_mat[di["m20"], di["m00"]]
        )
        std_v = jnp.average(jnp.array([std_modes[di["v%d" % _]] for _ in range(8)]))

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
        if v_min is None:
            self.v_min = ratio * std_v * 0.5
        else:
            self.v_min = v_min
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
        dm00 = -jnp.sqrt(2.0) * x[di["m22c"]]
        dm20 = -jnp.sqrt(6.0) * x[di["m42c"]]
        dm22c = (x[di["m00"]] - x[di["m40"]]) / jnp.sqrt(2.0)
        # TODO: Include spin-4 term. Will add it when we have M44
        dm22s = 0.0
        # TODO: Incldue the shear response of M40 in the future. This is not
        # required in the FPFS shear estimation (v1~v3), so I set it to zero
        # here (But if you are interested in playing with shear response of
        # this term, please contact me.)
        dm40 = 0.0
        dm42c = 0.0
        dm42s = 0.0
        out = jnp.stack(
            [
                dm00,
                dm20,
                dm22c,
                dm22s,
                dm40,
                dm42c,
                dm42s,
                x[di["v0_g1"]],
                x[di["v1_g1"]],
                x[di["v2_g1"]],
                x[di["v3_g1"]],
                x[di["v4_g1"]],
                x[di["v5_g1"]],
                x[di["v6_g1"]],
                x[di["v7_g1"]],
            ]
            + [0] * 16
        )
        return out

    def _dg2(self, x):
        """Returns shear response array [second component] of shapelet pytree"""
        dm00 = -jnp.sqrt(2.0) * x[di["m22s"]]
        dm20 = -jnp.sqrt(6.0) * x[di["m42s"]]
        # TODO: Include spin-4 term. Will add it when we have M44
        dm22c = 0.0
        dm22s = (x[di["m00"]] - x[di["m40"]]) / jnp.sqrt(2.0)
        # TODO: Incldue the shear response of M40 in the future. This is not
        # required in the FPFS shear estimation (v1~v3), so I set it to zero
        # here (But if you are interested in playing with shear response of
        # this term, please contact me.)
        dm40 = 0.0
        dm42c = 0.0
        dm42s = 0.0
        out = jnp.stack(
            [
                dm00,
                dm20,
                dm22c,
                dm22s,
                dm40,
                dm42c,
                dm42s,
                x[di["v0_g2"]],
                x[di["v1_g2"]],
                x[di["v2_g2"]],
                x[di["v3_g2"]],
                x[di["v4_g2"]],
                x[di["v5_g2"]],
                x[di["v6_g2"]],
                x[di["v7_g2"]],
            ]
            + [0] * 16
        )
        return out

    def _wsel(self, x):
        # selection on flux
        w0l = self.wfunc(x[di["m00"]], self.m00_min, self.sigma_m00)
        # w0u = self.ufunc(300.0 - x[di["m00"]], 0.0, self.sigma_m00)

        # selection on size (lower limit)
        # (M00 + M20) / M00 > r2_min
        r2l = x[di["m00"]] * (1.0 - self.r2_min) + x[di["m20"]]
        w2l = self.wfunc(r2l, self.sigma_r2, self.sigma_r2)

        # selection on size (upper limit)
        # (M00 + M20) / M00 < r2_max
        # M00 (1 - r2_max) + M20 < 0
        # M00 (r2_max - 1) - M20 > 0
        r2u = x[di["m00"]] * (self.r2_max - 1.0) - x[di["m20"]]
        w2u = self.wfunc(r2u, self.sigma_r2, self.sigma_r2)
        wsel = w0l * w2l * w2u
        return wsel

    def _wdet(self, x):
        npeak = 8
        # detection
        wdet = 1.0
        for i in range(0, npeak):
            wdet = wdet * self.wfunc(
                x[di["v%d" % i]],
                self.v_min,
                self.sigma_v,
            )
        return wdet

    def _e1(self, x):
        # ellipticity1
        denom = (x[di["m00"]] + self.C0) ** self.alpha * (
            x[di["m00"]] + x[di["m20"]] + self.C2
        ) ** self.beta
        e1 = x[di["m22c"]] / denom
        return e1

    def _e2(self, x):
        # ellipticity2
        denom = (x[di["m00"]] + self.C0) ** self.alpha * (
            x[di["m00"]] + x[di["m20"]] + self.C2
        ) ** self.beta
        e2 = x[di["m22s"]] / denom
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
        ("e1", "<f8"),
        ("e2", "<f8"),
        ("R1E", "<f8"),
        ("R2E", "<f8"),
    ]
    # make the output ndarray
    out = np.array(np.zeros(mm.size), dtype=types)

    # FPFS shape weight's inverse
    _w = mm["m00"] + const
    # FPFS ellipticity
    e1 = mm["m22c"] / _w
    e2 = mm["m22s"] / _w
    # FPFS spin-0 observables
    s0 = mm["m00"] / _w
    s4 = mm["m40"] / _w
    # intrinsic ellipticity
    e1e1 = e1 * e1
    e2e2 = e2 * e2

    # spin-2 properties
    out["e1"] = e1  # ellipticity
    out["e2"] = e2
    # response for ellipticity
    out["R1E"] = (s0 - s4 + 2.0 * e1e1) / np.sqrt(2.0)
    out["R2E"] = (s0 - s4 + 2.0 * e2e2) / np.sqrt(2.0)
    return out


def fpfscov_to_imptcov(data):
    """Converts FPFS noise Covariance elements into a covariance matrix of
    lensPT.

    Args:
        data (ndarray):     FPFS shapelet mode catalog
    Returns:
        out (ndarray):      Covariance matrix
    """
    # the colum names
    # M00 -> N00; v1 -> V1
    ll = [cn[5:].replace("M", "N").replace("v", "V") for cn in col_names]
    out = np.zeros((ncol, ncol))
    for i in range(ncol):
        for j in range(ncol):
            try:
                try:
                    cname = "fpfs_%s%s" % (ll[i], ll[j])
                    out[i, j] = data[cname][0]
                except (ValueError, KeyError):
                    cname = "fpfs_%s%s" % (ll[j], ll[i])
                    out[i, j] = data[cname][0]
            except (ValueError, KeyError):
                out[i, j] = 0.0
    return out


def imptcov_to_fpfscov(data):
    """Converts FPFS noise Covariance elements into a covariance matrix of
    lensPT.

    Args:
        data (ndarray):     impt covariance matrix
    Returns:
        out (ndarray):      FPFS covariance elements
    """
    # the colum names
    # M00 -> N00; v1 -> V1
    ll = [cn[5:].replace("M", "N").replace("v", "V") for cn in col_names]
    types = [(cn, "<f8") for cn in cov_names]
    out = np.zeros(1, dtype=types)
    for i in range(ncol):
        for j in range(i, ncol):
            cname = "fpfs_%s%s" % (ll[i], ll[j])
            if cname in cov_names:
                out[cname][0] = data[i, j]
    return out
