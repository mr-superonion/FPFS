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
import numpy as np


# functions used for selection
def tsfunc1(x, deriv=0, mu=0.0, sigma=1.5):
    """Returns the weight funciton [deriv=0], or the *multiplicative factor* to
    the weight function for first order derivative [deriv=1]. This is for C1
    function

    Args:
        deriv (int):    whether do derivative [deriv=1] or not [deriv=0]
        x (ndarray):    input data vector
        mu (float):     center of the cut
        sigma (float):  width of the selection function
    Returns:
        out (ndarray):  the weight funciton [deriv=0], or the *multiplicative
                        factor* to the weight function for first order
                        derivative [deriv=1]
    """
    t = (x - mu) / sigma
    if deriv == 0:
        return np.piecewise(
            t,
            [t < -1, (t >= -1) & (t <= 1), t > 1],
            [0.0, lambda t: 1.0 / 2.0 + np.sin(t * np.pi / 2.0) / 2.0, 1.0],
        )
    elif deriv == 1:
        # multiplicative factor (to weight) for derivative
        return np.piecewise(
            t,
            [t < -1 + 0.01, (t >= -1 + 0.01) & (t <= 1 - 0.01), t > 1 - 0.01],
            [
                0.0,
                lambda t: np.pi
                / 2.0
                / sigma
                * np.cos(t * np.pi / 2.0)
                / (1.0 + np.sin(t * np.pi / 2.0)),
                0.0,
            ],
        )
    else:
        raise ValueError("deriv should be 0 or 1")


def tsfunc2(x, mu=0.0, sigma=1.5, deriv=0):
    """Returns the weight funciton [deriv=0], or the *multiplicative factor* to
    the weight function for first order derivative [deriv=1]. This is for C2
    funciton

    Args:
        deriv (int):    whether do derivative [deriv=1] or not [deriv=0]
        x (ndarray):    input data vector
        mu (float):     center of the cut
        sigma (float):  width of the selection function
    Returns:
        out (ndarray):  the weight funciton [deriv=0], or the *multiplicative
                        factor* to the weight function for first order
                        derivative [deriv=1]
    """
    t = (x - mu) / sigma

    def func(t):
        return 1.0 / 2.0 + t / 2.0 + 1.0 / 2.0 / np.pi * np.sin(t * np.pi)

    def func2(t):
        # /(1./2.+t/2.+1./2./np.pi*np.sin(t*np.pi))
        return 1.0 / 2.0 / sigma + 1.0 / 2.0 / sigma * np.cos(np.pi * t)

    def func3(t):
        return -np.pi / 2.0 / sigma**2.0 * np.sin(np.pi * t)

    def func4(t):
        return -((np.pi) ** 2.0) / 2.0 / sigma**3.0 * np.cos(np.pi * t)

    if deriv == 0:
        return np.piecewise(t, [t < -1, (t >= -1) & (t <= 1), t > 1], [0.0, func, 1.0])
    elif deriv == 1:
        return np.piecewise(
            t,
            [t < -1 + 0.01, (t >= -1 + 0.01) & (t <= 1 - 0.01), t > 1 - 0.01],
            [0.0, lambda t: func2(t) / func(t), 0.0],
        )
    elif deriv == 2:
        return np.piecewise(
            t,
            [t < -1 + 0.01, (t >= -1 + 0.01) & (t <= 1 - 0.01), t > 1 - 0.01],
            [0.0, lambda t: func3(t) / func(t), 0.0],
        )
    elif deriv == 3:
        return np.piecewise(
            t,
            [t < -1 + 0.01, (t >= -1 + 0.01) & (t <= 1 - 0.01), t > 1 - 0.01],
            [0.0, lambda t: func4(t) / func(t), 0.0],
        )
    else:
        raise ValueError("deriv can only be 0,1,2,3")


def sigfunc(x, deriv=0, mu=0.0, sigma=1.5):
    """Returns the weight funciton [deriv=0], or the *multiplicative factor* to
    the weight function for first order derivative [deriv=1]

    Args:
        deriv (int):    whether do derivative [deriv=1] or not [deriv=0]
        x (ndarray):    input data vector
        mu (float):     center of the cut
        sigma (float):  width of the selection function
    Returns:
        out (ndarray):  the weight funciton [deriv=0], or the *multiplicative
                        factor* to the weight function for first order derivative
                        [deriv=1]
    """
    expx = np.exp(-(x - mu) / sigma)
    if deriv == 0:
        # sigmoid function
        return 1.0 / (1.0 + expx)
    elif deriv == 1:
        # multiplicative factor (to weight) for derivative
        return 1.0 / sigma * expx / (1.0 + expx)
    else:
        raise ValueError("deriv should be 0 or 1")


def get_wsel_eff(x, cut, sigma, use_sig, deriv=0):
    """Returns the weight funciton [deriv=0], or the *multiplicative
    factor* to the weight function for first order derivative [deriv=1]

    Args:
        x (ndarray):    input selection observable
        cut (float):    the cut on selection observable
        sigma (float):  width of the selection function
        use_sig (bool): whether use sigmoid [True] of truncated sine [False]
        deriv (int):    whether do derivative (1) or not (0)
    Returns:
        out (ndarray):  the weight funciton [deriv=0], or the *multiplicative
                        factor* to the weight function for first order
                        derivative [deriv=1]
    """
    if use_sig:
        out = sigfunc(x, deriv=deriv, mu=cut, sigma=sigma)
    else:
        out = tsfunc2(x, deriv=deriv, mu=cut, sigma=sigma)
    return out


def get_wbias(x, cut, sigma, use_sig, w_sel, rev=None):
    """Returns the weight bias due to shear dependence and noise bias [first
    order in w]

    Args:
        x (ndarray):        selection observable
        cut (float):        the cut on selection observable
        sigma (float):      width of the selection function
        use_sig (bool):     whether use sigmoid [True] of truncated sine [False]
        w_sel (ndarray):    selection weights as function of selection observable
        rev  (ndarray):     selection response array
    Returns:
        cor (float):        correction for shear response
    """
    if rev is None:
        cor = 0.0
    else:
        cor = np.sum(rev * w_sel * get_wsel_eff(x, cut, sigma, use_sig, deriv=1))
    return cor


# functions to get derived observables from fpfs modes
def fpfs_m2e(mm, const=1.0, nn=None):
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
        ("fpfs_ee", "<f8"),
        ("fpfs_s0", "<f8"),
        ("fpfs_s2", "<f8"),
        ("fpfs_s4", "<f8"),
        ("fpfs_R1E", "<f8"),
        ("fpfs_R2E", "<f8"),
        ("fpfs_RS0", "<f8"),
        ("fpfs_RS2", "<f8"),
    ]
    for i in range(8):
        types.append(("fpfs_R1Sv%d" % i, "<f8"))
        types.append(("fpfs_R2Sv%d" % i, "<f8"))

    # noirev
    if nn is not None:
        types = types + [
            ("fpfs_HE100", "<f8"),
            ("fpfs_HE200", "<f8"),
            ("fpfs_HR00", "<f8"),
            ("fpfs_HE120", "<f8"),
            ("fpfs_HE220", "<f8"),
            ("fpfs_HR20", "<f8"),
        ]
        for i in range(8):
            types.append(("fpfs_HRv%d" % i, "<f8"))
            types.append(("fpfs_HE1v%d" % i, "<f8"))
            types.append(("fpfs_HE2v%d" % i, "<f8"))
    # make the output ndarray
    out = np.array(np.zeros(mm.size), dtype=types)

    # FPFS shape weight's inverse
    _w = mm["fpfs_M00"] + const
    # FPFS ellipticity
    e1 = mm["fpfs_M22c"] / _w
    e2 = mm["fpfs_M22s"] / _w
    q1 = mm["fpfs_M42c"] / _w
    q2 = mm["fpfs_M42s"] / _w
    # FPFS spin-0 observables
    s0 = mm["fpfs_M00"] / _w
    s2 = mm["fpfs_M20"] / _w
    s4 = mm["fpfs_M40"] / _w
    # intrinsic ellipticity
    e1e1 = e1 * e1
    e2e2 = e2 * e2
    e_m22 = e1 * mm["fpfs_M22c"] + e2 * mm["fpfs_M22s"]
    e_m42 = e1 * mm["fpfs_M42c"] + e2 * mm["fpfs_M42s"]

    # shear response for detection process (not for deatection function)
    for i in range(8):
        out["fpfs_R1Sv%d" % (i)] = e1 * mm["fpfs_v%dr1" % (i)]
        out["fpfs_R2Sv%d" % (i)] = e2 * mm["fpfs_v%dr2" % (i)]

    # NOTE: START NOIST BIAS REVISION
    # noirev
    if nn is not None:
        # Selection
        out["fpfs_HR00"] = (
            -(
                nn["fpfs_N00N00"] * (const / _w + s4 - 4.0 * e1**2.0)
                - nn["fpfs_N00N40"]
            )
            / _w
            / np.sqrt(2.0)
        )
        out["fpfs_HR20"] = (
            -(
                nn["fpfs_N00N20"] * (const / _w + s4 - 4.0 * e2**2.0)
                - nn["fpfs_N20N40"]
            )
            / _w
            / np.sqrt(2.0)
        )
        out["fpfs_HE100"] = -(nn["fpfs_N00N22c"] - e1 * nn["fpfs_N00N00"]) / _w
        out["fpfs_HE200"] = -(nn["fpfs_N00N22s"] - e2 * nn["fpfs_N00N00"]) / _w
        out["fpfs_HE120"] = -(nn["fpfs_N20N22c"] - e1 * nn["fpfs_N00N20"]) / _w
        out["fpfs_HE220"] = -(nn["fpfs_N20N22s"] - e2 * nn["fpfs_N00N20"]) / _w
        ratio = nn["fpfs_N00N00"] / _w**2.0

        # Detection process and Shear Response
        for i in range(8):
            corr1 = (
                -1.0 * nn["fpfs_N22cV%dr1" % i] / _w
                + 1.0 * e1 * nn["fpfs_N00V%dr1" % i] / _w
                + 1.0 * nn["fpfs_N00N22c"] / _w**2.0 * mm["fpfs_v%dr1" % i]
            )
            corr2 = (
                -1.0 * nn["fpfs_N22sV%dr2" % i] / _w
                + 1.0 * e2 * nn["fpfs_N00V%dr2" % i] / _w
                + 1.0 * nn["fpfs_N00N22s"] / _w**2.0 * mm["fpfs_v%dr2" % i]
            )
            out["fpfs_R1Sv%d" % i] = (out["fpfs_R1Sv%d" % i] + corr1) / (1 + ratio)
            out["fpfs_R2Sv%d" % i] = (out["fpfs_R2Sv%d" % i] + corr2) / (1 + ratio)
            # Heissen
            out["fpfs_HRv%d" % i] = (
                -(
                    nn["fpfs_N00V%d" % i]
                    * (const / _w + s4 - 2.0 * e1**2.0 - 2.0 * e2**2.0)
                    - nn["fpfs_N40V%d" % i]
                )
                / _w
                / np.sqrt(2.0)
            )
            out["fpfs_HE1v%d" % i] = (
                -(nn["fpfs_N22cV%d" % i] - e1 * nn["fpfs_N00V%d" % i]) / _w
            )
            out["fpfs_HE2v%d" % i] = (
                -(nn["fpfs_N22sV%d" % i] - e2 * nn["fpfs_N00V%d" % i]) / _w
            )
        # intrinsic shape dispersion (not per component)
        e1e1 = (
            e1e1
            - (nn["fpfs_N22cN22c"]) / _w**2.0
            + 4.0 * (e1 * nn["fpfs_N00N22c"]) / _w**2.0
        ) - 3 * ratio * e1e1
        e2e2 = (
            e2e2
            - (nn["fpfs_N22sN22s"]) / _w**2.0
            + 4.0 * (e2 * nn["fpfs_N00N22s"]) / _w**2.0
        ) - 3 * ratio * e2e2
        e_m22 = (
            e_m22
            - (nn["fpfs_N22cN22c"] + nn["fpfs_N22sN22s"]) / _w
            + 2.0 * (nn["fpfs_N00N22c"] * e1 + nn["fpfs_N00N22s"] * e2) / _w
        ) / (1 + ratio)
        e_m42 = (
            e_m42
            - (nn["fpfs_N22cN42c"] + nn["fpfs_N22sN42s"]) / _w
            + 1.0 * (e1 * nn["fpfs_N00N42c"] + e2 * nn["fpfs_N00N42s"]) / _w
            + 1.0 * (q1 * nn["fpfs_N00N22c"] + q2 * nn["fpfs_N00N22s"]) / _w
        ) / (1 + ratio)
        # noise bias correction for ellipticity
        # (the following two expressions are the same to the second order of
        # noise)
        # e1 = (e1 + nn["fpfs_N00N22c"] / _w**2.0) / (1 + ratio)
        # e2 = (e2 + nn["fpfs_N00N22s"] / _w**2.0) / (1 + ratio)
        e1 = (e1 + nn["fpfs_N00N22c"] / _w**2.0) - ratio * e1
        e2 = (e2 + nn["fpfs_N00N22s"] / _w**2.0) - ratio * e2
        # noise bias correction for flux, size
        # (the following two expressions are the same to the second order of
        # noise)
        # s0 = (s0 + nn["fpfs_N00N00"] / _w**2.0) / (1 + ratio)
        # s2 = (s2 + nn["fpfs_N00N20"] / _w**2.0) / (1 + ratio)
        # s4 = (s4 + nn["fpfs_N00N40"] / _w**2.0) / (1 + ratio)
        s0 = (s0 + nn["fpfs_N00N00"] / _w**2.0) - ratio * s0
        s2 = (s2 + nn["fpfs_N00N20"] / _w**2.0) - ratio * s2
        s4 = (s4 + nn["fpfs_N00N40"] / _w**2.0) - ratio * s4
    # NOTE: END NOIST BIAS REVISION

    # spin-2 properties
    out["fpfs_e1"] = e1  # ellipticity
    out["fpfs_e2"] = e2
    del e1, e2
    # spin-0 properties
    out["fpfs_s0"] = s0  # flux
    out["fpfs_s2"] = s2  # size2
    out["fpfs_s4"] = s4  # size4
    out["fpfs_ee"] = e1e1 + e2e2  # shape noise
    # response for ellipticity
    out["fpfs_R1E"] = (s0 - s4 + 2.0 * e1e1) / np.sqrt(2.0)
    out["fpfs_R2E"] = (s0 - s4 + 2.0 * e2e2) / np.sqrt(2.0)
    del s0, s2, s4, e1e1, e2e2
    # response for selection process (not response for selection function)
    out["fpfs_RS0"] = -1.0 * e_m22 / np.sqrt(2.0)  # this has spin-4 leakage
    out["fpfs_RS2"] = -1.0 * e_m42 * np.sqrt(6.0) / 2.0  # this has spin-4 leakage
    del e_m22, e_m42
    return out


class summary_stats:
    def __init__(self, mm, ell, use_sig=False):
        """A class to get the summary statistics [e.g., mean shear] of from the
        moments and ellipticity.

        Args:
            mm (ndarray):   FPFS moments
            ell (ndarray):  FPFS ellipticity
            use_sig (bool): whether use sigmoid [True] of truncated sine [False]
        """
        self.use_sig = use_sig
        self.mm = mm
        self.ell = ell
        self.clear_outcomes()
        if "fpfs_HR00" in self.ell.dtype.names:
            self.noirev = True
        else:
            self.noirev = False
        return

    def clear_outcomes(self):
        """clears the outcome of the class"""
        self.nsel = 0
        self.ws = np.ones(self.ell.shape)
        # bias
        self.ncor = 0
        self.corE1 = 0.0  # selection bias in e1
        self.corE2 = 0.0  # selection bias in e2
        self.corR1 = 0.0  # selection bias in R1E (response)
        self.corR2 = 0.0  # selection bias in R2E (response)
        # signal
        self.sumE1 = 0.0  # sum of e1
        self.sumE2 = 0.0  # sum of e2
        self.sumR1 = 0.0  # sum of R1E (response)
        self.sumR2 = 0.0  # sum of R2E (response)
        return

    def update_selection_weight(self, snms, cuts, cutsigs):
        """Updates the selection weight term with the current selection weight"""
        if not isinstance(snms, np.ndarray):
            if (
                isinstance(snms, str)
                and isinstance(cuts, float)
                and isinstance(cutsigs, float)
            ):
                snms = np.array([snms])
                cuts = np.array([cuts])
                cutsigs = np.array([cutsigs])
            else:
                raise TypeError("snms, cuts and cutsigs should be str, float, float")
        for selnm, cut, cutsig in zip(snms, cuts, cutsigs):
            # print(selnm)
            if selnm == "detect":
                for iid in range(8):
                    self._update_selection_weight("det_v%d" % iid, cut, cutsig)
            elif selnm == "detect2":
                for iid in range(8):
                    self._update_selection_weight("det2_v%d" % iid, cut, cutsig)
            else:
                self._update_selection_weight(selnm, cut, cutsig)
        return

    def _update_selection_weight(self, selnm, cut, cutsig):
        """Updates the selection weight term with the current selection weight"""
        if not isinstance(selnm, str):
            raise TypeError("selnm should be str")
        if not isinstance(cut, float):
            raise TypeError("cut should be float")
        if not isinstance(cutsig, float):
            raise TypeError("cutsig should be float")

        cut_final = cut
        if selnm == "M00":
            scol = self.mm["fpfs_M00"]
        elif selnm == "M20":
            scol = -self.mm["fpfs_M20"]
        elif selnm == "R2":
            # M00+M20>cut*M00 (M00>0., we have mag cut to ensure it)
            scol = self.mm["fpfs_M00"] * (1.0 - cut) + self.mm["fpfs_M20"]
            cut_final = cutsig
        elif selnm == "R2_upp":
            # M00+M20<cut*M00 (M00>0.)
            scol = self.mm["fpfs_M00"] * (cut - 1.0) - self.mm["fpfs_M20"]
            cut_final = 0.0
        elif "det_" in selnm:
            vn = selnm.split("_")[-1]
            scol = self.mm["fpfs_%s" % vn]
        elif "det2_" in selnm:
            vn = selnm.split("_")[-1]
            scol = self.mm["fpfs_%s" % vn] - self.mm["fpfs_M00"] * cut
            cut_final = cutsig
        else:
            raise ValueError("Do not support selection vector name: %s" % selnm)
        # update weight
        ws = get_wsel_eff(scol, cut_final, cutsig, self.use_sig)
        self.ws = self.ws * ws
        # count the total number of selection cuts
        self.nsel = self.nsel + 1
        return

    def update_selection_bias(self, snms, cuts, cutsigs):
        """Updates the selection bias correction term with the current
        selection weight
        """
        if not isinstance(snms, np.ndarray):
            if (
                isinstance(snms, str)
                and isinstance(cuts, float)
                and isinstance(cutsigs, float)
            ):
                snms = [snms]
                cuts = [cuts]
                cutsigs = [cutsigs]
            else:
                raise TypeError(
                    "snms, cuts and cutsigs should be (lists of) str, float, float"
                )
        for selnm, cut, cutsig in zip(snms, cuts, cutsigs):
            # print(selnm)
            if selnm == "detect":
                for iid in range(8):
                    self._update_selection_bias("det_v%d" % iid, cut, cutsig)
            elif selnm == "detect2":
                for iid in range(8):
                    self._update_selection_bias("det2_v%d" % iid, cut, cutsig)
            else:
                self._update_selection_bias(selnm, cut, cutsig)
        assert self.nsel == self.ncor
        return

    def _update_selection_bias(self, selnm, cut, cutsig):
        """Updates the selection bias correction term with the current
        selection weight

        Args:
            selnm (str):    name of the selection variable ['M00', 'M20', 'R2']
            cut (float):    selection cut
            cutsig (float): width of the selection weight function. Note, it is
                            closer to heavy step when cutsig is smaller
        """
        if not isinstance(selnm, str):
            raise TypeError("selnm should be str")
        if not isinstance(cut, float):
            raise TypeError("cut should be float")
        if not isinstance(cutsig, float):
            raise TypeError("cutsig should be float")
        cut_final = cut
        if selnm == "M00":
            scol = self.mm["fpfs_M00"]
            # shear response
            ccol1 = self.ell["fpfs_RS0"]
            ccol2 = self.ell["fpfs_RS0"]
            if self.noirev:
                dcol = self.ell["fpfs_HR00"]
                ncol1 = self.ell["fpfs_HE100"]
                ncol2 = self.ell["fpfs_HE200"]
            else:
                dcol = None
                ncol1 = None
                ncol2 = None
        elif selnm == "M20":
            scol = -self.mm["fpfs_M20"]
            ccol1 = -self.ell["fpfs_RS2"]
            ccol2 = -self.ell["fpfs_RS2"]
            if self.noirev:
                dcol = -self.ell["fpfs_HR20"]
                ncol1 = -self.ell["fpfs_HE120"]
                ncol2 = -self.ell["fpfs_HE220"]
            else:
                dcol = None
                ncol1 = None
                ncol2 = None
        elif selnm == "R2" or selnm == "R2_upp":
            if "_upp" in selnm:
                fp = -1.0
            else:
                fp = 1.0
            # cut_final = 0.0
            cut_final = cutsig
            scol = (self.mm["fpfs_M00"] * (1.0 - cut) + self.mm["fpfs_M20"]) * fp
            ccol1 = (self.ell["fpfs_RS0"] * (1.0 - cut) + self.ell["fpfs_RS2"]) * fp
            ccol2 = (self.ell["fpfs_RS0"] * (1.0 - cut) + self.ell["fpfs_RS2"]) * fp
            if self.noirev:
                dcol = (
                    self.ell["fpfs_HR00"] * (1.0 - cut) + self.ell["fpfs_HR20"]
                ) * fp
                ncol1 = (
                    self.ell["fpfs_HE100"] * (1.0 - cut) + self.ell["fpfs_HE120"]
                ) * fp
                ncol2 = (
                    self.ell["fpfs_HE200"] * (1.0 - cut) + self.ell["fpfs_HE220"]
                ) * fp
            else:
                dcol = None
                ncol1 = None
                ncol2 = None
        elif "det_" in selnm:
            vn = selnm.split("_")[-1]
            scol = self.mm["fpfs_%s" % vn]
            ccol1 = self.ell["fpfs_R1S%s" % vn]
            ccol2 = self.ell["fpfs_R2S%s" % vn]
            if self.noirev:
                dcol = self.ell["fpfs_HR%s" % vn]
                ncol1 = self.ell["fpfs_HE1%s" % vn]
                ncol2 = self.ell["fpfs_HE2%s" % vn]
            else:
                dcol = None
                ncol1 = None
                ncol2 = None
        elif "det2_" in selnm:
            cut_final = cutsig
            vn = selnm.split("_")[-1]
            scol = self.mm["fpfs_%s" % vn] - self.mm["fpfs_M00"] * cut
            ccol1 = self.ell["fpfs_R1S%s" % vn] - self.ell["fpfs_RS0"] * cut
            ccol2 = self.ell["fpfs_R2S%s" % vn] - self.ell["fpfs_RS0"] * cut
            if self.noirev:
                dcol = self.ell["fpfs_HR%s" % vn] - self.ell["fpfs_HR00"] * cut
                ncol1 = self.ell["fpfs_HE1%s" % vn] - self.ell["fpfs_HE100"] * cut
                ncol2 = self.ell["fpfs_HE2%s" % vn] - self.ell["fpfs_HE200"] * cut
            else:
                dcol = None
                ncol1 = None
                ncol2 = None
        else:
            raise ValueError("Do not support selection vector name: %s" % selnm)
        cor_sel_r1 = get_wbias(scol, cut_final, cutsig, self.use_sig, self.ws, ccol1)
        cor_sel_r2 = get_wbias(scol, cut_final, cutsig, self.use_sig, self.ws, ccol2)
        cor_noise_r = get_wbias(scol, cut_final, cutsig, self.use_sig, self.ws, dcol)
        cor_noise_e1 = get_wbias(scol, cut_final, cutsig, self.use_sig, self.ws, ncol1)
        cor_noise_e2 = get_wbias(scol, cut_final, cutsig, self.use_sig, self.ws, ncol2)
        self.corR1 = self.corR1 + cor_sel_r1 + cor_noise_r
        self.corR2 = self.corR2 + cor_sel_r2 + cor_noise_r
        self.corE1 = self.corE1 + cor_noise_e1
        self.corE2 = self.corE2 + cor_noise_e2
        self.ncor = self.ncor + 1
        return

    def update_ellsum(self):
        """Updates the weighted sum of ellipticity and response with the currenct
        selection weight
        """
        self.sumE1 = np.sum(self.ell["fpfs_e1"] * self.ws)
        self.sumE2 = np.sum(self.ell["fpfs_e2"] * self.ws)
        self.sumR1 = np.sum(self.ell["fpfs_R1E"] * self.ws)
        self.sumR2 = np.sum(self.ell["fpfs_R2E"] * self.ws)
        return


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
