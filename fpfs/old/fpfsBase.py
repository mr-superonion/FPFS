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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib

import numba
import logging
import numpy as np
from . import imgutil
import numpy.lib.recfunctions as rfn
from .default import det_inds


@numba.njit
def get_klim(psf_array, sigma):
    """
    Get klim, the region outside klim is supressed by the shaplet Gaussian
    kernel in FPFS shear estimation method; therefore we set values in this
    region to zeros.

    Args:
        psf_array (ndarray):    PSF's Fourier power or Fourier transform

    Returns:
        klim (float):           the limit radius
    """
    ngrid = psf_array.shape[0]
    thres = 1.0e-10
    klim = ngrid // 2
    for dist in range(ngrid // 5, ngrid // 2 - 1):
        ave = abs(
            np.exp(-(dist**2.0) / 2.0 / sigma**2.0)
            / psf_array[ngrid // 2 + dist, ngrid // 2]
        )
        ave += abs(
            np.exp(-(dist**2.0) / 2.0 / sigma**2.0)
            / psf_array[ngrid // 2, ngrid // 2 + dist]
        )
        ave = ave / 2.0
        if ave <= thres:
            klim = dist
            break
    return klim


class fpfsTask:
    """
    A class to measure FPFS shapelet mode estimation.

    Args:
        psfData (ndarray):
            an average PSF image used to initialize the task
        beta (float):
            FPFS scale parameter
        nnord (int):
            the highest order of Shapelets radial components [default: 4]
        noiModel (ndarray):
            Models to be used to fit noise power function using the pixels at
            large k for each galaxy (if you wish FPFS code to estiamte noise
            power). [default: None]
        noiFit (ndarray):
            Estimated noise power function (if you have already estimated noise
            power) [default: None]
        deubg (bool):
            Whether debug or not [default: False]
    """

    _DefaultName = "fpfsTask"

    def __init__(self, psfData, beta, nnord=4, noiModel=None, noiFit=None, debug=False):
        if not isinstance(beta, (float, int)):
            raise TypeError("Input beta should be float.")
        if beta >= 1.0 or beta <= 0.0:
            raise ValueError("Input beta shoul be in range (0,1)")
        self.ngrid = psfData.shape[0]
        self._dk = 2.0 * np.pi / self.ngrid

        # Preparing noise
        self.noise_correct = False
        if noiFit is not None:
            self.noise_correct = True
            if isinstance(noiFit, np.ndarray):
                assert (
                    noiFit.shape == psfData.shape
                ), "the input noise power should have the same shape with \
                    input psf image"
                self.noiFit = np.array(noiFit, dtype="<f8")
            elif isinstance(noiFit, float):
                self.noiFit = (
                    np.ones(psfData.shape, dtype="<f8") * noiFit * (self.ngrid) ** 2.0
                )
            else:
                raise TypeError("noiFit should be either np.ndarray or float")
        else:
            self.noiFit = 0.0
        if noiModel is not None:
            self.noise_correct = True
            self.noiModel = np.array(noiModel, dtype="<f8")
        else:
            self.noiModel = None
        self.noiFit0 = self.noiFit  # we keep a copy of the initial noise Fourier power

        # Preparing PSF
        psfData = np.array(psfData, dtype="<f8")
        self.psfFou = np.fft.fftshift(np.fft.fft2(psfData))
        self.psfFou0 = self.psfFou.copy()  # we keep a copy of the initial PSF
        self.psfPow = imgutil.getFouPow(psfData)
        self.psfPow0 = self.psfPow.copy()  # we keep a copy of the initial PSF power

        # A few import scales
        # scale radius of PSF's Fourier transform (in units of pixel)
        sigmaPsf = imgutil.getRnaive(self.psfPow) * np.sqrt(2.0)
        # shapelet scale
        sigma_pix = max(min(sigmaPsf * beta, 6.0), 1.0)  # in pixel units
        self.sigmaF = sigma_pix * self._dk  # setting delta_pix=1
        # effective nyquest wave number
        self.klim_pix = get_klim(
            self.psfPow, sigma_pix / np.sqrt(2.0)
        )  # in pixel units
        self.klim = self.klim_pix * self._dk  # setting delta_pix=1
        # index selectors
        self._indX = np.arange(
            self.ngrid // 2 - self.klim_pix, self.ngrid // 2 + self.klim_pix + 1
        )
        self._indY = self._indX[:, None]
        self._ind2D = np.ix_(self._indX, self._indX)

        # Preparing shapelet basis
        # nm = m*(nnord+1)+n
        if nnord == 4:
            # This setup is for shear response only
            # Only uses M00, M20, M22 (real and img) and M40, M42
            self._indC = np.array([0, 10, 12, 20, 22])[:, None, None]
        elif nnord == 6:
            # This setup is able to derive kappa response as well as shear response
            # Only uses M00, M20, M22 (real and img), M40, M42(real and img), M60
            self._indC = np.array([0, 14, 16, 28, 30, 42])[:, None, None]
        else:
            raise ValueError("only support for nnord= 4 or nnord=6")
        self.nnord = nnord
        chi = imgutil.shapelets2D(self.ngrid, nnord, self.sigmaF).reshape(
            ((nnord + 1) ** 2, self.ngrid, self.ngrid)
        )[self._indC, self._indY, self._indX]
        self.prepare_ChiDeri(chi)
        if self.noise_correct:
            logging.info("measurement error covariance will be calculated")
            self.prepare_ChiCov(chi)
            self.prepare_ChiDet(chi)
        else:
            logging.info("measurement covariance will not be calculated")
        del chi

        # others
        if debug:
            self.stackPow = np.zeros(psfData.shape, dtype="<f8")
        else:
            self.stackPow = None
        return

    def reset_psf(self):
        """
        reset psf power to the average PSF used to initialize the task
        """
        self.psfFou = self.psfFou0
        self.psfPow = np.conjugate(self.psfFou) * self.psfFou
        return

    def reset_noiFit(self):
        """
        reset noiFit to the one used to initialize the task
        """
        self.noiFit = self.noiFit0
        return

    def setRlim(self, klim):
        """
        set klim, the area outside klim is supressed by the shaplet Gaussian
        kerenl
        """
        self.klim_pix = klim
        self._indX = np.arange(
            self.ngrid // 2 - self.klim_pix, self.ngrid // 2 + self.klim_pix + 1
        )
        self._indY = self._indX[:, None]
        self._ind2D = np.ix_(self._indX, self._indX)
        return

    def prepare_ChiDeri(self, chi):
        """
        prepare the basis to estimate Derivatives (or equivalent moments)
        Args:
            chi (ndarray):  2D shapelet basis
        """
        out = []
        if self.nnord == 4:
            out.append(chi.real[0])  # x00
            out.append(chi.real[1])  # x20
            out.append(chi.real[2])
            out.append(chi.imag[2])  # x22c,s
            out.append(chi.real[3])  # x40
            out.append(chi.real[4])
            out.append(chi.imag[4])  # x42c,s
            out = np.stack(out)
            self.deri_types = [
                ("fpfs_M00", "<f8"),
                ("fpfs_M20", "<f8"),
                ("fpfs_M22c", "<f8"),
                ("fpfs_M22s", "<f8"),
                ("fpfs_M40", "<f8"),
                ("fpfs_M42c", "<f8"),
                ("fpfs_M42s", "<f8"),
            ]
        elif self.nnord == 6:
            out.append(chi.real[0])  # x00
            out.append(chi.real[1])  # x20
            out.append(chi.real[2])
            out.append(chi.imag[2])  # x22c,s
            out.append(chi.real[3])  # x40
            out.append(chi.real[4])
            out.append(chi.imag[4])  # x42c,s
            out.append(chi.real[5])  # x60
            self.deri_types = [
                ("fpfs_M00", "<f8"),
                ("fpfs_M20", "<f8"),
                ("fpfs_M22c", "<f8"),
                ("fpfs_M22s", "<f8"),
                ("fpfs_M40", "<f8"),
                ("fpfs_M42c", "<f8"),
                ("fpfs_M42s", "<f8"),
                ("fpfs_M60", "<f8"),
            ]
        else:
            raise ValueError("only support for nnord=4 or nnord=6")
        assert len(out) == len(self.deri_types)
        self.chiD = out
        del out
        return

    def prepare_ChiCov(self, chi):
        """
        prepare the basis to estimate covariance of measurement error

        Args:
            chi (ndarray):    2D shapelet basis
        """
        out = []
        # diagonal terms
        out.append(chi.real[0] * chi.real[0])  # x00 x00
        out.append(chi.real[1] * chi.real[1])  # x20 x20
        out.append(chi.real[2] * chi.real[2])  # x22c x22c
        out.append(chi.imag[2] * chi.imag[2])  # x22s x22s
        out.append(chi.real[3] * chi.real[3])  # x40 x40
        # off-diagonal terms
        out.append(chi.real[0] * chi.real[1])  # x00 x20
        out.append(chi.real[0] * chi.real[2])  # x00 x22c
        out.append(chi.real[0] * chi.imag[2])  # x00 x22s
        out.append(chi.real[0] * chi.real[3])  # x00 x40
        out.append(chi.real[0] * chi.real[4])  # x00 x42c
        out.append(chi.real[0] * chi.imag[4])  # x00 x42s
        out.append(chi.real[2] * chi.real[4])  # x22c x42c
        out.append(chi.imag[2] * chi.imag[4])  # x22s x42s
        out = np.stack(out)
        self.cov_types = [
            ("fpfs_N00N00", "<f8"),
            ("fpfs_N20N20", "<f8"),
            ("fpfs_N22cN22c", "<f8"),
            ("fpfs_N22sN22s", "<f8"),
            ("fpfs_N40N40", "<f8"),
            ("fpfs_N00N20", "<f8"),
            ("fpfs_N00N22c", "<f8"),
            ("fpfs_N00N22s", "<f8"),
            ("fpfs_N00N40", "<f8"),
            ("fpfs_N00N42c", "<f8"),
            ("fpfs_N00N42s", "<f8"),
            ("fpfs_N22cN42c", "<f8"),
            ("fpfs_N22sN42s", "<f8"),
        ]
        assert len(out) == len(self.cov_types)
        self.chiCov = out
        del out
        return

    def prepare_ChiDet(self, chi):
        """
        prepare the basis to estimate covariance for detection
        Args:
            chi (ndarray):      2D shapelet basis
        """
        # get the Gaussian scale in Fourier space
        gKer, (k2grid, k1grid) = imgutil.gauss_kernel(
            self.ngrid, self.ngrid, self.sigmaF, do_shift=True, return_grid=True
        )
        q1Ker = (k1grid**2.0 - k2grid**2.0) / self.sigmaF**2.0 * gKer
        q2Ker = (2.0 * k1grid * k2grid) / self.sigmaF**2.0 * gKer
        d1Ker = (-1j * k1grid) * gKer
        d2Ker = (-1j * k2grid) * gKer
        out = []
        self.det_types = []
        for (j, i) in det_inds:
            y = j - 2
            x = i - 2
            r1 = (q1Ker + x * d1Ker - y * d2Ker) * np.exp(
                1j * (k1grid * x + k2grid * y)
            )
            r2 = (q2Ker + y * d1Ker + x * d2Ker) * np.exp(
                1j * (k1grid * x + k2grid * y)
            )
            r1 = r1[self._indY, self._indX] / self.ngrid**2.0
            r2 = r2[self._indY, self._indX] / self.ngrid**2.0
            out.append(chi.real[0] * r1)  # x00*phi;1
            out.append(chi.real[0] * r2)  # x00*phi;2
            out.append(chi.real[2] * r1)  # x22c*phi;1
            out.append(chi.imag[2] * r2)  # x22s*phi;2
            self.det_types.append(("pdet_N00F%d%dr1" % (j, i), "<f8"))
            self.det_types.append(("pdet_N00F%d%dr2" % (j, i), "<f8"))
            self.det_types.append(("pdet_N22cF%d%dr1" % (j, i), "<f8"))
            self.det_types.append(("pdet_N22sF%d%dr2" % (j, i), "<f8"))
        out = np.stack(out)
        assert len(out) == len(self.det_types)
        self.chiDet = out
        del out
        return

    def deconvolve(self, data, prder=1.0, frder=1.0):
        """
        Deconvolve input data with the PSF or PSF power

        Args:
            data (ndarray):
                galaxy power or galaxy Fourier transfer (ngrid//2,ngrid//2) is origin
            prder (float):
                deconvlove order of PSF FT power
            frder (float):
                deconvlove order of PSF FT

        Returns:
            out (ndarray):
                Deconvolved galaxy power (truncated at klim)
        """
        out = np.zeros(data.shape, dtype=np.complex64)
        out[self._ind2D] = (
            data[self._ind2D]
            / self.psfPow[self._ind2D] ** prder
            / self.psfFou[self._ind2D] ** frder
        )
        return out

    def itransform(self, data, out_type="Deri"):
        """
        Project image onto shapelet basis vectors

        Args:
            data (ndarray): image to transfer
            out_type (str): transform type ('Deri', 'Cov', or 'Det')

        Returns:
            out (ndarray):  projection in shapelet space
        """

        if out_type == "Deri":
            # Derivatives/Moments
            _ = np.sum(data[None, self._indY, self._indX] * self.chiD, axis=(1, 2)).real
            out = np.array(tuple(_), dtype=self.deri_types)
        elif out_type == "Cov":
            # covariance of moments
            _ = np.sum(
                data[None, self._indY, self._indX] * self.chiCov, axis=(1, 2)
            ).real
            out = np.array(tuple(_), dtype=self.cov_types)
        elif out_type == "Det":
            # covariance of pixels
            _ = np.sum(
                data[None, self._indY, self._indX] * self.chiDet, axis=(1, 2)
            ).real
            out = np.array(tuple(_), dtype=self.det_types)
        elif out_type == "PSF":
            # Derivatives/Moments
            _ = np.sum(data[None, self._indY, self._indX] * self.chiD, axis=(1, 2)).real
            out = np.array(tuple(_), dtype=self.psf_types)
        else:
            raise ValueError(
                "out_type can only be 'Deri', 'Cov' or 'Det',\
                    but the input is '%s'"
                % out_type
            )
        return out

    def measure(self, galData, psfFou=None, noiFit=None):
        """
        Measure the FPFS moments

        Args:
            galData (ndarray|list):     galaxy image
            psfFou (ndarray):           PSF's Fourier transform
            noiFit (ndarray):           noise Fourier power function

        Returns:
            out (ndarray):              FPFS moments
        """
        if psfFou is not None:
            self.psfFou = psfFou
            self.psfPow = (np.conjugate(psfFou) * psfFou).real
        if noiFit is not None:
            self.noiFit = noiFit
        if isinstance(galData, np.ndarray):
            assert galData.shape[-1] == galData.shape[-2]
            if len(galData.shape) == 2:
                # single galaxy
                out = self.__measure(galData)
                return out
            elif len(galData.shape) == 3:
                results = []
                for gal in galData:
                    _g = self.__measure(gal)
                    results.append(_g)
                out = rfn.stack_arrays(results, usemask=False)
                return out
            else:
                raise ValueError("Input galaxy data has wrong ndarray shape.")
        elif isinstance(galData, list):
            assert isinstance(galData[0], np.ndarray)
            # list of galaxies
            results = []
            for gal in galData:
                _g = self.__measure(gal)
                results.append(_g)
            out = rfn.stack_arrays(results, usemask=False)
            return out
        else:
            raise TypeError(
                "Input galaxy data has wrong type (neither list nor ndarray)."
            )

    def __measure(self, data):
        """
        Measure the FPFS moments

        Args:
            data (ndarray):     galaxy image array (centroid does not matter)

        Returns:
            mm (ndarray):       FPFS moments
        """
        if self.stackPow is not None:
            galPow = imgutil.getFouPow(data)
            self.stackPow += galPow
            return np.empty(1)
        galFou = np.fft.fftshift(np.fft.fft2(data))
        decG = self.deconvolve(galFou, prder=0.0, frder=1)
        mm = self.itransform(decG, out_type="Deri")
        del decG

        nn = None  # photon noise covariance array
        dd = None  # detection covariance array

        if self.noise_correct:
            # do noise covariance estimation
            if self.noiModel is not None:
                # fit the noise power from the galaxy power
                galPow = imgutil.getFouPow(data)
                noiFit = imgutil.fitNoiPow(
                    self.ngrid, galPow, self.noiModel, self.klim_pix
                )
                del galPow
            else:
                # use the input noise power
                noiFit = self.noiFit
            decNp = self.deconvolve(noiFit, prder=1, frder=0)
            nn = self.itransform(decNp, out_type="Cov")
            dd = self.itransform(decNp, out_type="Det")
            del decNp
            mm = rfn.merge_arrays([mm, nn, dd], flatten=True, usemask=False)
        return mm
