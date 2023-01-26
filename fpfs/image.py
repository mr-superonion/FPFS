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

import numba
import logging
import numpy as np
import numpy.lib.recfunctions as rfn
from . import imgutil


logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)


def detect_sources(
    img_data, psf_data, gsigma, thres=0.04, thres2=-0.01, klim=-1.0, pixel_scale=1.0
):
    """Returns the coordinates of detected sources

    Args:
        img_data (ndarray):      observed image
        psf_data (ndarray):      PSF image [must be well-centered]
        gsigma (float):         sigma of the Gaussian smoothing kernel in
                                *Fourier* space
        thres (float):          detection threshold
        thres2 (float):         peak identification difference threshold
        klim (float):           limiting wave number in Fourier space
        pixel_scale (float):    pixel scale in arcsec [set to 1]
    Returns:
        coords (ndarray):       peak values and the shear responses
    """

    psf_data = np.array(psf_data, dtype="<f8")
    assert (
        img_data.shape == psf_data.shape
    ), "image and PSF should have the same\
            shape. Please do padding before using this function."
    ny, nx = psf_data.shape
    psf_fourier = np.fft.rfft2(np.fft.ifftshift(psf_data))
    gauss_kernel, _ = imgutil.gauss_kernel(
        ny, nx, gsigma, return_grid=True, use_rfft=True
    )

    # convolved images
    img_fourier = np.fft.rfft2(img_data) / psf_fourier * gauss_kernel
    if klim > 0.0:
        # apply a truncation in Fourier space
        nxklim = int(klim * nx / np.pi / 2.0 + 0.5)
        nyklim = int(klim * ny / np.pi / 2.0 + 0.5)
        img_fourier[nyklim + 1 : -nyklim, :] = 0.0
        img_fourier[:, nxklim + 1 :] = 0.0
    else:
        # no truncation in Fourier space
        pass
    del psf_fourier, psf_data
    img_conv = np.fft.irfft2(img_fourier, (ny, nx))
    # the coordinates is not given, so we do another detection
    if not isinstance(thres, (int, float)):
        raise ValueError("thres must be float, but now got %s" % type(thres))
    if not isinstance(thres2, (int, float)):
        raise ValueError("thres2 must be float, but now got %s" % type(thres))
    coords = imgutil.find_peaks(img_conv, thres, thres2)
    return coords


@numba.njit
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
    klim = ngrid // 2 - 1
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


class measure_base:
    """A base class for measurement, which is extended to measure_source
    and measure_noise_cov
    Args:
        psf_data (ndarray):  an average PSF image used to initialize the task
        beta (float):       FPFS scale parameter
        nnord (int):        the highest order of Shapelets radial components
                            [default: 4]
        noise_ps (ndarray):   Estimated noise power function, if you have already
                            estimated noise power [default: None]
        pix_scale (float):  pixel scale in arcsec [default: 0.168 arcsec [HSC]]
    """

    _DefaultName = "measure_base"

    def __init__(
        self,
        psf_data,
        sigma_arcsec,
        nnord=4,
        pix_scale=0.168,
    ):
        if sigma_arcsec <= 0.0 or sigma_arcsec > 5.0:
            raise ValueError("sigma_arcsec should be positive and less than 5 arcsec")
        self.ngrid = psf_data.shape[0]
        self.nnord = nnord

        # Preparing PSF
        psf_data = np.array(psf_data, dtype="<f8")
        self.psf_fourier = np.fft.fftshift(np.fft.fft2(psf_data))
        self.psf_pow = imgutil.get_fourier_pow(psf_data)

        # A few import scales
        self.pix_scale = pix_scale  # this is only used to normalize basis functions
        self._dk = 2.0 * np.pi / self.ngrid  # assuming pixel scale is 1
        # # old function uses beta
        # # scale radius of PSF's Fourier transform (in units of dk)
        # sigmaPsf    =   imgutil.get_r_naive(self.psf_pow)*np.sqrt(2.)
        # # shapelet scale
        # sigma_pix   =   max(min(sigmaPsf*beta,6.),1.) # in units of dk
        # self.sigmaF =   sigma_pix*self._dk      # assume pixel scale is 1
        # sigma_arcsec  =   1./self.sigmaF*self.pix_scale

        self.sigmaF = self.pix_scale / sigma_arcsec
        sigma_pix = self.sigmaF / self._dk
        logging.info(
            "Gaussian kernel in configuration space: sigma= %.4f arcsec"
            % (sigma_arcsec)
        )
        # effective nyquest wave number
        self.klim_pix = get_klim(
            self.psf_pow, sigma_pix / np.sqrt(2.0)
        )  # in pixel units
        self.klim = self.klim_pix * self._dk  # assume pixel scale is 1
        # index bounds
        self._indx = np.arange(
            self.ngrid // 2 - self.klim_pix,
            self.ngrid // 2 + self.klim_pix + 1,
        )
        self._indy = self._indx[:, None]
        self._ind2d = np.ix_(self._indx, self._indx)
        return

    def set_klim(self, klim):
        """
        set klim, the area outside klim is supressed by the shaplet Gaussian
        kerenl
        """
        self.klim_pix = klim
        self._indx = np.arange(
            self.ngrid // 2 - self.klim_pix, self.ngrid // 2 + self.klim_pix + 1
        )
        self._indy = self._indx[:, None]
        self._ind2d = np.ix_(self._indx, self._indx)
        return

    def deconvolve(self, data, prder=1.0, frder=1.0):
        """Deconvolves input data with the PSF or PSF power

        Args:
            data (ndarray):
                galaxy power or galaxy Fourier transfer, origin is set to
                [ngrid//2,ngrid//2]
            prder (float):
                deconvlove order of PSF FT power
            frder (float):
                deconvlove order of PSF FT
        Returns:
            out (ndarray):
                Deconvolved galaxy power [truncated at klim]
        """
        out = np.zeros(data.shape, dtype=np.complex64)
        out[self._ind2d] = (
            data[self._ind2d]
            / self.psf_pow[self._ind2d] ** prder
            / self.psf_fourier[self._ind2d] ** frder
        )
        return out


class measure_noise_cov(measure_base):
    """A class to measure FPFS noise covariance of basis modes

    Args:
        psf_data (ndarray):  an average PSF image used to initialize the task
        beta (float):       FPFS scale parameter
        nnord (int):        the highest order of Shapelets radial components
                            [default: 4]
        noise_ps (ndarray):   Estimated noise power function, if you have already
                            estimated noise power [default: None]
        pix_scale (float):  pixel scale in arcsec [default: 0.168 arcsec [HSC]]
    """

    _DefaultName = "measure_noise_cov"

    def __init__(
        self,
        psf_data,
        sigma_arcsec,
        nnord=4,
        pix_scale=0.168,
    ):
        super().__init__(
            psf_data=psf_data,
            sigma_arcsec=sigma_arcsec,
            nnord=nnord,
            pix_scale=pix_scale,
        )
        bfunc, bnames = imgutil.fpfs_bases(
            self.ngrid,
            nnord,
            self.sigmaF,
        )
        self.bfunc = bfunc[:, self._indy, self._indx]
        self.bnames = bnames
        return

    def measure(self, noise_ps):
        """Prepares the basis to estimate covariance of measurement error

        Args:
            noise_ps (ndarray):     power spectrum (assuming homogeneous) of noise
        Return:
            cov_matrix (ndarray):   covariance matrix of FPFS basis modes
        """
        noise_ps = np.array(noise_ps, dtype="<f8")
        noise_ps_deconv = self.deconvolve(noise_ps, prder=1, frder=0)
        cov_matrix = (
            np.real(
                np.tensordot(
                    self.bfunc * noise_ps_deconv[None, self._indy, self._indx],
                    self.bfunc,
                    axes=((1, 2), (1, 2)),
                )
            )
            / self.pix_scale**4.0
        )
        return cov_matrix


class measure_source(measure_base):
    """A class to measure FPFS shapelet mode estimation

    Args:
        psf_data (ndarray):  an average PSF image used to initialize the task
        beta (float):       FPFS scale parameter
        nnord (int):        the highest order of Shapelets radial components
                            [default: 4]
        noise_mod (ndarray): Models to be used to fit noise power function using
                            the pixels at large k for each galaxy (if you wish
                            FPFS code to estiamte noise power). [default: None]
        noise_ps (ndarray):   Estimated noise power function, if you have already
                            estimated noise power [default: None]
        debug (bool):       Whether debug or not [default: False]
        pix_scale (float):  pixel scale in arcsec [default: 0.168 arcsec [HSC]]
    """

    _DefaultName = "measure_source"

    def __init__(
        self,
        psf_data,
        sigma_arcsec,
        nnord=4,
        noise_mod=None,
        noise_ps=None,
        debug=False,
        pix_scale=0.168,
    ):
        super().__init__(
            psf_data=psf_data,
            sigma_arcsec=sigma_arcsec,
            nnord=nnord,
            pix_scale=pix_scale,
        )
        if noise_ps is None:
            # estimated noise PS
            self.noise_ps = 0.0
            if noise_mod is not None:
                # PC models for noise PS
                self.noise_mod = np.array(noise_mod, dtype="<f8")
                self.noise_correct = True
            else:
                self.noise_mod = None
                self.noise_correct = False
        else:
            self.noise_correct = True
            self.noise_mod = None
            # Preparing noise
            if isinstance(noise_ps, np.ndarray):
                assert (
                    noise_ps.shape == psf_data.shape
                ), "the input noise power should have the same shape\
                    with input psf image"
                self.noise_ps = np.array(noise_ps, dtype="<f8")
            elif isinstance(noise_ps, float):
                self.noise_ps = (
                    np.ones_like(psf_data, dtype="<f8") * noise_ps * (self.ngrid) ** 2.0
                )
            else:
                raise TypeError("noise_ps should be either np.ndarray or float")

        # Preparing shapelet basis
        # nm = n*(nnord+1)+m
        # nnord is the maximum 'n' the code calculates
        if nnord == 4:
            # This setup is for shear response only
            # Only uses M00, M20, M22 (real and img) and M40, M42
            self._indM = np.array([0, 10, 12, 20, 22])[:, None, None]
            self._nameM = ["M00", "M20", "M22", "M40", "M42"]
        elif nnord == 6:
            # This setup is able to derive kappa response and shear response
            # Only uses M00, M20, M22 (real and img), M40, M42(real and img), M60
            self._indM = np.array([0, 14, 16, 28, 30, 42])[:, None, None]
            self._nameM = ["M00", "M20", "M22", "M40", "M42", "M60"]
        # TODO:Andy, please try to check which modes are necessay for shear
        # estimation with M_{42}
        else:
            raise ValueError(
                "only support for nnord= 4 or nnord=6, but your input\
                    is nnord=%d"
                % nnord
            )
        chi = imgutil.shapelets2d(self.ngrid, nnord, self.sigmaF).reshape(
            ((nnord + 1) ** 2, self.ngrid, self.ngrid)
        )[self._indM, self._indy, self._indx]
        psi = imgutil.detlets2d(self.ngrid, self.sigmaF)[:, :, self._indy, self._indx]
        self.prepare_chi(chi)
        self.prepare_psi(psi)
        if self.noise_correct:
            logging.info("measurement error covariance will be calculated")
            self.prepare_chicov(chi)
            self.prepare_detcov(chi, psi)
        else:
            logging.info("measurement covariance will not be calculated")
        del chi

        # others
        if debug:
            self.stackPow = np.zeros(psf_data.shape, dtype="<f8")
        else:
            self.stackPow = None
        return

    def prepare_chi(self, chi):
        """Prepares the basis to estimate shapelet modes

        Args:
            chi (ndarray):  2d shapelet basis
        """
        out = []
        if self.nnord == 4:
            out.append(chi.real[0])  # x00
            out.append(chi.real[1])  # x20
            out.append(chi.real[2])  # x22c
            out.append(chi.imag[2])  # x22s
            out.append(chi.real[3])  # x40
            out.append(chi.real[4])  # x42c
            out.append(chi.imag[4])  # x42s
            out = np.stack(out)
            self.chi_types = [
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
            out.append(chi.real[2])  # x22c
            out.append(chi.imag[2])  # x22s
            out.append(chi.real[3])  # x40
            out.append(chi.real[4])  # x42c
            out.append(chi.imag[4])  # x42s
            out.append(chi.real[5])  # x60
            self.chi_types = [
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
        assert len(out) == len(self.chi_types)
        self.chi = out
        del out
        return

    def prepare_psi(self, psi):
        """Prepares the basis to estimate chivatives (or equivalent moments)

        Args:
            psi (ndarray):  2d shapelet basis
        """
        self.psi_types = []
        out = []
        for _ in range(8):
            out.append(psi[_, 0])  # psi
            out.append(psi[_, 1])  # psi;1
            out.append(psi[_, 2])  # psi;2
            self.psi_types.append(("fpfs_v%d" % _, "<f8"))
            self.psi_types.append(("fpfs_v%dr1" % _, "<f8"))
            self.psi_types.append(("fpfs_v%dr2" % _, "<f8"))
        out = np.stack(out)
        assert len(out) == len(self.psi_types)
        self.psi = out
        self.psi0 = psi
        del out
        return

    def prepare_chicov(self, chi):
        """Prepares the basis to estimate covariance of measurement error

        Args:
            chi (ndarray):    2d shapelet basis
        """
        out = []
        # diagonal terms
        out.append(chi.real[0] * chi.real[0])  # x00 x00
        out.append(chi.real[1] * chi.real[1])  # x20 x20
        out.append(chi.real[2] * chi.real[2])  # x22c x22c
        out.append(chi.imag[2] * chi.imag[2])  # x22s x22s
        out.append(chi.real[3] * chi.real[3])  # x40 x40
        # off-diagonal terms
        #
        out.append(chi.real[0] * chi.real[1])  # x00 x20
        out.append(chi.real[0] * chi.real[2])  # x00 x22c
        out.append(chi.real[0] * chi.imag[2])  # x00 x22s
        out.append(chi.real[0] * chi.real[3])  # x00 x40
        out.append(chi.real[0] * chi.real[4])  # x00 x42c
        out.append(chi.real[0] * chi.imag[4])  # x00 x42s
        #
        out.append(chi.real[1] * chi.real[2])  # x20 x22c
        out.append(chi.real[1] * chi.imag[2])  # x20 x22s
        out.append(chi.real[1] * chi.real[3])  # x20 x40
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
            ("fpfs_N20N22c", "<f8"),
            ("fpfs_N20N22s", "<f8"),
            ("fpfs_N20N40", "<f8"),
            ("fpfs_N22cN42c", "<f8"),
            ("fpfs_N22sN42s", "<f8"),
        ]
        assert len(out) == len(self.cov_types)
        self.chicov = out
        del out
        return

    def prepare_detcov(self, chi, psi):
        """Prepares the basis to estimate covariance for detection

        Args:
            chi (ndarray):      2d shapelet basis
            psi (ndarray):      2d pixel basis
        """
        # get the Gaussian scale in Fourier space
        out = []
        self.det_types = []
        for _ in range(8):
            out.append(chi.real[0] * psi[_, 0])  # x00*psi
            out.append(chi.real[0] * psi[_, 1])  # x00*psi;1
            out.append(chi.real[0] * psi[_, 2])  # x00*psi;2
            out.append(chi.real[2] * psi[_, 0])  # x22c*psi
            out.append(chi.imag[2] * psi[_, 0])  # x22s*psi
            out.append(chi.real[2] * psi[_, 1])  # x22c*psi;1
            out.append(chi.imag[2] * psi[_, 2])  # x22s*psi;2
            out.append(chi.real[3] * psi[_, 0])  # x40*psi
            self.det_types.append(("fpfs_N00V%d" % _, "<f8"))
            self.det_types.append(("fpfs_N00V%dr1" % _, "<f8"))
            self.det_types.append(("fpfs_N00V%dr2" % _, "<f8"))
            self.det_types.append(("fpfs_N22cV%d" % _, "<f8"))
            self.det_types.append(("fpfs_N22sV%d" % _, "<f8"))
            self.det_types.append(("fpfs_N22cV%dr1" % _, "<f8"))
            self.det_types.append(("fpfs_N22sV%dr2" % _, "<f8"))
            self.det_types.append(("fpfs_N40V%d" % _, "<f8"))
        out = np.stack(out)
        assert len(out) == len(self.det_types)
        self.detcov = out
        del out
        return

    def itransform(self, data, out_type="chi"):
        """Projects image onto shapelet basis vectors

        Args:
            data (ndarray): image to transfer
            out_type (str): transform type ('chi', 'psi', 'cov', or 'detcov')
        Returns:
            out (ndarray):  projection in shapelet space
        """

        # Here we divide by self.pix_scale**2. for modes since pixel value are
        # flux in pixel (in unit of nano Jy for HSC). After dividing pix_scale**2.,
        # in units of (nano Jy/ arcsec^2), dk^2 has unit (1/ arcsec^2)
        # Correspondingly, covariances are divided by self.pix_scale**4.
        if out_type == "chi":
            # chivatives/Moments
            _ = (
                np.sum(data[None, self._indy, self._indx] * self.chi, axis=(1, 2)).real
                / self.pix_scale**2.0
            )
            out = np.array(tuple(_), dtype=self.chi_types)
        elif out_type == "psi":
            # chivatives/Moments
            _ = (
                np.sum(data[None, self._indy, self._indx] * self.psi, axis=(1, 2)).real
                / self.pix_scale**2.0
            )
            out = np.array(tuple(_), dtype=self.psi_types)
        elif out_type == "cov":
            # covariance of moments
            _ = (
                np.sum(
                    data[None, self._indy, self._indx] * self.chicov, axis=(1, 2)
                ).real
                / self.pix_scale**4.0
            )
            out = np.array(tuple(_), dtype=self.cov_types)
        elif out_type == "detcov":
            # covariance of pixels
            _ = (
                np.sum(
                    data[None, self._indy, self._indx] * self.detcov, axis=(1, 2)
                ).real
                / self.pix_scale**4.0
            )
            out = np.array(tuple(_), dtype=self.det_types)
        else:
            raise ValueError(
                "out_type can only be 'chi', 'cov' or 'Det',\
                    but the input is '%s'"
                % out_type
            )
        return out

    def measure(self, gal_data, psf_fourier=None, noise_ps=None):
        """Measures the FPFS moments

        Args:
            gal_data (ndarray|list):     galaxy image
            psf_fourier (ndarray):           PSF's Fourier transform
            noise_ps (ndarray):           noise Fourier power function
        Returns:
            out (ndarray):              FPFS moments
        """
        if psf_fourier is not None:
            self.psf_fourier = psf_fourier
            self.psf_pow = (np.conjugate(psf_fourier) * psf_fourier).real
        if noise_ps is not None:
            self.noise_ps = noise_ps
        if isinstance(gal_data, np.ndarray):
            assert gal_data.shape[-1] == gal_data.shape[-2]
            if len(gal_data.shape) == 2:
                # single galaxy
                out = self.__measure(gal_data)
                return out
            elif len(gal_data.shape) == 3:
                results = []
                for gal in gal_data:
                    _g = self.__measure(gal)
                    results.append(_g)
                out = rfn.stack_arrays(results, usemask=False)
                return out
            else:
                raise ValueError("Input galaxy data has wrong ndarray shape.")
        elif isinstance(gal_data, list):
            assert isinstance(gal_data[0], np.ndarray)
            # list of galaxies
            results = []
            for gal in gal_data:
                _g = self.__measure(gal)
                results.append(_g)
            out = rfn.stack_arrays(results, usemask=False)
            return out
        else:
            raise TypeError(
                "Input galaxy data has wrong type (neither list nor ndarray)."
            )

    def __measure(self, data):
        """Measures the FPFS moments

        Args:
            data (ndarray):     galaxy image array [centroid does not matter]
        Returns:
            mm (ndarray):       FPFS moments
        """
        if self.stackPow is not None:
            gal_pow = imgutil.get_fourier_pow(data)
            self.stackPow += gal_pow
            return np.empty(1)
        gal_fourier = np.fft.fftshift(np.fft.fft2(data))
        gal_deconv = self.deconvolve(gal_fourier, prder=0.0, frder=1)
        mm = self.itransform(gal_deconv, out_type="chi")
        mp = self.itransform(gal_deconv, out_type="psi")
        mm = rfn.merge_arrays([mm, mp], flatten=True, usemask=False)
        del gal_deconv, mp

        if self.noise_correct:
            # do noise covariance estimation
            if self.noise_mod is not None:
                # fit the noise power from the galaxy power
                gal_pow = imgutil.get_fourier_pow(data)
                noise_ps = imgutil.fit_noise_ps(
                    self.ngrid, gal_pow, self.noise_mod, self.klim_pix
                )
                del gal_pow
            else:
                # use the input noise power
                noise_ps = self.noise_ps
            noise_ps_deconv = self.deconvolve(noise_ps, prder=1, frder=0)
            nn = self.itransform(noise_ps_deconv, out_type="cov")
            dd = self.itransform(noise_ps_deconv, out_type="detcov")
            del noise_ps_deconv
            mm = rfn.merge_arrays([mm, nn, dd], flatten=True, usemask=False)
        return mm


class test_noise:
    _DefaultName = "fpfsTestNoi"

    def __init__(self, ngrid, noise_mod=None, noise_ps=None):
        self.ngrid = ngrid
        # Preparing noise Model
        self.noise_mod = noise_mod
        self.noise_ps = noise_ps
        self.rlim = int(ngrid // 4)
        return

    def test(self, gal_data):
        """Tests the noise subtraction

        Args:
            gal_data:    galaxy image [float array (list)]
        Returns:
            out :   FPFS moments
        """
        if isinstance(gal_data, np.ndarray):
            # single galaxy
            out = self.__test(gal_data)
            return out
        elif isinstance(gal_data, list):
            assert isinstance(gal_data[0], np.ndarray)
            # list of galaxies
            results = []
            for gal in gal_data:
                _g = self.__test(gal)
                results.append(_g)
            out = np.stack(results)
            return out

    def __test(self, data):
        """Tests the noise subtraction

        Args:
            data:    image array [centroid does not matter]
        """
        assert len(data.shape) == 2
        gal_pow = imgutil.get_fourier_pow(data)
        if (self.noise_ps is not None) or (self.noise_mod is not None):
            if self.noise_mod is not None:
                self.noise_ps = imgutil.fit_noise_ps(
                    self.ngrid, gal_pow, self.noise_mod, self.rlim
                )
            gal_pow = gal_pow - self.noise_ps
        return gal_pow
