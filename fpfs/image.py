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

import jax
import logging
import numpy as np
import jax.numpy as jnp
from . import imgutil
from functools import partial


logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)


def detect_sources(
    img_data,
    psf_data,
    gsigma,
    thres=0.04,
    thres2=-0.01,
    klim=-1.0,
    pixel_scale=1.0,
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

    psf_data = jnp.array(psf_data, dtype="<f8")
    assert (
        img_data.shape == psf_data.shape
    ), "image and PSF should have the same\
            shape. Please do padding before using this function."
    img_conv = imgutil.convolve2gausspsf(img_data, psf_data, gsigma, klim)
    # the coordinates is not given, so we do another detection
    if not isinstance(thres, (int, float)):
        raise ValueError("thres must be float, but now got %s" % type(thres))
    if not isinstance(thres2, (int, float)):
        raise ValueError("thres2 must be float, but now got %s" % type(thres))
    if not thres > 0.0:
        raise ValueError("detection threshold should be positive")
    if not thres2 <= 0.0:
        raise ValueError("difference threshold should be non-positive")
    dd = imgutil.find_peaks(img_conv, thres, thres2)
    coords = np.array(
        np.zeros(dd.size // 2),
        dtype=[("fpfs_y", "i4"), ("fpfs_x", "i4")],
    )
    coords["fpfs_y"] = dd[0]
    coords["fpfs_x"] = dd[1]
    del dd
    return coords


class measure_base:
    """A base class for measurement, which is extended to measure_source
    and measure_noise_cov
    Args:
        psf_data (ndarray):     an average PSF image used to initialize the task
        nnord (int):            the highest order of Shapelets radial
                                components [default: 4]
        sigma_arcsec (float):   Shapelet kernel size
        pix_scale (float):      pixel scale in arcsec [default: 0.168 arcsec]
        sigma_detect (float):   detection kernel size
    """

    _DefaultName = "measure_base"

    def __init__(
        self,
        psf_data,
        sigma_arcsec,
        nnord=4,
        pix_scale=0.168,
        sigma_detect=None,
    ):
        if sigma_arcsec <= 0.0 or sigma_arcsec > 5.0:
            raise ValueError("sigma_arcsec should be positive and less than 5 arcsec")
        if sigma_detect is None:
            sigma_detect = sigma_arcsec
        self.ngrid = psf_data.shape[0]
        self.nnord = nnord

        # Preparing PSF
        psf_data = jnp.array(psf_data, dtype="<f8")
        self.psf_fourier = jnp.fft.fftshift(jnp.fft.fft2(psf_data))
        self.psf_pow = jnp.array(imgutil.get_fourier_pow(psf_data))

        # A few import scales
        self.pix_scale = pix_scale  # this is only used to normalize basis functions
        self._dk = 2.0 * jnp.pi / self.ngrid  # assuming pixel scale is 1
        # # old function uses beta
        # # scale radius of PSF's Fourier transform (in units of dk)
        # sigmaPsf    =   imgutil.get_r_naive(self.psf_pow)*jnp.sqrt(2.)
        # # shapelet scale
        # sigma_pix   =   max(min(sigmaPsf*beta,6.),1.) # in units of dk
        # self.sigmaF =   sigma_pix*self._dk      # assume pixel scale is 1
        # sigma_arcsec  =   1./self.sigmaF*self.pix_scale

        # the following two assumes pixel_scale = 1
        self.sigmaF = self.pix_scale / sigma_arcsec
        self.sigmaF_det = self.pix_scale / sigma_detect
        sigma_pixf = self.sigmaF / self._dk
        sigma_pixf_det = self.sigmaF_det / self._dk
        logging.info(
            "Shapelet kernel in configuration space: sigma= %.4f arcsec"
            % (sigma_arcsec)
        )
        logging.info(
            "Detection kernel in configuration space: sigma= %.4f arcsec"
            % (sigma_detect)
        )
        # effective nyquest wave number
        self.klim_pix = imgutil.get_klim(
            self.psf_pow, (sigma_pixf + sigma_pixf_det) / 2.0 / jnp.sqrt(2.0)
        )  # in pixel units
        self.klim = self.klim_pix * self._dk  # assume pixel scale is 1
        # index bounds
        self._indx = jnp.arange(
            self.ngrid // 2 - self.klim_pix,
            self.ngrid // 2 + self.klim_pix + 1,
        )
        self._indy = self._indx[:, None]
        self._ind2d = jnp.ix_(self._indx, self._indx)
        return

    def set_klim(self, klim):
        """
        set klim, the area outside klim is supressed by the shaplet Gaussian
        kerenl
        """
        self.klim_pix = klim
        self._indx = jnp.arange(
            self.ngrid // 2 - self.klim_pix, self.ngrid // 2 + self.klim_pix + 1
        )
        self._indy = self._indx[:, None]
        self._ind2d = jnp.ix_(self._indx, self._indx)
        return

    @partial(jax.jit, static_argnames=["self"])
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
        out = jnp.zeros(data.shape, dtype="complex128")
        out2 = out.at[self._ind2d].set(
            data[self._ind2d]
            / self.psf_pow[self._ind2d] ** prder
            / self.psf_fourier[self._ind2d] ** frder
        )
        return out2


class measure_noise_cov(measure_base):
    """A class to measure FPFS noise covariance of basis modes

    Args:
        psf_data (ndarray):     an average PSF image used to initialize the task
        sigma_arcsec (float):   Shapelet kernel size
        nnord (int):            the highest order of Shapelets radial
                                components [default: 4]
        pix_scale (float):      pixel scale in arcsec [default: 0.168 arcsec]
        sigma_detect (float):   detection kernel size
    """

    _DefaultName = "measure_noise_cov"

    def __init__(
        self,
        psf_data,
        sigma_arcsec,
        nnord=4,
        pix_scale=0.168,
        sigma_detect=None,
    ):
        super().__init__(
            psf_data=psf_data,
            sigma_arcsec=sigma_arcsec,
            nnord=nnord,
            pix_scale=pix_scale,
            sigma_detect=sigma_detect,
        )
        bfunc, bnames = imgutil.fpfs_bases(
            self.ngrid,
            nnord,
            self.sigmaF,
            self.sigmaF_det,
        )
        self.bfunc = bfunc[:, self._indy, self._indx]
        self.bnames = bnames
        return

    def measure(self, noise_ps):
        """Estimate covariance of measurement error in impt form

        Args:
            noise_ps (ndarray):     power spectrum (assuming homogeneous) of noise
        Return:
            cov_matrix (ndarray):   covariance matrix of FPFS basis modes
        """
        noise_ps = jnp.array(noise_ps, dtype="<f8")
        noise_ps_deconv = self.deconvolve(noise_ps, prder=1, frder=0)
        cov_matrix = (
            jnp.real(
                jnp.tensordot(
                    self.bfunc * noise_ps_deconv[None, self._indy, self._indx],
                    jnp.conjugate(self.bfunc),
                    axes=((1, 2), (1, 2)),
                )
            )
            / self.pix_scale**4.0
        )
        return cov_matrix


class measure_source(measure_base):
    """A class to measure FPFS shapelet mode estimation

    Args:
        psf_data (ndarray):     an average PSF image used to initialize the task
        sigma_arcsec (float):   Shapelet kernel size
        nnord (int):            the highest order of Shapelets radial components
                                [default: 4]
        pix_scale (float):      pixel scale in arcsec [default: 0.168 arcsec]
        sigma_detect (float):   detection kernel size
    """

    _DefaultName = "measure_source"

    def __init__(
        self,
        psf_data,
        sigma_arcsec,
        nnord=4,
        pix_scale=0.168,
        sigma_detect=None,
    ):
        super().__init__(
            psf_data=psf_data,
            sigma_arcsec=sigma_arcsec,
            nnord=nnord,
            pix_scale=pix_scale,
            sigma_detect=sigma_detect,
        )
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
        psi = imgutil.detlets2d(
            self.ngrid,
            self.sigmaF_det,
        )[:, :, self._indy, self._indx]
        self.prepare_chi(chi)
        self.prepare_psi(psi)
        del chi
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
        out = jnp.stack(out)
        self.chi = out
        return

    def prepare_psi(self, psi):
        """Prepares the basis to estimate detection modes

        Args:
            psi (ndarray):  2d detection basis
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
        out = jnp.stack(out)
        assert len(out) == len(self.psi_types)
        self.psi = out
        return

    @partial(jax.jit, static_argnames=["self"])
    def _itransform_chi(self, data):
        """Projects image onto shapelet basis vectors

        Args:
            data (ndarray): image to transfer
        Returns:
            out (ndarray):  projection in shapelet space
        """

        # Here we divide by self.pix_scale**2. since pixel values are flux in
        # pixel (in unit of nano Jy for HSC). After dividing pix_scale**2., in
        # units of (nano Jy/ arcsec^2), dk^2 has unit (1/ arcsec^2)
        # Correspondingly, covariances are divided by self.pix_scale**4.
        # chivatives/Moments
        out = (
            jnp.sum(
                data[None, self._indy, self._indx] * self.chi,
                axis=(1, 2),
            ).real
            / self.pix_scale**2.0
        )
        return tuple(out)

    @partial(jax.jit, static_argnames=["self"])
    def _itransform_psi(self, data):
        """Projects image onto shapelet basis vectors

        Args:
            data (ndarray): image to transfer
        Returns:
            out (ndarray):  projection in shapelet space
        """

        # Here we divide by self.pix_scale**2. since pixel values are flux in
        # pixel (in unit of nano Jy for HSC). After dividing pix_scale**2., in
        # units of (nano Jy/ arcsec^2), dk^2 has unit (1/ arcsec^2)
        # Correspondingly, covariances are divided by self.pix_scale**4.
        # chivatives/Moments
        out = (
            jnp.sum(
                data[None, self._indy, self._indx] * self.psi,
                axis=(1, 2),
            ).real
            / self.pix_scale**2.0
        )
        return tuple(out)

    def measure(self, gal_data, psf_fourier=None):
        """Measures the FPFS moments

        Args:
            gal_data (ndarray|list):    galaxy image
            psf_fourier (ndarray):      PSF's Fourier transform
        Returns:
            out (ndarray):              FPFS moments
        """
        if psf_fourier is not None:
            self.psf_fourier = psf_fourier
            self.psf_pow = (jnp.conjugate(psf_fourier) * psf_fourier).real
        if isinstance(gal_data, np.ndarray):
            assert gal_data.shape[-1] == gal_data.shape[-2]
            if len(gal_data.shape) == 2:
                # single galaxy
                out = self.__results(self.__measure(gal_data))
                return out
            elif len(gal_data.shape) == 3:
                results = []
                for gal in gal_data:
                    _g = self.__measure(gal)
                    results.append(_g)
                out = self.__results((results))
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
            out = self.__results((results))
            return out
        else:
            raise TypeError(
                "Input galaxy data has wrong type (neither list nor ndarray)."
            )

    @partial(jax.jit, static_argnames=["self"])
    def __measure(self, data):
        """Measures the FPFS moments

        Args:
            data (ndarray):     galaxy image array [centroid does not matter]
        Returns:
            mm (ndarray):       FPFS moments
        """
        gal_fourier = jnp.fft.fftshift(jnp.fft.fft2(data))
        gal_deconv = self.deconvolve(gal_fourier, prder=0.0, frder=1)
        mm = self._itransform_chi(gal_deconv)  # FPFS shapelets
        mp = self._itransform_psi(gal_deconv)  # FPFS detection
        return mm + mp

    def __results(self, reslist):
        tps = self.chi_types + self.psi_types
        out = np.array(reslist, dtype=tps)
        return out
