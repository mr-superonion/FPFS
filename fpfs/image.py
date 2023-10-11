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


def results_coords(dd):
    coords = np.rec.fromarrays(
        dd.T,
        dtype=[("fpfs_y", "i4"), ("fpfs_x", "i4")],
    )
    return coords


class measure_base:
    """A base class for measurement, which is extended to measure_source
    and measure_noise_cov
    Args:
        psf_data (ndarray):     an average PSF image used to initialize the task
        pix_scale (float):      pixel scale in arcsec
        sigma_arcsec (float):   Shapelet kernel size
        sigma_detect (float):   detection kernel size
        nnord (int):            the highest order of Shapelets radial
                                components [default: 4]
    """

    _DefaultName = "measure_base"

    def __init__(
        self,
        psf_data,
        pix_scale,
        sigma_arcsec,
        sigma_detect=None,
        nnord=4,
    ):
        if sigma_arcsec <= 0.0 or sigma_arcsec > 5.0:
            raise ValueError("sigma_arcsec should be positive and less than 5 arcsec")
        self.ngrid = psf_data.shape[0]
        self.nnord = nnord
        if sigma_detect is None:
            sigma_detect = sigma_arcsec

        # Preparing PSF
        psf_data = jnp.array(psf_data, dtype="<f8")
        self.psf_fourier = jnp.fft.fftshift(jnp.fft.fft2(psf_data))
        self.psf_pow = imgutil.get_fourier_pow_fft(psf_data)

        # A few import scales
        self.pix_scale = pix_scale
        self._dk = 2.0 * jnp.pi / self.ngrid  # assuming pixel scale is 1

        # the following two assumes pixel_scale = 1
        self.sigmaf = float(self.pix_scale / sigma_arcsec)
        self.sigmaf_det = float(self.pix_scale / sigma_detect)
        sigma_pixf = self.sigmaf / self._dk
        sigma_pixf_det = self.sigmaf_det / self._dk
        logging.info("Order of the shear estimator: nnord=%d" % self.nnord)
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
            psf_array=self.psf_pow,
            sigma=(sigma_pixf + sigma_pixf_det) / 2.0 / jnp.sqrt(2.0),
            thres=1e-20,
        )  # in pixel units
        self.klim_pix = min(self.klim_pix, self.ngrid // 2 - 1)

        self.klim = float(self.klim_pix * self._dk)
        logging.info("Maximum |k| is %.3f" % (self.klim))

        self._indx = jnp.arange(
            self.ngrid // 2 - self.klim_pix,
            self.ngrid // 2 + self.klim_pix + 1,
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
        pix_scale (float):      pixel scale in arcsec
        sigma_arcsec (float):   Shapelet kernel size
        sigma_detect (float):   detection kernel size
        nnord (int):            the highest order of Shapelets radial
                                components [default: 4]
    """

    _DefaultName = "measure_noise_cov"

    def __init__(
        self,
        psf_data,
        pix_scale,
        sigma_arcsec,
        sigma_detect=None,
        nnord=4,
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
            self.sigmaf,
            self.sigmaf_det,
            self.klim,
        )
        self.bfunc = bfunc[:, self._indy, self._indx]
        self.bnames = bnames
        return

    def measure(self, noise_pf):
        """Estimate covariance of measurement error in impt form

        Args:
            noise_pf (ndarray):     power spectrum (assuming homogeneous) of noise
        Return:
            cov_matrix (ndarray):   covariance matrix of FPFS basis modes
        """
        noise_pf = jnp.array(noise_pf, dtype="<f8")
        noise_pf_deconv = self.deconvolve(noise_pf, prder=1, frder=0)
        cov_matrix = (
            jnp.real(
                jnp.tensordot(
                    self.bfunc * noise_pf_deconv[None, self._indy, self._indx],
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
        pix_scale (float):      pixel scale in arcsec
        sigma_arcsec (float):   Shapelet kernel size
        sigma_detect (float):   detection kernel size
        nnord (int):            the highest order of Shapelets radial components
                                [default: 4]
    """

    _DefaultName = "measure_source"

    def __init__(
        self,
        psf_data,
        pix_scale,
        sigma_arcsec,
        sigma_detect=None,
        nnord=4,
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
        elif nnord == 6:
            # This setup is able to derive kappa response and shear response
            # Only uses M00, M20, M22 (real and img), M40, M42(real and img), M60
            self._indM = np.array([0, 14, 16, 28, 30, 42])[:, None, None]
        else:
            raise ValueError(
                "only support for nnord= 4 or nnord=6, but your input\
                    is nnord=%d"
                % nnord
            )
        chi = imgutil.shapelets2d(
            self.ngrid,
            nnord,
            self.sigmaf,
            self.klim,
        )[self._indM, self._indy, self._indx]
        psi = imgutil.detlets2d(
            self.ngrid,
            self.sigmaf_det,
            self.klim,
        )[:, :, self._indy, self._indx]
        self.prepare_chi(chi)
        self.prepare_psi(psi)
        del chi, psi
        return

    def detect_sources(
        self,
        img_data,
        psf_data,
        thres,
        thres2,
        bound=None,
    ):
        """Returns the coordinates of detected sources

        Args:
            img_data (ndarray):         observed image
            psf_data (ndarray):         PSF image [must be well-centered]
            thres (float):              detection threshold
            thres2 (float):             peak identification difference threshold
            bound (int):                remove sources at boundary
        Returns:
            coords (ndarray):           peak values and the shear responses
        """
        if not isinstance(thres, (int, float)):
            raise ValueError("thres must be float, but now got %s" % type(thres))
        if not isinstance(thres2, (int, float)):
            raise ValueError("thres2 must be float, but now got %s" % type(thres))
        if not thres > 0.0:
            raise ValueError("detection threshold should be positive")
        if not thres2 <= 0.0:
            raise ValueError("difference threshold should be non-positive")
        psf_data = jnp.array(psf_data, dtype="<f8")
        assert (
            img_data.shape == psf_data.shape
        ), "image and PSF should have the same\
                shape. Please do padding before using this function."
        img_conv = imgutil.convolve2gausspsf(
            img_data,
            psf_data,
            self.sigmaf,
            self.klim,
        )
        img_conv_det = imgutil.convolve2gausspsf(
            img_data,
            psf_data,
            self.sigmaf_det,
            self.klim,
        )
        if bound is None:
            bound = self.ngrid // 2 + 5
        dd = imgutil.find_peaks(img_conv, img_conv_det, thres, thres2, bound).T
        return dd

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
            out.append(psi[_, 0])  # ps_i
            self.psi_types.append(("fpfs_v%d" % _, "<f8"))
        for j in [1, 2]:
            for i in range(8):
                out.append(psi[i, j])  # ps_i;j
                self.psi_types.append(("fpfs_v%dr%d" % (i, j), "<f8"))
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
        out = (
            jnp.sum(
                data[None, self._indy, self._indx] * self.chi,
                axis=(1, 2),
            ).real
            / self.pix_scale**2.0
        )
        return out

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
        return out

    def measure(self, exposure, coords=None):
        """Measures the FPFS moments

        Args:
            exposure (ndarray):         galaxy image
            psf_fourier (ndarray):      PSF's Fourier transform
        Returns:
            out (ndarray):              FPFS moments
        """
        if coords is None:
            coords = jnp.array(exposure.shape) // 2
        coords = jnp.atleast_2d(coords.T).T
        func = lambda xi: self.measure_coord(xi, jnp.array(exposure))
        return jax.lax.map(func, coords)

    @partial(jax.jit, static_argnames=["self"])
    def measure_coord(self, cc, image):
        """Measures the FPFS moments from a coordinate (jitted)

        Args:
            cc (ndarray):       galaxy peak coordinate
            image (ndarray):    exposure
        Returns:
            mm (ndarray):       FPFS moments
        """
        stamp = jax.lax.dynamic_slice(
            image,
            (cc[0] - self.ngrid // 2, cc[1] - self.ngrid // 2),
            (self.ngrid, self.ngrid),
        )
        return self.measure_stamp(stamp)

    @partial(jax.jit, static_argnames=["self"])
    def measure_stamp(self, data):
        """Measures the FPFS moments from a stamp (jitted)

        Args:
            data (ndarray):     galaxy image array
        Returns:
            mm (ndarray):       FPFS moments
        """
        gal_fourier = jnp.fft.fftshift(jnp.fft.fft2(data))
        gal_deconv = self.deconvolve(gal_fourier, prder=0.0, frder=1)
        mm = self._itransform_chi(gal_deconv)  # FPFS shapelets
        mp = self._itransform_psi(gal_deconv)  # FPFS detection
        # jax.debug.print("debug: {}", mm)
        return jnp.hstack([mm, mp])

    def get_results(self, out):
        tps = self.chi_types + self.psi_types
        res = np.rec.fromarrays(out.T, dtype=tps)
        return res
