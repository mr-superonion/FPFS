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
from . import util
from .shapelets import shapelets2d_real
from .detection import detlets2d


logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)


class measure_base(object):
    """A base class for measurement, which is extended to measure_source and
    measure_noise_cov

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
        detect_nrot=8,
    ):
        if sigma_arcsec <= 0.0 or sigma_arcsec > 5.0:
            raise ValueError("sigma_arcsec should be positive and less than 5 arcsec")
        self.ngrid = psf_data.shape[0]
        self.nnord = nnord
        self.detect_nrot = detect_nrot
        if sigma_detect is None:
            sigma_detect = sigma_arcsec

        # Preparing PSF
        psf_data = jnp.array(psf_data, dtype="<f8")
        self.psf_fourier = jnp.fft.fftshift(jnp.fft.fft2(psf_data))
        self.psf_pow = util.get_fourier_pow_fft(psf_data)

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
        self.klim_pix = util.get_klim(
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

    def prepare_fpfs_bases(self):
        """This fucntion prepare the FPFS bases (shapelets and detectlets)"""
        bfunc, snames = shapelets2d_real(
            self.ngrid,
            self.nnord,
            self.sigmaf,
            self.klim,
        )
        psi, dnames = detlets2d(
            self.ngrid,
            self.detect_nrot,
            self.sigmaf_det,
            self.klim,
        )
        bnames = snames + dnames
        bfunc = jnp.vstack([bfunc, jnp.vstack(psi)])
        self.bfunc = bfunc[:, self._indy, self._indx]
        self.bnames = [(_nn, "<f8") for _nn in bnames]
        return


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
        self.prepare_fpfs_bases()
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
        self.prepare_fpfs_bases()
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
        if not thres > 0.0:
            raise ValueError("detection threshold should be positive")
        if not thres2 <= 0.0:
            raise ValueError("difference threshold should be non-positive")
        psf_data = jnp.array(psf_data, np.float32)
        assert (
            img_data.shape == psf_data.shape
        ), "image and PSF should have the same\
                shape. Please do padding before using this function."
        if bound is None:
            bound = self.ngrid // 2 + 5
        return self.peak_detect(
            img_data=img_data,
            psf_data=psf_data,
            thres=thres,
            thres2=thres2,
            bound=bound,
        )

    def peak_detect(self, img_data, psf_data, thres, thres2, bound=20.0):
        """This function convolves an image to transform the PSF to a Gaussian

        Args:
        img_data (ndarray):     image data
        psf_data (ndarray):     psf data
        thres (float):          threshold of Gaussian flux
        thres2 (float):         threshold of peak detection
        bound (float):          minimum distance to the image boundary

        Returns:
        det (ndarray):          detection array
        """

        ny, nx = psf_data.shape
        # Fourier transform
        psf_fourier = jnp.fft.rfft2(jnp.fft.ifftshift(psf_data))
        sel = jnp.ones_like(psf_data, dtype=bool)

        # Gaussian kernel for shapelets
        gauss_kernel, (kygrids, kxgrids) = util.gauss_kernel_rfft(
            ny,
            nx,
            self.sigmaf_det,
            self.klim,
            return_grid=True,
        )
        for i in range(0, 8, 2):
            x = jnp.cos(jnp.pi / 4.0 * i)
            y = jnp.sin(jnp.pi / 4.0 * i)
            bb = (1.0 - jnp.exp(1j * (kxgrids * x + kygrids * y))) * gauss_kernel
            img_f = jnp.fft.rfft2(img_data) * bb / psf_fourier
            img_r = jnp.fft.irfft2(img_f, (ny, nx))
            sel = sel & (img_r > thres2)

        # Gaussian kernel for shapelets
        gauss_kernel = util.gauss_kernel_rfft(
            ny,
            nx,
            self.sigmaf,
            self.klim,
            return_grid=False,
        )
        # convolved images
        img_fourier = jnp.fft.rfft2(img_data) / psf_fourier * gauss_kernel
        img_conv = jnp.fft.irfft2(img_fourier, (ny, nx))
        sel = sel & (img_conv > thres)
        del psf_fourier, gauss_kernel

        r2_over_sigma2 = (kxgrids**2.0 + kygrids**2.0) / self.sigmaf**2.0
        # Set up Laguerre polynomials
        lfunc = np.zeros((3, ny, nx // 2 + 1), dtype=np.float32)
        lfunc[0, :, :] = 1.0
        lfunc[1, :, :] = 1.0 - r2_over_sigma2
        lfunc[2] = (2.0 - (1.0 + r2_over_sigma2) / 2) * lfunc[1] - 0.5 * lfunc[0]
        nn = 2
        w2 = pow(-1.0, nn // 2) * lfunc[nn // 2] * (1j) ** nn
        img_conv2 = jnp.fft.irfft2(img_fourier * w2, (ny, nx))
        sel = sel & (((img_conv + img_conv2) > 0.0) & ((img_conv - img_conv2) > 0.0))

        det = jnp.int_(jnp.argwhere(sel))
        det = jnp.hstack(
            [
                det,
                img_conv[sel][:, None] / self.pix_scale**2.0,
                img_conv2[sel][:, None] / self.pix_scale**2.0,
            ]
        )
        msk = (
            (det[:, 0] > bound)
            & (det[:, 0] < ny - bound)
            & (det[:, 1] > bound)
            & (det[:, 1] < nx - bound)
        )
        det = det[msk]
        return det

    def measure(self, exposure, coords=None):
        """This function measures the FPFS moments

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
        return jax.lax.map(jax.jit(func), coords)

    def measure_coord(self, cc, image):
        """This function measures the FPFS moments from a coordinate

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

    def measure_stamp(self, data):
        """This function measures the FPFS moments from a stamp

        Args:
        data (ndarray):     galaxy image array

        Returns:
        mm (ndarray):       FPFS moments
        """
        gal_fourier = jnp.fft.fftshift(jnp.fft.fft2(data))
        gal_deconv = self.deconvolve(gal_fourier, prder=0.0, frder=1)
        # jax.debug.print("debug: {}", mm)
        outcome = (
            jnp.real(
                jnp.sum(
                    gal_deconv[None, self._indy, self._indx] * self.bfunc,
                    axis=(-1, -2),
                )
            )
            / self.pix_scale**2.0
        )
        return outcome

    def get_results(self, data):
        outcome = np.rec.fromarrays(data.T, dtype=self.bnames)
        return outcome

    def get_results_detection(self, data):
        tps = [("y", "i4"), ("x", "i4"), ("m00", "<f8"), ("m20", "<f8")]
        coords = np.rec.fromarrays(
            data.T,
            dtype=tps,
        )
        return coords
