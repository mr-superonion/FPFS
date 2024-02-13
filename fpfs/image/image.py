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
from functools import partial
from . import util
from .shapelets import shapelets2d_real, get_shapelets_col_names
from .detection import detlets2d, get_det_col_names

det_ratio = 0.02

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)


class fpfs_base(object):
    def __init__(self, nord, detect_nrot):
        self.nord = nord
        self.detect_nrot = detect_nrot
        name_s, ind_s = get_shapelets_col_names(nord)
        name_d = get_det_col_names(detect_nrot)
        name_s = name_s + name_d
        self.di = {}
        for index, element in enumerate(name_s):
            self.di[element] = index
        self.ncol = len(name_s)
        return


class measure_base(fpfs_base):
    """A base class for measurement, which is extended to measure_source and
    measure_noise_cov

    Args:
    psf_data (ndarray):     an average PSF image used to initialize the task
    pix_scale (float):      pixel scale in arcsec
    sigma_arcsec (float):   Shapelet kernel size
    sigma_detect (float):   detection kernel size
    nord (int):            the highest order of Shapelets radial
                            components [default: 4]
    """

    _DefaultName = "measure_base"

    def __init__(
        self,
        psf_data,
        pix_scale,
        sigma_arcsec,
        sigma_detect=None,
        nord=4,
        detect_nrot=8,
    ):
        super().__init__(
            nord=nord,
            detect_nrot=detect_nrot,
        )
        if sigma_arcsec <= 0.0 or sigma_arcsec > 5.0:
            raise ValueError("sigma_arcsec should be positive and less than 5 arcsec")
        self.ngrid = psf_data.shape[0]
        if sigma_detect is None:
            sigma_detect = sigma_arcsec
        # NOTE: force them to be the same
        sigma_detect = sigma_arcsec

        # Preparing PSF
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
        logging.info("Order of the shear estimator: nord=%d" % self.nord)
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

        self.prepare_fpfs_bases()
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
        out = jnp.zeros(data.shape, dtype=jnp.complex128)
        out2 = out.at[self._ind2d].set(
            data[self._ind2d]
            / self.psf_pow[self._ind2d] ** prder
            / self.psf_fourier[self._ind2d] ** frder
        )
        return out2

    def prepare_fpfs_bases(self):
        """This fucntion prepare the FPFS bases (shapelets and detectlets)"""
        chi, snames = shapelets2d_real(
            self.ngrid,
            self.nord,
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
        bfunc = jnp.vstack([chi, psi])
        self.bfunc = jnp.array(bfunc[:, self._indy, self._indx])
        self.byps = [("fpfs_%s" % _nn, "<f8") for _nn in bnames]
        return


class measure_noise_cov(measure_base):
    """A class to measure FPFS noise covariance of basis modes

    Args:
    psf_data (ndarray):     an average PSF image used to initialize the task
    pix_scale (float):      pixel scale in arcsec
    sigma_arcsec (float):   Shapelet kernel size
    sigma_detect (float):   detection kernel size
    nord (int):             the highest order of Shapelets radial
                            components [default: 4]
    """

    _DefaultName = "measure_noise_cov"

    def __init__(
        self,
        psf_data,
        pix_scale,
        sigma_arcsec,
        sigma_detect=None,
        nord=4,
        detect_nrot=8,
    ):
        super().__init__(
            psf_data=psf_data,
            sigma_arcsec=sigma_arcsec,
            nord=nord,
            pix_scale=pix_scale,
            sigma_detect=sigma_detect,
            detect_nrot=detect_nrot,
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
    nord (int):             the highest order of Shapelets radial components
                            [default: 4]
    """

    def __init__(
        self,
        psf_data,
        pix_scale,
        sigma_arcsec,
        sigma_detect=None,
        nord=4,
        detect_nrot=8,
        detect_return_peak_modes=False,
    ):
        super().__init__(
            psf_data=psf_data,
            sigma_arcsec=sigma_arcsec,
            nord=nord,
            pix_scale=pix_scale,
            sigma_detect=sigma_detect,
            detect_nrot=detect_nrot,
        )
        self.detect_return_peak_modes = detect_return_peak_modes
        return

    def detect_sources(
        self,
        img_data,
        psf_data,
        cov_elem,
        thres,
        bound=None,
    ):
        """Returns the coordinates of detected sources

        Args:
        img_data (ndarray):         observed image
        psf_data (ndarray):         PSF image [must be well-centered]
        cov_elem (ndarray):         covariance matrix of the measurement error
        thres (float):              n-sigma detection threshold
        bound (int):                remove sources at boundary

        Returns:
        coords (ndarray):           peak values and the shear responses
        """
        logging.info("Running Detection")
        if not thres >= 5.0:
            raise ValueError("detection threshold should be positive")
        if bound is None:
            bound = self.ngrid // 2 + 5
        out = self.peak_detect(
            img_data=img_data,
            psf_data=psf_data,
            cov_elem=cov_elem,
            thres=thres,
            bound=bound,
        )
        logging.info("Finish Detection")
        logging.info("Detect sources %d" % len(out))
        return out

    def peak_detect(self, img_data, psf_data, cov_elem, thres, bound):
        """This function convolves an image to transform the PSF to a Gaussian

        Args:
        img_data (ndarray):     image data
        psf_data (ndarray):     psf data
        thres (float):          n-sigma threshold of Gaussian flux
        bound (float):          minimum distance to the image boundary

        Returns:
        det (ndarray):          detection array
        """

        std_modes = jnp.sqrt(jnp.diagonal(cov_elem))
        idm00 = self.di["m00"]
        thres_tmp = thres * std_modes[idm00] * self.pix_scale**2.0

        ny, nx = img_data.shape
        # Fourier transform
        npady = (ny - psf_data.shape[0]) // 2
        npadx = (nx - psf_data.shape[1]) // 2
        # Gaussian kernel for shapelets

        kernel, (kygrids, kxgrids) = util.gauss_kernel_rfft(
            ny,
            nx,
            self.sigmaf,
            self.klim,
            return_grid=True,
        )
        imf = (
            jnp.fft.rfft2(img_data)
            / jnp.fft.rfft2(
                jnp.fft.ifftshift(jnp.pad(psf_data, (npady, npadx), mode="constant"))
            )
            * kernel
        )
        img_conv = jnp.fft.irfft2(imf, (ny, nx))
        img_conv2 = jnp.fft.irfft2(
            imf * (2.0 - (kxgrids**2.0 + kygrids**2.0) / self.sigmaf**2.0),
            (ny, nx),
        )
        del imf, kernel, kxgrids, kygrids
        sel = get_pixel_detect_mask(
            jnp.logical_and(img_conv > thres_tmp, img_conv2 > 0.0),
            img_conv,
        )
        det = jnp.int_(jnp.argwhere(sel))

        # sel = jnp.ones_like(img_data, dtype=bool)
        # for irot in range(self.detect_nrot):
        #     x = jnp.cos(2.0 * jnp.pi / self.detect_nrot * irot)
        #     y = jnp.sin(2.0 * jnp.pi / self.detect_nrot * irot)
        #     bb = (1.0 - jnp.exp(1j * (kxgrids * x + kygrids * y)))
        #     img_r = jnp.fft.irfft2(imgf_use * bb, (ny, nx))
        #     sel = jnp.logical_and(sel, (img_r > thres2_tmp))
        # # Gaussian kernel for shapelets
        # gauss_kernel = util.gauss_kernel_rfft(
        #     ny,
        #     nx,
        #     self.sigmaf,
        #     self.klim,
        #     return_grid=False,
        # )
        # # convolved images
        # img_conv = jnp.fft.irfft2(imgf_use * gauss_kernel, (ny, nx))

        # # Set up Laguerre polynomials
        # img_conv2 = jnp.fft.irfft2(
        #     imgf_use * gauss_kernel * (
        #    1.0 - (kxgrids**2.0 + kygrids**2.0) / self.sigmaf**2.0),
        #     (ny, nx)
        # )
        # sel = sel & (((img_conv + img_conv2) > 0.0) & ((img_conv - img_conv2) > 0.0))

        # if self.detect_return_peak_modes:
        #     det = jnp.hstack(
        #         [
        #             det,
        #             img_conv[sel][:, None] / self.pix_scale**2.0,
        #             img_conv2[sel][:, None] / self.pix_scale**2.0,
        #         ]
        #     )
        #     gauss_kernel = util.gauss_kernel_rfft(
        #         ny,
        #         nx,
        #         self.sigmaf_det,
        #         self.klim,
        #         return_grid=False,
        #     )
        #     for irot in range(self.detect_nrot):
        #         x = jnp.cos(2.0 * jnp.pi / self.detect_nrot * irot)
        #         y = jnp.sin(2.0 * jnp.pi / self.detect_nrot * irot)
        #         bb = (1.0 - jnp.exp(1j * (kxgrids * x + kygrids * y))) * gauss_kernel
        #         img_r = jnp.fft.irfft2(imgf_use * bb, (ny, nx))
        #         det = jnp.hstack(
        #             [
        #                 det,
        #                 img_r[sel][:, None] / self.pix_scale**2.0,
        #             ]
        #         )
        #         del img_r, bb
        #     del gauss_kernel

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
        coords (ndarray):           coordinate array

        Returns:
        out (ndarray):              FPFS moments
        """
        if coords is None:
            coords = jnp.array(exposure.shape) // 2
        func = lambda xi: self.measure_coord(xi, jnp.array(exposure))
        return jax.lax.map(func, jnp.atleast_2d(coords.T).T)

    def measure_coord(self, cc, image):
        """This function measures the FPFS moments from a coordinate

        Args:
        cc (ndarray):       galaxy peak coordinate
        image (ndarray):    exposure

        Returns:
        mm (ndarray):       FPFS moments
        """
        y = cc[0].astype(int)
        x = cc[1].astype(int)
        stamp = jax.lax.dynamic_slice(
            image,
            (y - self.ngrid // 2, x - self.ngrid // 2),
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
            jnp.sum(
                gal_deconv[None, self._indy, self._indx] * self.bfunc,
                axis=(-1, -2),
            ).real
            / self.pix_scale**2.0
        )
        return outcome

    def get_results(self, data):
        outcome = np.rec.fromarrays(data.T, dtype=np.dtype(self.byps))
        return outcome

    def get_results_detection(self, data):
        tps = [
            ("fpfs_y", "i4"),
            ("fpfs_x", "i4"),
        ]
        if self.detect_return_peak_modes:
            tps = tps + [
                ("fpfs_m00", "<f8"),
                ("fpfs_m20", "<f8"),
            ]
            tps = tps + [("fpfs_v%d" % _, "<f8") for _ in range(self.detect_nrot)]
        coords = np.rec.fromarrays(
            data.T,
            dtype=np.dtype(tps),
        )
        return coords


@jax.jit
def get_pixel_detect_mask(sel, img):
    for ax in [-1, -2]:
        for shift in [-1, 1]:
            filtered = img - jnp.roll(img, shift=shift, axis=ax)
            sel = jnp.logical_and(sel, (filtered + det_ratio * img > 0.0))
    return sel
