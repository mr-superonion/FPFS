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

import logging

import jax
import jax.numpy as jnp
import numpy as np

from . import util
from .detection import detlets2d, get_det_col_names
from .shapelets import get_shapelets_col_names, shapelets2d

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)


class fpfs_base(object):
    def __init__(self, nord: int, det_nrot: int) -> None:
        self.nord = nord
        name_s, _ = get_shapelets_col_names(nord)
        name_d = get_det_col_names(det_nrot)
        name_a = name_s + name_d
        self.di = {}
        for index, element in enumerate(name_a):
            self.di[element] = index
        self.ncol = len(name_a)
        self.ndet = len(name_d)
        self.name_shapelets = name_s
        self.name_detect = name_d
        self.det_nrot = det_nrot
        self.prepare_dg1_mat()
        self.prepare_dg2_mat()
        # jax.debug.print("debug: {}", self.dg1_mat)
        return

    def prepare_dg1_mat(self):
        out = []
        for nn in self.name_shapelets:
            tmp = jnp.zeros(self.ncol)
            match nn:
                case "m00":
                    tmp = tmp.at[self.di["m22c"]].set(-jnp.sqrt(2.0))
                case "m20":
                    tmp = tmp.at[self.di["m42c"]].set(-jnp.sqrt(6.0))
                case "m22c":
                    tmp = tmp.at[self.di["m00"]].set(1.0 / jnp.sqrt(2.0))
                    tmp = tmp.at[self.di["m40"]].set(-1.0 / jnp.sqrt(2.0))
                    tmp = tmp.at[self.di["m44c"]].set(-jnp.sqrt(3.0))
                case "m22s":
                    tmp = tmp.at[self.di["m44s"]].set(-jnp.sqrt(3.0))
                case "m42c":
                    if self.nord >= 6:
                        tmp = tmp.at[self.di["m20"]].set(jnp.sqrt(6.0) / 2.0)
                        tmp = tmp.at[self.di["m60"]].set(-jnp.sqrt(6.0) / 2.0)
            out.append(tmp)
        for nn in self.name_detect[: self.det_nrot]:
            tmp = jnp.zeros(self.ncol)
            tmp = tmp.at[self.di[nn + "r1"]].set(1.0)
            out.append(tmp)
        for _ in range(self.det_nrot * 2):
            out.append(jnp.zeros(self.ncol))
        self.dg1_mat = jnp.vstack(out)
        return

    def prepare_dg2_mat(self):
        out = []
        for nn in self.name_shapelets:
            tmp = jnp.zeros(self.ncol)
            match nn:
                case "m00":
                    tmp = tmp.at[self.di["m22s"]].set(-jnp.sqrt(2.0))
                case "m20":
                    tmp = tmp.at[self.di["m42s"]].set(-jnp.sqrt(6.0))
                case "m22c":
                    tmp = tmp.at[self.di["m44s"]].set(-jnp.sqrt(3.0))
                case "m22s":
                    tmp = tmp.at[self.di["m00"]].set(1.0 / jnp.sqrt(2.0))
                    tmp = tmp.at[self.di["m40"]].set(-1.0 / jnp.sqrt(2.0))
                    tmp = tmp.at[self.di["m44c"]].set(jnp.sqrt(3.0))
                case "m42c":
                    if self.nord >= 6:
                        tmp = tmp.at[self.di["m20"]].set(jnp.sqrt(6.0) / 2.0)
                        tmp = tmp.at[self.di["m60"]].set(-jnp.sqrt(6.0) / 2.0)
            out.append(tmp)
        for nn in self.name_detect[: self.det_nrot]:
            tmp = jnp.zeros(self.ncol)
            tmp = tmp.at[self.di[nn + "r2"]].set(1.0)
            out.append(tmp)
        for _ in range(self.det_nrot * 2):
            out.append(jnp.zeros(self.ncol))
        self.dg2_mat = jnp.vstack(out)
        return

    def _dg1(self, x):
        return self.dg1_mat.dot(x)

    def _dg2(self, x):
        return self.dg2_mat.dot(x)


class measure_base(fpfs_base):
    """A base class for measurement

    Args:
    psf_array (ndarray):    an average PSF image used to initialize the task
    pix_scale (float):      pixel scale in arcsec
    sigma_arcsec (float):   Shapelet kernel size
    nord (int):             the highest order of Shapelets radial
                            components [default: 4]
    det_nrot (int):         number of rotation in the detection kernel
    """

    def __init__(
        self,
        psf_array,
        pix_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 8,
    ) -> None:
        super().__init__(
            nord=nord,
            det_nrot=det_nrot,
        )
        if sigma_arcsec <= 0.0 or sigma_arcsec > 5.0:
            raise ValueError("sigma_arcsec should be positive and less than 5 arcsec")
        self.ngrid = psf_array.shape[0]

        # Preparing PSF
        psf_f = jnp.fft.rfft2(psf_array)
        psf_pow = (jnp.abs(psf_f) ** 2.0).astype(jnp.float64)

        psf_rot = util.rotate90(psf_array)
        psf_rot_f = jnp.fft.rfft2(psf_rot)
        psf_rot_pow = (jnp.abs(psf_rot_f) ** 2.0).astype(jnp.float64)

        # A few import scales
        self.pix_scale = pix_scale
        self._dk = 2.0 * jnp.pi / self.ngrid  # assuming pixel scale is 1

        # the following two assumes pixel_scale = 1
        self.sigmaf = float(self.pix_scale / sigma_arcsec)
        logging.info("Order of the shear estimator: nord=%d" % self.nord)
        logging.info(
            "Shapelet kernel in configuration space: sigma= %.4f arcsec"
            % (sigma_arcsec)
        )
        # effective nyquest wave number
        self.klim_pix = util.get_klim(
            psf_pow=psf_pow,
            sigma=self.sigmaf / np.sqrt(2.0),
            thres=1e-20,
        )  # in pixel units
        self.klim = float(self.klim_pix * self._dk)
        logging.info("Maximum |k| is %.3f" % (self.klim))

        self.psf_f = util.truncate_psf_rfft(
            psf_f,
            self.klim_pix,
            self.ngrid,
        )
        self.psf_pow = util.truncate_psf_rfft(
            psf_pow,
            self.klim_pix,
            self.ngrid,
        )

        self.psf_rot_f = util.truncate_psf_rfft(
            psf_rot_f,
            self.klim_pix,
            self.ngrid,
        )
        self.psf_rot_pow = util.truncate_psf_rfft(
            psf_rot_pow,
            self.klim_pix,
            self.ngrid,
        )

        self.prepare_fpfs_bases()

        # Weight for rfft
        _w = jnp.ones(psf_pow.shape) * 2.0
        self._w = _w.at[:, 0].set(1.0).at[:, -1].set(1.0)
        return

    def prepare_fpfs_bases(self):
        """This fucntion prepare the FPFS bases (shapelets and detectlets)"""
        chi, snames = shapelets2d(
            ngrid=self.ngrid,
            nord=self.nord,
            sigma=self.sigmaf,
            klim=self.klim,
        )
        psi, dnames = detlets2d(
            ngrid=self.ngrid,
            sigma=self.sigmaf,
            klim=self.klim,
            det_nrot=self.det_nrot,
        )
        bnames = snames + dnames
        self.bfunc = jnp.vstack([chi, psi])
        self.byps = [("fpfs_%s" % _nn, "<f8") for _nn in bnames]
        return


class measure_noise_cov(measure_base):
    """A class to measure FPFS noise covariance of basis modes

    Args:
    psf_array (ndarray):     an average PSF image used to initialize the task
    pix_scale (float):      pixel scale in arcsec
    sigma_arcsec (float):   Shapelet kernel size
    nord (int):             the highest order of Shapelets radial
                            components [default: 4]
    det_nrot (int):         number of rotation in the detection kernel
    """

    def __init__(
        self,
        psf_array,
        pix_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 8,
    ) -> None:
        super().__init__(
            psf_array=psf_array,
            sigma_arcsec=sigma_arcsec,
            nord=nord,
            pix_scale=pix_scale,
            det_nrot=det_nrot,
        )
        return

    def measure(self, noise_pf):
        """Estimate covariance of measurement error in impt form

        Args:
        noise_pf (ndarray):     power spectrum (assuming homogeneous) of noise

        Return:
        cov_matrix (ndarray):   covariance matrix of FPFS basis modes
        """
        if noise_pf.shape == (self.ngrid, self.ngrid // 2 + 1):
            # rfft
            noise_pf = jnp.array(noise_pf, dtype=jnp.float64)
        elif noise_pf.shape == (self.ngrid, self.ngrid):
            # fft
            noise_pf = jnp.fft.ifftshift(noise_pf)
            noise_pf = jnp.array(noise_pf[:, : self.ngrid // 2 + 1], dtype=jnp.float64)
        else:
            raise ValueError("noise power not in correct shape")

        noise_pf_deconv = noise_pf / self.psf_pow
        cov_matrix = (
            jnp.tensordot(
                self.bfunc * (self._w * noise_pf_deconv)[jnp.newaxis, :, :],
                jnp.conjugate(self.bfunc),
                axes=((1, 2), (1, 2)),
            ).real
            / self.pix_scale**4.0
        )
        return cov_matrix


class measure_source(measure_base):
    """A class to measure FPFS shapelet mode estimation

    Args:
    psf_array (ndarray):     an average PSF image used to initialize the task
    pix_scale (float):      pixel scale in arcsec
    sigma_arcsec (float):   Shapelet kernel size
    nord (int):             the highest order of Shapelets radial components
                            [default: 4]
    det_nrot (int):         number of rotation in the detection kernel
    """

    def __init__(
        self,
        psf_array,
        pix_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 8,
    ):
        super().__init__(
            psf_array=psf_array,
            sigma_arcsec=sigma_arcsec,
            nord=nord,
            pix_scale=pix_scale,
            det_nrot=det_nrot,
        )
        return

    def detect_source(
        self,
        img_array,
        psf_array,
        cov_elem,
        fthres: float,
        pthres: float = 0.0,
        pratio: float = 0.02,
        bound: int | None = None,
        noise_array=None,
    ):
        """Returns the coordinates of detected sources

        Args:
        img_array (ndarray):         observed image
        psf_array (ndarray):         PSF image [must be well-centered]
        cov_elem (ndarray):         covariance matrix of the measurement error
        fthres (float):             n-sigma detection threshold (flux)
        pthres (float):             n-sigma detection threshold (peak)
        pratio (float):             ratio between difference and flux
        bound (int):                remove sources at boundary

        Returns:
        coords (ndarray):           peak values and the shear responses
        """
        logging.info("Running Detection")
        if not pthres >= 0.0:
            raise ValueError("Peak detection threshold should be positive")
        if bound is None:
            bound = self.ngrid // 2 + 5
        if noise_array is None:
            noise_array = jnp.zeros_like(img_array)
        out = self._detect_source(
            img_array=img_array,
            psf_array=psf_array,
            cov_elem=cov_elem,
            fthres=fthres,
            pthres=pthres,
            pratio=pratio,
            bound=bound,
            noise_array=noise_array,
        )
        return out

    def _detect_source(
        self,
        img_array,
        psf_array,
        cov_elem,
        fthres,
        pthres,
        pratio,
        bound,
        noise_array,
    ):
        """This function convolves an image to transform the PSF to a Gaussian

        Args:
        img_array (ndarray):     image data
        psf_array (ndarray):     psf data
        fthres (float):         n-sigma threshold of Gaussian flux
        pthres (float):         n-sigma detection threshold (difference)
        pratio (float):         ratio between difference and flux
        bound (float):          minimum distance to the image boundary

        Returns:
        det (ndarray):          detection array
        """

        std_modes = jnp.sqrt(jnp.diagonal(cov_elem))
        fcut = fthres * std_modes[self.di["m00"]] * self.pix_scale**2.0
        std_v = jnp.average(
            jnp.array([std_modes[self.di["v%d" % _]] for _ in range(self.det_nrot)])
        )
        pcut = std_v * self.pix_scale**2.0 * pthres

        ny, nx = img_array.shape
        # Fourier transform
        npady = (ny - psf_array.shape[0]) // 2
        npadx = (nx - psf_array.shape[1]) // 2

        # Gaussian kernel for shapelets
        img_conv = jnp.fft.irfft2(
            (
                jnp.fft.rfft2(img_array)
                / jnp.fft.rfft2(
                    jnp.fft.ifftshift(
                        jnp.pad(
                            psf_array,
                            (npady, npadx),
                            mode="constant",
                        ),
                    )
                )
                + jnp.fft.rfft2(noise_array)
                / jnp.fft.rfft2(
                    jnp.fft.ifftshift(
                        jnp.pad(
                            util.rotate90(psf_array),
                            (npady, npadx),
                            mode="constant",
                        ),
                    )
                )
            )
            * util.gauss_kernel_rfft(
                ny,
                nx,
                self.sigmaf,
                self.klim,
                return_grid=False,
            ),
            (ny, nx),
        )

        det = jnp.int_(
            jnp.argwhere(
                get_pixel_detect_mask(
                    (img_conv > fcut),
                    img_conv,
                    pcut,
                    pratio,
                )
            )
        )
        det = det[
            (
                (det[:, 0] > bound)
                & (det[:, 0] < ny - bound)
                & (det[:, 1] > bound)
                & (det[:, 1] < nx - bound)
            )
        ]
        func = lambda cc: self.determine_peak(cc, img_conv)
        det = jax.lax.map(func, jnp.atleast_2d(det))
        return det

    def determine_peak(self, cc, image):
        out = (
            (image[cc[0], cc[1]] - image[cc[0] + 1, cc[1]] >= 0.0)
            & (image[cc[0], cc[1]] - image[cc[0], cc[1] + 1] >= 0.0)
            & (image[cc[0], cc[1]] - image[cc[0] - 1, cc[1]] >= 0.0)
            & (image[cc[0], cc[1]] - image[cc[0], cc[1] - 1] >= 0.0)
        )
        out = jnp.append(cc, out)
        return out

    def measure(self, exposure, coords=None, psf_f=None):
        """This function measures the FPFS moments

        Args:
        exposure (ndarray):         galaxy image
        coords (ndarray|None):      coordinate array
        psf_f (ndarray|None):       rfft2 of PSF (image in real space centered
                                    at ngrid//2)

        Returns:
        src (ndarray):              FPFS linear observables
        """
        if coords is None:
            coords = jnp.array(exposure.shape) // 2
        if psf_f is None:
            psf_f = self.psf_f
        if not (
            psf_f.shape[-1] == self.ngrid // 2 + 1 and psf_f.shape[-2] == self.ngrid
        ):
            raise ValueError("psf_f not in correct shape")
        func = lambda x: self.measure_coord(
            exposure=exposure,
            cc=x,
            psf_f=psf_f,
        )
        src = jax.lax.map(
            func,
            jnp.atleast_2d(coords),
        )
        return src

    def measure_coord(self, exposure, cc, psf_f):
        """This function measures the FPFS moments from a coordinate

        Args:
        exposure (ndarray):     exposure
        cc (ndarray):           galaxy peak coordinate
        psf_f (ndarray):        rfft2 of PSF

        Returns:
        mm (ndarray):       FPFS moments
        """
        y = cc[0].astype(int)
        x = cc[1].astype(int)
        stamp = jax.lax.dynamic_slice(
            exposure,
            (y - self.ngrid // 2, x - self.ngrid // 2),
            (self.ngrid, self.ngrid),
        )
        gal_deconv = jnp.fft.rfft2(stamp) / psf_f
        # jax.debug.print("debug: {}", mm)
        outcome = (
            jnp.sum(
                (self._w * gal_deconv)[jnp.newaxis, :, :] * self.bfunc,
                axis=(-1, -2),
            ).real
            / self.pix_scale**2.0
        )
        return outcome

    def get_results(self, data):
        outcome = np.rec.fromarrays(data.T, dtype=np.dtype(self.byps))
        return outcome

    def get_results_detection(self, data):
        if data.shape[1] == 3:
            tps = [
                ("fpfs_y", "i4"),
                ("fpfs_x", "i4"),
                ("is_peak", "?"),
            ]
        elif data.shape[1] == 2:
            tps = [
                ("fpfs_y", "i4"),
                ("fpfs_x", "i4"),
            ]
        else:
            raise RuntimeError("detection has wrong number of columns")

        coords = np.rec.fromarrays(
            data.T,
            dtype=np.dtype(tps),
        )
        return coords


@jax.jit
def get_pixel_detect_mask(sel, img, pcut, pratio):
    thres = -1.0 * (pcut + pratio * img)
    for ax in [-1, -2]:
        for shift in [-1, 1]:
            sel = jnp.logical_and(
                sel, img - jnp.roll(img, shift=shift, axis=ax) > thres
            )
    return sel
