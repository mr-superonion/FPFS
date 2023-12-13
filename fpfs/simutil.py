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

import os
import gc
import galsim
import logging
import numpy as np
import astropy.io.fits as pyfits
from .default import __data_dir__

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)

nrot_default = 4
# use 4 rotations for ring test (to remove any spin-2 and spin-4 residuals in
# the simulated images)


# For ring tests
def make_ringrot_radians(nord=8):
    """Generates rotation angle array for ring test

    Args:
        nord (int):             up to 1/2**nord*pi rotation
    Returns:
        rot_array (ndarray):     rotation array [in units of radians]
    """
    rot_array = np.zeros(2**nord)
    nnum = 0
    for j in range(nord + 1):
        nj = 2**j
        for i in range(1, nj, 2):
            nnum += 1
            rot_array[nnum] = i / nj
    rot_array = rot_array * np.pi
    return rot_array


def coord_distort_1(x, y, xref, yref, gamma1, gamma2, kappa=0.0, inverse=False):
    """Distorts coordinates by shear

    Args:
        x (ndarray):    input coordinates [x]
        y (ndarray):    input coordinates [y]
        xref (float):   reference point [x]
        yref (float):   reference point [y]
        gamma1 (float): first component of shear distortion
        gamma2 (float): second component of shear distortion
        kappa (float):  kappa distortion [default: 0]
        inverse(bool):  if true, from source to lens; else, from lens to source
    Returns:
        x2 (ndarray):   distorted coordiantes [x]
        y2 (ndarray):   distorted coordiantes [y]
    """
    if inverse:
        xu = x - xref
        yu = y - yref
        x2 = (1 - kappa - gamma1) * xu - gamma2 * yu + xref
        y2 = -gamma2 * xu + (1 - kappa + gamma1) * yu + yref
    else:
        u_mag = 1.0 / (1 - kappa) ** 2.0 - gamma1**2.0 - gamma2**2.0
        xu = x - xref
        yu = y - yref
        x2 = ((1 - kappa + gamma1) * xu + gamma2 * yu + xref) * u_mag
        y2 = (gamma2 * xu + (1 - kappa - gamma1) * yu + yref) * u_mag
    return x2, y2


def coord_rotate(x, y, xref, yref, theta):
    """Rotates coordinates by an angle theta (anticlockwise)

    Args:
        x (ndarray):    input coordinates [x]
        y (ndarray):    input coordinates [y]
        xref (float):   reference point [x]
        yref (float):   reference point [y]
        theta (float):  rotation angle [rads]
    Returns:
        x2 (ndarray):   rotated coordiantes [x]
        y2 (ndarray):   rotated coordiantes [y]
    """
    xu = x - xref
    yu = y - yref
    x2 = np.cos(theta) * xu - np.sin(theta) * yu + xref
    y2 = np.sin(theta) * xu + np.cos(theta) * yu + yref
    return x2, y2


class sim_test:
    def __init__(self, shear, rng, scale=0.263, psf_fwhm=0.9, gal_hlr=0.5, ngrid=32):
        """Simulates an exponential object with moffat PSF, this class has the same
        observational setup as
        https://github.com/esheldon/ngmix/blob/38c379013840b5a650b4b11a96761725251772f5/examples/metacal/metacal.py#L199

        Args:
            shear (tuple):      tuple of [g1, g2]. The shear in each component
            rng (randState):    The random number generator
        """
        self.rng = rng
        dx = 0.5 * scale
        dy = 0.5 * scale

        psf = galsim.Moffat(beta=2.5, fwhm=psf_fwhm,).shear(
            g1=0.02,
            g2=-0.02,
        )
        psf = psf.shift(
            dx=dx,
            dy=dy,
        )

        obj0 = galsim.Exponential(half_light_radius=gal_hlr,).shear(
            g1=shear[0],
            g2=shear[1],
        )

        self.scale = scale

        self.obj = galsim.Convolve(psf, obj0)

        # define the psf and gal here which will be repeatedly used
        self.img0 = self.obj.drawImage(nx=ngrid, ny=ngrid, scale=scale).array
        self.psf = psf.drawImage(nx=ngrid, ny=ngrid, scale=scale).array
        self.ngrid = ngrid
        return

    def make_image(self, noise, psf_noise=0.0, do_shift=False):
        """Generates a galaxy image

        Args:
            noise (float):      Noise for the image
            psf_noise (float):  Noise for the PSF [defalut: 0.]
            do_shift (bool):    whether shift the galaxy [default: False]
        Returns:
            im (ndarray):       galaxy image
            psf_im (ndarray):   PSF image
        """
        if do_shift:
            dy, dx = self.rng.uniform(low=-self.scale / 2, high=self.scale / 2, size=2)
            obj = self.obj.shift(dx=dx, dy=dy)
            self.img = obj.drawImage(
                nx=self.ngrid, ny=self.ngrid, scale=self.scale
            ).array
        else:
            self.img = self.img0
        if noise > 1e-10:
            img = self.img + self.rng.normal(scale=noise, size=self.img.shape)
        else:
            img = self.img
        if psf_noise > 1e-10:
            psf = self.psf + self.rng.normal(scale=psf_noise, size=self.psf.shape)
        else:
            psf = self.psf
        return img, psf


def make_cosmo_sim(
    out_dir,
    psf_obj,
    gname,
    ind0,
    catname=None,
    ny=5000,
    nx=5000,
    rfrac=0.46,
    scale=0.168,
    do_write=True,
    return_array=False,
    magzero=27.0,
    rot_field=0.0,
    shear_value=0.02,
    nrot=nrot_default,
    rescale_min_max=None,
):
    """Makes cosmo-like blended galaxy image simulations.

    Args:
        out_dir (str):          output directory
        psf_obj (PSF):          input PSF object of galsim
        gname (str):            shear distortion setup
        ind0 (int):             index of the simulation
        catname (str):          input catalog Name [default: COSMOS 25.2 catalog]
        ny (int):               number of galaxies in y direction [default: 5000]
        nx (int):               number of galaxies in x direction [default: 5000]
        rfrac(float):           fraction of radius to minimum between nx and ny
        do_write (bool):        whether write output [default: True]
        return_array (bool):    whether return galaxy array [default: False]
        magzero (float):        magnitude zero point
        rot_field (float):      additional rotational angle [in units of radians]
        shear_value (float):    amplitude of the input shear
        nrot (int):             number of rotation, optional
        rescale_min_max (list|ndarray):
            lower and upper bounds of galaxy size rescaling factor, optional
    """

    if catname is None:
        catname = os.path.join(__data_dir__, "src_cosmos.fits")
    np.random.seed(ind0)
    out_fname = os.path.join(out_dir, "image-%d-%s.fits" % (ind0, gname))
    if os.path.isfile(out_fname):
        logging.info("Already have the outcome.")
        if do_write:
            logging.info("Nothing to write.")
        if return_array:
            return pyfits.getdata(out_fname)
        else:
            return None

    bigfft = galsim.GSParams(maximum_fft_size=10240)  # galsim setup
    # Get the shear information
    # Three choice on g(-shear_value,0,shear_value)
    shear_list = np.array([-shear_value, 0.0, shear_value])
    shear_list = shear_list[[eval(i) for i in gname.split("-")[-1]]]

    # number of galaxy
    # we only have `ngeff' galsim galaxies but with `nrot' rotation
    r2 = (min(nx, ny) * rfrac) ** 2.0
    density = int(out_dir.split("_psf")[0].split("_cosmo")[-1])
    ngal = max(int(r2 * np.pi * scale**2.0 / 3600.0 * density), nrot)
    ngal = int(ngal // (nrot * 2) * (nrot * 2))
    ngeff = ngal // (nrot * 2)
    logging.info(
        "We have %d galaxies in total, and each %d are the same" % (ngal, nrot)
    )

    # get the cosmos catalog
    cat_input = pyfits.getdata(catname)
    ntrain = len(cat_input)
    inds = np.random.randint(0, ntrain, ngeff)
    cat_input = cat_input[inds]

    # evenly distributed within a radius, min(nx,ny)*rfrac
    rarray = np.sqrt(r2 * np.random.rand(ngeff))  # radius
    tarray = np.random.uniform(0.0, np.pi / nrot, ngeff)  # theta (0,pi/nrot)
    tarray = tarray + rot_field
    xarray = rarray * np.cos(tarray) + nx // 2  # x
    yarray = rarray * np.sin(tarray) + ny // 2  # y
    if rescale_min_max is None:
        rescale_min_max = [0.95, 1.05]
    rsarray = np.random.uniform(rescale_min_max[0], rescale_min_max[1], ngeff)
    del rarray, tarray

    zbound = np.array([1e-5, 0.5477, 0.8874, 1.3119, 12.0])  # sim 3

    gal_image = galsim.ImageF(nx, ny, scale=scale)
    gal_image.setOrigin(0, 0)
    for ii in range(ngal):
        ig = ii // (nrot * 2)
        irot = ii % (nrot * 2)
        ss = cat_input[ig]
        # x,y
        xi = xarray[ig]
        yi = yarray[ig]
        # randomly rotate by an angle; we have 180/nrot deg pairs to remove
        # shape noise in additive bias estimation
        rot_ang = np.pi / nrot * irot
        ang = rot_ang * galsim.radians
        xi, yi = coord_rotate(xi, yi, nx // 2, ny // 2, rot_ang)
        # determine redshift
        shear_inds = np.where(
            (ss["zphot"] > zbound[:-1]) & (ss["zphot"] <= zbound[1:])
        )[0]
        if len(shear_inds) == 1:
            if gname.split("-")[0] == "g1":
                g1 = shear_list[shear_inds][0]
                g2 = 0.0
            elif gname.split("-")[0] == "g2":
                g1 = 0.0
                g2 = shear_list[shear_inds][0]
            else:
                raise ValueError("g1 or g2 must be in gname")
        else:
            g1 = 0.0
            g2 = 0.0

        gal = generate_cosmos_gal(ss, trunc_ratio=-1.0, gsparams=bigfft)
        # rescale the radius while keeping the surface brightness the same
        gal = gal.expand(rsarray[ig])
        # determine and assign flux
        # HSC's i-band coadds zero point is 27
        flux = 10 ** ((magzero - ss["mag_auto"]) / 2.5)
        gal = gal.withFlux(flux)
        # rotate by 'ang'
        gal = gal.rotate(ang)
        # lensing shear
        gal = gal.shear(g1=g1, g2=g2)
        # position and subpixel offset
        xi, yi = coord_distort_1(xi, yi, nx // 2, ny // 2, g1, g2)
        xu = int(xi)
        yu = int(yi)
        dx = (0.5 + xi - xu) * scale
        dy = (0.5 + yi - yu) * scale
        gal = gal.shift(dx, dy)
        # PSF
        gal = galsim.Convolve([psf_obj, gal], gsparams=bigfft)
        # Bounary
        r_grid = max(gal.getGoodImageSize(scale), 32)
        rx1 = np.min([r_grid, xu])
        rx2 = np.min([r_grid, nx - xu - 1])
        rx = int(min(rx1, rx2))
        del rx1, rx2
        ry1 = np.min([r_grid, yu])
        ry2 = np.min([r_grid, ny - yu - 1])
        ry = int(min(ry1, ry2))
        del ry1, ry2
        # draw galaxy
        b = galsim.BoundsI(xu - rx, xu + rx - 1, yu - ry, yu + ry - 1)
        sub_img = gal_image[b]
        gal.drawImage(sub_img, add_to_image=True)
        del gal, b, sub_img, xu, yu, xi, yi, r_grid
    gc.collect()
    del cat_input, psf_obj
    if do_write:
        gal_image.write(out_fname, clobber=True)
    if return_array:
        return gal_image.array


def generate_cosmos_gal(record, trunc_ratio=5.0, gsparams=None):
    """Generates COSMOS galaxies; modified version of
    https://github.com/GalSim-developers/GalSim/blob/releases/2.3/galsim/scene.py#L626

    Args:
        record (ndarray):   one row of the COSMOS galaxy catalog
        trunc_ratio (float):     truncation ratio
        gsparams:           An GSParams argument.
    Returns:
        gal:    Galsim galaxy
    """

    # record columns:
    # For 'sersicfit', the result is an array of 8 numbers for each:
    #     SERSICFIT[0]: intensity of light profile at the half-light radius.
    #     SERSICFIT[1]: half-light radius measured along the major axis, in
    #                   units of pixels in the COSMOS lensing data reductions
    #                   (0.03 arcsec).
    #     SERSICFIT[2]: Sersic n.
    #     SERSICFIT[3]: q, the ratio of minor axis to major axis length.
    #     SERSICFIT[4]: boxiness, currently fixed to 0, meaning isophotes are
    #                   all elliptical.
    #     SERSICFIT[5]: x0, the central x position in pixels.
    #     SERSICFIT[6]: y0, the central y position in pixels.
    #     SERSICFIT[7]: phi, the position angle in radians. If phi=0, the major
    #                   axis is lined up with the x axis of the image.
    # For 'bulgefit', the result is an array of 16 parameters that comes from
    # doing a 2-component sersic fit.  The first 8 are the parameters for the
    # disk, with n=1, and the last 8 are for the bulge, with n=4.
    def _galsim_round_sersic(n, sersic_prec):
        return float(int(n / sersic_prec + 0.5)) * sersic_prec

    bdparams = record["bulgefit"]
    sparams = record["sersicfit"]
    use_bulgefit = record["use_bulgefit"]
    if use_bulgefit:
        # Bulge parameters:
        # Minor-to-major axis ratio:
        bulge_hlr = record["hlr"][1]
        bulge_flux = record["flux"][1]
        disk_hlr = record["hlr"][2]
        disk_flux = record["flux"][2]
        if trunc_ratio <= 0.99:
            btrunc = None
            bulge = galsim.DeVaucouleurs(
                flux=bulge_flux, half_light_radius=bulge_hlr, gsparams=gsparams
            )
            disk = galsim.Exponential(
                flux=disk_flux, half_light_radius=disk_hlr, gsparams=gsparams
            )
        else:
            btrunc = bulge_hlr * trunc_ratio
            bulge = galsim.DeVaucouleurs(
                flux=bulge_flux,
                half_light_radius=bulge_hlr,
                trunc=btrunc,
                gsparams=gsparams,
            )
            dtrunc = disk_hlr * trunc_ratio
            disk = galsim.Sersic(
                1.0,
                flux=disk_flux,
                half_light_radius=disk_hlr,
                trunc=dtrunc,
                gsparams=gsparams,
            )
        # Apply shears for intrinsic shape.
        bulge_q = bdparams[11]
        bulge_beta = bdparams[15] * galsim.radians
        if bulge_q < 1.0:  # pragma: no branch
            bulge = bulge.shear(q=bulge_q, beta=bulge_beta)
        #
        disk_q = bdparams[3]
        disk_beta = bdparams[7] * galsim.radians
        if disk_q < 1.0:  # pragma: no branch
            disk = disk.shear(q=disk_q, beta=disk_beta)
        # Then combine the two components of the galaxy.
        # No center offset is included
        gal = bulge + disk
    else:
        # Do a similar manipulation to the stored quantities for the single
        # Sersic profiles.
        gal_n = sparams[2]
        # Fudge this if it is at the edge of the allowed n values.  Since
        # GalSim (as of #325 and #449) allow Sersic n in the range 0.3<=n<=6,
        # the only problem is that the fits occasionally go as low as n=0.2.
        # The fits in this file only go to n=6, so there is no issue with
        # too-high values, but we also put a guard on that side in case other
        # samples are swapped in that go to higher value of sersic n.
        if gal_n < 0.3:
            gal_n = 0.3
        if gal_n > 6.0:
            gal_n = 6.0

        # GalSim is much more efficient if only a finite number of Sersic n
        # values are used. This (optionally given constructor args) rounds n to
        # the nearest 0.05. change to 0.1 to speed up
        gal_n = _galsim_round_sersic(gal_n, 0.1)
        gal_hlr = record["hlr"][0]
        gal_flux = record["flux"][0]

        if trunc_ratio <= 0.99:
            strunc = None
            gal = galsim.Sersic(
                gal_n,
                flux=gal_flux,
                half_light_radius=gal_hlr,
                gsparams=gsparams,
            )
        else:
            strunc = gal_hlr * trunc_ratio
            gal = galsim.Sersic(
                gal_n,
                flux=gal_flux,
                half_light_radius=gal_hlr,
                trunc=strunc,
                gsparams=gsparams,
            )
        # Apply shears for intrinsic shape.
        gal_q = sparams[3]
        gal_beta = sparams[7] * galsim.radians
        if gal_q < 1.0:
            gal = gal.shear(q=gal_q, beta=gal_beta)
    return gal


def _generate_gal_fft(record, magzero, rng, gsparams):
    gal0 = generate_cosmos_gal(record, trunc_ratio=-1, gsparams=gsparams)
    # E.g., HSC's i-band coadds zero point is 27
    flux = 10 ** ((magzero - record["mag_auto"]) / 2.5)
    # flux_scaling=   2.587
    gal0 = gal0.withFlux(flux)
    # rescale the radius by 'rescale' and keep surface brightness the
    # same
    rescale = rng.np.uniform(0.95, 1.05)
    gal0 = gal0.expand(rescale)
    # rotate by a random angle
    ang = (rng.np.uniform(0.0, np.pi * 2.0)) * galsim.radians
    gal0 = gal0.rotate(ang)
    return gal0


def _generate_gal_mc(record, magzero, rng, gsparams, npoints):
    # need to truncate the profile since we do not want
    # Knots locate at infinity
    galp = generate_cosmos_gal(record, trunc_ratio=5.0)
    # accounting for zeropoint difference between COSMOS HST and HSC
    # HSC's i-band coadds zero point is 27
    flux = 10 ** ((magzero - record["mag_auto"]) / 2.5)
    galp = galp.withFlux(flux)
    # rescale the radius by 'rescale' and keep surface brightness the
    # same
    rescale = rng.np.uniform(0.95, 1.05)
    galp = galp.expand(rescale)
    # rotate by a random angle
    ang = (rng.np.uniform(0.0, np.pi * 2.0)) * galsim.radians
    galp = galp.rotate(ang)
    gal0 = galsim.RandomKnots(
        npoints=npoints,
        profile=galp,
        rng=rng,
        gsparams=gsparams,
    )
    return gal0


def make_exposure_stamp(
    sim_method,
    rng,
    magzero,
    psf_obj,
    scale,
    cat_input,
    ngalx,
    ngaly,
    ngrid,
    rot_field,
    g1,
    g2,
    nrot_per_gal,
    do_shift,
    buff=0,
    draw_method="auto",
):
    ngal = ngalx * ngaly
    gsparams = galsim.GSParams(maximum_fft_size=10240)
    gal0 = None
    gal_image_list = []
    npix_x = ngalx * ngrid + buff * 2.0
    npix_y = ngaly * ngrid + buff * 2.0
    for _ in range(len(rot_field)):
        gal_image = galsim.ImageF(npix_x, npix_y, scale=scale)
        gal_image.setOrigin(0, 0)
        gal_image_list.append(gal_image)
    for i in range(ngal):
        # boundary
        ix = i % ngalx
        iy = i // ngalx
        # each galaxy
        irot = i % nrot_per_gal
        igal = i // nrot_per_gal
        if irot == 0:
            # prepare the base galaxy
            del gal0
            if sim_method == "fft":
                gal0 = _generate_gal_fft(
                    cat_input[igal],
                    magzero,
                    rng,
                    gsparams,
                )
            elif sim_method == "mc":
                gal0 = _generate_gal_mc(
                    cat_input[igal],
                    magzero,
                    rng,
                    gsparams,
                    npoints=15,
                )
            else:
                raise ValueError("Does not support sim_method=%s" % sim_method)
        else:
            # rotate the base galaxy
            assert gal0 is not None
            ang = np.pi / nrot_per_gal * galsim.radians
            # update gal0
            gal0 = gal0.rotate(ang)

        s01 = rng.np.uniform(low=-0.5, high=0.5) * scale
        s02 = rng.np.uniform(low=-0.5, high=0.5) * scale
        for ii, rr in enumerate(rot_field):
            # base rotation and shear distortion
            gal = gal0.rotate(rr * galsim.radians).shear(g1=g1, g2=g2)
            gal = galsim.Convolve([psf_obj, gal], gsparams=gsparams)
            b = galsim.BoundsI(
                ix * ngrid + buff,
                (ix + 1) * ngrid - 1 + buff,
                iy * ngrid + buff,
                (iy + 1) * ngrid - 1 + buff,
            )
            sub_img = gal_image_list[ii][b]
            # shift with a random offset
            s1, s2 = coord_rotate(s01, s02, xref=0, yref=0, theta=rr)
            if do_shift:
                gal = gal.shift(s1, s2)
            # shift to (ngrid//2, ngrid//2)
            # since I set it as the default center of grids in the simulation
            gal = gal.shift(0.5 * scale, 0.5 * scale)
            gal.drawImage(sub_img, add_to_image=True, method=draw_method)
    gc.collect()
    exposures = [img.array for img in gal_image_list]
    return exposures


def read_cosmos_catalog(filename=None):
    # read the input COSMOS galaxy catalog
    if filename is None:
        filename = os.path.join(__data_dir__, "src_cosmos.fits")
    cat_input = pyfits.getdata(filename)
    return cat_input


class CosmosCatalog(object):
    def __init__(
        self,
        filename=None,
        max_mag=None,
        min_mag=None,
        max_hlr=None,
        min_hlr=None,
        gal_type="mixed",
    ):
        src = read_cosmos_catalog(filename)
        # Initializing selector mask with all Trues
        sel = np.ones(len(src), dtype=bool)
        # Filtering conditions
        if max_mag is not None:
            sel &= src["mag_auto"] < max_mag
        if min_mag is not None:
            sel &= src["mag_auto"] >= min_mag
        if max_hlr is not None:
            if gal_type == "mixed":
                sel &= (src["hlr"][:, 0:3] < max_hlr).all(axis=1)
            elif gal_type == "sersic":
                sel &= src["hlr"][:, 0] < max_hlr
            elif gal_type == "bulgedisk":
                sel &= (src["hlr"][:, 1:3] < max_hlr).all(axis=1)
        if min_hlr is not None:
            if gal_type == "mixed":
                sel &= (src["hlr"][:, 0:3] >= max_hlr).all(axis=1)
            elif gal_type == "sersic":
                sel &= src["hlr"][:, 0] >= min_hlr
            elif gal_type == "bulgedisk":
                sel &= (src["hlr"][:, 1:3] >= max_hlr).all(axis=1)
        # Applying selector mask
        src = src[sel]

        if gal_type == "mixed":
            logging.info("Creating Mixed galaxy profiles")
        elif gal_type == "sersic":
            logging.info("Creating single Sersic profiles")
            src = src[src["use_bulgefit"] == 0]
        elif gal_type == "bulgedisk":
            logging.info("Creating Bulge + Disk profiles")
            src = src[src["use_bulgefit"] != 0]
        elif gal_type == "debug":
            # This is used for debug
            logging.info("Creating profile for debug")
            src = src[src["use_bulgefit"] == 0]
            # src["sersicfit"][:, 3] = 1.0  # round galaxies
            # src["sersicfit"][:, 2] = 0.5  # only Gaussian
            src["sersicfit"][:, 2] = 1.0  # only Exponential
        else:
            raise ValueError("Do not support gal_type=%s" % gal_type)
        self.cat_input = src
        self.ntrain = len(src)
        return

    def make_catalog(self, rng, n):
        if self.ntrain < n:
            raise ValueError(
                "mag_cut is too small, and there is not enough number of galaxies"
            )
        # nrot_per_gal is the number of rotated galaxies in each subfield
        inds = rng.np.integers(low=0, high=self.ntrain, size=n)
        src = self.cat_input[inds]
        return src


def make_isolate_sim(
    ny,
    nx,
    psf_obj,
    gname,
    seed,
    catname=None,
    scale=0.168,
    magzero=27.0,
    rot_field=None,
    shear_value=0.02,
    ngrid=64,
    nrot_per_gal=nrot_default,
    max_mag=None,
    min_mag=None,
    max_hlr=None,
    min_hlr=None,
    gal_type="mixed",
    do_shift=False,
    npoints=30,
    sim_method="fft",
    buff=0,
    draw_method="auto",
):
    """Makes basic **isolated** galaxy image simulation.

    Args:
        ny (int):               number of pixels in y direction
        nx (int):               number of pixels in y direction
        psf_obj (PSF):          input PSF object of galsim
        gname (str):            shear distortion setup
        seed (int):             index of the simulation
        catname (str):          input catalog name
        scale (float):          pixel scale
        magzero (float):        magnitude zero point [27 for HSC]
        rot_field (list):       additional rotation angle
        shear_value (float):    shear distortion amplitude
        ngrid (int):            stampe size
        nrot_per_gal (int):             number of rotations
        max_mag (float):        maximum magnitude cut
        min_mag (float):        minimum magnitude cut
        max_hlr (float):        maximum half light radius cut [arcsec]
        min_hlr (float):        minimum half light radius cut [arcsec]
        gal_type (float):       galaxy morphology (mixed, sersic, or bulgedisk)
        do_shift (bool):        whether do shfits
        npoints (int):          number of random points when
        sim_method (str):       galaxy tpye ("fft" or "mc")
        buff (int):             buff size (zero padding near boundaries)
    """

    if nx % ngrid != 0:
        raise ValueError("nx is not divisible by ngrid")
    if ny % ngrid != 0:
        raise ValueError("ny is not divisible by ngrid")
    # Basic parameters
    ngalx = int(nx // ngrid)
    ngaly = int(ny // ngrid)
    ngal = ngalx * ngaly
    rng = galsim.BaseDeviate(seed)
    ngeff = max(ngal // nrot_per_gal, 1)
    cosmos_cat = CosmosCatalog(
        max_mag=max_mag,
        min_mag=min_mag,
        max_hlr=max_hlr,
        min_hlr=min_hlr,
        gal_type=gal_type,
    )
    cat_input = cosmos_cat.make_catalog(rng=rng, n=ngeff)
    # Get the shear information
    shear_list = np.array([-shear_value, shear_value, 0.0])
    dis_version = int(eval(gname.split("-")[-1]))
    assert dis_version < 3 and dis_version >= 0, "gname is not supported"
    shear_const = shear_list[dis_version]
    gver = gname.split("-")[0]
    if gver == "g1":
        g1 = shear_const
        g2 = 0.0
    elif gver == "g2":
        g1 = 0.0
        g2 = shear_const
    elif gver == "g1_g2":
        g1 = shear_const
        g2 = shear_const
    else:
        raise ValueError("cannot decide g1 or g2")
    logging.info("Processing for %s, and shear is %s." % (gname, shear_const))

    if rot_field is None:
        rot_field = [0.0]
    exposures = make_exposure_stamp(
        sim_method=sim_method,
        rng=rng,
        magzero=magzero,
        psf_obj=psf_obj,
        scale=scale,
        cat_input=cat_input,
        ngalx=ngalx,
        ngaly=ngaly,
        ngrid=ngrid,
        rot_field=rot_field,
        g1=g1,
        g2=g2,
        nrot_per_gal=nrot_per_gal,
        do_shift=do_shift,
        buff=buff,
        draw_method=draw_method,
    )
    return exposures


def make_noise_sim(
    out_dir,
    infname,
    ind0,
    ny=6400,
    nx=6400,
    scale=0.168,
    do_write=True,
    return_array=False,
):
    """Makes pure noise for galaxy image simulation.

    Args:
        out_dir (str):          output directory
        ind0 (int):             index of the simulation
        ny (int):               number of pixels in y direction
        nx (int):               number of pixels in x direction
        do_write (bool):        whether write output [default: True]
        return_array (bool):    whether return galaxy array [default: False]
    """
    logging.info("begining for field %04d" % (ind0))
    out_fname = os.path.join(out_dir, "noi%04d.fits" % (ind0))
    if os.path.exists(out_fname):
        if do_write:
            logging.info("Nothing to write.")
        if return_array:
            return pyfits.getdata(out_fname)
        else:
            return None
    logging.info("simulating noise for field %s" % (ind0))
    variance = 0.01
    ud = galsim.UniformDeviate(ind0 * 10000 + 1)

    # setup the galaxy image and the noise image
    noi_image = galsim.ImageF(nx, ny, scale=scale)
    noi_image.setOrigin(0, 0)
    noise_obj = galsim.getCOSMOSNoise(
        file_name=infname, rng=ud, cosmos_scale=scale, variance=variance
    )
    noise_obj.applyTo(noi_image)
    if do_write:
        pyfits.writeto(out_fname, noi_image.array)
    if return_array:
        return noi_image.array
    return


def make_gal_ssbg(shear, psf, rng, r1, r0=20.0):
    """This function is for the simulation for source photon noise. It
    simulates an exponential object with moffat PSF, given a SNR [r0] and a
    source background noise ratio [r0].

    Args:
        shear (tuple):          [g1, g2]. The shear in each component
        rng (randState):        The random number generator
        r1  (float):            The source background noise variance ratio
        r0  (float):            The SNR of galaxy
        psf (galsim.Moffat):    a Moffat PSF

    Returns:
       img (ndarray):           noisy image array
    """
    scale = 0.263
    gal_hlr = 0.5

    dy, dx = rng.uniform(low=-scale / 2, high=scale / 2, size=2)

    obj0 = (
        galsim.Exponential(
            half_light_radius=gal_hlr,
        )
        .shear(
            g1=shear[0],
            g2=shear[1],
        )
        .shift(
            dx=dx,
            dy=dy,
        )
    )
    obj = galsim.Convolve(psf, obj0)

    # define the psf and psf here which will be repeatedly used
    psf = psf.drawImage(scale=scale).array
    # galaxy image:
    img = obj.drawImage(scale=scale).array
    ngrid = img.shape[0]
    # noise image:
    noimg = rng.normal(scale=1.0, size=img.shape)
    # get the current flux using the 5x5 substamps centered at the stamp's center
    flux_tmp = np.sum(
        img[ngrid // 2 - 2 : ngrid // 2 + 3, ngrid // 2 - 2 : ngrid // 2 + 3]
    )
    # the current (expectation of) total noise std on the 5x5 substamps is 5
    # since for each pixel, the expecatation value of variance is 1; therefore,
    # the expectation value of variance is 25...
    std_tmp = 5
    # normalize both the galaxy image and noise image so that they will have
    # flux=1 and variance=1 (expectation value) in the 5x5 substamps
    img = img / flux_tmp
    noimg = noimg / std_tmp
    # now we can determine the flux and background variance using equation (3)
    source_flux = r0**2.0 * (1 + r1) / r1
    back_flux = source_flux / r1
    img = img * source_flux
    noimg = noimg * np.sqrt(back_flux)
    img = img + noimg
    return img


class Stamp(object):
    def __init__(self, coords=None, nn=32, scale=0.2):
        """Initialize the 2D stamp object. This class enables distorting
        an image by changing the samplinng position with non-affine
        transformation

        Args:
            nn (int):      number of grids on x and y direction
            scale (float): pixel scale in units of arcsec
        """
        if coords is None:
            indx = np.arange(-int(nn / 2), int((nn + 1) / 2), 1) * scale
            indy = np.arange(-int(nn / 2), int((nn + 1) / 2), 1) * scale
            inds = np.meshgrid(indy, indx, indexing="ij")
            self.coords = np.vstack([np.ravel(_) for _ in inds[::-1]])
        else:
            self.coords = coords
        self.pixel_values = None
        self.transformed = False
        if self.coords.shape[0] != 2:
            raise ValueError("Something wrong with the coordinate shape")
        self.shape = (nn, nn)
        return

    def sample_galaxy(self, gal_obj):
        """Sample the surface density field of a galaxy at the grids
        This function only conduct sampling; PSF and pixel response are
        not included.

        Args:
            gal_obj (galsim):   Galsim galaxy object to sample on the grids
        Returns:
            outcome (ndarray):  2D galaxy image on the grids
        """
        pixel_values = np.array([gal_obj.xValue(cc) for cc in self.coords.T])

        return np.reshape(pixel_values, self.shape)

    def transform_grids(self, transform_obj):
        if not hasattr(transform_obj, "transform"):
            raise TypeError("transform_obj is not in correct data type")
        self.coords = transform_obj.transform(self.coords)
        self.transformed = True
        return


class LensTransform1(object):
    def __init__(self, gamma1, gamma2, kappa):
        """Initialize the transform object of 2D grids
        Args:
            gamma1 (float):     the first component of lensing shear field
            gamma2 (float):     the second component of lensing shear field
            kappa (float):      the lensing convergence field
        """
        self.s2l_mat = np.array(
            [[1 - kappa - gamma1, -gamma2], [-gamma2, 1 - kappa + gamma1]]
        )
        return

    def transform(self, coords):
        """transform the center of pixels from lensed plane to pre-lensed plane
        Args:
            coords:   coordinates (x, y) of the pixel centers [arcsec]
        """
        return self.s2l_mat @ coords
