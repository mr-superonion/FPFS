import os
import gc
import json
import glob
import logging

import fitsio
import galsim
import numpy as np
from .util import make_isolate_sim
from configparser import ConfigParser, ExtendedInterpolation

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)


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


class SimulationTask(object):
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        self.img_dir = cparser.get("files", "img_dir")
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir, exist_ok=True)
        self.sim_method = cparser.get("simulation", "sim_method", fallback="fft")
        self.gal_type = cparser.get("simulation", "gal_type", fallback="mixed").lower()
        self.nrot = cparser.getint("simulation", "nrot")
        self.buff = cparser.getint("simulation", "buff")
        self.image_nx = cparser.getint("simulation", "image_nx")
        self.image_ny = cparser.getint("simulation", "image_ny")
        self.band = cparser.get("simulation", "band")
        self.do_shift = cparser.getboolean("simulation", "do_shift", fallback=False)
        self.scale = cparser.getfloat("survey", "pixel_scale")
        self.max_hlr = cparser.getfloat(
            "simulation",
            "max_hlr",
            fallback=1.5e0,
        )
        self.min_hlr = cparser.getfloat(
            "simulation",
            "min_hlr",
            fallback=1e-2,
        )
        assert self.image_ny == self.image_nx, "'image_nx' must equals 'image_ny'!"
        self.psf_obj = None
        assert self.sim_method in ["fft", "mc"]
        assert self.gal_type in ["mixed", "sersic", "bulgedisk", "debug"]
        # PSF
        seeing = cparser.getfloat("survey", "psf_fwhm", fallback=4.0 * self.scale)
        logging.info("Using modelled Moffat PSF with seeing %.2f arcsec. " % seeing)
        psffname = os.path.join(self.img_dir, "psf-%d.fits" % (seeing * 100))
        psf_beta = cparser.getfloat("survey", "psf_moffat_beta")
        psf_trunc = cparser.getfloat("survey", "psf_trunc_ratio")
        psf_e1 = cparser.getfloat("survey", "psf_e1")
        psf_e2 = cparser.getfloat("survey", "psf_e2")
        no_pixel = cparser.getboolean("survey", "no_pixel")
        if no_pixel:
            self.draw_method = "no_pixel"
        else:
            self.draw_method = "fft"
        if psf_trunc < 1.0:
            self.psf_obj = galsim.Moffat(
                beta=psf_beta,
                fwhm=seeing,
            ).shear(e1=psf_e1, e2=psf_e2)
        else:
            trunc = seeing * psf_trunc
            self.psf_obj = galsim.Moffat(beta=psf_beta, fwhm=seeing, trunc=trunc).shear(
                e1=psf_e1, e2=psf_e2
            )
        # write psf image
        psf_image = self.psf_obj.shift(
            0.5 * self.scale,
            0.5 * self.scale,
        ).drawImage(nx=64, ny=64, scale=self.scale, method=self.draw_method)
        psf_image.write(psffname)
        # Shear
        self.gver = cparser.get("distortion", "g_version")
        zlist = json.loads(cparser.get("distortion", "shear_z_list"))
        self.gname_list = ["%s-%s" % (self.gver, i1) for i1 in zlist]
        logging.info(
            "We will test the following constant shear distortion setups %s. "
            % self.gname_list
        )
        self.shear_value = cparser.getfloat("distortion", "shear_value")
        self.rot_list = [np.pi / self.nrot * i for i in range(self.nrot)]
        return

    def run(self, ifield):
        logging.info("start ID: %d" % (ifield))
        for gn in self.gname_list:
            # do basic stamp-like image simulation
            nfiles = len(
                glob.glob(
                    "%s/image-%05d_%s_rot*_%s.fits"
                    % (
                        self.img_dir,
                        ifield,
                        gn,
                        self.band,
                    )
                )
            )
            if nfiles == self.nrot:
                logging.info("We already have all the output files for %s" % gn)
                continue
            sim_img = make_isolate_sim(
                sim_method="fft",  # we use FFT method to render galaxy images
                psf_obj=self.psf_obj,
                gname=gn,
                seed=ifield,
                ny=self.image_ny,
                nx=self.image_nx,
                scale=self.scale,
                do_shift=self.do_shift,
                shear_value=self.shear_value,
                nrot_per_gal=1,
                min_hlr=self.min_hlr,  # set the minimum hlr to 0
                max_hlr=self.max_hlr,  # set maximum hlr (sersic fit)
                rot_field=self.rot_list,
                gal_type=self.gal_type,
                buff=self.buff,
                draw_method=self.draw_method,
            )
            for irot in range(self.nrot):
                gal_fname = "%s/image-%05d_%s_rot%d_%s.fits" % (
                    self.img_dir,
                    ifield,
                    gn,
                    irot,
                    self.band,
                )
                fitsio.write(gal_fname, sim_img[irot])
            gc.collect()
        logging.info("finish processing field ID: %d" % (ifield))
        return

    def clear(self, ifield):
        for gn in self.gname_list:
            for irot in range(self.nrot):
                gal_fname = "%s/image-%05d_%s_rot%d_%s.fits" % (
                    self.img_dir,
                    ifield,
                    gn,
                    irot,
                    self.band,
                )
                if os.path.isfile(gal_fname):
                    os.remove(gal_fname)
        logging.info("Cleaning results for field ID: %s" % (ifield))
        return

    def load_outcomes(self, ifield):
        outcomes = {}
        for gn in self.gname_list:
            for irot in range(self.nrot):
                gal_fname = "%s/image-%05d_%s_rot%d_%s.fits" % (
                    self.img_dir,
                    ifield,
                    gn,
                    irot,
                    self.band,
                )
                if os.path.isfile(gal_fname):
                    image = fitsio.read(gal_fname)
                    outcomes.update({"%s_rot%s" % (gn, irot): image})
        return outcomes
