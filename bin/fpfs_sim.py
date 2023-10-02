#!/usr/bin/env python
#
# FPFS shear estimator
# Copyright 20220312 Xiangchong Li.
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
import os
import gc
import glob
import fpfs
import json
import galsim
import schwimmbad
import numpy as np
from argparse import ArgumentParser
from configparser import ConfigParser, ExtendedInterpolation


class Worker(object):
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
        self.band_name = cparser.get("simulation", "band")
        self.do_shift = cparser.getboolean("simulation", "do_shift", fallback=False)
        self.scale = cparser.getfloat("survey", "pixel_scale")
        self.max_hlr = cparser.getfloat(
            "simulation",
            "max_hlr",
            fallback=self.scale * 32.0 / 5.0,
        )
        assert self.image_ny == self.image_nx, "'image_nx' must equals 'image_ny'!"
        self.psf_obj = None
        assert self.sim_method in ["fft", "mc"]
        assert self.gal_type in ["mixed", "sersic", "bulgedisk"]
        # PSF
        seeing = cparser.getfloat("survey", "psf_fwhm", fallback=4.0 * self.scale)
        print("Using modelled Moffat PSF with seeing %.2f arcsec. " % seeing)
        psffname = os.path.join(self.img_dir, "psf-%d.fits" % (seeing * 100))
        psf_beta = cparser.getfloat("survey", "psf_moffat_beta", fallback=3.5)
        psf_trunc = cparser.getfloat("survey", "psf_trunc_ratio", fallback=4)
        psf_e1 = cparser.getfloat("survey", "psf_e1", fallback=0.02)
        psf_e2 = cparser.getfloat("survey", "psf_e2", fallback=-0.02)
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
        ).drawImage(nx=64, ny=64, scale=self.scale)
        psf_image.write(psffname)
        # Shear
        self.gver = cparser.get("distortion", "g_version")
        zlist = json.loads(cparser.get("distortion", "shear_z_list"))
        self.gname_list = ["%s-%s" % (self.gver, i1) for i1 in zlist]
        print(
            "We will test the following constant shear distortion setups %s. "
            % self.gname_list
        )
        self.shear_value = cparser.getfloat("distortion", "shear_value")
        self.rot_list = [np.pi / self.nrot * i for i in range(self.nrot)]
        return

    def run(self, ifield):
        print("start ID: %d" % (ifield))
        for pp in self.gname_list:
            # do basic stamp-like image simulation
            nfiles = len(
                glob.glob(
                    "%s/image-%05d_%s_rot*_%s.fits"
                    % (
                        self.img_dir,
                        ifield,
                        pp,
                        self.band_name,
                    )
                )
            )
            if nfiles == self.nrot:
                print("We already have all the output files for %s" % pp)
                continue
            sim_img = fpfs.simutil.make_isolate_sim(
                sim_method="fft",  # we use FFT method to render galaxy images
                psf_obj=self.psf_obj,
                gname=pp,
                seed=ifield,
                ny=self.image_ny,
                nx=self.image_nx,
                scale=self.scale,
                do_shift=self.do_shift,
                shear_value=self.shear_value,
                nrot_per_gal=1,
                min_hlr=0.0,  # set the minimum hlr to 0
                max_hlr=self.max_hlr,  # set maximum hlr (sersic fit)
                rot_field=self.rot_list,
                gal_type=self.gal_type,
                buff=self.buff,
            )
            for irot in range(self.nrot):
                gal_fname = "%s/image-%05d_%s_rot%d_%s.fits" % (
                    self.img_dir,
                    ifield,
                    pp,
                    irot,
                    self.band_name,
                )
                fpfs.io.save_image(gal_fname, sim_img[irot])
            gc.collect()
        print("finish ID: %d" % (ifield))
        return

    def __call__(self, ifield):
        return self.run(ifield)


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs simulation")
    parser.add_argument(
        "--min_id",
        required=True,
        type=int,
        help="minimum id number, e.g. 0",
    )
    parser.add_argument(
        "--max_id",
        required=True,
        type=int,
        help="maximum id number, e.g. 4000",
    )
    parser.add_argument("--config", required=True, type=str, help="configure file name")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )
    args = parser.parse_args()
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    worker = Worker(args.config)
    refs = list(range(args.min_id, args.max_id))
    for r in pool.map(worker, refs):
        pass
    pool.close()
