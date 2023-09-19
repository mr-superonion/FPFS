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
from configparser import ConfigParser


class Worker(object):
    def __init__(self, config_name):
        cparser = ConfigParser()
        cparser.read(config_name)
        self.sim_method = cparser.get("simulation", "sim_method")
        self.gal_type = cparser.get("simulation", "gal_type").lower()
        self.imgdir = cparser.get("simulation", "img_dir")
        self.nrot = cparser.getint("simulation", "nrot")
        self.band_name = cparser.get("simulation", "band")
        self.scale = cparser.getfloat("survey", "pixel_scale")
        self.image_nx = cparser.getint("survey", "image_nx")
        self.image_ny = cparser.getint("survey", "image_ny")
        assert self.image_ny == self.image_nx, "'image_nx' must equals 'image_ny'!"
        self.psf_obj = None
        self.outdir = os.path.join(self.imgdir, self.sim_method)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)
        assert (
            self.sim_method in ["fft", "mc"]
        )
        assert (
            self.gal_type in ["mixed", "sersic", "bulgedisk"]
        )
        if cparser.has_option("survey", "psf_fwhm"):
            seeing = cparser.getfloat("survey", "psf_fwhm")
            self.prepare_psf(seeing, psf_type="moffat")
            print("Using modelled Moffat PSF with seeing %.2f arcsec. " % seeing)
        else:
            if not cparser.has_option("survey", "psf_filename"):
                raise ValueError("Do not have survey-psf_file option")
            else:
                self.psffname = cparser.get("survey", "psf_filename")
                print("Using PSF from input file. ")
        glist = []
        if cparser.getboolean("distortion", "test_g1"):
            glist.append("g1")
        if cparser.getboolean("distortion", "test_g2"):
            glist.append("g2")
        if len(glist) > 0:
            zlist = json.loads(cparser.get("distortion", "shear_z_list"))
            self.pendList = ["%s-%s" % (i1, i2) for i1 in glist for i2 in zlist]
        else:
            # this is for non-distorted image simulation
            self.pendList = ["g1-2"]
        print(
            "We will test the following constant shear distortion setups %s. "
            % self.pendList
        )
        self.shear_value = cparser.getfloat("distortion", "shear_value")
        self.rot_list = [np.pi / self.nrot * i for i in range(self.nrot)]
        return

    def prepare_psf(self, seeing, psf_type):
        psffname = os.path.join(self.outdir, "psf-%d.fits" % (seeing * 100))
        if psf_type.lower() == "moffat":
            self.psf_obj = galsim.Moffat(
                beta=3.5, fwhm=seeing, trunc=seeing * 4.0
            ).shear(e1=0.02, e2=-0.02)
        else:
            raise ValueError("Only support moffat PSF.")
        psf_image = self.psf_obj.shift(
            0.5 * self.scale, 0.5 * self.scale
        ).drawImage(nx=64, ny=64, scale=self.scale)
        psf_image.write(psffname)
        return

    def run(self, ifield):
        print("start ID: %d" % (ifield))
        if self.psf_obj is None:
            if "%" in self.psffname:
                psffname = self.psffname % ifield
            else:
                psffname = self.psffname
            assert os.path.isfile(psffname), "Cannot find input PSF file"
            psf_image = galsim.fits.read(psffname)
            self.psf_obj = galsim.InterpolatedImage(
                psf_image, scale=self.scale, flux=1.0
            )
            del psf_image
        for pp in self.pendList:
            # do basic stamp-like image simulation
            nfiles = len(glob.glob("%s/image-%05d_%s_rot*_%s.fits" % (
                self.outdir,
                ifield,
                pp,
                self.band_name,
            )))
            if nfiles == self.nrot:
                print("We already have all the output files for %s" % pp)
                continue
            sim_img = fpfs.simutil.make_isolate_sim(
                sim_method="fft",
                psf_obj=self.psf_obj,
                gname=pp,
                seed=ifield,
                ny=self.image_ny,
                nx=self.image_nx,
                scale=self.scale,
                do_shift=False,
                shear_value=self.shear_value,
                nrot=1,
                rot2=self.rot_list,
                gal_type=self.gal_type,
            )
            for irot in range(self.nrot):
                gal_fname = "%s/image-%05d_%s_rot%d_%s.fits" % (
                    self.outdir,
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
        "--min_id", required=True, type=int, help="minimum id number, e.g. 0"
    )
    parser.add_argument(
        "--max_id", required=True, type=int, help="maximum id number, e.g. 4000"
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
        "--mpi", dest="mpi", default=False, action="store_true", help="Run with MPI."
    )
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    worker = Worker(args.config)
    refs = list(range(args.min_id, args.max_id))
    # worker(1)
    for r in pool.map(worker, refs):
        pass
    pool.close()
