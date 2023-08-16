#!/usr/bin/env python
#
# FPFS shear estimator
# Copyright 20221013 Xiangchong Li.
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
import glob
import schwimmbad
import numpy as np
import astropy.io.fits as pyfits
from fpfs.io import save_image
from argparse import ArgumentParser
from configparser import ConfigParser


band_map = {
    "g": 0,
    "r": 1,
    "i": 2,
    "z": 3,
}


version = 1
if version == 1:
    nstd_map = {
        "g": 0.315,
        "r": 0.371,
        "i": 0.595,
        "z": 1.155,
        "a": 0.2186,
    }
    w_map = {
        "g": 0.48179905,
        "r": 0.34732755,
        "i": 0.13503710,
        "z": 0.03583629,
    }
elif version == 2:
    nstd_map = {
        "g": 0.315,
        "r": 0.371,
        "i": 0.595,
        "z": 1.155,
        "a": 0.27934,
    }
    w_map = {
        "g": 0.12503653,
        "r": 0.47022727,
        "i": 0.30897575,
        "z": 0.09576044,
    }


class Worker(object):
    def __init__(self, config_name):
        cparser = ConfigParser()
        cparser.read(config_name)
        self.img_dir = cparser.get("procsim", "img_dir")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find input images directory!")
        print("The input directory for galaxy images is %s. " % self.img_dir)
        return

    def prepare(self, fname):
        header = pyfits.getheader(fname)
        self.image_nx = int(header["NAXIS1"])
        return

    def prepare_image(self, fname):
        blist = ["g", "r", "i", "z"]
        self.prepare(fname)
        gal_array = np.zeros((self.image_nx, self.image_nx), dtype=np.float32)
        weight_all = 0.0
        for band in blist:
            print("processing %s band" % band)
            weight = w_map[band]
            weight_all = weight_all + weight
            fname2 = fname.replace("_g.fits", "_%s.fits" % band)
            this_gal = pyfits.getdata(fname2)
            gal_array = gal_array + this_gal * weight
        gal_array = gal_array / weight_all
        return gal_array

    def run(self, fname):
        out_dir = self.img_dir
        out_fname = os.path.join(out_dir, fname.split("/")[-1])
        out_fname = out_fname.replace("_g.fits", "_a.fits") + "4"
        if os.path.exists(out_fname):
            print("Already has measurement for this simulation. ")
            return
        exposure = self.prepare_image(fname)
        save_image(out_fname, exposure)
        return


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="configure file name",
    )
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
    fname_list = glob.glob(os.path.join(worker.img_dir, "image-*_g.fits"))
    fname_list = np.sort(fname_list)  # [0:1]
    for r in pool.map(worker.run, fname_list):
        pass
    pool.close()
