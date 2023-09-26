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
import time
import fpfs
import schwimmbad
import numpy as np
import astropy.io.fits as pyfits
from argparse import ArgumentParser
from configparser import ConfigParser

band_map = {
    "g": 0,
    "r": 1,
    "i": 2,
    "z": 3,
    "a": 4,
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


def get_sim_fname(directory, ftype, min_id, max_id, nshear, nrot, band):
    """Generate filename for simulations
    Args:
        ftype (str):    file type ('src' for source, and 'image' for exposure
        min_id (int):   minimum id
        max_id (int):   maximum id
        nshear (int):   number of shear
        nrot (int):     number of rotations
        band (str):     'grizy' or 'a'
    Returns:
        out (list):     a list of file name
    """
    out = [
        os.path.join(
            directory, "%s-%05d_g1-%d_rot%d_%s.fits" % (ftype, fid, gid, rid, band)
        )
        for fid in range(min_id, max_id)
        for gid in range(nshear)
        for rid in range(nrot)
    ]
    return out


def get_seed_from_fname(fname, band):
    fid = int(fname.split("image-")[-1].split("_")[0]) + 212
    rid = int(fname.split("rot")[1][0])
    bid = band_map[band]
    return (fid * 2 + rid) * 4 + bid


class Worker(object):
    def __init__(self, config_name):
        cparser = ConfigParser()
        cparser.read(config_name)
        self.imgdir = cparser.get("files", "img_dir")
        self.catdir = cparser.get("files", "cat_dir")
        if not os.path.isdir(self.imgdir):
            raise FileNotFoundError("Cannot find input images directory!")
        print("The input directory for galaxy images is %s. " % self.imgdir)
        if not os.path.isdir(self.catdir):
            os.makedirs(self.catdir, exist_ok=True)
        print("The output directory for shear catalogs is %s. " % self.catdir)

        # setup FPFS task
        self.psf_fname = cparser.get("files", "psf_fname")
        self.sigma_as = cparser.getfloat("FPFS", "sigma_as")
        self.sigma_det = cparser.getfloat("FPFS", "sigma_det")
        self.rcut = cparser.getint("FPFS", "rcut")
        self.nnord = cparser.getint("FPFS", "nnord", fallback=4)
        if self.nnord not in [4, 6]:
            raise ValueError(
                "Only support for nnord= 4 or nnord=6, but your input\
                    is nnord=%d"
                % self.nnord
            )

        # setup survey parameters
        self.noise_ratio = cparser.getfloat("survey", "noise_ratio")
        self.ncov_fname = os.path.join(self.catdir, "cov_matrix.fits")
        self.magz = cparser.getfloat("survey", "mag_zero")
        self.band = cparser.get("survey", "band")
        self.scale = cparser.getfloat("survey", "pixel_scale")
        self.nstd_f = nstd_map[self.band] * self.noise_ratio
        ngrid = 2 * self.rcut
        self.noise_pow = np.ones((ngrid, ngrid)) * self.nstd_f**2.0 * ngrid**2.0
        return

    def prepare_noise_psf(self, fname):
        exposure = pyfits.getdata(fname)
        self.image_nx = exposure.shape[1]
        psf_array2 = pyfits.getdata(self.psf_fname)
        npad = (self.image_nx - psf_array2.shape[0]) // 2
        psf_array3 = np.pad(psf_array2, (npad, npad), mode="constant")
        if not os.path.isfile(self.ncov_fname):
            # FPFS noise cov task
            noise_task = fpfs.image.measure_noise_cov(
                psf_array2,
                sigma_arcsec=self.sigma_as,
                sigma_detect=self.sigma_det,
                nnord=self.nnord,
                pix_scale=self.scale,
            )
            cov_elem = np.array(noise_task.measure(self.noise_pow))
            pyfits.writeto(self.ncov_fname, cov_elem, overwrite=True)
        else:
            cov_elem = pyfits.getdata(self.ncov_fname)
        return psf_array2, psf_array3, cov_elem

    def prepare_image(self, fname):
        gal_array = np.zeros((self.image_nx, self.image_nx))
        print("processing %s band" % self.band)
        gal_array = pyfits.getdata(fname)
        if self.nstd_f > 1e-10:
            # noise
            seed = get_seed_from_fname(fname, self.band)
            rng = np.random.RandomState(seed)
            print("Using noisy setup with std: %.2f" % self.nstd_f)
            print("The random seed is %d" % seed)
            gal_array = gal_array + rng.normal(
                scale=self.nstd_f,
                size=gal_array.shape,
            )
        else:
            print("Using noiseless setup")
        return gal_array

    def process_image(self, gal_array, psf_array2, psf_array3, cov_elem):
        # measurement task
        meas_task = fpfs.image.measure_source(
            psf_array2,
            sigma_arcsec=self.sigma_as,
            sigma_detect=self.sigma_det,
            nnord=self.nnord,
            pix_scale=self.scale,
        )
        print(
            "The upper limit of Fourier wave number is %s pixels" % (meas_task.klim_pix)
        )

        std_modes = np.sqrt(np.diagonal(cov_elem))
        idm00 = fpfs.catalog.indexes["m00"]
        idv0 = fpfs.catalog.indexes["v0"]
        # Temp fix for 4th order estimator
        if self.nnord == 6:
            idv0 += 1
        if std_modes[idm00] > 1e-10:
            thres = 9.5 * std_modes[idm00] * self.scale**2.0
            thres2 = -1.5 * std_modes[idv0] * self.scale**2.0
        else:
            cutmag = self.magz * 0.935
            thres = 10 ** ((self.magz - cutmag) / 2.5) * self.scale**2.0
            thres2 = -0.05

        coords = fpfs.image.detect_sources(
            gal_array,
            psf_array3,
            sigmaf=meas_task.sigmaf,
            sigmaf_det=meas_task.sigmaf_det,
            thres=thres,
            thres2=thres2,
            bound=self.rcut + 5,
        )
        print("pre-selected number of sources: %d" % len(coords))
        out = meas_task.measure(gal_array, coords)
        out = meas_task.get_results(out)
        sel = (out["fpfs_M00"] + out["fpfs_M20"]) > 0.0
        out = out[sel]
        print("final number of sources: %d" % len(out))
        coords = coords[sel]
        coords = np.rec.fromarrays(coords.T, dtype=[("fpfs_y", "i4"), ("fpfs_x", "i4")])
        return out, coords

    def run(self, fname):
        out_fname = os.path.join(self.catdir, fname.split("/")[-1])
        out_fname = out_fname.replace("image-", "src-")

        det_fname = os.path.join(self.catdir, fname.split("/")[-1])
        det_fname = det_fname.replace("image-", "det-")
        if os.path.isfile(out_fname) and os.path.isfile(det_fname):
            print("Already has measurement for this simulation. ")
            return
        psf_array2, psf_array3, cov_elem = self.prepare_noise_psf(fname)
        gal_array = self.prepare_image(fname)
        start_time = time.time()
        cat, det = self.process_image(gal_array, psf_array2, psf_array3, cov_elem)
        # Stop the timer
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        # Print the elapsed time
        print(f"Elapsed time: {elapsed_time} seconds")
        fpfs.io.save_catalog(det_fname, det, dtype="position", nnord=str(self.nnord))
        fpfs.io.save_catalog(out_fname, cat, dtype="shape", nnord=str(self.nnord))
        return


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs_process_sims")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="configure file name",
    )
    parser.add_argument(
        "--min_id",
        required=True,
        type=int,
        help="minimum ID, e.g. 0",
    )
    parser.add_argument(
        "--max_id",
        required=True,
        type=int,
        help="maximum ID, e.g. 4000",
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
    fname_list = get_sim_fname(
        worker.imgdir,
        "image",
        args.min_id,
        args.max_id,
        2,
        2,
        worker.band,
    )
    for r in pool.map(worker.run, fname_list):
        pass
    pool.close()
