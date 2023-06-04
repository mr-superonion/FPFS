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
import fpfs
import glob
import pickle
import schwimmbad
import numpy as np
import astropy.io.fits as pyfits
import lsst.geom as lsstgeom
import lsst.afw.image as afwimage
from descwl_shear_sims.psfs import make_dm_psf
from argparse import ArgumentParser
from configparser import ConfigParser

band_map = {
    "g": 0,
    "r": 1,
    "i": 2,
    "z": 3,
    "a": 4,
}

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


def get_seed_from_fname(fname, band):
    fid = int(fname.split("image-")[-1].split("_")[0]) + 212
    rid = int(fname.split("rot")[1][0])
    bid = band_map[band]
    return (fid * 2 + rid) * 4 + bid


class Worker(object):
    def __init__(self, config_name):
        cparser = ConfigParser()
        cparser.read(config_name)
        self.imgdir = cparser.get("procsim", "img_dir")
        self.catdir = cparser.get("procsim", "cat_dir")
        if not os.path.isdir(self.imgdir):
            raise FileNotFoundError("Cannot find input images directory!")
        print("The input directory for galaxy images is %s. " % self.imgdir)
        if not os.path.isdir(self.catdir):
            os.makedirs(self.catdir, exist_ok=True)
        print("The output directory for shear catalogs is %s. " % self.catdir)

        # setup FPFS task
        self.psf_fname = cparser.get("procsim", "psf_fname")
        self.sigma_as = cparser.getfloat("FPFS", "sigma_as")
        self.sigma_det = cparser.getfloat("FPFS", "sigma_det")
        self.rcut = cparser.getint("FPFS", "rcut")
        self.nnord = cparser.getint("FPFS", "nnord", fallback=4)

        # setup survey parameters
        self.noi_ratio = cparser.getfloat("survey", "noise_ratio")
        self.ncov_fname = os.path.join(self.catdir, "cov_matrix.fits")
        self.magz = cparser.getfloat("survey", "mag_zero")
        self.band = cparser.get("survey", "band")
        self.nstd_f = nstd_map[self.band] * self.noi_ratio
        if self.noi_ratio > 1e-5:
            ngrid = 2 * self.rcut
            self.noise_pow = np.ones((ngrid, ngrid)) * self.nstd_f**2.0 * ngrid**2.0
        return

    def prepare_psf(self, exposure, rcut):
        # pad to (64, 64) and then cut off
        ngrid = 64
        beg = ngrid // 2 - rcut
        end = beg + 2 * rcut
        bbox = exposure.getBBox()
        bcent = bbox.getCenter()
        psf_model = exposure.getPsf()
        psf_array = psf_model.computeImage(lsstgeom.Point2I(bcent)).getArray()
        npad = (ngrid - psf_array.shape[0]) // 2
        psf_array2 = np.pad(psf_array, (npad + 1, npad), mode="constant")[
            beg:end, beg:end
        ]
        del npad
        # pad to exposure size
        npad = (self.image_nx - psf_array2.shape[0]) // 2
        psf_array3 = np.pad(psf_array2, (npad, npad), mode="constant")
        return psf_array2, psf_array3

    def prepare_noise_psf(self, fname):
        exposure = afwimage.ExposureF.readFits(fname)
        wcs = exposure.getInfo().getWcs()
        self.scale = wcs.getPixelScale().asArcseconds()
        with open(self.psf_fname, "rb") as f:
            psf_dict = pickle.load(f)
        psf_dm = make_dm_psf(**psf_dict)
        exposure.setPsf(psf_dm)
        # zero_flux = exposure.getPhotoCalib().getInstFluxAtZeroMagnitude()
        # magz = np.log10(zero_flux) * 2.5
        self.image_nx = exposure.getWidth()
        psf_array2, psf_array3 = self.prepare_psf(
            exposure,
            self.rcut,
        )
        # FPFS Tasks
        # noise cov task
        if self.noi_ratio > 1e-10:
            if not os.path.isfile(self.ncov_fname):
                noise_task = fpfs.image.measure_noise_cov(
                    psf_array2,
                    sigma_arcsec=self.sigma_as,
                    sigma_detect=self.sigma_det,
                    pix_scale=self.scale,
                )
                cov_elem = np.array(noise_task.measure(self.noise_pow))
                pyfits.writeto(self.ncov_fname, cov_elem, overwrite=True)
            else:
                cov_elem = pyfits.getdata(self.ncov_fname)
        else:
            cov_elem = np.zeros((31, 31))
        return psf_array2, psf_array3, cov_elem

    def prepare_image(self, fname):
        if self.band != "a":
            blist = [self.band]
        else:
            blist = ["g", "r", "i", "z"]

        gal_array = np.zeros((self.image_nx, self.image_nx))
        weight_all = 0.0
        for band in blist:
            print("processing %s band" % band)
            noi_std = nstd_map[band] * self.noi_ratio
            weight = w_map[band]
            weight_all = weight_all + weight
            fname2 = fname.replace("_g.fits", "_%s.fits" % band)
            exposure = afwimage.ExposureF.readFits(fname2)
            mi = exposure.getMaskedImage()
            im = mi.getImage()
            gal_array = gal_array + im.getArray() * weight

            if noi_std > 1e-5:
                # noise
                seed = get_seed_from_fname(fname, band)
                rng = np.random.RandomState(seed)
                print("Using noisy setup with std: %.2f" % noi_std)
                print("The random seed is %d" % seed)
                gal_array = (
                    gal_array
                    + rng.normal(
                        scale=noi_std,
                        size=gal_array.shape,
                    )
                    * weight
                )
            else:
                print("Using noiseless setup")
        gal_array = gal_array / weight_all
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
        if std_modes[idm00] > 1e-10:
            thres = 8.5 * std_modes[idm00] * self.scale**2.0
            thres2 = -1.0 * std_modes[idv0] * self.scale**2.0
        else:
            magz = 30.0
            cutmag = 26.5
            thres = 10 ** ((magz - cutmag) / 2.5) * self.scale**2.0
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
        out = out[(out["fpfs_M00"] + out["fpfs_M20"]) > 0.0]
        print("final number of sources: %d" % len(out))
        return out

    def run(self, fname):
        out_fname = os.path.join(self.catdir, fname.split("/")[-1])
        out_fname = out_fname.replace("image-", "src-").replace("_g.fits", ".fits")
        if os.path.exists(out_fname):
            print("Already has measurement for this simulation. ")
            return
        psf_array2, psf_array3, cov_elem = self.prepare_noise_psf(fname)
        gal_array = self.prepare_image(fname)
        cat = self.process_image(gal_array, psf_array2, psf_array3, cov_elem)
        pyfits.writeto(out_fname, cat)
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
    fname_list = glob.glob(os.path.join(worker.imgdir, "image-*_g.fits"))
    for r in pool.map(worker.run, fname_list):
        pass
    pool.close()
