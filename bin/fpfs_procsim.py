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
import fpfs
import json
import galsim
import schwimmbad
import numpy as np
import astropy.io.fits as pyfits
from argparse import ArgumentParser
from configparser import ConfigParser
import numpy.lib.recfunctions as rfn

_DefaultImgSize = 6400


class Worker(object):
    def __init__(self, config_name):
        cparser = ConfigParser()
        cparser.read(config_name)
        # setup processor
        self.imgdir = cparser.get("procsim", "img_dir")
        self.catdir = cparser.get("procsim", "cat_dir")
        self.simname = cparser.get("procsim", "sim_name")
        proc_name = cparser.get("procsim", "proc_name")
        self.do_det = cparser.getboolean("FPFS", "do_det")
        self.sigma_as = cparser.getfloat("FPFS", "sigma_as")
        self.rcut = cparser.getint("FPFS", "rcut")
        self.indir = os.path.join(self.imgdir, self.simname)
        if not os.path.exists(self.indir):
            raise FileNotFoundError("Cannot find input images directory!")
        self.outdir = os.path.join(self.catdir, "src_%s_%s" % (self.simname, proc_name))
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)
        print("The input directory for galaxy images is %s. " % self.indir)
        print("The output directory for shear catalogs is %s. " % self.outdir)

        # setup survey parameters
        self.scale = cparser.getfloat("survey", "pixel_scale")
        self.psffname = os.path.join(self.indir, cparser.get("survey", "psf_filename"))
        if not os.path.isfile(self.psffname):
            raise FileNotFoundError("Cannot find the PSF file: %s" % self.psffname)
        self.noi_var = cparser.getfloat("survey", "noi_var")
        if self.noi_var > 1e-20:
            self.noidir = os.path.join(self.imgdir, "noise")
            if not os.path.exists(self.noidir):
                raise FileNotFoundError("Cannot find input noise directory!")
            self.noiPfname = cparser.get("survey", "noiP_filename")
            print("The input directory for noise images is %s. " % self.noidir)
        else:
            self.noidir = None
            self.noiPfname = None
        # size of the image
        self.image_nx = cparser.getint("survey", "image_nx")
        self.image_ny = cparser.getint("survey", "image_ny")
        self.magz = cparser.getfloat("survey", "mag_zero")
        assert self.image_ny == self.image_nx, "image_nx must equals image_ny!"
        assert self.image_ny <= _DefaultImgSize, (
            "image_nx (image_ny) should \
                    not be greater than %d !"
            % _DefaultImgSize
        )

        # setup WL distortion parameter
        glist = []
        # this is for const shear tests
        if cparser.getboolean("distortion", "test_g1"):
            glist.append("g1")
        if cparser.getboolean("distortion", "test_g2"):
            glist.append("g2")
        if len(glist) > 0:
            zlist = json.loads(cparser.get("distortion", "shear_z_list"))
            self.pendList = ["%s-%s" % (i1, i2) for i1 in glist for i2 in zlist]
        else:
            # this is for non-distorted image simulation
            self.pendList = ["g1-1111"]
        return

    def prepare_PSF(self, psffname, rcut, ngrid2):
        ngrid = 64
        beg = ngrid // 2 - rcut
        end = beg + 2 * rcut
        psfData = pyfits.getdata(psffname)
        npad = (ngrid - psfData.shape[0]) // 2
        psfData2 = np.pad(psfData, (npad + 1, npad), mode="constant")[beg:end, beg:end]
        del npad
        npad = (ngrid2 - psfData.shape[0]) // 2
        psfData3 = np.pad(psfData, (npad + 1, npad), mode="constant")
        return psfData2, psfData3

    def run(self, Id):
        print("running for galaxy field: %d" % (Id))
        if "cosmo" in self.simname:
            print("cosmos parametric galaxies for blended galaxy image simulation. ")
        elif "basic" in self.simname:
            print("cosmos parametric galaxies for isolate galaxy image simulation. ")
        elif "small" in self.simname:
            print("Random walk galaxies for isolate galaxy image simulation. ")
        else:
            raise ValueError("simname not supported. ")

        nbegin = (_DefaultImgSize - self.image_nx) // 2
        nend = nbegin + self.image_nx
        gal_dir = os.path.join(self.imgdir, self.simname)
        # PSF
        if "%" in self.psffname:
            psffname = self.psffname % Id
        else:
            psffname = self.psffname

        psfInt = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0)
        psfInt = psfInt.shear(e1=0.02, e2=-0.02).shift(
            0.5 * self.scale, 0.5 * self.scale
        )
        psfData2 = psfInt.drawImage(nx=64, ny=64, scale=self.scale).array
        self.ppt = psfData2
        psfData2, psfData3 = self.prepare_PSF(psffname, self.rcut, self.image_nx)

        # FPFS Task
        if self.noi_var > 1e-20:
            # noise
            print("Using noisy setup with variance: %.3f" % self.noi_var)
            assert self.noidir is not None
            noiFname = os.path.join(self.noidir, "noi%04d.fits" % Id)
            if not os.path.isfile(noiFname):
                print("Cannot find input noise file: %s" % noiFname)
                return
            # multiply by 10 since the noise has variance 0.01
            noiData = (
                pyfits.getdata(noiFname)[nbegin:nend, nbegin:nend]
                * 10.0
                * np.sqrt(self.noi_var)
            )
            # Also times 100 for the noivar model
            powIn = (
                np.load(self.noiPfname, allow_pickle=True).item()["%s" % self.rcut]
                * self.noi_var
                * 100
            )
            powModel = np.zeros((1, powIn.shape[0], powIn.shape[1]))
            powModel[0] = powIn
            measTask = fpfs.image.measure_source(
                psfData2, sigma_arcsec=self.sigma_as, noiFit=powModel[0]
            )
        else:
            print("Using noiseless setup")
            # by default noiFit=None
            measTask = fpfs.image.measure_source(psfData2, sigma_arcsec=self.sigma_as)
            noiData = 0.0
        print("The upper limit of wave number is %s pixels" % (measTask.klim_pix))

        for ishear in self.pendList:
            print("FPFS measurement on simulation: %04d, %s" % (Id, ishear))
            galFname = os.path.join(gal_dir, "image-%s-%s.fits" % (Id, ishear))
            if not os.path.isfile(galFname):
                print("Cannot find input galaxy file: %s" % galFname)
                return
            galData = pyfits.getdata(galFname) + noiData
            assert galData.shape == (self.image_ny, self.image_nx)
            outFname = os.path.join(self.outdir, "src-%04d-%s.fits" % (Id, ishear))
            pp = "cut%d" % self.rcut
            outFname = os.path.join(
                self.outdir, "fpfs-%s-%04d-%s.fits" % (pp, Id, ishear)
            )
            if os.path.exists(outFname):
                print("Already has measurement for this simulation. ")
                continue
            if not self.do_det:
                if "Center" in gal_dir:
                    # fake detection
                    indX = np.arange(32, self.image_nx, 64)
                    indY = np.arange(32, self.image_ny, 64)
                    inds = np.meshgrid(indY, indX, indexing="ij")
                    coords = np.array(
                        np.zeros(inds[0].size),
                        dtype=[("fpfs_y", "i4"), ("fpfs_x", "i4")],
                    )
                    coords["fpfs_y"] = np.ravel(inds[0])
                    coords["fpfs_x"] = np.ravel(inds[1])
                    del indX, indY, inds
                else:
                    raise ValueError(
                        "Do not support the case without detection on galaxies with center offsets. "
                    )
            else:
                if self.sigma_as < 0.5:
                    cutmag = 25.5
                else:
                    cutmag = 26.0
                thres = 10 ** ((self.magz - cutmag) / 2.5) * self.scale**2.0
                thres2 = -thres / 20.0
                coords = fpfs.image.detect_sources(
                    galData,
                    psfData3,
                    gsigma=measTask.sigmaF,
                    thres=thres,
                    thres2=thres2,
                    klim=measTask.klim,
                )
            # # for test
            # coords  =   np.array(np.zeros(1),dtype=[('fpfs_y','i4'),('fpfs_x','i4')])
            # coords['fpfs_y'][0]=   self.image_ny//2
            # coords['fpfs_x'][0]=   self.image_nx//2
            print("pre-selected number of sources: %d" % len(coords))
            imgList = [
                galData[
                    cc["fpfs_y"] - self.rcut : cc["fpfs_y"] + self.rcut,
                    cc["fpfs_x"] - self.rcut : cc["fpfs_x"] + self.rcut,
                ]
                for cc in coords
            ]
            out = measTask.measure(imgList)
            out = rfn.merge_arrays([coords, out], flatten=True, usemask=False)
            pyfits.writeto(outFname, out)
            del imgList, out, coords, galData, outFname
            gc.collect()
        print("finish %s" % (Id))
        return

    def __call__(self, Id):
        print("start ID: %d" % (Id))
        return self.run(Id)


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument(
        "--minId", required=True, type=int, help="minimum id number, e.g. 0"
    )
    parser.add_argument(
        "--maxId", required=True, type=int, help="maximum id number, e.g. 4000"
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
    refs = list(range(args.minId, args.maxId))
    for r in pool.map(worker, refs):
        pass
    pool.close()
