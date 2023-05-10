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
import fpfs
import schwimmbad
import numpy as np
import pandas as pd
from fpfs.default import (
    sigP,
    sigM,
    sigR,
    cutP,
    cutM,
    cutR,
)
import astropy.io.fits as pyfits
from argparse import ArgumentParser
from configparser import ConfigParser


class Worker(object):
    def __init__(self, config_name, gver="g1"):
        cparser = ConfigParser()
        cparser.read(config_name)
        # survey parameter
        self.magz = cparser.getfloat("survey", "mag_zero")

        # setup processor
        self.catdir = cparser.get("procsim", "cat_dir")
        self.simname = cparser.get("procsim", "sim_name")
        proc_name = cparser.get("procsim", "proc_name")
        self.do_noirev = cparser.getboolean("FPFS", "do_noirev")
        self.rcut = cparser.getint("FPFS", "rcut")

        self.selnm = []
        self.cutsig = []
        self.cut = []
        self.do_detcut = cparser.getboolean("FPFS", "do_detcut")  # detection cut
        if self.do_detcut:
            self.selnm.append("detect2")
            self.cutsig.append(sigP)
            self.cut.append(cutP)
        self.do_magcut = cparser.getboolean("FPFS", "do_magcut")  # magnitude cut
        if self.do_magcut:
            self.selnm.append("M00")
            self.cutsig.append(sigM)
            self.cut.append(10 ** ((self.magz - cutM) / 2.5))
        self.do_magcut = cparser.getboolean("FPFS", "do_rcut")  # resolution cut
        if self.do_magcut:
            self.selnm.append("R2")
            self.cutsig.append(sigR)
            self.cut.append(cutR)
        assert len(self.selnm) >= 1, "Must do at least one selection."
        self.selnm = np.array(self.selnm)
        self.cutsig = np.array(self.cutsig)
        self.cut = np.array(self.cut)

        # This task change the cut on one observable and see how the biases changes.
        # Here is  the observable used for test
        self.test_name = cparser.get("FPFS", "test_name")
        assert self.test_name in self.selnm
        self.test_ind = np.where(self.selnm == self.test_name)[0]
        self.cutB = cparser.getfloat("FPFS", "cutB")
        self.dcut = cparser.getfloat("FPFS", "dcut")
        self.ncut = cparser.getint("FPFS", "ncut")

        self.indir = os.path.join(self.catdir, "src_%s_%s" % (self.simname, proc_name))
        if not os.path.exists(self.indir):
            raise FileNotFoundError("Cannot find input directory: %s!" % self.indir)
        print("The input directory for galaxy shear catalogs is %s. " % self.indir)
        # setup WL distortion parameter
        self.gver = gver
        self.Const = cparser.getfloat("FPFS", "weighting_c")
        return

    def run(self, Id):
        pp = "cut%d" % self.rcut
        in_nm1 = os.path.join(
            self.indir, "fpfs-%s-%04d-%s-0000.fits" % (pp, Id, self.gver)
        )
        in_nm2 = os.path.join(
            self.indir, "fpfs-%s-%04d-%s-2222.fits" % (pp, Id, self.gver)
        )
        assert os.path.isfile(in_nm1) & os.path.isfile(in_nm2), (
            "Cannot find\
                input galaxy shear catalog distorted by positive and negative shear\
                : %s , %s"
            % (in_nm1, in_nm2)
        )
        mm1 = pyfits.getdata(in_nm1)
        mm2 = pyfits.getdata(in_nm2)
        ell1 = fpfs.catalog.fpfs_m2e(mm1, const=self.Const, noirev=self.do_noirev)
        ell2 = fpfs.catalog.fpfs_m2e(mm2, const=self.Const, noirev=self.do_noirev)

        fs1 = fpfs.catalog.summary_stats(mm1, ell1, use_sig=False, ratio=1.0)
        fs2 = fpfs.catalog.summary_stats(mm2, ell2, use_sig=False, ratio=1.0)

        # names= [('cut','<f8'), ('de','<f8'), ('eA1','<f8'), ('eA2','<f8'),
        # ('res1','<f8'), ('res2','<f8')]
        out = np.zeros((6, self.ncut))
        for i in range(self.ncut):
            fs1.clear_outcomes()
            fs2.clear_outcomes()
            icut = self.cutB + self.dcut * i
            if self.test_name == "M00":
                self.cut[self.test_ind] = 10 ** ((self.magz - icut) / 2.5)
            else:
                self.cut[self.test_ind] = icut
            fs1.update_selection_weight(self.selnm, self.cut, self.cutsig)
            fs2.update_selection_weight(self.selnm, self.cut, self.cutsig)
            fs1.update_selection_bias(self.selnm, self.cut, self.cutsig)
            fs2.update_selection_bias(self.selnm, self.cut, self.cutsig)
            fs1.update_ellsum()
            fs2.update_ellsum()
            out[0, i] = icut
            out[1, i] = fs2.sumE1 - fs1.sumE1
            out[2, i] = (fs1.sumE1 + fs2.sumE1) / 2.0
            out[3, i] = (fs1.sumE1 + fs2.sumE1 + fs1.corE1 + fs2.corE1) / 2.0
            out[4, i] = (fs1.sumR1 + fs2.sumR1) / 2.0
            out[5, i] = (fs1.sumR1 + fs2.sumR1 + fs1.corR1 + fs2.corR1) / 2.0
        return out

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

    cparser = ConfigParser()
    cparser.read(args.config)
    glist = []
    if cparser.getboolean("distortion", "test_g1"):
        glist.append("g1")
    if cparser.getboolean("distortion", "test_g2"):
        glist.append("g2")
    if len(glist) < 1:
        raise ValueError("Cannot test nothing!! Must test g1 or test g2. ")
    shear_value = cparser.getfloat("distortion", "shear_value")

    for gver in glist:
        print("Testing for %s . " % gver)
        worker = Worker(args.config, gver=gver)
        refs = list(range(args.minId, args.maxId))
        outs = []
        for r in pool.map(worker, refs):
            outs.append(r)
        outs = np.stack(outs)
        nsims = outs.shape[0]
        summary_dirname = "summary_output"
        os.makedirs(summary_dirname, exist_ok=True)
        pyfits.writeto(
            os.path.join(
                summary_dirname,
                "bin_%s_sim_%s.fits" % (worker.test_name, worker.simname),
            ),
            outs,
            overwrite=True,
        )

        res = np.average(outs, axis=0)
        err = np.std(outs, axis=0)
        mbias = (res[1] / res[5] / 2.0 - shear_value) / shear_value
        merr = (err[1] / res[5] / 2.0) / shear_value / np.sqrt(nsims)
        cbias = res[3] / res[5]
        cerr = err[3] / res[5] / np.sqrt(nsims)
        df = pd.DataFrame(
            {
                "simname": worker.simname.split("galaxy_")[-1],
                "binave": res[0],
                "mbias": mbias,
                "merr": merr,
                "cbias": cbias,
                "cerr": cerr,
            }
        )
        df.to_csv(
            os.path.join(
                summary_dirname,
                "bin_%s_sim_%s.csv" % (worker.test_name, worker.simname),
            ),
            index=False,
        )

        print("Separate galaxies into %d bins: %s" % (len(res[0]), res[0]))
        print("Multiplicative biases for those bins are: ", mbias)
        print("Errors are: ", merr)
        print("Additive biases for those bins are: ", cbias)
        print("Errors are: ", cerr)
        del worker
    pool.close()
