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
import jax
import impt
import fitsio
import schwimmbad
import numpy as np

from argparse import ArgumentParser
from configparser import ConfigParser


def get_processor_count(pool, args):
    if isinstance(pool, schwimmbad.MPIPool):
        # MPIPool
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_size() - 1
    elif isinstance(pool, schwimmbad.MultiPool):
        # MultiPool
        return args.n_cores
    else:
        # SerialPool
        return 1


class Worker(object):
    def __init__(
        self,
        config_name,
        gver="g1",
        magcut=27.0,
        min_id=0,
        max_id=1000,
        ncores=1,
    ):
        cparser = ConfigParser()
        cparser.read(config_name)
        # survey parameter
        self.magz = cparser.getfloat("survey", "mag_zero")

        nids = max_id - min_id
        self.n_per_c = nids // ncores
        self.mid = nids % ncores
        self.min_id = min_id
        self.max_id = max_id
        self.rest_list = list(np.arange(ncores * self.n_per_c, nids) + min_id)
        print("number of files per core is: %d" % self.n_per_c)

        # setup processor
        self.catdir = cparser.get("procsim", "cat_dir")
        self.sum_dir = cparser.get("procsim", "sum_dir")
        self.do_noirev = cparser.getboolean("FPFS", "do_noirev")
        if self.do_noirev:
            ncov_fname = os.path.join(self.catdir, "cov_matrix.fits")
            self.cov_mat = fitsio.read(ncov_fname)
        else:
            self.cov_mat = np.zeros((31, 31))
        self.shear_value = cparser.getfloat("distortion", "shear_value")
        # survey parameter
        self.magz = cparser.getfloat("survey", "mag_zero")
        # This task change the cut on one observable and see how the biases
        # changes.
        # Here is  the observable used for test
        self.upper_mag = magcut
        self.lower_m00 = 10 ** ((self.magz - self.upper_mag) / 2.5)
        # setup WL distortion parameter
        self.gver = gver
        self.ofname = os.path.join(
            self.sum_dir,
            "bin_%s.fits" % (self.upper_mag),
        )
        return

    def prepare_functions(self):
        params = impt.fpfs.FpfsParams(
            Const=10.0,
            lower_m00=self.lower_m00,
            lower_r2=0.03,
            upper_r2=2.0,
            lower_v=0.10,
            sigma_m00=0.4,
            sigma_r2=0.6,
            sigma_v=0.15,
        )
        funcnm = "ss2"
        e1 = impt.fpfs.FpfsWeightE1(params, func_name=funcnm)
        enoise = impt.BiasNoise(e1, self.cov_mat)
        res1 = impt.RespG1(e1)
        rnoise = impt.BiasNoise(res1, self.cov_mat)
        return e1, enoise, res1, rnoise

    def get_sum_e_r(self, in_nm, e1, enoise, res1, rnoise):
        assert os.path.isfile(
            in_nm
        ), "Cannot find input galaxy shear catalogs : %s " % (in_nm)
        mm = impt.fpfs.read_catalog(in_nm)
        # noise bias

        def fune(carry, ss):
            y = e1._obs_func(ss) - enoise._obs_func(ss)
            return carry + y, y

        def funr(carry, ss):
            y = res1._obs_func(ss) - rnoise._obs_func(ss)
            return carry + y, y

        e1_sum, _ = jax.lax.scan(fune, 0.0, mm)
        r1_sum, _ = jax.lax.scan(funr, 0.0, mm)
        del mm
        gc.collect()
        return e1_sum, r1_sum

    def get_range(self, icore):
        ibeg = self.min_id + icore * self.n_per_c
        iend = min(ibeg + self.n_per_c, self.max_id)
        id_range = list(range(ibeg, iend))
        if icore < len(self.rest_list):
            id_range.append(self.rest_list[icore])
        return id_range

    def run(self, icore):
        id_range = self.get_range(icore)
        out = np.empty((len(id_range), 3))
        print("start core: %d, with id: %s" % (icore, id_range))
        for icount, ifield in enumerate(id_range):
            e1, enoise, res1, rnoise = self.prepare_functions()
            in_nm1 = os.path.join(
                self.catdir,
                "src_%05d-%s_01-rot_0.fits" % (ifield, self.gver),
            )
            e1_1, r1_1 = self.get_sum_e_r(in_nm1, e1, enoise, res1, rnoise)

            in_nm2 = os.path.join(
                self.catdir,
                "src_%05d-%s_00-rot_0.fits" % (ifield, self.gver),
            )
            e1_2, r1_2 = self.get_sum_e_r(in_nm2, e1, enoise, res1, rnoise)
            out[icount, 0] = e1_1 - e1_2
            out[icount, 1] = (e1_1 + e1_2) / 2.0
            out[icount, 2] = (r1_1 + r1_2) / 2.0
            del e1, enoise, res1, rnoise
            jax.clear_backends()
            gc.collect()
        return out


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument(
        "--runid",
        default=0,
        type=int,
        help="id number, e.g. 0",
    )
    parser.add_argument(
        "--min_id",
        default=0,
        type=int,
        help="id number, e.g. 0",
    )
    parser.add_argument(
        "--max_id",
        default=1000,
        type=int,
        help="id number, e.g. 1000",
    )
    parser.add_argument(
        "--magcut",
        default=27.0,
        type=float,
        help="id number, e.g. 0",
    )
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
    cparser = ConfigParser()
    cparser.read(args.config)
    shear_value = cparser.getfloat("distortion", "shear_value")

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    ncores = get_processor_count(pool, args)
    assert isinstance(ncores, int)
    core_list = np.arange(ncores)
    worker = Worker(
        args.config,
        gver="g1",
        magcut=args.magcut,
        min_id=args.min_id,
        max_id=args.max_id,
        ncores=ncores,
    )
    summary_dirname = worker.sum_dir
    os.makedirs(summary_dirname, exist_ok=True)

    olist = pool.map(worker.run, core_list)
    pool.close()
    outcome = np.vstack(olist)
    fitsio.write(worker.ofname, outcome)
