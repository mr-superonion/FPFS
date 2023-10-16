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

import schwimmbad
from argparse import ArgumentParser
from configparser import ConfigParser, ExtendedInterpolation
from fpfs.tasks import ProcessSimulationTask


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs measurement")
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
    worker = ProcessSimulationTask(args.config)
    refs = list(range(args.min_id, args.max_id))
    for r in pool.map(worker.run, refs):
        pass
    pool.close()
