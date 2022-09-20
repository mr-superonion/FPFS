#!/usr/bin/env python
# LSST Data Management System
# Copyright 20082014 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
# Copyright 20220312 Xiangchong Li.
import gc
import sys
import fpfs
from fpfs.default import *

class Worker(object):
    def __init__(self,config_name):
        p1List=['g1']
        # p1List=['g1','g2']
        p2List=['0000','2222']
        self.expDir=
        # expDir  =   "small0_psf60"
        # expDir  =   "galaxy_basic_psf60"
        # expDir  =   "galaxy_basic3Center_psf60"
        self.expDir  =   "galaxy_basic3Shift_psf60"
        # expDir  =   "galaxy_unif3_cosmo170_psf60"
        # expDir  =   "galaxy_unif3_cosmo085_psf60"
        self.pendList=['%s-%s' %(i1,i2) for i1 in p1List for i2 in p2List]
        return

    def run(self,Id):
        for pp in self.pendList:
            fpfs.simutil.make_basic_sim(self.expDir,pp,Id)
            gc.collect()
        print('finish ID: %d' %(Id))
        return 0

    def __call__(self,Id):
        return self.run(Id)

def main(pool,minId,maxId,config_name):
    worker=Worker(config_name)
    refs = list(range(minId,maxId))
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    pool.map(worker,refs)
    pool.close()
    return

if __name__=='__main__':
    import schwimmbad
    from argparse import ArgumentParser

    parser = ArgumentParser(description="fpfs simulation")
    parser.add_argument('--minId', required=True,type=int,
                        help='minimum id number, e.g. 0')
    parser.add_argument('--maxId', required=True,type=int,
                        help='maximum id number, e.g. 4000')
    parser.add_argument('--config', required=True,type=str,
                        help='configure file name')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    main(pool,args.minId,args.maxId,args.config)
