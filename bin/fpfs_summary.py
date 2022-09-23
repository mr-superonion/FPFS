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
import astropy.io.fits as pyfits
from argparse import ArgumentParser
from fpfs.default import *
from configparser import ConfigParser

class Worker(object):
    def __init__(self,config_name):
        cparser     =   ConfigParser()
        cparser.read(config_name)
        # setup processor
        self.catdir =   cparser.get('procsim', 'cat_dir')
        self.simname=   cparser.get('procsim', 'sim_name')
        proc_name   =   cparser.get('procsim', 'proc_name')
        self.do_det =   cparser.getboolean('FPFS', 'do_det')
        self.do_noirev= cparser.getboolean('FPFS', 'do_noirev')
        self.rcut= cparser.getint('FPFS', 'rcut')
        self.indir =   os.path.join(self.catdir,'src_%s_%s'%(self.simname,proc_name))
        if not os.path.exists(self.indir):
            raise FileNotFoundError('Cannot find input directory!')
        # setup WL distortion parameter
        self.gver='g1'
        self.Const=20.
        return

    def run(self,Id):
        pp    = 'cut%d' %self.rcut
        mm1   = pyfits.getdata(os.path.join(self.indir,'fpfs-%s-%04d-%s-0000.fits' %(pp,Id,self.gver)))
        mm2   = pyfits.getdata(os.path.join(self.indir,'fpfs-%s-%04d-%s-2222.fits' %(pp,Id,self.gver)))
        ellM1 = fpfs.catalog.fpfsM2E(mm1,const=self.Const,noirev=self.do_noirev)
        ellM2 = fpfs.catalog.fpfsM2E(mm2,const=self.Const,noirev=self.do_noirev)

        fs1 =   fpfs.catalog.summary_stats(mm1,ellM1,use_sig=False,ratio=1.)
        fs2 =   fpfs.catalog.summary_stats(mm2,ellM2,use_sig=False,ratio=1.)
        # selnm=  ['M00']
        selnm=  []
        dcc =   -0.6
        cutB=   29.5
        #cutsig= [sigM]
        cutsig= []
        ncut=   7

        #names= [('cut','<f8'), ('de','<f8'), ('eA1','<f8'), ('eA2','<f8'), ('res1','<f8'), ('res2','<f8')]
        out=np.zeros((6,ncut))
        for i in range(ncut):
            fs1.clear_outcomes()
            fs2.clear_outcomes()
            mcut=cutB+dcc*i
            # cut=[10**((27.-mcut)/2.5)]
            cut=[]
            # weight array
            fs1.update_selection_weight(selnm,cut,cutsig);fs2.update_selection_weight(selnm,cut,cutsig)
            fs1.update_selection_bias(selnm,cut,cutsig);fs2.update_selection_bias(selnm,cut,cutsig)
            fs1.update_ellsum();fs2.update_ellsum()
            out[0,i]= mcut
            out[1,i]= fs2.sumE1-fs1.sumE1
            out[2,i]= (fs1.sumE1+fs2.sumE1)/2.
            out[3,i]= (fs1.sumE1+fs2.sumE1+fs1.corE1+fs2.corE1)/2.
            out[4,i]= (fs1.sumR1+fs2.sumR1)/2.
            out[5,i]= (fs1.sumR1+fs2.sumR1+fs1.corR1+fs2.corR1)/2.
        return out

    def __call__(self,Id):
        print('start ID: %d' %(Id))
        return self.run(Id)

if __name__=='__main__':
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument('--minId', required=True,type=int,
                        help='minimum id number, e.g. 0')
    parser.add_argument('--maxId', required=True,type=int,
                        help='maximum id number, e.g. 4000')
    parser.add_argument('--config', required=True,type=str,
                        help='configure file name')
    group   =   parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args    =   parser.parse_args()
    pool    =   schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)

    worker  =   Worker(args.config)
    refs    =   list(range(args.minId,args.maxId))
    outs    =   []
    for r in pool.map(worker,refs):
        outs.append(r)
    print(np.vstack(outs).shape)
    outs=np.vstack(outs)
    if len(outs.shape)==3:
        outs    =   np.sum(outs,axis=0)

    print((outs[1]/outs[5]/2.-0.02)/0.02)
    pool.close()
