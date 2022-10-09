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
from fpfs.default import *
import astropy.io.fits as pyfits
from argparse import ArgumentParser
from configparser import ConfigParser

class Worker(object):
    def __init__(self,config_name,gver='g1'):
        cparser     =   ConfigParser()
        cparser.read(config_name)
        # setup processor
        self.catdir =   cparser.get('procsim', 'cat_dir')
        self.simname=   cparser.get('procsim', 'sim_name')
        proc_name   =   cparser.get('procsim', 'proc_name')
        self.do_noirev= cparser.getboolean('FPFS', 'do_noirev')
        self.rcut   =   cparser.getint('FPFS', 'rcut')

        self.selnm  =   []
        self.cutsig =   []
        self.cut    =   []
        self.do_detcut =   cparser.getboolean('FPFS', 'do_detcut')
        if self.do_detcut:
            self.selnm.append('detect2')
            self.cutsig.append(sigP)
            self.cut.append(cutP)
        self.do_magcut= cparser.getboolean('FPFS', 'do_magcut')
        if self.do_magcut:
            self.selnm.append('M00')
            self.cutsig.append(sigM)
            self.cut.append(10**((27.-cutM)/2.5))
        assert len(self.selnm)>=1, "Must do at least one selection."
        self.selnm  =   np.array(self.selnm)
        self.cutsig =   np.array(self.cutsig)
        self.cut    =   np.array(self.cut)
        self.test_name= cparser.get('FPFS', 'test_name')
        assert self.test_name in self.selnm
        self.test_ind=  np.where(self.selnm==self.test_name)[0]

        self.indir  =   os.path.join(self.catdir,'src_%s_%s'%(self.simname,proc_name))
        if not os.path.exists(self.indir):
            raise FileNotFoundError('Cannot find input directory!')
        print('The input directory for galaxy shear catalogs is %s. ' %self.indir)
        # setup WL distortion parameter
        self.gver   =   gver
        self.Const  =   cparser.getfloat('FPFS', 'weighting_c')
        return

    def run(self,Id):
        pp    = 'cut%d' %self.rcut
        in_nm1= os.path.join(self.indir,'fpfs-%s-%04d-%s-0000.fits' %(pp,Id,self.gver))
        in_nm2= os.path.join(self.indir,'fpfs-%s-%04d-%s-2222.fits' %(pp,Id,self.gver))
        assert os.path.isfile(in_nm1) & os.path.isfile(in_nm2), 'Cannot find\
                input galaxy shear catalog distorted by positive and negative shear'
        mm1   = pyfits.getdata(in_nm1)
        mm2   = pyfits.getdata(in_nm2)
        ellM1 = fpfs.catalog.fpfsM2E(mm1,const=self.Const,noirev=self.do_noirev)
        ellM2 = fpfs.catalog.fpfsM2E(mm2,const=self.Const,noirev=self.do_noirev)

        fs1 =   fpfs.catalog.summary_stats(mm1,ellM1,use_sig=False,ratio=1.)
        fs2 =   fpfs.catalog.summary_stats(mm2,ellM2,use_sig=False,ratio=1.)
        if self.test_name=='M00':
            ncut=   6
            dcc =   -0.6
            cutB=   27.5
        else:
            raise ValueError('only support mag cut')

        #names= [('cut','<f8'), ('de','<f8'), ('eA1','<f8'), ('eA2','<f8'), ('res1','<f8'), ('res2','<f8')]
        out =   np.zeros((6,ncut))
        for i in range(ncut):
            fs1.clear_outcomes()
            fs2.clear_outcomes()
            icut=cutB+dcc*i
            if self.test_name=='M00':
                self.cut[self.test_ind]=10**((27.-icut)/2.5)
            fs1.update_selection_weight(self.selnm,self.cut,self.cutsig)
            fs2.update_selection_weight(self.selnm,self.cut,self.cutsig)
            fs1.update_selection_bias(self.selnm,self.cut,self.cutsig)
            fs2.update_selection_bias(self.selnm,self.cut,self.cutsig)
            fs1.update_ellsum();fs2.update_ellsum()
            out[0,i]= icut
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

    cparser =   ConfigParser()
    cparser.read(args.config)
    glist=[]
    if cparser.getboolean('distortion','test_g1'):
        glist.append('g1')
    if cparser.getboolean('distortion','test_g2'):
        glist.append('g2')
    if len(glist)<1:
        raise ValueError('Cannot test nothing!! Must test g1 or test g2. ')
    shear_value =   cparser.getfloat('distortion','shear_value')

    for gver in glist:
        print('Testing for %s . ' %gver)
        worker  =   Worker(args.config,gver=gver)
        refs    =   list(range(args.minId,args.maxId))
        outs    =   []
        for r in pool.map(worker,refs):
            outs.append(r)
        outs    =   np.vstack(outs)
        if len(outs.shape)==3:
            res     =   np.average(outs,axis=0)
            err     =   np.std(outs,axis=0)
        else:
            res     =   outs
            err     =   np.zeros_like(res)
        mbias   =   (res[1]/res[5]/2.-shear_value)/shear_value
        merr    =   (err[1]/res[5]/2.)/shear_value
        cbias   =   res[2]/res[5]
        cerr    =   err[2]/res[5]
        df      =   pd.DataFrame({
            'simname': eval(worker.simname.split('Center')[-1]),
            'binave': res[0],
            'mbias': mbias,
            'merr': merr,
            'cbias': cbias,
            'cerr': cerr,
            })
        summary_base_fname='summary_output'
        os.makedirs(summary_base_fname,exist_ok=True)
        df.to_csv(os.path.join(summary_base_fname,'shear_%s.csv' %worker.simname),index=False)

        print('Separate galaxies into %d bins: %s'  %(len(res[0]),res[0]))
        print('Multiplicative biases for those bins are: ', mbias)
        print(merr)
        print(cbias)
        print(cerr)
        del worker
    pool.close()
