#!/usr/bin/env python
# Copyright 20220312 Xiangchong Li.
import os
import sys
import fpfs
import fitsio
import argparse
import numpy as np
from schwimmbad import MPIPool
from default import *

def do_process(ref):
    noirev =True
    use_sig=False
    Const=  20.
    ver  =  'try2'
    gver =  'basic3'
    dver =  'cut32'
    nver =  'var7em3'
    wrkDir= os.environ['homeWrk']
    simDir= os.path.join(wrkDir,'FPFS2/sim/')
    # read noiseless data
    mm1  =  fitsio.read(os.path.join(simDir,'srcfs3_%sCenter-%s_%s/psf60/fpfs-%s-%04d-g1-0000.fits' %(gver,nver,ver,dver,ref)))
    mm2  =  fitsio.read(os.path.join(simDir,'srcfs3_%sCenter-%s_%s/psf60/fpfs-%s-%04d-g1-2222.fits' %(gver,nver,ver,dver,ref)))

    ellM1  =fpfs.catalog.fpfsM2E(mm1,const=Const,noirev=noirev)
    ellM2  =fpfs.catalog.fpfsM2E(mm2,const=Const,noirev=noirev)

    fs1=fpfs.catalog.summary_stats(mm1,ellM1,use_sig)
    fs2=fpfs.catalog.summary_stats(mm2,ellM2,use_sig)
    selnm=['detect','R2','M00']
    dcc=0.1
    cutB=-0.2
    cutsig=[sigP,sigR,sigM]
    ncut=8

    #names= [('cut','<f8'), ('de','<f8'), ('eA1','<f8'), ('eA2','<f8'), ('res1','<f8'), ('res2','<f8')]
    out=np.zeros((6,ncut))
    for i in range(ncut):
        # clean outcome
        fs1.clear_outcomes()
        fs2.clear_outcomes()
        rcut=cutB+dcc*i
        cut=[cutP,rcut,10**((27.-cutM)/2.5)]
        # weight array
        fs1.update_selection_weight(selnm,cut,cutsig);fs2.update_selection_weight(selnm,cut,cutsig)
        fs1.update_selection_bias(selnm,cut,cutsig);fs2.update_selection_bias(selnm,cut,cutsig)
        fs1.update_ellsum();fs2.update_ellsum()
        out[0,i]= rcut
        out[1,i]= fs2.sumE1-fs1.sumE1
        out[2,i]= (fs1.sumE1+fs2.sumE1)/2.
        out[3,i]= (fs1.sumE1+fs2.sumE1+fs1.corE1+fs2.corE1)/2.
        out[4,i]= (fs1.sumR1+fs2.sumR1)/2.
        out[5,i]= (fs1.sumR1+fs2.sumR1+fs1.corR1+fs2.corR1)/2.
    return out

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--minId', required=True,type=int,
                        help='minimum id number, e.g. 0')
    parser.add_argument('--maxId', required=True,type=int,
                        help='maximum id number, e.g. 1024')
    args = parser.parse_args()
    refs    =   list(range(args.minId,args.maxId))
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    results=pool.map(do_process,refs)
    out =   np.stack(results)
    print(out.shape)
    fitsio.write('detect_r2cut.fits',out)
    pool.close()
    return

if __name__=='__main__':
    main()
