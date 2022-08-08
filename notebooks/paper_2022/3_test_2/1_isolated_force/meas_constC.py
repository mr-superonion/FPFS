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
    ver  =  'try2'
    gver =  'basic3'
    dver =  'cut32'
    nver =  'var7em3'
    wrkDir= os.environ['homeWrk']
    simDir= os.path.join(wrkDir,'FPFS2/sim/')
    # read noiseless data
    mm1  =  fitsio.read(os.path.join(simDir,'srcfs3_%sCenter-%s_%s/psf60/fpfs-%s-%04d-g1-0000.fits' %(gver,nver,ver,dver,ref)))
    mm2  =  fitsio.read(os.path.join(simDir,'srcfs3_%sCenter-%s_%s/psf60/fpfs-%s-%04d-g1-2222.fits' %(gver,nver,ver,dver,ref)))
    ntest=  20
    clist=  np.logspace(-1.5,1.5,ntest)
    out  =  np.zeros((4,ntest))
    for i in range(ntest):
        Const=  clist[i]
        ellM1  =    fpfs.catalog.fpfsM2E(mm1,const=Const,noirev=noirev)
        ellM2  =    fpfs.catalog.fpfsM2E(mm2,const=Const,noirev=noirev)
        e1sum1 =    np.sum(ellM1['fpfs_e1'])
        e1sum2 =    np.sum(ellM2['fpfs_e1'])
        R1sum1 =    np.sum(ellM1['fpfs_R1E'])
        R1sum2 =    np.sum(ellM2['fpfs_R1E'])
        out[0,i]=   Const
        out[1,i]=   e1sum2-e1sum1
        out[2,i]=   (e1sum1+e1sum2)/2.
        out[3,i]=   (R1sum1+R1sum2)/2.
    return out

def main():
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--minId', required=True, type=int,
                        help='minimum id number, e.g. 0')
    parser.add_argument('--maxId', required=True, type=int,
                        help='maximum id number, e.g. 1024')
    parser.add_argument('--noirev', dest='noirev', action='store_true')
    parser.add_argument('--no-noirev', dest='noirev', action='store_false')
    parser.set_defaults(noirev=True)
    args = parser.parse_args()
    global noirev
    noirev=args.noirev
    refs = list(range(args.minId,args.maxId))
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    results=pool.map(do_process,refs)
    out =   np.stack(results)
    if noirev:
        fitsio.write('center_constC_noirev.fits',out)
    else:
        fitsio.write('center_constC.fits',out)
    pool.close()
    return

if __name__=='__main__':
    main()
