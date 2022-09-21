#!/usr/bin/env python
#
# Copyright 20220312 Xiangchong Li.
#
import os
import gc
import fpfs
import json
import logging
import schwimmbad
import numpy as np
import astropy.io.fits as pyfits
from argparse import ArgumentParser
from configparser import ConfigParser
import numpy.lib.recfunctions as rfn

class Worker(object):
    def __init__(self,config_name):
        cparser      =   ConfigParser()
        cparser.read(config_name)
        self.simname=cparser.get('simulation', 'sim_name')
        self.infname=cparser.get('simulation','input_name')
        self.scale=cparser.getfloat('survey','pixel_scale')
        self.psfInt=None
        if not os.path.exists(self.simname):
            os.mkdir(self.simname)
        return

    def run(self,Id):
        gal_dir      =   "galaxy_cosmo170"
        ngrid       =   64
        pixScale    =   0.168
        psfFWHM     =   0.6

        # FPFS Basic
        # sigma_as  =   0.5944 #[arcsec] try2
        sigma_as    =   0.45 #[arcsec] try3
        rcut        =   32

        beg         =   ngrid//2-rcut
        end         =   beg+2*rcut
        scale       =   0.168
        outDir      =   'srcfs3_unif3_cosmo170-var7em3_try3'

        if 'small' in gal_dir:
            logging.info('Using small galaxies')
            if "var0em0" not in outDir:
                gid  =   Id//8
            else:
                gid  =   Id
            gbegin=0;gend=6400
            ngrid2   =   6400
        elif 'star' in gal_dir:
            logging.info('Using stars')
            if "var0em0" not in outDir:
                raise ValueError("stars do not support noiseless simulations")
            gid  =   0
            gbegin=0;gend=6400
            ngrid2  =   6400
        elif 'basic' in gal_dir:
            logging.info('Using cosmos parametric galaxies to simulate the isolated case.')
            if Id >= 3000: # 3000 is enough for dm<0.1%
                return
            gid  =  Id
            gbegin=0;gend=6400
            ngrid2  =   6400
        elif 'cosmo' in gal_dir:
            logging.info('Using cosmos parametric galaxies to simulate the blended case.')
            if Id >= 10000: # 10000 is enough for dm<0.15%
                return
            gid  =   Id
            gbegin=700;gend=5700
            ngrid2  =   5000
        else:
            raise ValueError("gal_dir should cantain either 'small', 'star', 'basic' or 'cosmo'")
        logging.info('running for galaxy field: %s, noise field: %s' %(gid,Id))

        # PSF
        psfFname    =   os.path.join(gal_dir,'psf-%s.fits' %psfFWHM)
        psfData     =   pyfits.open(psfFname)[0].data
        npad        =   (ngrid-psfData.shape[0])//2
        psfData2    =   np.pad(psfData,(npad+1,npad),mode='constant')
        psfData2    =   psfData2[beg:end,beg:end]
        # PSF2
        npad        =   (ngrid2-psfData.shape[0])//2
        psfData3    =   np.pad(psfData,(npad+1,npad),mode='constant')
        noi_var     =   7e-3

        # FPFS Task
        if noi_var>1e-20:
            # noise
            _tmp        =   outDir.split('var')[-1]
            noi_var      =   eval(_tmp[0])*10**(-1.*eval(_tmp[3]))
            logging.info('noisy setup with variance: %.3f' %noi_var)
            noiFname    =   os.path.join('noise','noi%04d.fits' %Id)
            if not os.path.isfile(noiFname):
                logging.info('Cannot find input noise file: %s' %noiFname)
                return
            # multiply by 10 since the noise has variance 0.01
            noiData     =   pyfits.open(noiFname)[0].data*10.*np.sqrt(noi_var)
            # Also times 100 for the noivar model
            powIn       =   np.load('corPre/noiPows3.npy',allow_pickle=True).item()['%s'%rcut]*noi_var*100
            powModel    =   np.zeros((1,powIn.shape[0],powIn.shape[1]))
            powModel[0] =   powIn
            measTask    =   fpfs.image.measure_source(psfData2,sigma_arcsec=sigma_as,noiFit=powModel[0])
        else:
            noi_var      =   1e-20
            logging.info('We are using noiseless setup')
            # by default noiFit=None
            measTask    =   fpfs.image.measure_source(psfData2,sigma_arcsec=sigma_as)
            noiData     =   None
        logging.info('The upper limit of wave number is %s pixels' %(measTask.klim_pix))
        # isList        =   ['g1-0000','g2-0000','g1-2222','g2-2222']
        # isList        =   ['g1-1111']
        isList          =   ['g1-0000','g1-2222']
        # isList        =   ['g1-0000']
        for ishear in isList:
            galFname    =   os.path.join(gal_dir,'image-%s-%s.fits' %(gid,ishear))
            if not os.path.isfile(galFname):
                logging.info('Cannot find input galaxy file: %s' %galFname)
                return
            galData     =   pyfits.getdata(galFname)
            if noiData is not None:
                galData =   galData+noiData[gbegin:gend,gbegin:gend]

            outFname    =   os.path.join(outDir,'src-%04d-%s.fits' %(Id,ishear))
            pp  =   'cut%d' %rcut
            outFname    =   os.path.join(outDir,'fpfs-%s-%04d-%s.fits' %(pp,Id,ishear))
            if not os.path.exists(outFname):
                logging.info('FPFS measurement: %04d, %s' %(Id,ishear))
                if 'Center' in gal_dir and 'det' not in pp:
                    # fake detection
                    indX    =   np.arange(32,ngrid2,64)
                    indY    =   np.arange(32,ngrid2,64)
                    inds    =   np.meshgrid(indY,indX,indexing='ij')
                    coords  =   np.array(np.zeros(inds[0].size),dtype=[('fpfs_y','i4'),('fpfs_x','i4')])
                    coords['fpfs_y']=   np.ravel(inds[0])
                    coords['fpfs_x']=   np.ravel(inds[1])
                    del indX,indY,inds
                else:
                    magz        =   27.
                    if  sigma_as<0.5:
                        cutmag      =   25.5
                    else:
                        cutmag      =   26.0
                    thres       =   10**((magz-cutmag)/2.5)*scale**2.
                    thres2      =   -thres/20.
                    coords  =   fpfs.image.detect_sources(galData,psfData3,gsigma=measTask.sigmaF,\
                                thres=thres,thres2=thres2,klim=measTask.klim)
                logging.info('pre-selected number of sources: %d' %len(coords))
                imgList =   [galData[cc['fpfs_y']-rcut:cc['fpfs_y']+rcut,\
                            cc['fpfs_x']-rcut:cc['fpfs_x']+rcut] for cc in coords]
                out     =   measTask.measure(imgList)
                out     =   rfn.merge_arrays([coords,out],flatten=True,usemask=False)
                pyfits.writeto(outFname,out)
                del imgList,out,coords
            else:
                logging.info('Skip FPFS measurement: %04d, %s' %(Id,ishear))
            del galData,outFname
            gc.collect()
        logging.info('finish %s' %(Id))

        return

    def __call__(self,Id):
        logging.info('start ID: %d' %(Id))
        return self.run(Id)

if __name__=='__main__':
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
    print(pool)

    worker  =   Worker(args.config)
    refs    =   list(range(args.minId,args.maxId))
    print(refs)
    # worker(1)
    for r in pool.map(worker,refs):
        pass
    pool.close()
