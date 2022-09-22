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
        cparser     =   ConfigParser()
        cparser.read(config_name)
        # setup processor
        self.imgdir =   cparser.get('procsim', 'img_dir')
        self.catdir =   cparser.get('procsim', 'cat_dir')
        self.simname=   cparser.get('procsim', 'sim_name')
        self.indir  =   os.path.join(self.imgdir,self.simname)
        self.do_det =   cparser.get('procsim', 'do_det')
        if not os.path.exists(self.indir):
            raise ValueError('Cannot find input images')
        if not os.path.exists(os.path.join(self.imgdir,'noise')):
            raise ValueError('Cannot find input noises')
        self.outdir=os.path.join(self.catdir,self.simname)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # setup survey parameters
        self.scale=cparser.getfloat('survey','pixel_scale')
        self.psffname=cparser.get('survey','psf_filename')
        self.noi_var=cparser.getfloat('survey','noi_var')

        # setup WL distortion parameter
        glist=[]
        if cparser.getboolean('distortion','test_g1'):
            glist.append('g1')
        if cparser.getboolean('distortion','test_g2'):
            glist.append('g2')
        if len(glist)>0:
            zlist=json.loads(cparser.get('distortion','shear_z_list'))
            self.pendList=['%s-%s' %(i1,i2) for i1 in glist for i2 in zlist]
        else:
            raise ValueError('Cannot process, at least test on g1 or g2.')
        return

    def run(self,Id):
        logging.info('running for galaxy field: %d' %(Id))
        gal_dir     =   os.path.join(self.imgdir,self.simname)
        ngrid       =   64
        # FPFS Basic
        # sigma_as  =   0.5944 #[arcsec] try2
        sigma_as    =   0.45 #[arcsec] try3
        rcut        =   32

        beg         =   ngrid//2-rcut
        end         =   beg+2*rcut
        # PSF
        if '%' in self.psffname:
            psffname=   self.psffname %Id
        else:
            psffname=   self.psffname
        psfData     =   pyfits.open(psffname)[0].data
        npad        =   (ngrid-psfData.shape[0])//2
        psfData2    =   np.pad(psfData,(npad+1,npad),mode='constant')
        psfData2    =   psfData2[beg:end,beg:end]
        outDir      =   'srcfs3_unif3_cosmo170-var7em3_try3'

        if 'cosmo' in gal_dir:
            logging.info('Using cosmos parametric galaxies to simulate the blended case.')
            gbegin=700;gend=5700 # 10000 is enough for dm<0.15%
        else:
            gbegin=0;gend=6400   # 3000 is enough for dm<0.1%
        ngrid2   =   gend-gbegin
        npad        =   (ngrid2-psfData.shape[0])//2
        psfData3    =   np.pad(psfData,(npad+1,npad),mode='constant')

        # FPFS Task
        if self.noi_var>1e-20:
            # noise
            logging.info('Using noisy setup with variance: %.3f' %self.noi_var)
            noiFname    =   os.path.join('noise','noi%04d.fits' %Id)
            if not os.path.isfile(noiFname):
                logging.info('Cannot find input noise file: %s' %noiFname)
                return
            # multiply by 10 since the noise has variance 0.01
            noiData     =   pyfits.getdata(noiFname)[gbegin:gend,gbegin:gend]*10.*np.sqrt(self.noi_var)
            # Also times 100 for the noivar model
            powIn       =   np.load('corPre/noiPows3.npy',allow_pickle=True).item()['%s'%rcut]*self.noi_var*100
            powModel    =   np.zeros((1,powIn.shape[0],powIn.shape[1]))
            powModel[0] =   powIn
            measTask    =   fpfs.image.measure_source(psfData2,sigma_arcsec=sigma_as,noiFit=powModel[0])
        else:
            logging.info('Using noiseless setup')
            # by default noiFit=None
            measTask    =   fpfs.image.measure_source(psfData2,sigma_arcsec=sigma_as)
            noiData     =   0.
        logging.info('The upper limit of wave number is %s pixels' %(measTask.klim_pix))
        for ishear in self.pendList:
            galFname    =   os.path.join(gal_dir,'image-%s-%s.fits' %(Id,ishear))
            if not os.path.isfile(galFname):
                logging.info('Cannot find input galaxy file: %s' %galFname)
                return
            galData     =   pyfits.getdata(galFname)
            galData =   galData+noiData

            outFname    =   os.path.join(outDir,'src-%04d-%s.fits' %(Id,ishear))
            pp  =   'cut%d' %rcut
            outFname    =   os.path.join(outDir,'fpfs-%s-%04d-%s.fits' %(pp,Id,ishear))
            if  os.path.exists(outFname):
                logging.info('Already has measurement: %04d, %s' %(Id,ishear))
                continue
            logging.info('FPFS measurement: %04d, %s' %(Id,ishear))
            if not self.do_det:
                if 'Center' in gal_dir:
                    # fake detection
                    indX    =   np.arange(32,ngrid2,64)
                    indY    =   np.arange(32,ngrid2,64)
                    inds    =   np.meshgrid(indY,indX,indexing='ij')
                    coords  =   np.array(np.zeros(inds[0].size),dtype=[('fpfs_y','i4'),('fpfs_x','i4')])
                    coords['fpfs_y']=   np.ravel(inds[0])
                    coords['fpfs_x']=   np.ravel(inds[1])
                    del indX,indY,inds
                else:
                    raise ValueError('Do not support the case without detection on galaxies with center offsets.')
            else:
                magz        =   27.
                if  sigma_as<0.5:
                    cutmag      =   25.5
                else:
                    cutmag      =   26.0
                thres       =   10**((magz-cutmag)/2.5)*self.scale**2.
                thres2      =   -thres/20.
                coords  =   fpfs.image.detect_sources(galData,psfData3,gsigma=measTask.sigmaF,\
                            thres=thres,thres2=thres2,klim=measTask.klim)
            logging.info('pre-selected number of sources: %d' %len(coords))
            imgList =   [galData[cc['fpfs_y']-rcut:cc['fpfs_y']+rcut,\
                        cc['fpfs_x']-rcut:cc['fpfs_x']+rcut] for cc in coords]
            out     =   measTask.measure(imgList)
            out     =   rfn.merge_arrays([coords,out],flatten=True,usemask=False)
            pyfits.writeto(outFname,out)
            del imgList,out,coords,galData,outFname
            gc.collect()
        logging.info('finish %s' %(Id))
        return

    def __call__(self,Id):
        logging.info('start ID: %d' %(Id))
        return self.run(Id)

if __name__=='__main__':
    parser = ArgumentParser(description="fpfs procsim")
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
