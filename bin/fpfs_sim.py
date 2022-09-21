#!/usr/bin/env python
#
# Copyright 20220312 Xiangchong Li.
#
import os
import gc
import fpfs
import json
import galsim
import logging
from configparser import ConfigParser

class Worker(object):
    def __init__(self,config_name):
        cparser      =   ConfigParser()
        cparser.read(config_name)
        self.simname=cparser.get('simulation', 'sim_name')
        self.infname=cparser.get('simulation','input_name')
        self.scale=cparser.getfloat('survey','pixel_scale')
        if cparser.has_option('survey','psf_fwhm'):
            seeing=cparser.getfloat('survey','psf_fwhm')
            self.prepare_psf(seeing,psf_type='moffat')
            logging.info('Using PSF model')
        else:
            if not cparser.has_option('survey','psf_filename'):
                raise ValueError('Do not have survey-psf_file option')
            else:
                self.psffname=cparser.get('survey','psf_filename')
                self.psfInt=None
            logging.info('Using PSF from input file')
        if not os.path.exists(self.simname):
            os.mkdir(self.simname)
        if 'galaxy' in self.simname:
            assert 'basic' in self.simname or 'small' in self.simname or 'cosmo' in self.simname
            glist=[]
            if cparser.getboolean('distortions','test_g1'):
                glist.append('g1')
            if cparser.getboolean('distortions','test_g2'):
                glist.append('g2')
            if len(glist)>0:
                zlist=json.loads(cparser.get('sources','zbound'))
                self.pendList=['%s-%s' %(i1,i2) for i1 in glist for i2 in zlist]
            else:
                raise ValueError('Cannot process, at least test on g1 or g2.')
        elif 'noise' in self.simname:
            self.pendList=[0]
        else:
            raise ValueError("Cannot setup the task for sim_name=%s!!\\ \
                    Must contain 'galaxy' or 'noise'" %self.simname)
        return 0

    def prepare_psf(self,seeing,psf_type):
        if psf_type.lower()=="moffat":
            psfInt  =   galsim.Moffat(beta=3.5,fwhm=seeing,trunc=seeing*4.)
            self.psfInt  =   psfInt.shear(e1=0.02,e2=-0.02)
        else:
            raise ValueError('Only support moffat PSF.')
        psfImg  =   psfInt.drawImage(nx=45,ny=45,scale=self.scale)
        psffname=   os.path.join(self.simname,'psf-%d.fits' %(seeing*100))
        psfImg.write(psffname)
        return 0

    def run(self,Id):
        if self.psfInt is None:
            if '%' in self.psffname:
                psffname=   self.psffname %Id
            else:
                psffname=   self.psffname
            assert os.path.isfile(psffname), 'Cannot find input PSF file'
            psfImg      =   galsim.fits.read(psffname)
            self.psfInt =   galsim.InterpolatedImage(psfImg,scale=self.scale,flux = 1.)
            del psfImg
        for pp in self.pendList:
            if 'basic' in self.simname or 'small' in self.simname:
                # do basic stamp-like image simulation
                fpfs.simutil.make_basic_sim(self.simname,self.infname,self.psfInt,pp,Id,scale=self.scale)
            elif 'cosmo' in self.simname:
                # do blended cosmo-like image simulation
                fpfs.simutil.make_cosmo_sim(self.simname,self.infname,self.psfInt,pp,Id,scale=self.scale)
            elif 'noise' in self.simname:
                # do pure noise image simulation
                fpfs.simutil.make_noise_sim(self.simname,self.infname,Id,scale=self.scale)
            gc.collect()
        logging.info('finish ID: %d' %(Id))
        return 0

    def __call__(self,Id):
        return self.run(Id)

def main(pool,minId,maxId,config_name):
    worker  =   Worker(config_name)
    refs    =   list(range(minId,maxId))
    pool.map(worker,refs)
    pool.close()
    return 0

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
