#!/usr/bin/env python
#
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
#
# python lib
import os
import galsim
import numpy as np
import astropy.io.fits as pyfits

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError


class cgcSimConfig(pexConfig.Config):
    outDir      =   pexConfig.Field(dtype=str, default='sim20210301/galaxy/', doc = 'directory to store outputs')
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)

class cgcSimTask(pipeBase.CmdLineTask):
    _DefaultName = "cgcSim"
    ConfigClass = cgcSimConfig
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)


    @pipeBase.timeMethod
    def run(self,ifield):
        self.log.info('begining for field %04d' %(ifield))
        ngrid       =   64
        nx          =   100
        ny          =   nx
        ndata       =   nx*ny
        nrot        =   2
        scale       =   0.168
        bigfft      =   galsim.GSParams(maximum_fft_size=10240)
        flux_scaling=   2.587

        variance    =   0.0035
        ud          =   galsim.UniformDeviate(ifield*10000+1)
        np.random.seed(ifield*10000+1)

        # training data
        catName     =   'real_galaxy_catalog_25.2.fits'
        directory   =   '../../galsim_train/COSMOS_25.2_training_sample/'
        cosmos_cat  =   galsim.COSMOSCatalog(catName, dir=directory)

        # index
        index_use   =   cosmos_cat.orig_index
        # parametric catalog
        param_cat   =   cosmos_cat.param_cat[index_use]
        ngAll       =   len(index_use)

        # Get the psf and nosie information
        psfFname    =   os.path.join('psfPre','psf%04d.fits'%(ifield))
        psfImg      =   galsim.fits.read(psfFname)

        psfInt      =   galsim.InterpolatedImage(psfImg,scale=scale,flux = 1.)
        g2          =   0.

        outFname1   =   os.path.join(self.config.outDir,'gal%04d-00.fits' %(ifield))
        outFname2   =   os.path.join(self.config.outDir,'gal%04d-20.fits' %(ifield))
        if os.path.exists(outFname1):
            self.log.info('Already have the outcome: %s' %outFname1)
            return
        gal_image1  =   galsim.ImageF(nx*ngrid,ny*ngrid,scale=scale)
        gal_image1.setOrigin(0,0)
        gal_image2  =   galsim.ImageF(nx*ngrid,ny*ngrid,scale=scale)
        gal_image2.setOrigin(0,0)

        data_rows   =   []
        i           =   0
        while i <ndata:
            # Prepare the subimage
            ix      =   i%nx
            iy      =   i//nx
            b       =   galsim.BoundsI(ix*ngrid,(ix+1)*ngrid-1,iy*ngrid,(iy+1)*ngrid-1)
            sub_image1 = gal_image1[b]
            sub_image2 = gal_image2[b]
            #simulate the galaxy
            if i%nrot==0:
                # update galaxy
                index   =   np.random.randint(0,ngAll)
                ss      =   param_cat[index]

                # prepare the galaxies
                gal0    =   cosmos_cat.makeGalaxy(gal_type='parametric',index=index,gsparams=bigfft)
                gal0    *=  flux_scaling

                # rotate the galaxy
                ang     =   ud()*2.*np.pi * galsim.radians
                gal0    =   gal0.rotate(ang)
            else:
                gal0    =   gal0.rotate(1./nrot*np.pi*galsim.radians)
            gal         =   gal0.shear(g1=0.,g2=0.)
            final       =   galsim.Convolve([psfInt,gal],gsparams=bigfft)
            final.drawImage(sub_image1,method='no_pixel')
            gal         =   gal0.shear(g1=0.02,g2=0.)
            final       =   galsim.Convolve([psfInt,gal],gsparams=bigfft)
            final.drawImage(sub_image2,method='no_pixel')
            row=(i,index,ss['IDENT'],ss['RA'],ss['DEC'],ss['MAG'],iparent)
            data_rows.append(row)
            i   +=  1
            del gal,final,row
            gc.collect
        pyfits.writeto(outFname,gal_image.array)
        del gal_image1,gal_image2
        gc.collect()
        return

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        doBatch = kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        return parser
    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass
    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass
    def writeMetadata(self, dataRef):
        pass
    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass

class cgcSimBatchConfig(pexConfig.Config):
    perGroup =   pexConfig.Field(dtype=int, default=100, doc = 'sims per group')
    cgcSim = pexConfig.ConfigurableField(
        target = cgcSimTask,
        doc = "cgcSim task to run on multiple cores"
    )

class cgcSimRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minGroup    =  parsedCmd.minGroup
        maxGroup    =  parsedCmd.maxGroup
        return [(ref, kwargs) for ref in range(minGroup,maxGroup)]

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)
class cgcSimBatchTask(BatchPoolTask):
    ConfigClass = cgcSimBatchConfig
    RunnerClass = cgcSimRunner
    _DefaultName = "cgcSimBatch"
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))
    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.makeSubtask("cgcSim")

    @abortOnError
    def runDataRef(self,Id):
        self.log.info('beginning group %d' %(Id))
        perGroup=   self.config.perGroup
        fMin    =   perGroup*Id
        fMax    =   perGroup*(Id+1)
        #Prepare the pool
        pool    =   Pool("cgcSim")
        pool.cacheClear()
        fieldList=  range(fMin,fMax)
        pool.map(self.process,fieldList)
        self.log.info('finish group %d'%(Id) )
        return

    def process(self,cache,ifield):
        self.cgcSim.run(ifield)
        return

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument('--minGroup', type= int,
                        default=0,
                        help='minimum group number')
        parser.add_argument('--maxGroup', type= int,
                        default=1,
                        help='maximum group number')
        return parser
    @classmethod
    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass
    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass
    def writeMetadata(self, dataRef):
        pass
    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass
    def _getConfigName(self):
        return None
    def _getEupsVersionsName(self):
        return None
    def _getMetadataName(self):
        return None
