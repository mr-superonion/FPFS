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
from fpsBase import fpsBaseTask

# lsst Tasks
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.image as afwImg
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.meas.base as meaBase
import lsst.meas.algorithms as meaAlg
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.base import SingleFrameMeasurementTask


from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError


class ten9SimNoNoiseConfig(pexConfig.Config):
    "config"
    doDetect  = pexConfig.Field(
        dtype=bool, default=True,
        doc = "whether to detect galaxies"
    )
    doNoise  = pexConfig.Field(
        dtype=bool, default=True,
        doc = "whether to add noise"
    )
    detection = pexConfig.ConfigurableField(
        target = SourceDetectionTask,
        doc = "Detect sources"
    )
    measurement = pexConfig.ConfigurableField(
        target = SingleFrameMeasurementTask,
        doc = "Measure sources"
    )
    fpsBase = pexConfig.ConfigurableField(
        target = fpsBaseTask,
        doc = "Subtask to run measurement of fps method"
    )
    
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        self.detection.thresholdValue = 5.0
        self.detection.isotropicGrow  = True
        self.detection.reEstimateBackground=False
        # measure and apply aperture correction; note: measuring and applying aperture
        # minimal set of measurements needed to determine PSF
        self.measurement.plugins.names = [
            "base_SdssCentroid",
            'base_GaussianFlux',
            "base_SdssShape",
            "base_PsfFlux",
            ]
        self.measurement.slots.apFlux       =   None
        self.measurement.slots.instFlux     =   None
        self.measurement.slots.calibFlux    =   None

class ten9SimNoNoiseTask(pipeBase.CmdLineTask):
    _DefaultName = "ten9SimNoNoise"
    ConfigClass = ten9SimNoNoiseConfig
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask('detection', schema=self.schema)
        self.makeSubtask('measurement', schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask('fpsBase', schema=self.schema)
        self.schema.addField("detected", type=int, doc="wheter galaxy is detected by hscpipe")
    
    def SimulateExposure(self,ifield,ngrid):
        # Basic parameters
        nx          =   20 
        ny          =   nx
        ndata       =   nx*ny
        scale       =   0.168
        ngridTot    =   ngrid*nx
        bigfft      =   galsim.GSParams(maximum_fft_size=10240)
        gal_image   =   galsim.ImageF((nx+2)*ngrid,(ny+2)*ngrid,scale=scale)
        g1          =   0.
        g2          =   0.
    
        
        # Get the PSF image
        psf_beta    =   3.5 
        psf_fwhm    =   0.6         # arcsec
        psf_trunc   =   4.*psf_fwhm # arcsec (=pixels)
        psf_e1      =   0.          #
        psf_e2      =   0.025       #
        psf         =   galsim.Moffat(beta=psf_beta, 
                        fwhm=psf_fwhm,trunc=psf_trunc)
        psf         =   psf.shear(e1=psf_e1,e2=psf_e2)        
        psfGalsim   =   psf.drawImage(nx=45,ny=45,scale=scale)
        
        # Get the  galaxy image      
        flux_scaling=   2.587
        # Load data
        catName     =   'real_galaxy_catalog_25.2.fits'
        dir         =   '../galsim_train/COSMOS_25.2_training_sample/'
        cosmos_cat  =   galsim.COSMOSCatalog(catName, dir=dir)
        # index
        index_use   =   cosmos_cat.orig_index
        # parametric catalog
        param_cat   =   cosmos_cat.param_cat[index_use]
        for i in range(ndata):
            index       =   (ifield*ndata+i)
            ud          =   galsim.UniformDeviate(index)
            # prepare the galaxies        
            gal         =   cosmos_cat.makeGalaxy(gal_type='parametric',index=index%81400,gsparams=bigfft)
            # Rotate the galaxy 
            ang     =   ud()*2.*np.pi * galsim.radians
            gal     =   gal.rotate(ang)
            # Shear the galaxy
            gal     =   gal.shear(g1=g1,g2=g2)
            gal     =   galsim.Convolve([psf,gal],gsparams=bigfft)
            # Prepare the subimage
            ix      =   i%nx+1
            iy      =   i/nx+1
            b       =   galsim.BoundsI(ix*ngrid+1, (ix+1)*ngrid,iy*ngrid+1,(iy+1)*ngrid)
            sub_gal_image = gal_image[b]
            # Draw the galaxy image
            gal.drawImage(sub_gal_image)
        if self.config.doNoise:
            # Get the noise image
            ud          =   galsim.UniformDeviate(ifield)
            varNoi      =   0.008769248618965289
            corNoise    =   galsim.GaussianNoise(ud)
            corNoise    =   corNoise.withVariance(varNoi)
            corNoise.applyTo(gal_image)
        exposure    =   afwImg.ExposureF((nx+2)*ngrid,(ny+2)*ngrid)
        exposure.getMaskedImage().getImage().getArray()[:,:]=gal_image.array
        exposure.getMaskedImage().getVariance().getArray()[:,:]=0.001
        print 'finish making exposure'
        psfImg      =   afwImg.ImageF(45,45)
        psfImg.getArray()[:,:]= psfGalsim.array
        psfImg      =   psfImg.convertD()
        kernel      =   afwMath.FixedKernel(psfImg)
        kernelPSF   =   meaAlg.KernelPsf(kernel)
        exposure.setPsf(kernelPSF)
        return exposure

    def addFootPrint(self,exposure,sources,ngrid):
        for ss in sources:
            ss['detected']  =   1
        offset  =   afwGeom.Extent2I(10,10)
        dims    =   afwGeom.Extent2I(20,20)
        mskData =   exposure.getMaskedImage().getMask().getArray()
        imageBBox=  exposure.getBBox()
        for j in range(1,21):
            for i in range(1,21):
                centx   =   i*ngrid+ngrid/2
                centy   =   j*ngrid+ngrid/2
                hasSource=  False
                for jj in range(centy-15,centy+15):
                    if hasSource:
                        break
                    for ii in range(centx-15,centx+15):
                        if mskData[jj,ii]>1:
                            hasSource=True
                            break
                if not hasSource:
                    record = sources.addNew()
                    position = afwGeom.Point2I(centx,centy)
                    bbox = afwGeom.Box2I(position-offset, dims)
                    footprint = afwDet.Footprint(bbox, imageBBox)
                    footprint.addPeak(position.getX(), position.getY(),1.)
                    record.setFootprint(footprint)
                    record['detected']  =   0
        return
    
    def measureSource(self,catOutput,exposure,ngrid):
        table           =   afwTable.SourceTable.make(self.schema)
        table.setMetadata(self.algMetadata)
        if self.config.doDetect:
            detRes      =   self.detection.run(table=table, exposure=exposure, doSmooth=True)
            sources     =   detRes.sources
        else:
            sources     =   afwTable.SourceCatalog(table)
        # get the preMeasurements
        self.addFootPrint(exposure,sources,ngrid)
        # do measurement
        self.measurement.run(measCat=sources, exposure=exposure)
        # run fps method
        self.fpsBase.run(sources,exposure)
        return sources
    
    @pipeBase.timeMethod
    def run(self,ifield):
        # output 
        iout    =   ifield/200
        outputdir=  './outNoise%d/' %(iout)
        if not os.path.exists(outputdir):
            os.popen('mkdir %s' %(outputdir) )
        print 'processing field %s' %(ifield)
        catOutput=  outputdir+'src-%04d.fits'%(ifield)
        if os.path.exists(catOutput):
            print 'already has the outcome'
            return
        print 'begin simulation'
        ngrid   =   48
        exposure=   self.SimulateExposure(ifield,ngrid)
        print 'begin measurement'
        sources =   self.measureSource(catOutput,exposure,ngrid)
        if len(sources)>0:
            print 'writing %d sources' %(len(sources))
            flags   =   afwTable.SOURCE_IO_NO_FOOTPRINTS
            sources.writeFits(catOutput,mode='w',flags=flags)
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

class ten9SimNoNoiseBatchConfig(pexConfig.Config):
    minField =   pexConfig.Field(dtype=int, default=0, doc = 'minField')
    maxField =   pexConfig.Field(dtype=int, default=100, doc = 'minField')
    ten9SimNoNoise = pexConfig.ConfigurableField(
        target = ten9SimNoNoiseTask,
        doc = "ten9SimNoNoise task to run on multiple cores"
    )
    
class ten9SimNoNoiseRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(ref, kwargs) for ref in range(1)] 

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class ten9SimNoNoiseBatchTask(BatchPoolTask):
    ConfigClass = ten9SimNoNoiseBatchConfig
    RunnerClass = ten9SimNoNoiseRunner
    _DefaultName = "ten9SimNoNoiseBatch"

    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))

    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.makeSubtask("ten9SimNoNoise")
    
    @abortOnError
    def run(self,Id):
        minField=   self.config.minField
        maxField=   self.config.maxField
        fMin    =   200*minField
        fMax    =   200*(maxField)
        #Prepare the pool
        pool    =   Pool("ten9SimNoNoise")
        pool.cacheClear()
        fieldList=  range(fMin,fMax)
        pool.map(self.process,fieldList)
        return
        
    def process(self,cache,ifield):
        return self.ten9SimNoNoise.run(ifield)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
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
