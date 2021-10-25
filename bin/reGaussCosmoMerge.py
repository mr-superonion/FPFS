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
import gc
import glob
import scipy
import catutil
import imgSimutil
import numpy as np
import astropy.io.fits as pyfits
from astropy.table import Table,vstack,hstack
from pixel3D import cartesianGrid3D
from configparser import ConfigParser

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class reGaussCosmoMergeBatchConfig(pexConfig.Config):
    rootDir     = pexConfig.Field(
        dtype=str,
        default="./",
        doc="Root Diectory"
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)
class reGaussCosmoMergeRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        p2List  =   ['0000','2222','2000','0200','0020','0002']
        p1List  =   ['g1']
        pendList=   ['%s-%s' %(i1,i2) for i1 in p1List for i2 in p2List]
        return [(ref, kwargs) for ref in pendList]
def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)
class reGaussCosmoMergeBatchTask(BatchPoolTask):
    ConfigClass = reGaussCosmoMergeBatchConfig
    RunnerClass = reGaussCosmoMergeRunner
    _DefaultName = "reGaussCosmoMergeBatch"
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))
    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        return

    @abortOnError
    def runDataRef(self,pend):
        self.log.info('begining for %s' %(pend))
        #Prepare the storeSet
        pool    =   Pool("reGaussCosmoMergeBatch")
        pool.cacheClear()
        pool.storeSet(pend=pend)
        hpList  =  imgSimutil.cosmoHSThpix[:-1]
        #hpList  =  imgSimutil.cosmoHSThpix[:67]
        #Prepare the pool
        pool.map(self.process,hpList)
        return

    @abortOnError
    def process(self,cache,pixId):
        self.log.info('processing healpix: %d' %pixId)
        pend    =   cache.pend
        inDir   =   os.path.join(self.config.rootDir,'outCosmo-var36em4','src-psf75-%s' %(pixId))
        outFname=os.path.join(inDir,'stackAll-%s_2.fits' %pend)
        names=['ext_shapeHSM_HsmShapeRegauss_e1','ext_shapeHSM_HsmShapeRegauss_e2','base_SdssShape_x','base_SdssShape_y',\
           'modelfit_CModel_instFlux','modelfit_CModel_instFluxErr','ext_shapeHSM_HsmShapeRegauss_resolution']
        pltDir='../../galSim-HSC/s19/s19-1/anaCat_newS19Mask_fdeltacut/plot/optimize_weight/'

        pix_scale=  0.168/3600.
        cosmo252=   imgSimutil.cosmoHSTGal('252')
        #for index in range(133):
        dd      =   cosmo252.hpInfo[cosmo252.hpInfo['pix']==pixId]
        nx      =   int(dd['dra']/pix_scale)
        ny      =   int(dd['ddec']/pix_scale)

        hstcat  =   cosmo252.readHpixSample(pixId)
        msk     =   (hstcat['xI']>32)&(hstcat['yI']>32)&(hstcat['xI']<nx-32)&(hstcat['yI']<ny-32)
        hstcat  =   hstcat[msk]
        xyRef   =   np.vstack([hstcat['xI'],hstcat['yI']]).T
        tree    =   scipy.spatial.cKDTree(xyRef)
        del msk,xyRef
        gc.collect()

        fnList  =   glob.glob(os.path.join(inDir,'src*-%s.fits' %pend))
        dataAll =   []
        for fname in fnList:
            assert os.path.isfile(fname)
            data=   self.readFits(fname)
            data['a_i']=0.
            wlmsk   =   catutil.get_wl_flags(data)
            data    =   data[names]
            xyDat   =   np.vstack([data['base_SdssShape_x'],data['base_SdssShape_y']]).T
            dis,inds=   tree.query(xyDat,k=1)
            mask    =   (dis<=(1.5/0.168))
            dis     =   dis[mask]
            inds    =   inds[mask]
            data    =   data[mask]
            wlmsk   =   wlmsk[mask]
            matcat  =   hstcat[inds]
            data    =   Table(data.as_array(), names=names)
            sigmae  =   catutil.get_sigma_e_model(data,pltDir)
            erms    =   catutil.get_erms_model(data,pltDir)
            data['i_hsmshaperegauss_derived_sigma_e']=   sigmae
            data['i_hsmshaperegauss_derived_rms_e']  =   erms
            data['i_hsmshaperegauss_derived_weight'] =   1./(sigmae**2 + erms**2)
            data['mag_auto']    =   matcat['mag_auto']
            data['zphot']       =   matcat['zphot']
            data['cosmo_index'] =   matcat['index']
            data['match_dPix']  =   dis
            data['wlmask']      =   wlmsk
            dataAll.append(data)
            del xyDat,dis,inds,wlmsk,matcat,data
        dataAll=vstack(dataAll)
        dataAll.write(outFname,overwrite=True)
        del dataAll,hstcat
        gc.collect()
        return

    def readFits(self,filename):
        dd=Table.read(filename)
        # Load the header to get proper name of flags
        header = pyfits.getheader(filename, 1)
        n_flag = dd["flags"].shape[1]
        for i in range(n_flag):
            dd[header["TFLAG%s" % (i+1)]] = dd["flags"][:, i]

        # Then, apply mask for permissive cuts
        mask =  (~(dd['base_SdssCentroid_flag'])) &\
                (~dd['ext_shapeHSM_HsmShapeRegauss_flag']) &\
                (dd['base_ClassificationExtendedness_value'] > 0) &\
                (~np.isnan(dd['modelfit_CModel_instFlux']))&\
                (~np.isnan(dd['modelfit_CModel_instFluxErr'])) &\
                (~np.isnan(dd['ext_shapeHSM_HsmShapeRegauss_resolution'])) &\
                (~np.isnan(dd['ext_shapeHSM_HsmPsfMoments_xx'])) &\
                (~np.isnan(dd['ext_shapeHSM_HsmPsfMoments_yy'])) &\
                (~np.isnan(dd['ext_shapeHSM_HsmPsfMoments_xy'])) &\
                (~np.isnan(dd['base_Variance_value']))&\
                (~np.isnan(dd['modelfit_CModel_instFlux']))&\
                (~np.isnan(dd['modelfit_CModel_instFluxErr']))&\
                (~np.isnan(dd['ext_shapeHSM_HsmShapeRegauss_resolution']))&\
                (dd['deblend_nChild']  ==  0)
        dd =   dd[mask]
        return dd

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
