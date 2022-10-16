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
from astropy.table import Table, vstack, hstack
import numpy.lib.recfunctions as rfn

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError


class reGaussCosmoMeasBatchConfig(pexConfig.Config):
    rootDir = pexConfig.Field(dtype=str, default="./", doc="Root Diectory")

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

    def validate(self):
        pexConfig.Config.validate(self)


class reGaussCosmoMeasRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        p2List = ["0000", "2222", "2000", "0200", "0020", "0002"]
        # p2List  =   ['0002']
        p1List = ["g1"]
        pendList = ["%s-%s" % (i1, i2) for i1 in p1List for i2 in p2List]
        return [(ref, kwargs) for ref in pendList]


def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)


class reGaussCosmoMeasBatchTask(BatchPoolTask):
    ConfigClass = reGaussCosmoMeasBatchConfig
    RunnerClass = reGaussCosmoMeasRunner
    _DefaultName = "reGaussCosmoMeasBatch"

    def __reduce__(self):
        """Pickler"""
        return unpickle, (
            self.__class__,
            [],
            dict(
                config=self.config,
                name=self._name,
                parentTask=self._parentTask,
                log=self.log,
            ),
        )

    def __init__(self, **kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        return

    @abortOnError
    def runDataRef(self, pend):
        psfFWHM = "60"  #'60','HSC'
        npend = "outCosmoR-var36em4"
        outDir = os.path.join(self.config.rootDir, npend, "mag245-res03-bm38-dis4")
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
        self.log.info("beginning for %s, seeing %s: " % (pend, psfFWHM))
        # Prepare the storeSet
        pool = Pool("reGaussCosmoMeasBatch")
        pool.cacheClear()
        pool.storeSet(pend=pend)
        pool.storeSet(psfFWHM=psfFWHM)
        pool.storeSet(npend=npend)
        # Prepare the pool
        resList = pool.map(self.process, np.arange(1000))
        resList = [x for x in resList if x is not None]
        if len(resList) > 1:
            newTab = Table(
                rows=resList,
                names=(
                    "e1_z1",
                    "n_z1",
                    "e1_z2",
                    "n_z2",
                    "e1_z3",
                    "n_z3",
                    "e1_z4",
                    "n_z4",
                ),
            )
            finOname = os.path.join(outDir, "e1_%s_psf%s.fits" % (pend, psfFWHM))
            newTab.write(finOname, overwrite=True)
        return

    @abortOnError
    def process(self, cache, ifield):
        hpList = imgSimutil.cosmoHSThpix
        # nhp     =   len(hpList)
        # if int(ifield//30>=nhp):
        #    return
        # pixId   =   hpList[ifield//30]
        pixId = 1743743
        self.log.info("process healpix: %d, field: %d " % (pixId, ifield))
        pend = cache.pend
        npend = cache.npend
        psfFWHM = cache.psfFWHM
        inDir = os.path.join(
            self.config.rootDir, npend, "src-psf%s-%s" % (psfFWHM, pixId)
        )
        fname = os.path.join(inDir, "src%04d-%s.fits" % (ifield, pend))
        assert os.path.isfile(fname)
        data = self.readMatch(fname, pixId)

        # zbound  =   np.array([0.,0.561,0.906,1.374,5.410])     #before sim 3
        zbound = np.array([0.005, 0.5477, 0.8874, 1.3119, 3.0])  # sim 3
        nzs = len(zbound) - 1
        eAll = ()
        for iz in range(nzs):
            msk = (data["zphot"] > zbound[iz]) & (data["zphot"] <= zbound[iz + 1])
            eTmp = np.average(data["ext_shapeHSM_HsmShapeRegauss_e1"][msk])
            nn = np.sum(msk)
            eAll = eAll + (eTmp, nn)
            del eTmp, msk, nn
        del data
        gc.collect()
        eAll = list(eAll)
        return eAll

    def readMatch(self, fname, pixId):
        names = [
            "ext_shapeHSM_HsmShapeRegauss_e1",
            "ext_shapeHSM_HsmShapeRegauss_e2",
            "base_SdssShape_x",
            "base_SdssShape_y",
            "modelfit_CModel_instFlux",
            "modelfit_CModel_instFluxErr",
            "ext_shapeHSM_HsmShapeRegauss_resolution",
        ]
        pltDir = "../../galSim-HSC/s19/s19-1/anaCat_newS19Mask_fdeltacut/plot/optimize_weight/"

        # Match
        pix_scale = 0.168 / 3600.0
        cosmo252 = imgSimutil.cosmoHSTGal("252")
        cosmo252E = imgSimutil.cosmoHSTGal("252E")

        info = cosmo252.hpInfo[cosmo252.hpInfo["pix"] == pixId]
        nx = int(info["dra"] / pix_scale)
        ny = int(info["ddec"] / pix_scale)

        if "CosmoE" in fname:
            hstcat = pyfits.getdata("hstcatE-dis4.fits")
        elif "CosmoR" in fname:
            hstcat = pyfits.getdata("hstcatR-dis4.fits")

        msk = (
            (hstcat["xI"] > 32)
            & (hstcat["yI"] > 32)
            & (hstcat["xI"] < nx - 32)
            & (hstcat["yI"] < ny - 32)
        )
        hstcat = hstcat[msk]
        xyRef = np.vstack([hstcat["xI"], hstcat["yI"]]).T
        tree = scipy.spatial.cKDTree(xyRef)
        del msk, xyRef

        dd = self.readFits(fname)
        xyDat = np.vstack([dd["base_SdssShape_x"], dd["base_SdssShape_y"]]).T
        dis, inds = tree.query(xyDat, k=1)
        mask = dis <= (0.85 / 0.168)
        dis = dis[mask]
        inds = inds[mask]
        dd = dd[mask]

        wlmsk = (
            (catutil.get_imag(dd) < 24.5)
            & (catutil.get_abs_ellip(dd) <= 2.0)
            & (catutil.get_res(dd) >= 0.3)
            & (catutil.get_snr(dd) >= 10.0)
            & (catutil.get_imag_A10(dd) < 25.5)
            & (catutil.get_logb(dd) <= -0.38)
        )
        dd = dd[wlmsk]
        inds = inds[wlmsk]
        matcat = hstcat[inds]
        del mask, inds, dis
        dd = dd[names]

        dd = Table(dd.as_array(), names=names)
        sigmae = catutil.get_sigma_e_model(dd, pltDir)
        erms = catutil.get_erms_model(dd, pltDir)
        dd["i_hsmshaperegauss_derived_weight"] = 1.0 / (sigmae**2 + erms**2)
        # dd['i_hsmshaperegauss_derived_sigma_e']=   sigmae
        # dd['i_hsmshaperegauss_derived_rms_e']  =   erms
        dd["zphot"] = matcat["zphot"]
        dd["mag_auto"] = matcat["mag_auto"]
        del xyDat, wlmsk, matcat, hstcat, sigmae, erms
        gc.collect()
        return dd

    def readFits(self, filename):
        dd = Table.read(filename)
        # Load the header to get proper name of flags
        header = pyfits.getheader(filename, 1)
        n_flag = dd["flags"].shape[1]
        for i in range(n_flag):
            dd[header["TFLAG%s" % (i + 1)]] = dd["flags"][:, i]

        # Then, apply mask for permissive cuts
        mask = (
            (~(dd["base_SdssCentroid_flag"]))
            & (~dd["ext_shapeHSM_HsmShapeRegauss_flag"])
            & (dd["base_ClassificationExtendedness_value"] > 0)
            & (~np.isnan(dd["modelfit_CModel_instFlux"]))
            & (~np.isnan(dd["modelfit_CModel_instFluxErr"]))
            & (~np.isnan(dd["ext_shapeHSM_HsmShapeRegauss_resolution"]))
            & (~np.isnan(dd["ext_shapeHSM_HsmPsfMoments_xx"]))
            & (~np.isnan(dd["ext_shapeHSM_HsmPsfMoments_yy"]))
            & (~np.isnan(dd["ext_shapeHSM_HsmPsfMoments_xy"]))
            & (~np.isnan(dd["base_Variance_value"]))
            & (~np.isnan(dd["modelfit_CModel_instFlux"]))
            & (~np.isnan(dd["modelfit_CModel_instFluxErr"]))
            & (~np.isnan(dd["ext_shapeHSM_HsmShapeRegauss_resolution"]))
            & (dd["deblend_nChild"] == 0)
        )
        dd = dd[mask]
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
