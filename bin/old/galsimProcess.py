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
import galsim
import numpy as np
import fitsio

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError


class galsimProcessBatchConfig(pexConfig.Config):
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

    def validate(self):
        pexConfig.Config.validate(self)


class galsimProcessRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minGroup = parsedCmd.minGroup
        maxGroup = parsedCmd.maxGroup
        return [(ref, kwargs) for ref in range(minGroup, maxGroup)]


def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)


class galsimProcessBatchTask(BatchPoolTask):
    ConfigClass = galsimProcessBatchConfig
    RunnerClass = galsimProcessRunner
    _DefaultName = "galsimProcessBatch"

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
    def runDataRef(self, Id):
        self.log.info("begining for group %d" % (Id))
        # Prepare the storeSet
        pool = Pool("galsimProcessBatch")
        pool.cacheClear()
        expDir = "sim20210301/galaxy_basic_psf75"
        assert os.path.isdir(expDir)
        pool.storeSet(expDir=expDir)
        pool.storeSet(Id=Id)

        # Prepare the pool
        p2List = ["0000", "1111", "2222"]
        p1List = ["g1", "g2"]
        pendList = ["%s-%s" % (i1, i2) for i1 in p1List for i2 in p2List]
        pool.map(self.process, pendList)
        self.log.info("finish group %d" % (Id))
        return

    def process(self, cache, pend):
        Id = cache.Id
        inFname = os.path.join(cache.expDir, "image-%d-%s.fits" % (Id, pend))
        assert os.path.exists(inFname)
        gal_image = galsim.fits.read(inFname)

        psfFWHM = cache.expDir.split("_psf")[-1]
        psfFname = os.path.join(cache.expDir, "psf-%s.fits" % (psfFWHM))
        assert os.path.exists(psfFname)
        psf_img = galsim.fits.read(psfFname)

        # Basic parameters
        scale = 0.168
        ngrid = 64
        nx = 100
        ny = 100
        ngal = nx * ny

        # Get the shear information
        gList = np.array([-0.02, 0.0, 0.02])
        gList = gList[[eval(i) for i in pend.split("-")[-1]]]
        self.log.info("Processing for %s" % pend)
        self.log.info("shear List is for %s" % gList)
        types = [
            ("regauss_e1", ">f8"),
            ("regauss_e2", ">f8"),
            ("regauss_detR", ">f8"),
            ("regauss_resolution", ">f8"),
        ]

        data = []
        for i in range(ngal):
            ix = i % nx
            iy = i // nx
            b = galsim.BoundsI(
                ix * ngrid, (ix + 1) * ngrid - 1, iy * ngrid, (iy + 1) * ngrid - 1
            )
            sub_img = gal_image[b]
            result = galsim.hsm.EstimateShear(sub_img, psf_img, strict=False)
            data.append(
                (
                    result.corrected_e1,
                    result.corrected_e2,
                    result.moments_sigma,
                    result.resolution_factor,
                )
            )
            del result, sub_img
            gc.collect()
        out = np.array(data, types)
        outFname = os.path.join(cache.expDir, "hsm-%d-%s.fits" % (Id, pend))
        fitsio.write(outFname, out)
        del data, out
        gc.collect()
        return

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument(
            "--minGroup", type=int, default=0, help="minimum group number"
        )
        parser.add_argument(
            "--maxGroup", type=int, default=1, help="maximum group number"
        )
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
