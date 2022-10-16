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
import astropy.io.fits as pyfits

# LSST Base
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

# LSST Tasks
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError


class noiSimBatchConfig(pexConfig.Config):
    perGroup = pexConfig.Field(dtype=int, default=100, doc="sims per group")
    outDir = pexConfig.Field(
        dtype=str, default="noise/", doc="directory to store exposures"
    )

    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

    def validate(self):
        pexConfig.Config.validate(self)
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)


class noiSimRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minIndex = parsedCmd.minIndex
        maxIndex = parsedCmd.maxIndex
        return [(ref, kwargs) for ref in range(minIndex, maxIndex)]


def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)


class noiSimBatchTask(BatchPoolTask):
    ConfigClass = noiSimBatchConfig
    RunnerClass = noiSimRunner
    _DefaultName = "noiSimBatch"

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

    @abortOnError
    def runDataRef(self, Id):
        self.log.info("beginning group %d" % (Id))
        perGroup = self.config.perGroup
        fMin = perGroup * Id
        fMax = perGroup * (Id + 1)
        # Prepare the pool
        pool = Pool("noiSim")
        pool.cacheClear()
        fieldList = range(fMin, fMax)
        pool.map(self.process, fieldList)
        self.log.info("finish group %d" % (Id))
        return

    def process(self, cache, ifield):
        self.log.info("begining for field %04d" % (ifield))
        outFname = os.path.join(self.config.outDir, "noi%04d.fits" % (ifield))
        if os.path.exists(outFname):
            self.log.info("Already have the outcome")
            return
        self.log.info("simulating noise for field %s" % (ifield))

        ngrid = 64
        nx = 100
        ny = nx
        scale = 0.168

        variance = 0.01
        ud = galsim.UniformDeviate(ifield * 10000 + 1)

        # setup the galaxy image and the noise image
        noi_image = galsim.ImageF(nx * ngrid, ny * ngrid, scale=scale)
        noi_image.setOrigin(0, 0)
        corNoise = galsim.getCOSMOSNoise(
            file_name="./corPre/correlation.fits",
            rng=ud,
            cosmos_scale=scale,
            variance=variance,
        )
        corNoise.applyTo(noi_image)
        pyfits.writeto(outFname, noi_image.array)
        return

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument(
            "--minIndex", type=int, default=0, help="minimum group index"
        )
        parser.add_argument(
            "--maxIndex", type=int, default=1, help="maximum group index"
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
