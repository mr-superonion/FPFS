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
import fpfs
import numpy as np

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from lsst.ctrl.pool.pool import Pool
from lsst.pipe.base import TaskRunner
from lsst.utils.timer import timeMethod
from lsst.ctrl.pool.parallel import BatchPoolTask


class cgcSimBasicBatchConfig(pexConfig.Config):
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

    def validate(self):
        pexConfig.Config.validate(self)


class cgcSimBasicRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minIndex = parsedCmd.minIndex
        maxIndex = parsedCmd.maxIndex
        return [(ref, kwargs) for ref in range(minIndex, maxIndex)]


class cgcSimBasicBatchTask(BatchPoolTask):
    ConfigClass = cgcSimBasicBatchConfig
    RunnerClass = cgcSimBasicRunner
    _DefaultName = "cgcSimBasicBatch"

    def __init__(self, **kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        return

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument(
            "--minIndex", type=int, default=0, help="minimum group number"
        )
        parser.add_argument(
            "--maxIndex", type=int, default=1, help="maximum group number"
        )
        return parser

    @timeMethod
    def runDataRef(self, index):
        self.log.info("begining for group %d" % (index))
        # Prepare the storeSet
        pool = Pool("cgcSimBasicBatch")
        pool.cacheClear()
        # expDir  =   "galaxy_basic_psf60"
        # expDir  =   "small0_psf60"
        expDir = "galaxy_basic3Shift_psf60"
        # expDir  =   "galaxy_basic3Center_psf60"
        if not os.path.isdir(expDir):
            os.mkdir(expDir)
        pool.storeSet(expDir=expDir)
        fieldList = np.arange(100 * index, 100 * (index + 1))
        pool.map(self.process, fieldList)
        return

    @timeMethod
    def process(self, cache, Id):
        # Prepare the pool
        p2List = ["0000", "2222"]
        p1List = ["g1"]
        # p1List=['g1','g2']
        pendList = ["%s-%s" % (i1, i2) for i1 in p1List for i2 in p2List]
        for pp in pendList:
            fpfs.simutil.make_basic_sim(cache.expDir, pp, Id)
            gc.collect()
        self.log.info("finish ID: %d" % (Id))
        return

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
