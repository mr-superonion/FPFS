# Copyright 20210905 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# python lib
import os
import galsim
import numpy as np
import astropy.io.fits as pyfits
import numpy.lib.recfunctions as rfn


class cosmoHSTGal:
    def __init__(self, version):
        self.version = version
        if version == "252":
            self.directory = os.path.join(
                os.environ["homeWrk"],
                "COSMOS/galsim_train/COSMOS_25.2_training_sample/",
            )
            self.catName = "real_galaxy_catalog_25.2.fits"
        elif version == "252E":
            self.directory = os.path.join(
                os.environ["homeWrk"], "COSMOS/galsim_train/COSMOS_25.2_extended/"
            )
        else:
            raise ValueError("Does not support version=%s" % version)
        self.finName = os.path.join(self.directory, "cat_used.fits")
        self.catused = np.array(pyfits.getdata(self.finName))
        return

    def prepare_sample(self):
        """Reads the HST galaxy training sample"""
        if not os.path.isfile(self.finName):
            if self.version == "252":
                cosmos_cat = galsim.COSMOSCatalog(self.catName, dir=self.directory)
                # used index
                index_use = cosmos_cat.orig_index
                # used catalog
                paracat = cosmos_cat.param_cat[index_use]
                # parametric catalog
                oricat = np.array(pyfits.getdata(cosmos_cat.real_cat.getFileName()))[
                    index_use
                ]
                ra = oricat["RA"]
                dec = oricat["DEC"]
                index_new = np.arange(len(ra), dtype=int)
                __tmp = np.stack([ra, dec, index_new]).T
                radec = np.array(
                    [tuple(__t) for __t in __tmp],
                    dtype=[("ra", ">f8"), ("dec", ">f8"), ("index", "i8")],
                )
                catfinal = rfn.merge_arrays(
                    [paracat, radec], flatten=True, usemask=False
                )
                pyfits.writeto(self.finName, catfinal)
                self.catused = catfinal
            else:
                return
        return
