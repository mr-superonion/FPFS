#!/usr/bin/env python
import gc
import os
import fpfs
import numpy as np
import fitsio
from multiprocessing import Pool

vdir = "outSmall2-var7em3/psf60/"


def mean_shape(ifield):
    C = 2000.0
    noiRev = True
    fname1 = os.path.join(vdir, "fpfs-cut16-%04d-g1-2222.fits" % (ifield))
    moments1 = fitsio.read(fname1)
    elli1 = fpfs.fpfsBase.fpfsM2E(moments1, C, rev=noiRev)

    fname2 = os.path.join(vdir, "fpfs-cut16-%04d-g1-0000.fits" % (ifield))
    moments2 = fitsio.read(fname2)
    elli2 = fpfs.fpfsBase.fpfsM2E(moments2, C, rev=noiRev)
    a = np.average(elli1["fpfs_e1"])
    b = np.average(elli2["fpfs_e1"])
    c = np.average(elli1["fpfs_RE"])
    d = np.average(elli2["fpfs_RE"])
    del elli1, elli2, moments1, moments2
    gc.collect()
    return a, b, c, d


if __name__ == "__main__":
    names = [("e1p", "f8"), ("e1m", "f8"), ("rep", "f8"), ("rem", "f8")]
    with Pool(20) as p:
        data = np.array(p.map(mean_shape, range(4000)), dtype=names)
    denom = np.average(data["rep"] + data["rem"])
    mArray = (data["e1p"] - data["e1m"]) / denom * 2.0 / 0.04 - 1
    cArray = (data["e1p"] + data["e1m"]) / denom
    print(np.average(mArray), np.std(mArray) / np.sqrt(len(mArray)))
    print(np.average(cArray), np.std(cArray) / np.sqrt(len(cArray)))
