#!/usr/bin/env python
import glob
import numpy as np
import astropy.io.fits as pyfits


def main(tList):
    for t in tList:
        stack(t)
    return


def stack(t):
    fitList=glob.glob('debug/%s_*.fits' %t)
    dataStack=np.zeros((64,64))
    for fit in fitList:
        data=pyfits.getdata(fit)
        dataStack+=data
    dataStack/=len(fitList)
    pyfits.writeto('%sStack.fits' %t,dataStack )
    return



if __name__=='__main__':
    tList=['galPow','minPow']
    main(tList)
