#!/usr/bin/env python
import os
import numpy as np
import astropy.table as astTab


if __name__=='__main__':
    catDir  =   'catPre'
    if not os.path.exists(catDir):
        os.mkdir(catDir)
    outFname=   os.path.join(catDir,'control_cat.csv')
    nGroup  =   1000
    nser1   =   np.linspace(1.,4.,nGroup)+3./nGroup
    nser2   =   np.linspace(1.,4.,nGroup)
    rgal1   =   np.linspace(.2,.8,nGroup)+.6/nGroup
    rgal2   =   np.linspace(.2,.8,nGroup)
    flux1   =   15.*2.**(np.linspace(0.,4.,nGroup)+4./nGroup)
    flux2   =   15.*2.**(np.linspace(0.,4.,nGroup))
    dist    =   np.linspace(0.4,3.0,nGroup)
    fwhm    =   np.linspace(0.39,0.81,nGroup)
    varNoi  =   np.linspace(0.0015,0.19,nGroup)

    np.random.shuffle(nser1)
    np.random.shuffle(nser2)
    np.random.shuffle(rgal1)
    np.random.shuffle(rgal2)
    np.random.shuffle(flux1)
    np.random.shuffle(flux2)
    np.random.shuffle(dist)
    np.random.shuffle(fwhm)
    np.random.shuffle(varNoi)
    names=('nser1','nser2','rgal1','rgal2','flux1','flux2','dist','fwhm','varNoi')
    data=(nser1,nser2,rgal2,rgal2,flux2,flux2,dist,fwhm,varNoi)
    tab=astTab.Table(data=data,names=names)

    tab.write(outFname)
