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
    nser1   =   np.linspace(1.,4.,nGroup)+3./nGroup/2.
    nser2   =   np.linspace(1.,4.,nGroup)
    rgal1   =   np.linspace(.2,.8,nGroup)+.6/nGroup/2.
    rgal2   =   np.linspace(.2,.8,nGroup)
    flux1   =   15.*2.**(np.linspace(0.,4.,nGroup)+4./nGroup/2.)
    flux2   =   15.*2.**(np.linspace(0.,4.,nGroup))
    e1gal1  =   np.linspace(-0.6,0.6,nGroup)
    e2gal1  =   np.linspace(-0.6,0.6,nGroup)
    e1gal2  =   np.linspace(-0.6,0.6,nGroup)
    e2gal2  =   np.linspace(-0.6,0.6,nGroup)
    dist    =   2.0*2**np.linspace(0,.7,nGroup) #arcsec
    fwhm    =   np.linspace(0.39,0.81,nGroup)   #arcsec
    beta    =   np.linspace(3.0,4.0,nGroup)     #arcsec
    e1psf   =   np.linspace(-0.14,0.14,nGroup)  
    e2psf   =   np.linspace(-0.14,0.14,nGroup)
    varNoi  =   10**(np.linspace(-3.,-0.8,nGroup))

    np.random.shuffle(nser1)
    np.random.shuffle(nser2)
    np.random.shuffle(rgal1)
    np.random.shuffle(rgal2)
    np.random.shuffle(e1gal1)
    np.random.shuffle(e2gal1)
    np.random.shuffle(e1gal2)
    np.random.shuffle(e2gal2)
    np.random.shuffle(flux1)
    np.random.shuffle(flux2)
    np.random.shuffle(dist)
    np.random.shuffle(fwhm)
    np.random.shuffle(beta)
    np.random.shuffle(e1psf)
    np.random.shuffle(e2psf)
    np.random.shuffle(varNoi)
    names=('nser1','nser2','rgal1','rgal2','e1gal1','e2gal1','e1gal2','e2gal2','flux1','flux2','dist','fwhm','beta','e1psf','e2psf','varNoi')
    data=(nser1,nser2,rgal1,rgal2,e1gal1,e2gal1,e1gal2,e2gal2,flux1,flux2,dist,fwhm,beta,e1psf,e2psf,varNoi)
    tab=astTab.Table(data=data,names=names)
    tab.write(outFname,overwrite=True)
