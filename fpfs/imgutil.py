# FPFS shear estimator
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib

import math
import numpy as np

def try_numba_njit():
    try:
        import numba
        return numba.njit
    except ImportError:
        return lambda func: func

def _gauss_kernel(ny,nx,sigma,do_shift=False,return_grid=False):
    """
    Generate a Gaussian kernel on grids for np.fft.fft transform

    Parameters:
        ny:    		    grid size in y-direction
        nx:    		    grid size in x-direction
        sigma:		    scale of Gaussian in Fourier space
        do_shift:       Whether doing
        return_grid:    return grids or not

    Returns:
        Gaussian on grids and (if return_grid) grids for (y, x) axes.
    """
    out = np.empty((ny,nx))
    x   = np.fft.fftfreq(nx,1/np.pi/2.)
    y   = np.fft.fftfreq(ny,1/np.pi/2.)
    if do_shift:
        x=np.fft.fftshift(x)
        y=np.fft.fftshift(y)
    Y,X = np.meshgrid(y,x,indexing='ij')
    r2  = X**2.+Y**2.
    out = np.exp(-r2/2./sigma**2.)
    if not return_grid:
        return out
    else:
        return out,(Y,X)

def _gauss_kernel_rfft(ny,nx,sigma,return_grid=False):
    """
    Generate a Gaussian kernel on grids for np.fft.rfft transform

    Parameters:
        ny:    		    grid size in y-direction
        nx:    		    grid size in x-direction
        sigma:		    scale of Gaussian in Fourier space
        return_grid:    return grids or not

    Returns:
        Gaussian on grids and (if return_grid) grids for (y, x) axes.
    """
    out = np.empty((ny,nx//2+1))
    x   = np.fft.rfftfreq(nx,1/np.pi/2.)
    y   = np.fft.fftfreq(ny,1/np.pi/2.)
    Y,X = np.meshgrid(y,x,indexing='ij')
    r2  = X**2.+Y**2.
    out = np.exp(-r2/2./sigma**2.)
    if not return_grid:
        return out
    else:
        return out,(Y,X)

def gauss_kernel(ny,nx,sigma,do_shift=False,return_grid=False,use_rfft=False):
    """
    Generate a Gaussian kernel in Fourier space on grids

    Parameters:
        ny:    		    grid size in y-direction
        nx:    		    grid size in x-direction
		sigma:		    scale of Gaussian
		do_shift:	    center at (0,0) or (ny/2,nx/2) [bool]
        return_grid:    return grids or not [bool]
        use_rfft:       whether use rfft or not [bool]
    """
    if not isinstance(ny,int):
        raise TypeError('ny should be int')
    if not isinstance(nx,int):
        raise TypeError('nx should be int')
    if not isinstance(sigma,(float,int)):
        raise TypeError('sigma should be float or int')
    if sigma<=0.:
        raise ValueError('sigma should be positive')

    if not use_rfft:
        return _gauss_kernel(ny,nx,sigma,do_shift,return_grid)
    else:
        if do_shift:
            raise ValueError('do not support shifting centroid if use_rfft=True')
        return _gauss_kernel_rfft(ny,nx,sigma,return_grid)

def getFouPow_rft(arrayIn: np.ndarray) -> np.ndarray:
    """
    Get Fourier power function

    Parameters:
    -----
    arrayIn:    array_like
                image array (centroid does not matter)

    Returns:
    ----
    galpow:     array_like
                Fourier Power
    """

    ngrid   =   arrayIn.shape[0]
    tmp     =   np.abs(np.fft.rfft2(arrayIn))**2.
    tmp     =   np.fft.fftshift(tmp,axes=0)
    # Get power function and subtract noise power
    galpow  =   np.empty((ngrid,ngrid),dtype=np.float64)
    tmp2    =   np.roll(np.flip(tmp),axis=0,shift=1)
    galpow[:,:ngrid//2+1] =  tmp2
    galpow[:,ngrid//2:]   =  tmp[:,:-1]
    return galpow

def getFouPow(arrayIn: np.ndarray) -> np.ndarray:
    """
    Get Fourier power function

    Parameters:
    -----
    arrayIn:    array_like
                image array (centroid does not matter)

    Returns:
    ----
    galpow:     array_like
                Fourier Power (centered at (ngrid//2,ngrid//2))
    """

    ngrid   =   arrayIn.shape[0]
    galpow  =   np.empty((ngrid,ngrid),dtype=np.float64)
    # Get power function and subtract noise power
    galpow[:,:]  =   np.abs(np.fft.fft2(arrayIn))**2.
    galpow  =   np.fft.fftshift(galpow)
    return galpow

def getRnaive(arrayIn):
    """
    A naive way to estimate Radius.
    Note, this naive estimation is heavily influenced by noise.

    Parameters:
        arrayIn:    image array (centroid does not matter)

    Returns:
        Fourier Power (centered at (ngrid//2,ngrid//2))
    """

    arrayIn2=   np.abs(arrayIn)
    # Get the half light radius of noiseless PSF
    thres   =   arrayIn2.max()*0.5
    sigma   =   np.sum(arrayIn2>thres)
    sigma   =   np.sqrt(sigma/np.pi)
    return sigma

def shapelets2D(ngrid,nord,sigma):
    """
    Generate shapelets function in Fourier space
    (only support square stamps: ny=nx=ngrid)

    Parameters:
        ngrid:      number of pixels in x and y direction
        nord:       radial order of the shaplets
        sigma:      scale of shapelets in Fourier space

    Returns:
        2D shapelet basis in shape of [nord,nord,ngrid,ngrid]
    """

    mord    =   nord
    # Set up the r and theta function
    xy1d    =   np.fft.fftshift(np.fft.fftfreq(ngrid,d=sigma/2./np.pi))
    xfunc,yfunc=np.meshgrid(xy1d,xy1d)
    rfunc   =   np.sqrt(xfunc**2.+yfunc**2.)
    gaufunc =   np.exp(-rfunc*rfunc/2.)
    rmask   =   (rfunc!=0.)
    xtfunc  =   np.zeros((ngrid,ngrid),dtype=np.float64)
    ytfunc  =   np.zeros((ngrid,ngrid),dtype=np.float64)
    np.divide(xfunc,rfunc,where=rmask,out=xtfunc)
    np.divide(yfunc,rfunc,where=rmask,out=ytfunc)
    eulfunc = xtfunc+1j*ytfunc
    lfunc   =   np.zeros((nord+1,mord+1,ngrid,ngrid),dtype=np.float64)
    chi     =   np.zeros((nord+1,mord+1,ngrid,ngrid),dtype=np.complex64)
    # Set up l function
    lfunc[0,:,:,:]=1.
    lfunc[1,:,:,:]=1.-rfunc*rfunc+np.arange(mord+1)[None,:,None,None]
    #
    for n in range(2,nord+1):
        for m in range(mord+1):
            lfunc[n,m,:,:]=(2.+(m-1.-rfunc*rfunc)/n)*lfunc[n-1,m,:,:]-(1.+(m-1.)/n)*lfunc[n-2,m,:,:]
    for nn in range(nord+1):
        for mm in range(nn,-1,-2):
            c1=(nn-abs(mm))//2
            d1=(nn+abs(mm))//2
            cc=math.factorial(c1)+0.
            dd=math.factorial(d1)+0.
            cc=cc/dd/np.pi
            chi[nn,mm,:,:]=pow(-1.,d1)/sigma*pow(cc,0.5)*lfunc[c1,abs(mm),:,:]\
                    *pow(rfunc,abs(mm))*gaufunc*eulfunc**mm
    return chi

def fitNoiPow(ngrid,galPow,noiModel,rlim):
    """
    Fit the noise power from observed galaxy power and remove it

    Parameters:
        ngrid:      number of pixels in x and y direction
        galPow:     galaxy Fourier power function

    Returns:
        list:       (power after removing noise power,subtracted noise power)
    """

    rlim2=  int(max(ngrid*0.4,rlim))
    indX=   np.arange(ngrid//2-rlim2,ngrid//2+rlim2+1)
    indY=   indX[:,None]
    mask=   np.ones((ngrid,ngrid),dtype=bool)
    mask[indY,indX]=False
    vl  =   galPow[mask]
    nl  =   noiModel[:,mask]
    par =   np.linalg.lstsq(nl.T,vl,rcond=None)[0]
    noiSub=np.sum(par[:,None,None]*noiModel,axis=0)
    return noiSub

def pcaimages(X,nmodes):
    """
    Estimate the principal components of array list X

    Parameters:
        X:          input data array
        nmodes:     number of pcs to keep

    Returns:
        list:       pc images, stds on the axis
    """

    assert len(X.shape)==3
    # vectorize
    nobj,nn2,nn1=   X.shape
    dim         =   nn1*nn2
    # X is (x1,x2,x3..,xnobj).T [x_i is column vectors of data]
    X           =   X.reshape((nobj,dim))
    # Xave  = X.mean(axis=0)
    # X     = X-Xave
    # Xave  = Xave.reshape((1,nn2,nn1))
    # out =   np.vstack([Xave,V])

    # Get covariance matrix
    M   =   np.dot(X,X.T)/(nobj-1)
    # Solve the Eigen function of the covariance matrix
    # e is eigen value and eV is eigen vector
    # eV: (p1,p2,..,pnobj) [p_i is column vectors of parameters]
    e,eV=   np.linalg.eigh(M)
    # The Eigen vector tells the combination of ndata
    tmp =   np.dot(eV.T,X)
    # Rank from maximum eigen value to minimum
    # and only keep the first nmodes
    V   =   tmp[::-1][:nmodes]
    e   =   e[::-1][:nmodes+10]
    stds=   np.sqrt(e)
    out =   V.reshape((nmodes,nn2,nn1))
    eVout=  eV.T[:nmodes]
    return out,stds,eVout
