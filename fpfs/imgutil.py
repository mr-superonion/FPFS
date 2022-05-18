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
        out:            Gaussian on grids and (if return_grid) grids for (y, x) axes.
        (Y,X)           Grids
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
        sigma:		    scale of Gaussian in Fourier space
        do_shift:	    center at (0,0) or (ny/2,nx/2) [bool]
        return_grid:    return grids or not [bool]
        use_rfft:       whether use rfft or not [bool]

    Returns:
        out:            Gaussian kernel
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

def getFouPow_rft(arrayIn: np.ndarray)->np.ndarray:
    """
    Get Fourier power function

    Parameters:
        arrayIn:        image array (centroid does not matter)

    Returns:
        galpow:         Fourier Power
    """

    ngrid   =   arrayIn.shape[0]
    tmp     =   np.abs(np.fft.rfft2(arrayIn))**2.
    tmp     =   np.fft.fftshift(tmp,axes=0)
    # Get power function and subtract noise power
    foupow  =   np.empty((ngrid,ngrid),dtype=np.float64)
    tmp2    =   np.roll(np.flip(tmp),axis=0,shift=1)
    foupow[:,:ngrid//2+1] =  tmp2
    foupow[:,ngrid//2:]   =  tmp[:,:-1]
    return foupow

def getFouPow(arrayIn: np.ndarray, noiPow=None)->np.ndarray:
    """
    Get Fourier power function

    Parameters:
        arrayIn:        image array (centroid does not matter) [np.ndarray]

    Returns:
        out:            Fourier Power (centered at (ngrid//2,ngrid//2)) [ndarray]
    """
    out =   np.fft.fftshift(np.abs(np.fft.fft2(arrayIn))**2.).astype(np.float64)
    if isinstance(noiPow,float):
        out =   out-np.ones(arrayIn.shape)*noiPow*arrayIn.size
    elif isinstance(noiPow,np.ndarray):
        out =   out-noiPow
    return out

def getRnaive(arrayIn:np.ndarray)->float:
    """
    A naive way to estimate Radius.
    Note, this naive estimation is heavily influenced by noise.

    Parameters:
        arrayIn:        image array (centroid does not matter)

    Returns:
        sigma:          effective radius
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
        ngrid:          number of pixels in x and y direction
        nord:           radial order of the shaplets
        sigma:          scale of shapelets in Fourier space

    Returns:
        chi:            2D shapelet basis in shape of [nord,nord,ngrid,ngrid]
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

def fitNoiPow(ngrid,galPow,noiModel,rlim)->np.ndarray:
    """
    Fit the noise power from observed galaxy power

    Parameters:
        ngrid:      number of pixels in x and y direction
        galPow:     galaxy Fourier power function

    Returns:
        noiSub:     noise power to be subtracted
    """

    rlim2=  int(max(ngrid*0.4,rlim))
    indX=   np.arange(ngrid//2-rlim2,ngrid//2+rlim2+1)
    indY=   indX[:,None]
    mask=   np.ones((ngrid,ngrid),dtype=bool)
    mask[indY,indX]=False
    vl  =   galPow[mask]
    nl  =   noiModel[:,mask]
    par =   np.linalg.lstsq(nl.T,vl,rcond=None)[0]
    noiSub= np.sum(par[:,None,None]*noiModel,axis=0)
    return noiSub

class pcaVector():
    def __init__(self,X=None,fname=None):
        """Building PC space

        Parameters:
            X (ndarray):        input data array (mean subtracted) [shape=(nobj,ndim)]

        Atributes:
            shape (tuple):      shape of the data vector
            bases (ndarray):    principal vectors
            stds (ndarray):     stds on theses axes
            projs (ndarray):    projection coefficients of the initializing data
        """

        if X is not None:
            if fname is not None:
                raise ValueError('fname should be None when X is not None')
            # initialize with X
            assert len(X.shape)==2
            nobj=X.shape[0]

            # subtract average
            self.ave=np.average(X,axis=0)
            X   =   X-self.ave
            # normalize data vector
            self.norm = np.sqrt(np.average(X**2.,axis=0))
            X   =   X/self.norm
            self.data=X

            # Get covariance matrix
            Cov =   np.dot(X,X.T)/(nobj-1)
            # Solve the Eigen function of the covariance matrix
            # e is eigen value and eVec is eigen vector
            eVal,eVec=   np.linalg.eigh(Cov)

            # The Eigen vector tells the combination of these data vectors
            # Rank from maximum eigen value to minimum and only keep the first nmodes
            bases=  np.dot(eVec.T,X)[::-1]
            var  =  eVal[::-1]
            projs=  eVec[:,::-1]

            # remove those bases with extremely small stds
            msk  =  var>var[0]/1e8
            self.stds=np.sqrt(var[msk])
            self.bases=bases[msk]
            self.projs=projs[:,msk]
            base_norm=np.sum(self.bases**2.,axis=1)
            self.bases_inv=self.bases/base_norm[:,None]
        elif fname is not None:
            # initialize with fname
            pass
        else:
            raise ValueError('X and fname cannot all be None')

        return

    def transform(self,X):
        """ transform from data space to pc coefficients
        Parameters:
            X (ndarray): input data vectors [shape=(nobj,ndim)]
        Returns:
            proj (ndarray): projection array
        """
        assert len(X.shape)==2
        X   =   X-self.ave
        X   =   X/self.norm
        proj=   X.dot(self.bases_inv.T)
        return proj

    def itransform(self,projs):
        """ transform from pc space to data
        Parameters:
            projs (ndarray): projection coefficients
        Returns:
            X (ndarray): data vector
        """
        assert len(projs.shape)==2
        nm= projs.shape[1]
        X= projs.dot(self.bases[0:nm])
        X= X*self.norm
        X= X+self.ave
        return X

def cut_img(img,rcut):
    """
    cutout img into postage stamp with width=2rcut

    Parameters:
        img:            input image
        rcut:           cutout radius

    Returns:
        out:            stamps (cut-out)
    """
    ngrid   =   img.shape[0]
    beg     =   ngrid//2-rcut
    end     =   beg+2*rcut
    out     =   img[beg:end,beg:end]
    return out
