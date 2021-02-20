import numpy as np

def getFouPow(arrayIn):
    """
    # get Fourier power function

    Parameters:
    -----------
    arrayIn:    image array (centroid does not matter)

    Returns:
    ----------
    galpow:     Fourier Power (centered at (ngrid//2,ngrid//2))
    """

    arrayIn.astype(np.float64)
    # Get power function and subtract noise power
    galpow  =   np.abs(np.fft.fft2(arrayIn))**2.
    galpow  =   np.fft.fftshift(galpow)
    return galpow

def getRnaive(arrayIn):
    """
    # A naive way to estimate Radius
    # Note that this naive estimation is
    # heavily influenced by noise

    Parameters:
    -----------
    arrayIn:    image array (centroid does not matter)

    Returns:
    ----------
    galpow:     Fourier Power (centered at (ngrid//2,ngrid//2))
    """

    arrayIn2=   np.abs(arrayIn)
    # Get the half light radius of noiseless PSF
    thres   =   arrayIn2.max()*0.5
    sigma   =   np.sum(arrayIn2>thres)
    sigma   =   np.sqrt(sigma/np.pi)
    return sigma

def shapelets2D(ngrid,nord,sigma):
    """
    # Generate the shapelets function

    Parameters:
    -----------
    ngrid:      number of pixels in x and y direction
    nord:       radial order of the shaplets
    sigma:      scale of shapelets

    Returns:
    ----------
    chi:        2D shapelet basis
                in shape of [nord,nord,ngrid,ngrid]
    """

    mord    =   nord
    # Set up the r and theta function
    xy1d    =   np.fft.fftshift(np.fft.fftfreq(ngrid,d=sigma/ngrid))
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
            cc=np.math.factorial(c1)+0.
            dd=np.math.factorial(d1)+0.
            cc=cc/dd/np.pi
            chi[nn,mm,:,:]=pow(-1.,d1)/sigma*pow(cc,0.5)*lfunc[c1,abs(mm),:,:]*pow(rfunc,abs(mm))*gaufunc*eulfunc**mm
    return chi

def removeNoiPow(ngrid,galPow,noiModel,rlim):
    #TODO: finish it
    """
    # fit the noise power and remove it

    Parameters:
    -----------
    ngrid:      number of pixels in x and y direction
    galPow:     galaxy Fourier power function

    Returns:
    ----------
    minPow:     power after removing noise power
    noiSub:     subtracted noise power
    """

    rlim2       =   max(ngrid*0.4,rlim)
    noiList     =   []
    valList     =   []
    for j in range(ngrid):
        for i in range(ngrid):
            ii=i-ngrid/2.
            jj=j-ngrid/2.
            r   =   np.sqrt(ii**2.+jj**2.)
            if r>rlim2:
                valList.append(galPow[j,i])
                noiList.append(noiModel[:,j,i])
    vl  =   np.array(valList)
    nl  =   np.array(noiList)
    nl  =   np.hstack([nl,np.ones((nl.shape[0],1))])
    par =   np.linalg.lstsq(nl,vl)[0]
    noiSub   =   np.zeros((ngrid,ngrid))
    npar=   len(par)
    for ipc in range(npar-1):
        noiSub+=(par[ipc]*noiModel[ipc])
    noiSub  +=  par[-1]
    minPow  =   galPow-noiSub
    return minPow,noiSub
