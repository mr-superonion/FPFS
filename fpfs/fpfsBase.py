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

import numba
import logging
from . import imgutil
import numpy as np
import numpy.lib.recfunctions as rfn

_default_inds=[(1,2),(2,1),(2,2),(2,3),(3,2)]
_gsigma=3.*2*np.pi/64.

@numba.njit
def get_Rlim(psf_array,sigma):
    """
    Get rlim, the area outside rlim is supressed by the shaplet Gaussian kernel
    in FPFS shear estimation method.

    Parameters:
    ----
    psf_array:     power of PSF or PSF array [np.ndarray]

    Returns:
    ----
        the limit radius [float]
    """
    ngrid   =   psf_array.shape[0]
    thres   =   1.e-3
    rlim    =   ngrid//2
    for dist in range(ngrid//5,ngrid//2-1):
        ave =  abs(np.exp(-dist**2./2./sigma**2.)\
                /psf_array[ngrid//2+dist,ngrid//2])
        ave +=  abs(np.exp(-dist**2./2./sigma**2.)\
                /psf_array[ngrid//2,ngrid//2+dist])
        ave =   ave/2.
        if ave<=thres:
            rlim=   dist
            break
    return rlim

class fpfsTask():
    """
    A class to measure FPFS shapelet mode estimation.

    Parameters:
    ----
    psfData:    2D array of PSF image
    beta:       FPFS scale parameter
    noiModel:   Models to be used to fit noise power function using the pixels
                at large k for each galaxy (if you wish FPFS code to estiamte
                noise power).
                [default: None]
    noiFit:     Estimated noise power function (if you have already estimated
                noise power)
                [default: None]
    det_gsigma: Gaussian sigma for detection kernel [float, default: None]
    deubg:      Whether debug or not [bool, default: False]
    """
    _DefaultName = "fpfsTask"
    def __init__(self,psfData,beta,noiModel=None,noiFit=None,det_gsigma=None,debug=False):
        if not isinstance(beta,(float,int)):
            raise TypeError('Input beta should be float.')
        if beta>=1. or beta<=0.:
            raise ValueError('Input beta shoul be in range (0,1)')
        self.ngrid  =   psfData.shape[0]
        self._dk    =   2.*np.pi/self.ngrid
        self.base_types=[('fpfs_M00','>f8'),\
                    ('fpfs_M22c','>f8'),\
                    ('fpfs_M22s','>f8'),\
                    ('fpfs_M40','>f8')]
        psfData     =   np.array(psfData,dtype='>f8')
        self.noise_correct=False
        if noiFit is not None:
            self.noise_correct=True
            if isinstance(noiFit,np.ndarray):
                assert noiFit.shape==psfData.shape, \
                'the input noise power should have the same shape with input psf image'
                self.noiFit  =  np.array(noiFit,dtype='>f8')
            elif isinstance(noiFit,float):
                self.noiFit  =  np.ones(psfData.shape,dtype='>f8')*noiFit*(self.ngrid)**2.
        else:
            self.noiFit  =  None
        if noiModel is not None:
            self.noiModel=  np.array(noiModel,dtype='>f8')
            self.noise_correct=True
        else:
            self.noiModel=  None
        self.psfPow =   imgutil.getFouPow(psfData)
        # Preparing PSF
        # scale radius of PSF
        sigmaPsf    =   imgutil.getRnaive(self.psfPow)
        # shapelet scale
        self.sigmaPx=   max(min(sigmaPsf*beta,4.),1.)
        self.rlim   =   get_Rlim(self.psfPow,self.sigmaPx)
        self._indX=np.arange(self.ngrid//2-self.rlim,self.ngrid//2+self.rlim+1)
        self._indY=self._indX[:,None]
        self._ind2D=np.ix_(self._indX,self._indX)
        # Preparing shapelets (reshaped)
        nnord       =   4
        # Only uses M00, M22 (real and img) and M40
        self._indC  =   np.array([0,12,20])[:,None,None]
        self.chi    =   imgutil.shapelets2D(self.ngrid,nnord,self.sigmaPx*self._dk)\
                        .reshape(((nnord+1)**2,self.ngrid,self.ngrid))
        if self.noise_correct:
            self.prepare_ChiCov()
            logging.info('Will correct noise bias')
            if det_gsigma is not None:
                logging.info('Will correct detection bias')
                self.prepare_ChiDet(det_gsigma)
                self.det_gsigma=det_gsigma
                self.psfFou = np.fft.fftshift(np.fft.fft2(psfData))
            else:
                self.det_gsigma=None
                logging.info('Do not correct for detection bias')
        else:
            if det_gsigma is not None:
                raise ValueError('Cannot fully correct detection bias without \
                        noise power. Please input noise power.')
        if debug:
            self.stackRes=np.zeros(psfData.shape,dtype='>f8')
        else:
            self.stackRes=None
        return

    def setRlim(self,rlim):
        """
        set rlim, the area outside rlim is supressed by the shaplet Gaussian
        kerenl
        """
        self.rlim=   rlim
        self._indX=np.arange(self.ngrid//2-self.rlim,self.ngrid//2+self.rlim+1)
        self._indY=self._indX[:,None]
        self._ind2D=np.ix_(self._indX,self._indX)
        return

    def prepare_ChiCov(self):
        """
        prepare the basis to estimate covariance
        """
        _chiU   =   self.chi[self._indC,self._indY,self._indX]
        out     =   []
        out.append(_chiU.real[0]*_chiU.real[0])
        out.append(_chiU.real[1]*_chiU.real[1])
        out.append(_chiU.imag[1]*_chiU.imag[1])
        out.append(_chiU.real[2]*_chiU.real[2])
        out.append(_chiU.real[0]*_chiU.real[1])
        out.append(_chiU.real[0]*_chiU.imag[1])
        out.append(_chiU.real[0]*_chiU.real[2])
        out     =   np.stack(out)
        self.chiCov =   out
        self.cov_types= [('fpfs_N00N00','>f8'),\
                    ('fpfs_N22cN22c','>f8'),('fpfs_N22sN22s','>f8'),\
                    ('fpfs_N40N40','>f8'),\
                    ('fpfs_N00N22c','>f8'),('fpfs_N00N22s','>f8'),\
                    ('fpfs_N00N40','>f8')]
        return

    def prepare_ChiDet(self,gsigma=_gsigma):
        """
        prepare the basis to estimate covariance for detection
        """
        _chiU  =    self.chi[self._indC,self._indY,self._indX]
        gKer,(k2grid,k1grid)=imgutil.gauss_kernel(self.ngrid,self.ngrid,gsigma,\
                        do_shift=True,return_grid=True)
        q1Ker  =    (k1grid**2.-k2grid**2.)/gsigma**2.*gKer
        q2Ker  =    (2.*k1grid*k2grid)/gsigma**2.*gKer
        d1Ker  =    (-1j*k1grid)*gKer
        d2Ker  =    (-1j*k2grid)*gKer
        out    =    []
        self.det_types= []
        for (j,i) in _default_inds:
            y  =    j-2
            x  =    i-2
            r1 =    (q1Ker+x*d1Ker-y*d2Ker)*np.exp(1j*(k1grid*x+k2grid*y))
            r2 =    (q2Ker+y*d1Ker+x*d2Ker)*np.exp(1j*(k1grid*x+k2grid*y))
            r1 =    r1[self._indY,self._indX]/self.ngrid**2.
            r2 =    r2[self._indY,self._indX]/self.ngrid**2.
            out.append(_chiU.real[0]*r1) #x00*phi1
            out.append(_chiU.real[0]*r2) #x00*phi2
            out.append(_chiU.real[1]*r1) #x22c*phi1
            out.append(_chiU.imag[1]*r2) #x22s*phi2
            self.det_types.append(('pdet_N00F%d%dr1'  %(j,i),'>f8'))
            self.det_types.append(('pdet_N00F%d%dr2'  %(j,i),'>f8'))
            self.det_types.append(('pdet_N22cF%d%dr1' %(j,i),'>f8'))
            self.det_types.append(('pdet_N22sF%d%dr2' %(j,i),'>f8'))
        out     =   np.stack(out)
        self.chiDet =   out
        return

    def deconvolvePow(self,data,order=1.):
        """
        Deconvolve the galaxy power with the PSF power

        Parameters:
        ----
        data :  galaxy power or galaxy Fourier transfer (ngrid//2,ngrid//2) is origin
        order:  deconvolve order of PSF power

        Returns :
        ----
            Deconvolved galaxy power (truncated at rlim)

        """
        out  =   np.zeros(data.shape,dtype=np.float64)
        out[self._ind2D]=data[self._ind2D]/self.psfPow[self._ind2D]**order
        return out

    def deconvolve2(self,data,prder=1.,frder=1.):
        """
        Deconvolve the galaxy power with the PSF power

        Parameters:
        ----
        data :  galaxy power or galaxy Fourier transfer (ngrid//2,ngrid//2) is origin
        prder:  deconvlove order of PSF FT power
        prder:  deconvlove order of PSF FT

        Returns :
        ----
            Deconvolved galaxy power (truncated at rlim)

        """
        out  =   np.zeros(data.shape,dtype=np.complex64)
        out[self._ind2D]=data[self._ind2D]\
                /self.psfPow[self._ind2D]**prder\
                /self.psfFou[self._ind2D]**frder
        return out

    def itransform(self,data):
        """
        Project image onto shapelet basis vectors

        Parameters:
        ----
        data:   image to transfer

        Returns:
        ----
            projection in shapelet space
        """

        # Moments
        M       =   np.sum(data[None,self._indY,self._indX]\
                    *self.chi[self._indC,self._indY,self._indX],axis=(1,2))
        out     =   np.array((M.real[0],\
                    M.real[1],M.imag[1],\
                    M.real[2]),dtype=self.base_types)
        return out

    def itransformCov(self,data):
        """
        Project the (PP+PD)/P^2 to measure the covariance of shapelet modes.

        Parameters:
        data:   data to transfer

        Returns:
            projection in shapelet space
        """
        # Moments
        dataU   =   data[None,self._indY,self._indX]

        N       =   np.sum(self.chiCov*dataU,axis=(1,2))
        out     =   np.array(tuple(N),dtype=self.cov_types)
        return out

    def itransformDet(self,data):
        dataU   =   data[None,self._indY,self._indX]
        D       =   np.sum(self.chiDet*dataU,axis=(1,2)).real
        out     =   np.array(tuple(D),dtype=self.det_types)
        return out

    def measure(self,galData):
        """
        Measure the FPFS moments

        Parameters:
        ----
        galData:    galaxy image

        Returns:
        ----
            FPFS moments
        """
        if isinstance(galData,np.ndarray):
            assert galData.shape[-1]==galData.shape[-2]
            if len(galData.shape) == 2:
                # single galaxy
                out =   self.__measure(galData)
                return out
            elif len(galData.shape)==3:
                results=[]
                for gal in galData:
                    _g=self.__measure(gal)
                    results.append(_g)
                out =   rfn.stack_arrays(results,usemask=False)
                return out
            else:
                raise ValueError("Input galaxy data has wrong ndarray shape.")
        elif isinstance(galData,list):
            assert isinstance(galData[0],np.ndarray)
            # list of galaxies
            results=[]
            for gal in galData:
                _g=self.__measure(gal)
                results.append(_g)
            out =   rfn.stack_arrays(results,usemask=False)
            return out
        else:
            raise TypeError("Input galaxy data has wrong type (neither list nor ndarray).")

    def __measure(self,data):
        """
        Measure the FPFS moments

        Parameters:
        ----
        data:    image array (centroid does not matter)

        Returns:
        ----
            FPFS moments
        """
        assert len(data.shape)==2
        nn          =   None
        dd          =   None
        galPow      =   imgutil.getFouPow(data)
        if self.noiModel is not None:
            self.noiFit  =   imgutil.fitNoiPow(self.ngrid,galPow,self.noiModel,self.rlim)
        if self.noise_correct:
            assert self.noiFit is not None
            galPow  =   galPow-self.noiFit
            epcor   =   2.*(self.noiFit*self.noiFit+2.*self.noiFit*galPow)
            decEP   =   self.deconvolvePow(epcor,order=2.)
            nn      =   self.itransformCov(decEP)
            if self.det_gsigma is not None:
                _tmp=   np.fft.fftshift(np.fft.fft2(data))
                _tmp=   2.*_tmp*self.noiFit
                decDet= self.deconvolve2(_tmp,prder=1.,frder=1)
                dd  =   self.itransformDet(decDet)
        decPow      =   self.deconvolvePow(galPow,order=1.)
        mm          =   self.itransform(decPow)
        if (nn is not None) and (dd is not None):
            mm      =   rfn.merge_arrays([mm,nn,dd],flatten=True,usemask=False)
        elif nn is not None:
            mm      =   rfn.merge_arrays([mm,nn],flatten=True,usemask=False)
        if self.stackRes is not None:
            self.stackRes+=galPow
        return mm

def fpfsM2E(moments,const=1.,noirev=False,flipsign=False):
    """
    Estimate FPFS ellipticities from fpfs moments

    Parameters:
    ----
    moments:    input FPFS moments     [float array]
    const:      the weighting Constant [float]
    mcalib:     multiplicative bias [float array]
    noirev:     revise the second-order noise bias? [bool]
    flipsign:   flip the sign of response? [bool] (if you are using the convention in FPFSv1, set it to True)

    Returns:
    ----
        an array of (FPFS ellipticities, FPFS ellipticity response, FPFS flux ratio, and FPFS selection response)
    """
    #Get weight
    weight  =   moments['fpfs_M00']+const
    #Ellipticity
    e1      =   moments['fpfs_M22c']/weight
    e2      =   moments['fpfs_M22s']/weight
    e1sq    =   e1*e1
    e2sq    =   e2*e2
    #FPFS flux ratio
    s0      =   moments['fpfs_M00']/weight
    s4      =   moments['fpfs_M40']/weight
    #FPFS sel Respose (part1)
    e1sqS0  =   e1sq*s0
    e2sqS0  =   e2sq*s0

    if noirev:
        assert 'fpfs_N00N00' in moments.dtype.names
        assert 'fpfs_N00N22c' in moments.dtype.names
        assert 'fpfs_N00N22s' in moments.dtype.names
        ratio=  moments['fpfs_N00N00']/weight**2.
        e1  =   (e1+moments['fpfs_N00N22c']\
                /weight**2.)/(1+ratio)
        e2  =   (e2+moments['fpfs_N00N22s']\
                /weight**2.)/(1+ratio)
        e1sq=   (e1sq-moments['fpfs_N22cN22c']/weight**2.\
                +4.*e1*moments['fpfs_N00N22c']/weight**2.)\
                /(1.+3*ratio)
        e2sq=   (e2sq-moments['fpfs_N22sN22s']/weight**2.\
                +4.*e2*moments['fpfs_N00N22s']/weight**2.)\
                /(1.+3*ratio)
        s0  =   (s0+moments['fpfs_N00N00']\
                /weight**2.)/(1+ratio)
        s4  =   (s4+moments['fpfs_N00N40']\
                /weight**2.)/(1+ratio)

        e1sqS0= (e1sqS0+3.*e1sq*moments['fpfs_N00N00']/weight**2.\
                -s0*moments['fpfs_N22cN22c']/weight**2.)/(1+6.*ratio)
        e2sqS0= (e2sqS0+3.*e2sq*moments['fpfs_N00N00']/weight**2.\
                -s0*moments['fpfs_N22sN22s']/weight**2.)/(1+6.*ratio)

    eSq     =   e1sq+e2sq
    eSqS0   =   e1sqS0+e2sqS0
    #Response factor
    RE      =   1./np.sqrt(2.)*(s0-s4+e1sq+e2sq)
    types   =   [('fpfs_e1','>f8'),('fpfs_e2','>f8'),('fpfs_RE','>f8'),\
                ('fpfs_s0','>f8'), ('fpfs_eSquare','>f8'), ('fpfs_RS','>f8')]
    ellDat  =   np.array(np.zeros(moments.size),dtype=types)
    ellDat['fpfs_e1']   =   e1
    ellDat['fpfs_e2']   =   e2
    # In Li et. al (2018) there is a minus sign difference
    ellDat['fpfs_RE']   =   RE*(2.*flipsign.real-1.)
    ellDat['fpfs_s0']   =   s0
    ellDat['fpfs_eSquare']  =   eSq
    ellDat['fpfs_RS']   =   (eSq-eSqS0)/np.sqrt(2.)
    return ellDat

def fpfsM2Err(moments,const=1.):
    """
    Estimate FPFS measurement errors from the fpfs
    moments and the moments covariances

    Parameters:
    ----
    moments:    input FPFS moments     [float array]
    const:      the weighting Constant [float]
    mcalib:     multiplicative bias [float array]

    Returns:
    ----
        an array of (measurement error, FPFS ellipticity, FPFS flux ratio)
    """
    assert 'fpfs_N00N00' in moments.dtype.names
    assert 'fpfs_N00N22c' in moments.dtype.names
    assert 'fpfs_N00N22s' in moments.dtype.names
    assert 'fpfs_N00N40' in moments.dtype.names

    #Get weight
    weight  =   moments['fpfs_M00']+const
    #FPFS Ellipticity
    e1      =   moments['fpfs_M22c']/weight
    e2      =   moments['fpfs_M22s']/weight
    #FPFS flux ratio
    s0      =   moments['fpfs_M00']/weight
    e1sq    =   e1*e1
    e2sq    =   e2*e2
    s0sq    =   s0*s0
    ratio   =   moments['fpfs_N00N00']/weight**2.

    e1Err    =   moments['fpfs_N22cN22c']/weight**2.\
            -4.*e1*moments['fpfs_N00N22c']/weight**2.\
            +3*ratio*e1sq
    e2Err    =   moments['fpfs_N22sN22s']/weight**2.\
            -4.*e2*moments['fpfs_N00N22s']/weight**2.\
            +3*ratio*e2sq
    s0Err    =   moments['fpfs_N00N00']/weight**2.\
            -4.*s0*moments['fpfs_N00N00']/weight**2.\
             +3*ratio*s0sq

    e1s0Cov =   moments['fpfs_N00N22c']/weight**2.\
            -2.*s0*moments['fpfs_N00N22c']/weight**2.\
            -2.*e1*moments['fpfs_N00N00']/weight**2.\
            +3*ratio*e1*s0

    e2s0Cov =   moments['fpfs_N00N22s']/weight**2.\
            -2.*s0*moments['fpfs_N00N22s']/weight**2.\
            -2.*e2*moments['fpfs_N00N00']/weight**2.\
            +3*ratio*e2*s0

    types   =   [('fpfs_e1Err','>f8'),('fpfs_e2Err','>f8'),('fpfs_s0Err','>f8'),\
                    ('fpfs_e1s0Cov','>f8'),('fpfs_e2s0Cov','>f8')]
    errDat  =   np.array(np.zeros(len(moments)),dtype=types)
    errDat['fpfs_e1Err']   =   e1Err
    errDat['fpfs_e2Err']   =   e2Err
    errDat['fpfs_s0Err']   =   s0Err
    errDat['fpfs_e1s0Cov'] =   e1s0Cov
    errDat['fpfs_e2s0Cov'] =   e2s0Cov
    return errDat

#def fpfsM2E_v3(moments,const=1.,mcalib=0.):
#    """
#    (This is for higher order shapelest moments
#    but the implementation of this function is unfinished)
#    # Estimate FPFS ellipticities from fpfs moments

#    Parameters:
#    moments:    input FPFS moments
#    const:      the weighting Constant
#    mcalib:     multiplicative biases

#    Returns:
#       an array of FPFS ellipticities, FPFS ellipticity response, FPFS flux ratio
#
#
#    """
#    #Get weight
#    weight  =   moments['fpfs_M20']+const
#    #FPFS flux
#    flux    =   moments['fpfs_M00']/weight
#    #Ellipticity
#    e1      =   moments['fpfs_M22c']/weight
#    e2      =   moments['fpfs_M22s']/weight
#    e41     =   moments['fpfs_M42c']/weight
#    e42     =   moments['fpfs_M42s']/weight
#    #Response factor
#    R1      =   1./np.sqrt(2.)*(moments['fpfs_M00']-moments['fpfs_M40'])/weight+np.sqrt(6)*(e1*e41)
#    R2      =   1./np.sqrt(2.)*(moments['fpfs_M00']-moments['fpfs_M40'])/weight+np.sqrt(6)*(e2*e42)
#    RE      =   (R1+R2)/2.
#    types   =   [('fpfs_e1','>f8'),('fpfs_e2','>f8'),('fpfs_RE','>f8'),('fpfs_flux','>f8')]
#    ellDat  =   np.array(np.zeros(moments.size),dtype=types)
#    ellDat['fpfs_e1']=e1
#    ellDat['fpfs_e2']=e2
#    ellDat['fpfs_RE']=RE
#    ellDat['fpfs_flux']=flux
#    return ellDat

#class fpfsTestNoi():
#    _DefaultName = "fpfsTestNoi"
#    def __init__(self,ngrid,noiModel=None,noiFit=None):
#        self.ngrid  =   ngrid
#        # Preparing noise Model
#        self.noiModel=  noiModel
#        self.noiFit =   noiFit
#        self.rlim   =   int(ngrid//4)
#        return

#    def test(self,galData):
#        """
#        # test the noise subtraction

#        Parameters:
#        galData:    galaxy image [float array (list)]

#        Returns:
#        out :   FPFS moments
#
#
#        """
#        if isinstance(galData,np.ndarray):
#            # single galaxy
#            out =   self.__test(galData)
#            return out
#        elif isinstance(galData,list):
#            assert isinstance(galData[0],np.ndarray)
#            # list of galaxies
#            results=[]
#            for gal in galData:
#                _g=self.__test(gal)
#                results.append(_g)
#            out =   np.stack(results)
#            return out

#    def __test(self,data):
#        """
#        # test the noise subtraction

#        Parameters:
#        data:    image array (centroid does not matter)
#        """
#        assert len(data.shape)==2
#        galPow  =   imgutil.getFouPow(data)
#        if (self.noiFit is not None) or (self.noiModel is not None):
#            if self.noiModel is not None:
#                self.noiFit  =   imgutil.fitNoiPow(self.ngrid,galPow,self.noiModel,self.rlim)
#            galPow  =   galPow-self.noiFit
#        return galPow
