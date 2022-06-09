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
import numpy as np
from . import imgutil
import numpy.lib.recfunctions as rfn

det_inds=[(2,2),(1,2),(3,2),(2,1),(2,3)]
"""list: a list of pixel index, where (2,2) is the centroid
"""
# _gsigma=3.*2*np.pi/64.
# """float: default gaussian smoothing kernel
# """

@numba.njit
def get_klim(psf_array,sigma):
    """
    Get klim, the region outside klim is supressed by the shaplet Gaussian
    kernel in FPFS shear estimation method; therefore we set values in this
    region to zeros.

    Parameters:
        psf_array (ndarray):    PSF's Fourier power or Fourier transform

    Returns:
        klim (float):           the limit radius
    """
    ngrid   =   psf_array.shape[0]
    thres   =   1.e-10
    klim    =   ngrid//2
    for dist in range(ngrid//5,ngrid//2-1):
        ave =  abs(np.exp(-dist**2./2./sigma**2.)\
                /psf_array[ngrid//2+dist,ngrid//2])
        ave +=  abs(np.exp(-dist**2./2./sigma**2.)\
                /psf_array[ngrid//2,ngrid//2+dist])
        ave =   ave/2.
        if ave<=thres:
            klim=   dist
            break
    return klim

class fpfsTask():
    """
    A class to measure FPFS shapelet mode estimation.

    Parameters:
        psfData (ndarray):
            an average PSF image used to initialize the task
        beta (float):
            FPFS scale parameter
        nnord (int):
            the highest order of Shapelets radial components [default: 4]
        noiModel (ndarray):
            Models to be used to fit noise power function using the pixels at
            large k for each galaxy (if you wish FPFS code to estiamte noise
            power). [default: None]
        noiFit (ndarray):
            Estimated noise power function (if you have already estimated noise
            power) [default: None]
        deubg (bool):
            Whether debug or not [default: False]
    """
    _DefaultName = "fpfsTask"
    def __init__(self,psfData,beta,nnord=4,noiModel=None,noiFit=None,debug=False):
        if not isinstance(beta,(float,int)):
            raise TypeError('Input beta should be float.')
        if beta>=1. or beta<=0.:
            raise ValueError('Input beta shoul be in range (0,1)')
        self.ngrid  =   psfData.shape[0]
        self._dk    =   2.*np.pi/self.ngrid

        # Preparing noise
        self.noise_correct=False
        if noiFit is not None:
            self.noise_correct=True
            if isinstance(noiFit,np.ndarray):
                assert noiFit.shape==psfData.shape, \
                'the input noise power should have the same shape with input psf image'
                self.noiFit  =  np.array(noiFit,dtype='<f8')
            elif isinstance(noiFit,float):
                self.noiFit  =  np.ones(psfData.shape,dtype='<f8')*noiFit*(self.ngrid)**2.
            else:
                raise TypeError('noiFit should be either np.ndarray or float')
        else:
            self.noiFit  =  0.
        if noiModel is not None:
            self.noise_correct=True
            self.noiModel=  np.array(noiModel,dtype='<f8')
        else:
            self.noiModel=  None
        self.noiFit0=   self.noiFit # we keep a copy of the initial noise Fourier power

        # Preparing PSF
        psfData     =   np.array(psfData,dtype='<f8')
        self.psfFou =   np.fft.fftshift(np.fft.fft2(psfData))
        self.psfFou0=   self.psfFou.copy() # we keep a copy of the initial PSF
        self.psfPow =   imgutil.getFouPow(psfData)
        self.psfPow0=   self.psfPow.copy() # we keep a copy of the initial PSF power

        # A few import scales
        # scale radius of PSF's Fourier transform (in units of pixel)
        sigmaPsf    =   imgutil.getRnaive(self.psfPow)*np.sqrt(2.)
        # shapelet scale
        sigmaPx=   max(min(sigmaPsf*beta,4.),1.)
        self.rlim   =   get_Rlim(self.psfPow,sigmaPx)
        self.sigmaF2=   sigmaPx*self._dk
        self.sigmaF =   self.sigmaF2*np.sqrt(2.)
        self._indX=np.arange(self.ngrid//2-self.rlim,self.ngrid//2+self.rlim+1)
        self._indY=self._indX[:,None]
        self._ind2D=np.ix_(self._indX,self._indX)

        # Preparing shapelet basis
        ## nm = m*(nnord+1)+n
        if nnord == 4:
            # This setup is for shear response only
            # Only uses M00, M20, M22 (real and img) and M40, M42
            self._indC  =   np.array([0,10,12,20,22])[:,None,None]
        elif nnord== 6:
            # This setup is able to derive kappa response as well as shear response
            # Only uses M00, M20, M22 (real and img), M40, M42(real and img), M60
            self._indC =    np.array([0, 14, 16, 28, 30, 42])[:,None,None]
        else:
            raise ValueError('only support for nnord= 4 or nnord=6')
        self.nnord=nnord
        chi   =    imgutil.shapelets2D(self.ngrid,nnord,self.sigmaF2)\
                        .reshape(((nnord+1)**2,self.ngrid,self.ngrid))
        self.prepare_ChiDeri(chi)
        self.prepare_ChiDet(chi)
        self.psfFou = np.fft.fftshift(np.fft.fft2(psfData))

        if self.noise_correct:
            logging.info('measurement error covariance will be calculated')
            self.prepare_ChiCov(chi)
            logging.info('Will correct noise bias')
        del chi

        # measure PSF properties
        self.psf_types=[
                    ('fpfs_psf_M00','<f8'),('fpfs_psf_M20','<f8'),\
                    ('fpfs_psf_M22c','<f8'),('fpfs_psf_M22s','<f8'),\
                    ('fpfs_psf_M40','<f8'),\
                    ('fpfs_psf_M42c','<f8'),('fpfs_psf_M42s','<f8')
                    ]
        mm_psf      =   self.itransform(self.psfFou,out_type='PSF')

        # others
        if debug:
            self.stackPow=np.zeros(psfData.shape,dtype='<f8')
        else:
            self.stackPow=None
        return

    def reset_psf(self):
        """
        reset psf power to the average PSF used to initialize the task
        """
        self.psfFou= self.psfFou0
        self.psfPow= np.conjugate(self.psfFou)*self.psfFou
        return

    def reset_noiFit(self):
        """
        reset noiFit to the one used to initialize the task
        """
        self.noiFit=self.noiFit0
        return

    def setRlim(self,klim):
        """
        set klim, the area outside klim is supressed by the shaplet Gaussian
        kerenl
        """
        self.klim_pix=   klim
        self._indX=np.arange(self.ngrid//2-self.klim_pix,self.ngrid//2+self.klim_pix+1)
        self._indY=self._indX[:,None]
        self._ind2D=np.ix_(self._indX,self._indX)
        return

    def prepare_ChiDeri(self,chi):
        """
        prepare the basis to estimate Derivatives (or equivalent moments)
        Parameters:
            chi (ndarray):  2D shapelet basis
        """
        out     =   []
        if self.nnord==4:
            out.append(chi.real[0]) #x00
            out.append(chi.real[1]) #x20
            out.append(chi.real[2]);out.append(chi.imag[2]) # x22c,s
            out.append(chi.real[3]) # x40
            out.append(chi.real[4]);out.append(chi.imag[4]) # x42c,s
            out     =   np.stack(out)
            self.deri_types=[
                        ('fpfs_M00','<f8'),('fpfs_M20','<f8'),\
                        ('fpfs_M22c','<f8'),('fpfs_M22s','<f8'),\
                        ('fpfs_M40','<f8'),\
                        ('fpfs_M42c','<f8'),('fpfs_M42s','<f8')
                        ]
        elif self.nnord==6:
            out.append(chi.real[0]) #x00
            out.append(chi.real[1]) #x20
            out.append(chi.real[2]);out.append(chi.imag[2]) # x22c,s
            out.append(chi.real[3]) # x40
            out.append(chi.real[4]);out.append(chi.imag[4]) # x42c,s
            out.append(chi.real[5]) # x60
            self.deri_types=[
                        ('fpfs_M00','<f8'),('fpfs_M20','<f8'),\
                        ('fpfs_M22c','<f8'),('fpfs_M22s','<f8'),\
                        ('fpfs_M40','<f8'),\
                        ('fpfs_M42c','<f8'),('fpfs_M42s','<f8'),\
                        ('fpfs_M60','<f8')
                        ]
        else:
            raise ValueError('only support for nnord=4 or nnord=6')
        assert len(out)==len(self.deri_types)
        self.chiD =   out
        del out
        return

    def prepare_ChiCov(self,chi):
        """
        prepare the basis to estimate covariance of measurement error

        Parameters:
            chi (ndarray):    2D shapelet basis
        """
        out     =   []
        # diagonal terms
        out.append(chi.real[0]*chi.real[0])#x00 x00
        out.append(chi.real[1]*chi.real[1])#x20 x20
        out.append(chi.real[2]*chi.real[2])#x22c x22c
        out.append(chi.imag[2]*chi.imag[2])#x22s x22s
        out.append(chi.real[3]*chi.real[3])#x40 x40
        # off-diagonal terms
        out.append(chi.real[0]*chi.real[1])#x00 x40
        out.append(chi.real[0]*chi.real[2])#x00 x22c
        out.append(chi.real[0]*chi.imag[2])#x00 x22s
        out.append(chi.real[0]*chi.real[3])#x00 x40
        out.append(chi.real[0]*chi.real[4])#x00 x42c
        out.append(chi.real[0]*chi.imag[4])#x00 x42s
        out.append(chi.real[2]*chi.real[4])#x22c x42c
        out.append(chi.imag[2]*chi.imag[4])#x22s x42s
        out     =   np.stack(out)
        self.cov_types= [('fpfs_N00N00','<f8'),('fpfs_N20N20','<f8'),\
                    ('fpfs_N22cN22c','<f8'),('fpfs_N22sN22s','<f8'),\
                    ('fpfs_N40N40','<f8'),\
                    ('fpfs_N00N20','<f8'),\
                    ('fpfs_N00N22c','<f8'),('fpfs_N00N22s','<f8'),\
                    ('fpfs_N00N40','<f8'),\
                    ('fpfs_N00N42c','<f8'),('fpfs_N00N42s','<f8'),\
                    ('fpfs_N22cN42c','<f8'),('fpfs_N22sN42s','<f8'),\
                    ]
        assert len(out)==len(self.cov_types)
        self.chiCov =   out
        del out
        return

    def prepare_ChiDet(self,chi):
        """
        prepare the basis to estimate covariance for detection
        Parameters:
            chi (ndarray):      2D shapelet basis
        """
        _chiU  =    chi[self._indC,self._indY,self._indX]
        gKer,(k2grid,k1grid)=imgutil.gauss_kernel(self.ngrid,self.ngrid,self.sigmaF,\
                        do_shift=True,return_grid=True)
        q1Ker  =    (k1grid**2.-k2grid**2.)/self.sigmaF**2.*gKer
        q2Ker  =    (2.*k1grid*k2grid)/self.sigmaF**2.*gKer
        d1Ker  =    (-1j*k1grid)*gKer
        d2Ker  =    (-1j*k2grid)*gKer
        out    =    []
        self.det_types= []
        for (j,i) in det_inds:
            y  =    j-2
            x  =    i-2
            r1 =    (q1Ker+x*d1Ker-y*d2Ker)*np.exp(1j*(k1grid*x+k2grid*y))
            r2 =    (q2Ker+y*d1Ker+x*d2Ker)*np.exp(1j*(k1grid*x+k2grid*y))
            r1 =    r1[self._indY,self._indX]/self.ngrid**2.
            r2 =    r2[self._indY,self._indX]/self.ngrid**2.
            out.append(chi.real[0]*r1) #x00*phi;1
            out.append(chi.real[0]*r2) #x00*phi;2
            out.append(chi.real[2]*r1) #x22c*phi;1
            out.append(chi.imag[2]*r2) #x22s*phi;2
            self.det_types.append(('pdet_N00F%d%dr1'  %(j,i),'<f8'))
            self.det_types.append(('pdet_N00F%d%dr2'  %(j,i),'<f8'))
            self.det_types.append(('pdet_N22cF%d%dr1' %(j,i),'<f8'))
            self.det_types.append(('pdet_N22sF%d%dr2' %(j,i),'<f8'))
        out     =   np.stack(out)
        assert len(out)==len(self.det_types)
        self.chiDet =   out
        del out
        return

    def deconvolve(self,data,prder=1.,frder=1.):
        """
        Deconvolve input data with the PSF or PSF power

        Parameters:
            data (ndarray):
                galaxy power or galaxy Fourier transfer (ngrid//2,ngrid//2) is origin
            prder (float):
                deconvlove order of PSF FT power
            frder (float):
                deconvlove order of PSF FT

        Returns:
            out (ndarray):
                Deconvolved galaxy power (truncated at klim)
        """
        out  =   np.zeros(data.shape,dtype=np.complex64)
        out[self._ind2D]=data[self._ind2D]\
                /self.psfPow[self._ind2D]**prder\
                /self.psfFou[self._ind2D]**frder
        return out

    def itransform(self,data,out_type='Deri'):
        """
        Project image onto shapelet basis vectors

        Parameters:
            data (ndarray): image to transfer
            out_type (str): transform type ('Deri', 'Cov', or 'Det')

        Returns:
            out (ndarray):  projection in shapelet space
        """

        if out_type=='Deri':
            # Derivatives/Moments
            _       =   np.sum(data[None,self._indY,self._indX]\
                        *self.chiD,axis=(1,2)).real
            out     =   np.array(tuple(_),dtype=self.deri_types)
        elif out_type=='Cov':
            # covariance of moments
            _       =   np.sum(data[None,self._indY,self._indX]\
                        *self.chiCov,axis=(1,2)).real
            out     =   np.array(tuple(_),dtype=self.cov_types)
        elif out_type=='Det':
            # covariance of pixels
            _       =   np.sum(data[None,self._indY,self._indX]*\
                        self.chiDet,axis=(1,2)).real
            out     =   np.array(tuple(_),dtype=self.det_types)
        elif out_type=='PSF':
            # Derivatives/Moments
            _       =   np.sum(data[None,self._indY,self._indX]\
                        *self.chiD,axis=(1,2)).real
            out     =   np.array(tuple(_),dtype=self.psf_types)
        else:
            raise ValueError("out_type can only be 'Deri', 'Cov' or 'Det',\
                    but the input is '%s'" %out_type)
        return out

    def measure(self,galData,psfFou=None,noiFit=None):
        """
        Measure the FPFS moments

        Parameters:
            galData (ndarray|list):     galaxy image
            psfFou (ndarray):           PSF's Fourier transform
            noiFit (ndarray):           noise Fourier power function

        Returns:
            out (ndarray):              FPFS moments
        """
        if psfFou is not None:
            self.psfFou= psfFou
            self.psfPow= (np.conjugate(psfFou)*psfFou).real
        if noiFit is not None:
            self.noiFit=noiFit
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
            data (ndarray):     galaxy image array (centroid does not matter)

        Returns:
            mm (ndarray):       FPFS moments
        """
        nn          =   None
        dd          =   None
        galPow      =   imgutil.getFouPow(data)
        if self.noiModel is not None:
            noiFit  =   imgutil.fitNoiPow(self.ngrid,galPow,self.noiModel,self.rlim)
        else:
            noiFit  =   self.noiFit
        if self.noise_correct:
            galPow  =   galPow-noiFit*1.
            epcor   =   2.*(noiFit*noiFit+2.*galPow*noiFit)
            decEP   =   self.deconvolvePow(epcor,order=2.)
            nn      =   self.itransformCov(decEP)
        decPow      =   self.deconvolvePow(galPow,order=1.)
        mm          =   self.itransform(decPow)
        _tmp        =   np.fft.fftshift(np.fft.fft2(data))
        _tmp        =   2.*_tmp*noiFit
        decDet      =   self.deconvolve2(_tmp,prder=1.,frder=1)
        dd          =   self.itransformDet(decDet)
        if nn is not None:
            mm      =   rfn.merge_arrays([mm,nn,dd],flatten=True,usemask=False)
        else:
            mm      =   rfn.merge_arrays([mm,nn,dd],flatten=True,usemask=False)

        if self.stackPow is not None:
            galPow  =   imgutil.getFouPow(data)
            self.stackPow+=galPow
            return np.empty(1)
        galFou  =   np.fft.fftshift(np.fft.fft2(data))
        decG    =   self.deconvolve(galFou,prder=0.,frder=1)
        mm      =   self.itransform(decG,out_type='Deri')
        del decG

        nn          =   None # photon noise covariance array
        dd          =   None # detection covariance array

        if self.noise_correct:
            # do noise covariance estimation
            if self.noiModel is not None:
                # fit the noise power from the galaxy power
                galPow  =   imgutil.getFouPow(data)
                noiFit  =   imgutil.fitNoiPow(self.ngrid,galPow,self.noiModel,self.klim_pix)
                del galPow
            else:
                # use the input noise power
                noiFit  =   self.noiFit
            decNp   =   self.deconvolve(noiFit,prder=1,frder=0)
            nn      =   self.itransform(decNp,out_type='Cov')
            dd      =   self.itransform(decNp,out_type='Det')
            del decNp
            mm      =   rfn.merge_arrays([mm,nn,dd],flatten=True,usemask=False)
        return mm

def fpfsM2E(moments,dets=None,const=1.,noirev=False,flipsign=False):
    """
    Estimate FPFS ellipticities from fpfs moments

    Parameters:
        moments (ndarray):
            input FPFS moments
        dets (ndarray):
            detection array [default: None]
        const (float):
            the weight constant [default:1]
        noirev (bool):
            revise the second-order noise bias? [default: False]
        flipsign (bool):
            flip the sign of response? [default: False] (if you are using the
            convention in Li et. al (2018), set it to True)

    Returns:
        out (ndarray):
            an array of (FPFS ellipticities, FPFS ellipticity response, FPFS
            flux ratio, and FPFS selection response)
    """
    # ellipticity, q-ellipticity, sizes, e^2, eq
    types   =   [('fpfs_e1','<f8'), ('fpfs_e2','<f8'),  ('fpfs_RE','<f8'), \
                ('fpfs_q1','<f8'), ('fpfs_q2','<f8'), \
                ('fpfs_s0','<f8') , ('fpfs_s2','<f8') , ('fpfs_s4','<f8'),\
                ('fpfs_ee','<f8'), ('fpfs_eq','<f8'),\
                ('fpfs_RS0','<f8'),('fpfs_RS2','<f8')]
    # FPFS shape weight's inverse
    _w      =   moments['fpfs_M00']+const
    # FPFS ellipticity
    e1      =   moments['fpfs_M22c']/_w
    e2      =   moments['fpfs_M22s']/_w
    q1      =   moments['fpfs_M42c']/_w #New
    q2      =   moments['fpfs_M42s']/_w
    e1sq    =   e1*e1
    e2sq    =   e2*e2
    e1q1    =   e1*q1   #New
    e2q2    =   e2*q2
    # FPFS flux ratio
    s0      =   moments['fpfs_M00']/_w
    s2      =   moments['fpfs_M20']/_w
    s4      =   moments['fpfs_M40']/_w
    # FPFS sel respose
    e1sqS0  =   e1sq*s0
    e2sqS0  =   e2sq*s0
    e1sqS2  =   e1sq*s2 #New
    e2sqS2  =   e2sq*s2

    if dets is not None:
        # shear response for detection
        dDict=  {}
        for (j,i) in det_inds:
            types.append(('fpfs_e1v%d%dr1'%(j,i),'<f8'))
            types.append(('fpfs_e2v%d%dr2'%(j,i),'<f8'))
            dDict['fpfs_e1v%d%dr1' %(j,i)]=e1*dets['pdet_v%d%dr1' %(j,i)]
            dDict['fpfs_e2v%d%dr2' %(j,i)]=e2*dets['pdet_v%d%dr2' %(j,i)]
    else:
        dDict = None

    if noirev:
        # do second-order noise bias correction
        assert 'fpfs_N00N00' in moments.dtype.names
        assert 'fpfs_N00N22c' in moments.dtype.names
        assert 'fpfs_N00N22s' in moments.dtype.names
        ratio=  moments['fpfs_N00N00']/_w**2.
        if dDict is not None:
            # correction detection shear response for noise bias
            for (j,i) in det_inds:
                dDict['fpfs_e1v%d%dr1'%(j,i)]=dDict['fpfs_e1v%d%dr1'%(j,i)]\
                    -1.*dets['pdet_N22cV%d%dr1'%(j,i)]/_w\
                    +1.*e1*dets['pdet_N00V%d%dr1'%(j,i)]/_w\
                    +1.*moments['fpfs_N00N22c']/_w**2.*dets['pdet_v%d%dr1'%(j,i)]
                dDict['fpfs_e1v%d%dr1'%(j,i)]=dDict['fpfs_e1v%d%dr1'%(j,i)]/(1+ratio)
                dDict['fpfs_e2v%d%dr2'%(j,i)]=dDict['fpfs_e2v%d%dr2'%(j,i)]\
                    -1.*dets['pdet_N22sV%d%dr2'%(j,i)]/_w\
                    +1.*e2*dets['pdet_N00V%d%dr2'%(j,i)]/_w\
                    +1.*moments['fpfs_N00N22s']/_w**2.*dets['pdet_v%d%dr2'%(j,i)]
                dDict['fpfs_e2v%d%dr2'%(j,i)]=dDict['fpfs_e2v%d%dr2'%(j,i)]/(1+ratio)

        # equation (22) of Li et.al (2022)
        e1  =   (e1+moments['fpfs_N00N22c']\
                /_w**2.)/(1+ratio)
        e2  =   (e2+moments['fpfs_N00N22s']\
                /_w**2.)/(1+ratio)
        # e1^2, e2^2
        e1sq=   (e1sq-moments['fpfs_N22cN22c']/_w**2.\
                +4.*e1*moments['fpfs_N00N22c']/_w**2.)\
                /(1.+3*ratio)
        e2sq=   (e2sq-moments['fpfs_N22sN22s']/_w**2.\
                +4.*e2*moments['fpfs_N00N22s']/_w**2.)\
                /(1.+3*ratio)
        # noise bias correction for flux ratio
        s0  =   (s0+moments['fpfs_N00N00']\
                /_w**2.)/(1+ratio)
        s2  =   (s2+moments['fpfs_N00N20']\
                /_w**2.)/(1+ratio)
        s4  =   (s4+moments['fpfs_N00N40']\
                /_w**2.)/(1+ratio)
        # correction for selection (by s0) shear response
        # e1^2s0, e2^2s0
        e1sqS0= (e1sqS0+3.*e1sq*moments['fpfs_N00N00']/_w**2.\
                -s0*moments['fpfs_N22cN22c']/_w**2.)/(1+6.*ratio)
        e2sqS0= (e2sqS0+3.*e2sq*moments['fpfs_N00N00']/_w**2.\
                -s0*moments['fpfs_N22sN22s']/_w**2.)/(1+6.*ratio)
        # shear response of resolution selection
        e1sqS2= (e1sqS2+3.*e1sq*moments['fpfs_N00N20']/_w**2.\
                -s2*moments['fpfs_N22cN22c']/_w**2.)/(1+6.*ratio)
        e2sqS2= (e2sqS2+3.*e2sq*moments['fpfs_N00N20']/_w**2.\
                -s2*moments['fpfs_N22sN22s']/_w**2.)/(1+6.*ratio)
        e1q1=   (e1q1-moments['fpfs_N22cN42c']/_w**2.)/(1.+3*ratio)
        e2q2=   (e2q2-moments['fpfs_N22sN42s']/_w**2.)/(1.+3*ratio)

    # make the output ndarray
    out  =   np.array(np.zeros(moments.size),dtype=types)
    if dDict is not None:
        for (j,i) in det_inds:
            out['fpfs_e1v%d%dr1'%(j,i)] = dDict['fpfs_e1v%d%dr1'%(j,i)]
            out['fpfs_e2v%d%dr2'%(j,i)] = dDict['fpfs_e2v%d%dr2'%(j,i)]
        del dDict

    # spin-2 properties
    out['fpfs_e1']  =   e1      # ellipticity
    out['fpfs_e2']  =   e2
    out['fpfs_q1']  =   q1      # epsilon
    out['fpfs_q2']  =   q2
    del e1,e2,q1,q2

    # spin-0 properties
    out['fpfs_s0']  =   s0      # flux
    out['fpfs_s2']  =   s2      # size2
    out['fpfs_s4']  =   s4      # size4
    eSq  =  e1sq+e2sq       # e_1^2+e_2^2
    RE   =  1./np.sqrt(2.)*(s0-s4+eSq) # shear response of e
    # Li et. al (2018) has a minus sign difference
    out['fpfs_RE']  =   RE*(2.*flipsign.real-1.)
    out['fpfs_ee']  =   eSq     # shape noise
    del s0,s2,s4,RE,e1sq,e2sq

    # for selection bias correction
    eSqS0=  e1sqS0+e2sqS0   # (e_1^2+e_2^2)s_0 for selection bias correcton
    del e1sqS0,e2sqS0
    eqeq =  e1q1+e2q2
    eSqS2=  e1sqS2+e2sqS2   # (e_1^2+e_2^2)s_2 for selection bias correcton
    del e1q1,e2q2
    out['fpfs_RS0'] =   (eSq-eSqS0)/np.sqrt(2.) # selection bias correction for flux
    out['fpfs_RS2'] =   (eqeq*np.sqrt(3.)-eSqS2)/np.sqrt(2.) # selection bias correction for resolution
    return out

def fpfsM2Err(moments,const=1.):
    """
    Estimate FPFS measurement errors from the fpfs
    moments and the moments covariances

    Parameters:
        moments (ndarray):
            input FPFS moments
        const (float):
            the weighting Constant
        mcalib (ndarray):
            multiplicative bias

    Returns:
        errDat (ndarray):
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

    types   =   [('fpfs_e1Err','<f8'),('fpfs_e2Err','<f8'),('fpfs_s0Err','<f8'),\
                    ('fpfs_e1s0Cov','<f8'),('fpfs_e2s0Cov','<f8')]
    errDat  =   np.array(np.zeros(len(moments)),dtype=types)
    errDat['fpfs_e1Err']   =   e1Err
    errDat['fpfs_e2Err']   =   e2Err
    errDat['fpfs_s0Err']   =   s0Err
    errDat['fpfs_e1s0Cov'] =   e1s0Cov
    errDat['fpfs_e2s0Cov'] =   e2s0Cov
    return errDat
