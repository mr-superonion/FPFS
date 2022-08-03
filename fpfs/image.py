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
import numpy.lib.recfunctions as rfn

from . import imgutil

def detect_sources(imgData,psfData,gsigma,thres=0.04,thres2=-0.01,klim=-1.,pixel_scale=1.):
    """Returns the coordinates of detected sources
    Args:
        imgData (ndarray):      observed image
        psfData (ndarray):      PSF image (well centered)
        gsigma (float):         sigma of the Gaussian smoothing kernel in *Fourier* space
        thres (float):          detection threshold
        thres2 (float):         peak identification difference threshold
        klim (float):           limiting wave number in Fourier space
        pixel_scale (float):    pixel scale in arcsec
    Returns:
        coords (ndarray):       peak values and the shear responses
    """

    assert imgData.shape==psfData.shape, 'image and PSF should have the same\
            shape. Please do padding before using this function.'
    ny,nx   =   psfData.shape
    psfF    =   np.fft.rfft2(np.fft.ifftshift(psfData))
    gKer,_  =   imgutil.gauss_kernel(ny,nx,gsigma,return_grid=True,use_rfft=True)

    # convolved images
    imgF    =   np.fft.rfft2(imgData)/psfF*gKer
    if klim>0.:
        # apply a truncation in Fourier space
        nxklim  =   int(klim*nx/np.pi/2.+0.5)
        nyklim  =   int(klim*ny/np.pi/2.+0.5)
        imgF[nyklim+1:-nyklim,:] = 0.
        imgF[:,nxklim+1:] = 0.
    else:
        # no truncation in Fourier space
        pass
    del psfF,psfData
    imgCov  =   np.fft.irfft2(imgF,(ny,nx))
    # the coordinates is not given, so we do another detection
    if not isinstance(thres,(int, float)):
        raise ValueError('thres must be float, but now got %s' %type(thres))
    if not isinstance(thres2,(int, float)):
        raise ValueError('thres2 must be float, but now got %s' %type(thres))
    coords  =   imgutil.find_peaks(imgCov,thres,thres2)
    return coords

@numba.njit
def get_klim(psf_array,sigma,thres=1e-20):
    """
    Get klim, the region outside klim is supressed by the shaplet Gaussian
    kernel in FPFS shear estimation method; therefore we set values in this
    region to zeros.
    Args:
        psf_array (ndarray):    PSF's Fourier power or Fourier transform
        sigma (float):          one sigma of Gaussian Fourier power
        thres (float):          the threshold for a tuncation on Gaussian (default: 1e-20)
    Returns:
        klim (float):           the limit radius
    """
    ngrid   =   psf_array.shape[0]
    klim    =   ngrid//2-1
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

class measure_source():
    """
    A class to measure FPFS shapelet mode estimation.

    Args:
        psfData (ndarray):  an average PSF image used to initialize the task
        beta (float):       FPFS scale parameter
        nnord (int):        the highest order of Shapelets radial components [default: 4]
        noiModel (ndarray): Models to be used to fit noise power function using the pixels at
                            large k for each galaxy (if you wish FPFS code to estiamte noise
                            power). [default: None]
        noiFit (ndarray):   Estimated noise power function (if you have already estimated noise
                            power) [default: None]
        deubg (bool):       Whether debug or not [default: False]
        pix_scale (float):  pixel scale in arcsec [default: 0.168 arcsec (HSC)]
    """
    _DefaultName = "measure_source"
    def __init__(self,psfData,sigma_arcsec,nnord=4,noiModel=None,noiFit=None,debug=False,pix_scale=0.168):
        if sigma_arcsec<=0. or sigma_arcsec>5.:
            raise ValueError('sigma_arcsec should be positive and less than 5 arcsec')
        self.ngrid  =   psfData.shape[0]
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
        self.pix_scale= pix_scale # this is only used to normalize basis functions
        self._dk    =   2.*np.pi/self.ngrid # assuming pixel scale is 1
        # beta
        # # scale radius of PSF's Fourier transform (in units of dk)
        # sigmaPsf    =   imgutil.getRnaive(self.psfPow)*np.sqrt(2.)
        # # shapelet scale
        # sigma_pix   =   max(min(sigmaPsf*beta,6.),1.) # in units of dk
        # self.sigmaF =   sigma_pix*self._dk      # assume pixel scale is 1
        # sigma_arcsec  =   1./self.sigmaF*self.pix_scale

        self.sigmaF =   self.pix_scale/sigma_arcsec
        sigma_pix   =   self.sigmaF/self._dk
        logging.info('Gaussian in configuration space: sigma= %.4f arcsec' %(sigma_arcsec))
        # effective nyquest wave number
        self.klim_pix=  get_klim(self.psfPow,sigma_pix/np.sqrt(2.)) # in pixel units
        self.klim   =   self.klim_pix*self._dk  # assume pixel scale is 1
        # index selectors
        self._indX=np.arange(self.ngrid//2-self.klim_pix,self.ngrid//2+self.klim_pix+1)
        self._indY=self._indX[:,None]
        self._ind2D=np.ix_(self._indX,self._indX)

        # Preparing shapelet basis
        ## nm = m*(nnord+1)+n
        if nnord == 4:
            # This setup is for shear response only
            # Only uses M00, M20, M22 (real and img) and M40, M42
            self._indM  =   np.array([0,10,12,20,22])[:,None,None]
            self._nameM =   ['M00','M20','M22','M40','M42']
        elif nnord== 6:
            # This setup is able to derive kappa response and shear response
            # Only uses M00, M20, M22 (real and img), M40, M42(real and img), M60
            self._indM =    np.array([0, 14, 16, 28, 30, 42])[:,None,None]
            self._nameM =   ['M00','M20','M22','M40','M42','M60']
        else:
            raise ValueError('only support for nnord= 4 or nnord=6, but your input\
                    is nnord=%d' %nnord)
        self.nnord=nnord
        chi   = imgutil.shapelets2D(self.ngrid,nnord,self.sigmaF)\
                    .reshape(((nnord+1)**2,self.ngrid,self.ngrid))[self._indM,self._indY,self._indX]
        psi   = imgutil.detlets2D(self.ngrid,self.sigmaF)[:,:,self._indY,self._indX]
        self.prepare_Chi(chi)
        self.prepare_Psi(psi)
        if self.noise_correct:
            logging.info('measurement error covariance will be calculated')
            self.prepare_ChiCov(chi)
            self.prepare_detCov(chi,psi)
        else:
            logging.info('measurement covariance will not be calculated')
        del chi

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

    def prepare_Chi(self,chi):
        """
        prepare the basis to estimate shapelet modes
        Args:
            chi (ndarray):  2D shapelet basis
        """
        out     =   []
        if self.nnord==4:
            out.append(chi.real[0]) # x00
            out.append(chi.real[1]) # x20
            out.append(chi.real[2]);out.append(chi.imag[2]) # x22c,s
            out.append(chi.real[3]) # x40
            out.append(chi.real[4]);out.append(chi.imag[4]) # x42c,s
            out     =   np.stack(out)
            self.chi_types=[
                        ('fpfs_M00','<f8'),('fpfs_M20','<f8'),\
                        ('fpfs_M22c','<f8'),('fpfs_M22s','<f8'),\
                        ('fpfs_M40','<f8'),\
                        ('fpfs_M42c','<f8'),('fpfs_M42s','<f8')\
                        ]
        elif self.nnord==6:
            out.append(chi.real[0]) # x00
            out.append(chi.real[1]) # x20
            out.append(chi.real[2]);out.append(chi.imag[2]) # x22c,s
            out.append(chi.real[3]) # x40
            out.append(chi.real[4]);out.append(chi.imag[4]) # x42c,s
            out.append(chi.real[5]) # x60
            self.chi_types=[
                        ('fpfs_M00','<f8'),('fpfs_M20','<f8'),\
                        ('fpfs_M22c','<f8'),('fpfs_M22s','<f8'),\
                        ('fpfs_M40','<f8'),\
                        ('fpfs_M42c','<f8'),('fpfs_M42s','<f8'),\
                        ('fpfs_M60','<f8')
                        ]
        else:
            raise ValueError('only support for nnord=4 or nnord=6')
        assert len(out)==len(self.chi_types)
        self.Chi =   out
        del out
        return

    def prepare_Psi(self,psi):
        """
        prepare the basis to estimate Chivatives (or equivalent moments)
        Args:
            psi (ndarray):  2D shapelet basis
        """
        self.psi_types=[]
        out     =   []
        for _ in range(8):
            out.append(psi[_,0]) #psi
            out.append(psi[_,1]) #psi;1
            out.append(psi[_,2]) #psi;2
            self.psi_types.append(('fpfs_v%d'  %_,'<f8'))
            self.psi_types.append(('fpfs_v%dr1'%_,'<f8'))
            self.psi_types.append(('fpfs_v%dr2'%_,'<f8'))
        out     =   np.stack(out)
        assert len(out)==len(self.psi_types)
        self.Psi=   out
        del out
        return

    def prepare_ChiCov(self,chi):
        """
        prepare the basis to estimate covariance of measurement error
        Args:
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
        #
        out.append(chi.real[0]*chi.real[1])#x00 x20
        out.append(chi.real[0]*chi.real[2])#x00 x22c
        out.append(chi.real[0]*chi.imag[2])#x00 x22s
        out.append(chi.real[0]*chi.real[3])#x00 x40
        out.append(chi.real[0]*chi.real[4])#x00 x42c
        out.append(chi.real[0]*chi.imag[4])#x00 x42s
        #
        out.append(chi.real[1]*chi.real[2])#x20 x22c
        out.append(chi.real[1]*chi.imag[2])#x20 x22s
        out.append(chi.real[1]*chi.real[3])#x20 x40
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
                    ('fpfs_N20N22c','<f8'),('fpfs_N20N22s','<f8'),\
                    ('fpfs_N20N40','<f8'),\
                    ('fpfs_N22cN42c','<f8'),('fpfs_N22sN42s','<f8'),\
                    ]
        assert len(out)==len(self.cov_types)
        self.chiCov =   out
        del out
        return

    def prepare_detCov(self,chi,psi):
        """
        prepare the basis to estimate covariance for detection
        Args:
            chi (ndarray):      2D shapelet basis
            psi (ndarray):      2D pixel basis
        """
        # get the Gaussian scale in Fourier space
        out    =    []
        self.det_types= []
        for _ in range(8):
            out.append(chi.real[0]*psi[_,0]) #x00*psi
            out.append(chi.real[0]*psi[_,1]) #x00*psi;1
            out.append(chi.real[0]*psi[_,2]) #x00*psi;2
            out.append(chi.real[2]*psi[_,0]) #x22c*psi
            out.append(chi.imag[2]*psi[_,0]) #x22s*psi
            out.append(chi.real[2]*psi[_,1]) #x22c*psi;1
            out.append(chi.imag[2]*psi[_,2]) #x22s*psi;2
            out.append(chi.real[3]*psi[_,0]) #x40*psi
            self.det_types.append(('fpfs_N00V%d'   %_,'<f8'))
            self.det_types.append(('fpfs_N00V%dr1' %_,'<f8'))
            self.det_types.append(('fpfs_N00V%dr2' %_,'<f8'))
            self.det_types.append(('fpfs_N22cV%d'  %_,'<f8'))
            self.det_types.append(('fpfs_N22sV%d'  %_,'<f8'))
            self.det_types.append(('fpfs_N22cV%dr1'%_,'<f8'))
            self.det_types.append(('fpfs_N22sV%dr2'%_,'<f8'))
            self.det_types.append(('fpfs_N40V%d'   %_,'<f8'))
        out     =   np.stack(out)
        assert len(out)==len(self.det_types)
        self.detCov =   out
        del out
        return

    def deconvolve(self,data,prder=1.,frder=1.):
        """
        Deconvolve input data with the PSF or PSF power
        Args:
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

    def itransform(self,data,out_type='Chi'):
        """
        Project image onto shapelet basis vectors
        Args:
            data (ndarray): image to transfer
            out_type (str): transform type ('Chi', 'Psi', 'Cov', or 'detCov')
        Returns:
            out (ndarray):  projection in shapelet space
        """

        # Here we divide by self.pix_scale**2. for modes since pixel value are
        # flux in pixel (in unit of nano Jy for HSC). After dividing pix_scale**2.,
        # in units of (nano Jy/ arcsec^2), dk^2 has unit (1/ arcsec^2)
        # Correspondingly, covariances are divided by self.pix_scale**4.
        if out_type=='Chi':
            # Chivatives/Moments
            _       =   np.sum(data[None,self._indY,self._indX]\
                        *self.Chi,axis=(1,2)).real/self.pix_scale**2.
            out     =   np.array(tuple(_),dtype=self.chi_types)
        elif out_type=='Psi':
            # Chivatives/Moments
            _       =   np.sum(data[None,self._indY,self._indX]\
                        *self.Psi,axis=(1,2)).real/self.pix_scale**2.
            out     =   np.array(tuple(_),dtype=self.psi_types)
        elif out_type=='Cov':
            # covariance of moments
            _       =   np.sum(data[None,self._indY,self._indX]\
                        *self.chiCov,axis=(1,2)).real/self.pix_scale**4.
            out     =   np.array(tuple(_),dtype=self.cov_types)
        elif out_type=='detCov':
            # covariance of pixels
            _       =   np.sum(data[None,self._indY,self._indX]*\
                        self.detCov,axis=(1,2)).real/self.pix_scale**4.
            out     =   np.array(tuple(_),dtype=self.det_types)
        else:
            raise ValueError("out_type can only be 'Chi', 'Cov' or 'Det',\
                    but the input is '%s'" %out_type)
        return out

    def measure(self,galData,psfFou=None,noiFit=None):
        """
        Measure the FPFS moments

        Args:
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
        Args:
            data (ndarray):     galaxy image array (centroid does not matter)
        Returns:
            mm (ndarray):       FPFS moments
        """
        if self.stackPow is not None:
            galPow  =   imgutil.getFouPow(data)
            self.stackPow+=galPow
            return np.empty(1)
        galFou  =   np.fft.fftshift(np.fft.fft2(data))
        decG    =   self.deconvolve(galFou,prder=0.,frder=1)
        mm      =   self.itransform(decG,out_type='Chi')
        mp      =   self.itransform(decG,out_type='Psi')
        mm      =   rfn.merge_arrays([mm,mp],flatten=True,usemask=False)
        del decG,mp

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
            dd      =   self.itransform(decNp,out_type='detCov')
            del decNp
            mm      =   rfn.merge_arrays([mm,nn,dd],flatten=True,usemask=False)
        return mm

class test_noise():
    _DefaultName = "fpfsTestNoi"
    def __init__(self,ngrid,noiModel=None,noiFit=None):
        self.ngrid  =   ngrid
        # Preparing noise Model
        self.noiModel=  noiModel
        self.noiFit =   noiFit
        self.rlim   =   int(ngrid//4)
        return

    def test(self,galData):
        """
        # test the noise subtraction

        Parameters:
        galData:    galaxy image [float array (list)]

        Returns:
        out :   FPFS moments


        """
        if isinstance(galData,np.ndarray):
            # single galaxy
            out =   self.__test(galData)
            return out
        elif isinstance(galData,list):
            assert isinstance(galData[0],np.ndarray)
            # list of galaxies
            results=[]
            for gal in galData:
                _g=self.__test(gal)
                results.append(_g)
            out =   np.stack(results)
            return out

    def __test(self,data):
        """
        # test the noise subtraction

        Parameters:
        data:    image array (centroid does not matter)
        """
        assert len(data.shape)==2
        galPow  =   imgutil.getFouPow(data)
        if (self.noiFit is not None) or (self.noiModel is not None):
            if self.noiModel is not None:
                self.noiFit  =   imgutil.fitNoiPow(self.ngrid,galPow,self.noiModel,self.rlim)
            galPow  =   galPow-self.noiFit
        return galPow
