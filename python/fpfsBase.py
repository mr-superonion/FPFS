import numpy as np
import imgutil

class fpfsTask():
    _DefaultName = "fpfsBase"
    def __init__(self,psfData,noiModel=None):
        self.ngrid  =   psfData.shape[0]
        self.psfPow =   imgutil.getFouPow(psfData)
        # Preparing PSF model
        self.beta   =   0.85
        sigmaPsf    =   imgutil.getRnaive(self.psfPow)
        self.sigma  =   max(min(sigmaPsf*self.beta,4.),1.)
        self._prepareRlim()
        # Preparing shapelets
        self.chi    =   imgutil.shapelets2D(self.ngrid,4,self.sigma)
        # Preparing noise Model
        self.noiModel=  noiModel
        return

    def _prepareRlim(self):
        """
        # Get rlim, the area outside rlim is supressed by
        # the shaplet Gaussian kerenl
        """
        thres   =   1.e-4
        for dist in range(12,30):
            ave =  abs(np.exp(-dist**2./2./self.sigma**2.)/self.psfPow[self.ngrid//2+dist,self.ngrid//2])
            ave +=  abs(np.exp(-dist**2./2./self.sigma**2.)/self.psfPow[self.ngrid//2,self.ngrid//2+dist])
            ave =   ave/2.
            if ave<=thres:
                self.rlim=   dist
                break
        self._indX=np.arange(self.ngrid//2-self.rlim,self.ngrid//2+self.rlim+1)
        self._indY=self._indX[:,None]
        self._ind2D=np.ix_(self._indX,self._indX)
        return

    def deconvolvePow(self,arrayIn):
        """
        # Deconvolve the galaxy power with the PSF power
        Parameters:
        -------------
        arrayIn:    galaxy power, centerred at middle

        """
        arrayDec  =   np.zeros(arrayIn.shape,dtype=np.float64)
        arrayDec[self._ind2D]=arrayIn[self._ind2D]/self.psfPow[self._ind2D]
        return arrayDec

    def itransform(self,data):
        """
        # Deconvolve the galaxy power with the PSF power
        Parameters:
        -------------
        data:   data to transfer
        """

        # Moments
        M       =   np.sum(data[None,None,self._indY,self._indX]*self.chi[::2,:4:2,self._indY,self._indX],axis=(2,3))
        types   =   [('fpfs_M00','>f8'),\
                     ('fpfs_M20','>f8') ,('fpfs_M22c','>f8'),('fpfs_M22s','>f8'),\
                    ('fpfs_M40','>f8'),('fpfs_M42c','>f8'),('fpfs_M42s','>f8')]
        out     =   np.array((M.real[0,0],\
                              M.real[1,0],M.real[1,1],M.imag[1,1],\
                              M.real[2,0],M.real[2,1],M.imag[2,1]),dtype=types)
        return out

    def measure(self,galData):
        """
        # measure the FPFS moments

        Parameters:
        -----------
        galData:    galaxy image
        """
        if len(galData.shape)==2:
            # single galaxy
            out =   self._measure(galData)
            return out

        elif len(galData.shape)==3:
            # list of galaxies
            results=[]
            for gal in galData:
                _g=self._measure(gal)
                results.append(_g)
            out =   np.vstack(results)
            return out
        else:
            pass

    def _measure(self,arrayIn):
        """
        # measure the FPFS moments

        Parameters:
        -----------
        arrayIn:    image array (centroid does not matter)
        """

        galPow  =   imgutil.getFouPow(arrayIn)
        if self.noiModel is not None:
            minPow,noiFit   =   imgutil.removeNoiPow(self.ngrid,galData,self.noiModel,self.rlim)
        else:
            minPow          =   galPow
        decPow  =   self.deconvolvePow(minPow)
        mm      =   self.itransform(decPow)
        return mm
