import imgutil
import numpy as np
import numpy.lib.recfunctions as rfn

class fpfsTask():
    _DefaultName = "fpfsBase"
    def __init__(self,psfData,noiModel=None,noiSub=None):
        if noiSub is not None:
            print('testing')
        self.ngrid  =   psfData.shape[0]
        self.psfPow =   imgutil.getFouPow(psfData)
        # Preparing PSF model
        self.beta   =   0.85
        sigmaPsf    =   imgutil.getRnaive(self.psfPow)
        self.sigma  =   max(min(sigmaPsf*self.beta,4.),1.)
        self.__prepareRlim()
        # Preparing shapelets (reshaped)
        self.chi    =   imgutil.shapelets2D(self.ngrid,4,self.sigma).reshape((25,self.ngrid,self.ngrid))
        self._indC  =   np.array([0,12,20])[:,None,None]
        # Preparing noise Model
        self.noiModel=  noiModel
        return

    def __prepareRlim(self):
        """
        # Get rlim, the area outside rlim is supressed by
        # the shaplet Gaussian kerenl
        # (part of __init__)
        """
        thres   =   1.e-4
        for dist in range(self.ngrid//5,int(self.ngrid*0.45)):
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

    def deconvolvePow(self,arrayIn,order=1.):
        """
        # Deconvolve the galaxy power with the PSF power
        Parameters:
        -------------
        arrayIn :   galaxy power, centerred at middle

        Returns :
        -------------
        out     :   Deconvolved galaxy power (truncated at rlim)
        """
        out  =   np.zeros(arrayIn.shape,dtype=np.float64)
        out[self._ind2D]=arrayIn[self._ind2D]/self.psfPow[self._ind2D]**order
        return out

    def itransformCov(self,data):
        """
        # project data onto shapelet basis
        Parameters:
        -------------
        data:   data to transfer

        Returns:
        -------------
        out :   projection in shapelet space
        """

        # Moments
        _chiU   =   self.chi[self._indC,self._indY,self._indX]
        chiUList=   []
        chiUList.append(_chiU.real[0]*_chiU.real[0])
        chiUList.append(_chiU.real[1]*_chiU.real[1])
        chiUList.append(_chiU.imag[1]*_chiU.imag[1])
        chiUList.append(_chiU.real[2]*_chiU.real[2])
        chiUList.append(_chiU.real[0]*_chiU.real[1])
        chiUList.append(_chiU.real[0]*_chiU.imag[1])
        chiUList.append(_chiU.real[0]*_chiU.real[2])
        chiUList=   np.stack(chiUList)
        dataU   =   data[None,self._indY,self._indX]

        N       =   2.*np.sum(chiUList*dataU,axis=(1,2))
        types   =   [('fpfs_N00N00','>f8'),\
                    ('fpfs_N22cN22c','>f8'),('fpfs_N22sN22s','>f8'),\
                    ('fpfs_N40N40','>f8'),\
                    ('fpfs_N00N22c','>f8'),('fpfs_N00N22s','>f8'),\
                    ('fpfs_N00N40','>f8')\
                    ]
        out     =   np.array(tuple(N),dtype=types)
        return out

    def itransform(self,data):
        """
        # Project the (PP+PD)/P^2 to get the covariance
        Parameters:
        -------------
        data:   data to transfer

        Returns:
        -------------
        out :   projection in shapelet space
        """

        # Moments
        M       =   np.sum(data[None,self._indY,self._indX]*self.chi[self._indC,self._indY,self._indX],axis=(1,2))
        types   =   [('fpfs_M00','>f8'),\
                    ('fpfs_M22c','>f8'),('fpfs_M22s','>f8'),\
                    ('fpfs_M40','>f8')\
                    ]
        # types =   [('fpfs_M00','>f8'),\
        #           ('fpfs_M20','>f8') ,('fpfs_M22c','>f8'),('fpfs_M22s','>f8'),\
        #           ('fpfs_M40','>f8'),('fpfs_M42c','>f8'),('fpfs_M42s','>f8')]
        out     =   np.array((M.real[0],\
                        M.real[1],M.imag[1],\
                        M.real[2]),dtype=types)
        return out

    def measure(self,galData):
        """
        # measure the FPFS moments

        Parameters:
        -----------
        galData:    galaxy image

        Returns:
        -------------
        out :   FPFS moments
        """
        if len(galData.shape)==2:
            # single galaxy
            out =   self.__measure(galData)
            return out

        elif len(galData.shape)==3:
            # list of galaxies
            results=[]
            for gal in galData:
                _g=self.__measure(gal)
                results.append(_g)
            out =   np.vstack(results)
            return out
        else:
            pass

    def __measure(self,arrayIn):
        """
        # measure the FPFS moments

        Parameters:
        -----------
        arrayIn:    image array (centroid does not matter)
        """

        galPow  =   imgutil.getFouPow(arrayIn)
        if self.noiModel is not None:
            noiFit  =   imgutil.fitNoiPow(self.ngrid,galPow,self.noiModel,self.rlim)
            galPow  =   galPow-noiFit
            epcor   =   noiFit*noiFit+2*noiFit*galPow
            decEP   =   self.deconvolvePow(epcor,order=2.)
            nn      =   self.itransformCov(decEP)

        decPow      =   self.deconvolvePow(galPow,order=1.)
        mm          =   self.itransform(decPow)
        if self.noiModel is not None:
            mm  =   rfn.merge_arrays([mm,nn], flatten = True, usemask = False)
        return mm

def fpfsM2E(moments,const=1.,mcalib=0.,ver=1):
    """
    # Estimate FPFS ellipticities from fpfs moments

    Parameters:
    -----------
    moments:    input FPFS moments
    const:      the weighting Constant
    mcalib:     multiplicative biases

    Returns:
    -------------
    out :       FPFS ellipticities
    """
    if ver==1:
        #Get weight
        weight  =   moments['fpfs_M00']+const
        #FPFS flux
        flux    =   moments['fpfs_M00']/weight
        #Ellipticity
        e1      =   -moments['fpfs_M22c']/weight
        e2      =   -moments['fpfs_M22s']/weight
        #Response factor
        R1      =   1./np.sqrt(2.)*(moments['fpfs_M00']-moments['fpfs_M40'])/weight+np.sqrt(2)*(e1*e1)
        R2      =   1./np.sqrt(2.)*(moments['fpfs_M00']-moments['fpfs_M40'])/weight+np.sqrt(2)*(e2*e2)
        RA      =   (R1+R2)/2.
        types   =   [('fpfs_e1','>f8'),('fpfs_e2','>f8'),('fpfs_RA','>f8'),('fpfs_flux','>f8')]
        ellDat  =   np.array(np.zeros(len(e1)),dtype=types)
        ellDat['fpfs_e1']=e1.transpose()
        ellDat['fpfs_e2']=e2.transpose()
        ellDat['fpfs_RA']=RA.transpose()
        ellDat['fpfs_flux']=flux.transpose()
    elif ver==2:
        #Get weight
        weight  =   moments['fpfs_M20']+const
        #FPFS flux
        flux    =   moments['fpfs_M00']/weight
        #Ellipticity
        e1      =   -moments['fpfs_M22c']/weight
        e2      =   -moments['fpfs_M22s']/weight
        e41     =   -moments['fpfs_M42c']/weight
        e42     =   -moments['fpfs_M42s']/weight
        #Response factor
        R1      =   1./np.sqrt(2.)*(moments['fpfs_M00']-moments['fpfs_M40'])/weight+np.sqrt(6)*(e1*e41)
        R2      =   1./np.sqrt(2.)*(moments['fpfs_M00']-moments['fpfs_M40'])/weight+np.sqrt(6)*(e2*e42)
        RA      =   (R1+R2)/2.
        types   =   [('fpfs_e1','>f8'),('fpfs_e2','>f8'),('fpfs_RA','>f8'),('fpfs_flux','>f8')]
        ellDat  =   np.array(np.zeros(len(e1)),dtype=types)
        ellDat['fpfs_e1']=e1.transpose()
        ellDat['fpfs_e2']=e2.transpose()
        ellDat['fpfs_RA']=RA.transpose()
        ellDat['fpfs_flux']=flux.transpose()
    return ellDat
