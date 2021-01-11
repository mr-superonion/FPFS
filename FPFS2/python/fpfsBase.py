import numpy as np

class fpfsBaseTask():
    _DefaultName = "fpfsBase"
    def __init__(self,psfData):
        self.ngrid   =   psfData.shape[0]
        self.psfPow  =   self.getPow(psfData)
        # Get PSF power and radius
        self.beta    =   0.2
        self.sigma   =   self.getHLRnaive(self.psfPow,self.beta)
        self.prepareRlim()
        self.chi     =   self.shapeletsPrepare(4)
        return

    def getHLRnaive(self,imgData,beta):
        imgData2=   np.abs(imgData)
        # Get the half light radius of noiseless PSF
        thres   =   imgData2.max()*0.5
        sigma   =   np.sum(imgData2>thres)
        sigma   =   np.sqrt(sigma/np.pi)*beta
        sigma   =   max(1.,min(sigma,4.))
        return sigma

    def prepareRlim(self):
        # Get rlim
        thres   =   1.e-3
        for dist in range(12,30):
            ave =  abs(np.exp(-dist**2./2./self.sigma**2.)/self.psfPow[ngrid//2+dist,ngrid//2])
            ave +=  abs(np.exp(-dist**2./2./self.sigma**2.)/self.psfPow[ngrid//2,ngrid//2+dist])
            ave =   ave/2.
            if ave<=thres:
                self.rlim=   dist
                break
        self.indX=np.arange(self.ngrid//2-self.rlim,self.ngrid//2+self.rlim+1)
        self.indY=self.indX[:,None]
        self.ind2D=np.ix_(self.indX,self.indX)
        return

    def shapeletsPrepare(self,nord):
        # Prepare the shapelets function
        ngrid   =   self.ngrid
        mord    =   nord
        # Set up the r and theta function
        xy1d    =   np.fft.fftshift(np.fft.fftfreq(ngrid,d=self.sigma/ngrid))
        xfunc,yfunc=  np.meshgrid(xy1d,xy1d)
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
                chi[nn,mm,:,:]=pow(-1.,d1)/self.sigma*pow(cc,0.5)*lfunc[c1,abs(mm),:,:]*pow(rfunc,abs(mm))*gaufunc*eulfunc**mm
        return chi

    def getPow(self,galData):
        galData.astype(np.float64)
        galData=np.fft.ifftshift(galData)
        # Get power function and subtract noise power
        galpow  =   np.fft.fft2(galData)#np.abs(np.fft.fft2(galData))**2.
        galpow  =   np.fft.fftshift(galpow)
        return galpow

    def deconvolvePow(self,galData,noiData=None):
        # Deconvolve the galaxy power with the PSF power

        # Subtract the noiPow
        ngrid   =   galData.shape[0]
        if noiData is not None:
            minPow,noiPow2  =   self.removeNoiPow(ngrid,galData,noiData,self.rlim)
        else:
            minPow=galData;noiPow2=None
        decPow  =   np.zeros(galData.shape,dtype=np.complex64)
        decPow[self.ind2D]=minPow[self.ind2D]/self.psfPow[self.ind2D]
        return decPow,noiPow2,minPow

    def removeNoiPow(self,ngrid,galPow,noiPowR,rlim):
        rlim2       =   max(27,rlim)
        noiList     =   []
        valList     =   []
        for j in range(ngrid):
            for i in range(ngrid):
                ii=i-ngrid/2.
                jj=j-ngrid/2.
                r   =   np.sqrt(ii**2.+jj**2.)
                if r>rlim2:
                    valList.append(galPow[j,i])
                    noiList.append(noiPowR[:,j,i])
        vl  =   np.array(valList)
        nl  =   np.array(noiList)
        nl  =   np.hstack([nl,np.ones((nl.shape[0],1))])
        par =   np.linalg.lstsq(nl,vl)[0]
        #self.log.info('%s' %par)
        noiSub   =   np.zeros((ngrid,ngrid))
        npar=   len(par)
        for ipc in range(npar-1):
            noiSub+=(par[ipc]*noiPowR[ipc])
        noiSub  +=  par[-1]
        minPow  =   galPow-noiSub
        return minPow,noiSub

    def measMoments(self,data):
        height  =   data.shape[0]
        width   =   data.shape[1]
        print(np.abs(data.imag).max())
        MAll    =   np.sum(data[None,None,self.indY,self.indX]*self.chi[::2,:4:2,self.indY,self.indX],axis=(2,3))
        MC      =   MAll.real
        MS      =   MAll.imag
        types=[('fpfs_M00','>f8'),('fpfs_M20','>f8') ,('fpfs_M22c','>f8'),('fpfs_M22s','>f8'), \
               ('fpfs_M40','>f8'),('fpfs_M42c','>f8'),('fpfs_M42s','>f8')]
        M00 =MC[0,0];M20 =MC[1,0];M40 =MC[2,0]
        M22c=MC[1,1];M22s=MS[1,1]
        M42c=MC[2,1];M42s=MS[2,1]
        return np.array((M00,M20,M22c,M22s,M40,M42c,M42s),dtype=types)

    def measure(self,galData):
        if len(galData.shape)==2:
            return self.measureSingle(galData)
        elif len(galData.shape)==3:
            results=[]
            for gal in galData:
                results.append(self.measureSingle(gal))
            return np.vstack(results)
        else:
            pass

    def measureSingle(self,galData):
        galPow  =   self.getPow(galData)
        #get the shapelets file
        decPow,noiPowModel,minPow  =   self.deconvolvePow(galPow)
        return self.measMoments(decPow)
