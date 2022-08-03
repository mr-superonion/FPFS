# FPFS shear estimator
# Copyright 20210805 Xiangchong Li.
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
import numpy as np

# functions used for selection
def tsfunc1(x,deriv=0,mu=0.,sigma=1.5):
    """Returns the weight funciton (deriv=0), or the *multiplicative factor* to
    the weight function for first order derivative (deriv=1)
    Args:
        deriv (int):    whether do derivative (1) or not (0)
        x (ndarray):    input data vector
        mu (float):     center of the cut
        sigma (sigma):  widthe of the selection function
    Returns:
        out (ndarray):  the weight funciton (deriv=0), or the *multiplicative
                        factor* to the weight function for first order derivative
                        (deriv=1)
    """
    t=(x-mu)/sigma
    if deriv==0:
        return np.piecewise(t, [t<-1,(t>=-1)&(t<=1),t>1], [0.,lambda t:1./2.+np.sin(t*np.pi/2.)/2., 1.])
    elif deriv==1:
        # multiplicative factor (to weight) for derivative
        return np.piecewise(t,[t<-1+0.01,(t>=-1+0.01)&(t<=1-0.01),t>1-0.01],\
                [0.,lambda t: np.pi/2./sigma*np.cos(t*np.pi/2.)/(1.+np.sin(t*np.pi/2.)), 0.] )
    else:
        raise ValueError('deriv should be 0 or 1')

def tsfunc2(x,mu=0.,sigma=1.5,deriv=0):
    t=(x-mu)/sigma
    func=lambda t:1./2.+t/2.+1./2./np.pi*np.sin(t*np.pi)

    if deriv==0:
        return np.piecewise(t, [t<-1,(t>=-1)&(t<=1),t>1], [0.,func, 1.])
    elif deriv==1:
        func2= lambda t:(1./2./sigma+1./2./sigma*np.cos(np.pi*t))#/(1./2.+t/2.+1./2./np.pi*np.sin(t*np.pi))
        return np.piecewise(t, [t<-1+0.01,(t>=-1+0.01)&(t<=1-0.01),t>1-0.01],[0.,lambda t: func2(t)/func(t),0.] )
    elif deriv==2:
        func3= lambda t:(-np.pi/2./sigma**2.*np.sin(np.pi*t))
        return np.piecewise(t, [t<-1+0.01,(t>=-1+0.01)&(t<=1-0.01),t>1-0.01],[0.,lambda t: func3(t)/func(t),0.] )
    elif deriv==3:
        func4= lambda t:(-(np.pi)**2./2./sigma**3.*np.cos(np.pi*t))
        return np.piecewise(t, [t<-1+0.01,(t>=-1+0.01)&(t<=1-0.01),t>1-0.01],[0.,lambda t: func4(t)/func(t),0.] )
    else:
        raise ValueError('deriv can only be 0,1,2,3')

def sigfunc(x,deriv=0,mu=0.,sigma=1.5):
    """Returns the weight funciton (deriv=0), or the *multiplicative factor* to
    the weight function for first order derivative (deriv=1)
    Args:
        deriv (int):    whether do derivative (1) or not (0)
        x (ndarray):    input data vector
        mu (float):     center of the cut
        sigma (sigma):  widthe of the selection function
    Returns:
        out (ndarray):  the weight funciton (deriv=0), or the *multiplicative
                        factor* to the weight function for first order derivative
                        (deriv=1)
    """
    expx=np.exp(-(x-mu)/sigma)
    if deriv==0:
        # sigmoid function
        return 1./(1. + expx)
    elif deriv==1:
        # multiplicative factor (to weight) for derivative
        return 1./sigma*expx/(1. + expx)
    else:
        raise ValueError('deriv should be 0 or 1')

def get_wsel_eff(x,cut,sigma,use_sig,deriv=0):
    """Returns the weight funciton (deriv=0), or the *multiplicative
    factor* to the weight function for first order derivative (deriv=1)
    Args:
        x (ndarray):    input selection observable
        cut (float):    the cut on selection observable
        sigma (sigma):  width of the selection function
        use_sig (bool): whether use sigmoid (True) of truncated sine (False)
        deriv (int):    whether do derivative (1) or not (0)
    Returns:
        out (ndarray):  the weight funciton (deriv=0), or the *multiplicative
                        factor* to the weight function for first order
                        derivative (deriv=1)
    """
    if use_sig:
        out = sigfunc(x,deriv=deriv,mu=cut,sigma=sigma)
    else:
        out = tsfunc1(x,deriv=deriv,mu=cut,sigma=sigma)
    return out

def get_wbias(x,cut,sigma,use_sig,w_sel,rev=None):
    """Returns the weight bias due to shear dependence and noise bias (first
    order in w)
    Args:
        x (ndarray):        selection observable
        cut (float):        the cut on selection observable
        sigma (sigma):      width of the selection function
        use_sig (bool):     whether use sigmoid (True) of truncated sine (False)
        w_sel (ndarray):    selection weights as function of selection observable
        rev  (ndarray):     selection response array
    Returns:
        cor (float):        correction for shear response
    """
    if rev is None:
        cor = 0.
    else:
        cor = np.sum(rev*w_sel*get_wsel_eff(x,cut,sigma,use_sig,deriv=1))
    return cor

# functions to get derived observables from fpfs modes
def fpfsM2E(mm,const=1.,noirev=False):
    """
    Estimate FPFS ellipticities from fpfs moments
    Args:
        mm (ndarray):
            input FPFS moments
        const (float):
            the weight constant [default:1]
        noirev (bool):
            revise the second-order noise bias? [default: False]
    Returns:
        out (ndarray):
            an array of (FPFS ellipticities, FPFS ellipticity response, FPFS
            flux, size and FPFS selection response)
    """
    # ellipticity, q-ellipticity, sizes, e^2, eq
    types   =   [('fpfs_e1','<f8'), ('fpfs_e2','<f8') , ('fpfs_ee','<f8'), \
                ('fpfs_s0','<f8') , ('fpfs_s2','<f8') , ('fpfs_s4','<f8'), \
                ('fpfs_R1E','<f8'), ('fpfs_R2E','<f8'), ('fpfs_RS0','<f8'),\
                ('fpfs_RS2','<f8')]
    for i in range(8):
        types.append(('fpfs_R1Sv%d'%i,'<f8'))
        types.append(('fpfs_R2Sv%d'%i,'<f8'))
    if noirev:
        types=  types+[('fpfs_HE100','<f8'),('fpfs_HE200','<f8'),('fpfs_HR00','<f8'),\
                       ('fpfs_HE120','<f8'),('fpfs_HE220','<f8'),('fpfs_HR20','<f8')]
        for i in range(8):
            types.append(('fpfs_HRv%d' %i,'<f8'))
            types.append(('fpfs_HE1v%d' %i,'<f8'))
            types.append(('fpfs_HE2v%d' %i,'<f8'))
    # make the output ndarray
    out  =   np.array(np.zeros(mm.size),dtype=types)

    # FPFS shape weight's inverse
    _w      =   mm['fpfs_M00']+const
    # FPFS ellipticity
    e1      =   mm['fpfs_M22c']/_w
    e2      =   mm['fpfs_M22s']/_w
    q1      =   mm['fpfs_M42c']/_w
    q2      =   mm['fpfs_M42s']/_w
    # FPFS spin-0 observables
    s0      =   mm['fpfs_M00']/_w
    s2      =   mm['fpfs_M20']/_w
    s4      =   mm['fpfs_M40']/_w
    # intrinsic ellipticity
    e1e1    =   e1*e1
    e2e2    =   e2*e2
    eM22    =   e1*mm['fpfs_M22c']+e2*mm['fpfs_M22s']
    eM42    =   e1*mm['fpfs_M42c']+e2*mm['fpfs_M42s']

    # shear response for detection process (not for deatection function)
    for i in range(8):
        out['fpfs_R1Sv%d' %(i)]=e1*mm['fpfs_v%dr1'%(i)]
        out['fpfs_R2Sv%d' %(i)]=e2*mm['fpfs_v%dr2'%(i)]

    if noirev:
        out['fpfs_HR00']=-(mm['fpfs_N00N00']*(const/_w+s4-4.*e1**2.)\
                -mm['fpfs_N00N40'])/_w/np.sqrt(2.)
        out['fpfs_HR20']=-(mm['fpfs_N00N20']*(const/_w+s4-4.*e2**2.)\
                -mm['fpfs_N20N40'])/_w/np.sqrt(2.)
        out['fpfs_HE100']=-(mm['fpfs_N00N22c']-e1*mm['fpfs_N00N00'])/_w
        out['fpfs_HE200']=-(mm['fpfs_N00N22s']-e2*mm['fpfs_N00N00'])/_w
        out['fpfs_HE120']=-(mm['fpfs_N20N22c']-e1*mm['fpfs_N00N20'])/_w
        out['fpfs_HE220']=-(mm['fpfs_N20N22s']-e2*mm['fpfs_N00N20'])/_w
        ratio=  mm['fpfs_N00N00']/_w**2.
        # correction for detection process shear response for noise bias
        for i in range(8):
            corr1=-1.*mm['fpfs_N22cV%dr1'%i]/_w\
                +1.*e1*mm['fpfs_N00V%dr1'%i]/_w\
                +1.*mm['fpfs_N00N22c']/_w**2.*mm['fpfs_v%dr1'%i]
            corr2=-1.*mm['fpfs_N22sV%dr2'%i]/_w\
                +1.*e2*mm['fpfs_N00V%dr2'%i]/_w\
                +1.*mm['fpfs_N00N22s']/_w**2.*mm['fpfs_v%dr2'%i]
            out['fpfs_R1Sv%d'%i]=(out['fpfs_R1Sv%d'%i]+corr1)/(1+ratio)
            out['fpfs_R2Sv%d'%i]=(out['fpfs_R2Sv%d'%i]+corr2)/(1+ratio)
            # Heissen
            out['fpfs_HRv%d' %i]=-(mm['fpfs_N00V%d' %i]*(const/_w+s4-2.*e1**2.-2.*e2**2.)\
                    -mm['fpfs_N40V%d'%i])/_w/np.sqrt(2.)
            out['fpfs_HE1v%d'%i]=-(mm['fpfs_N22cV%d'%i]-e1*mm['fpfs_N00V%d'%i])/_w
            out['fpfs_HE2v%d'%i]=-(mm['fpfs_N22sV%d'%i]-e2*mm['fpfs_N00V%d'%i])/_w
        # intrinsic shape dispersion (not per component)
        e1e1    =   (e1e1-(mm['fpfs_N22cN22c'])/_w**2.\
                    +4.*(e1*mm['fpfs_N00N22c'])/_w**2.)\
                    /(1.+3*ratio)
        e2e2    =   (e2e2-(mm['fpfs_N22sN22s'])/_w**2.\
                    +4.*(e2*mm['fpfs_N00N22s'])/_w**2.)\
                    /(1.+3*ratio)
        eM22    =   (eM22-(mm['fpfs_N22cN22c']+mm['fpfs_N22sN22s'])/_w \
                    +2.*(mm['fpfs_N00N22c']*e1+mm['fpfs_N00N22s']*e2)/_w)\
                    /(1+ratio)
        eM42    =   (eM42-(mm['fpfs_N22cN42c']+mm['fpfs_N22sN42s'])/_w\
                    +1.*(e1*mm['fpfs_N00N42c']+e2*mm['fpfs_N00N42s'])/_w\
                    +1.*(q1*mm['fpfs_N00N22c']+q2*mm['fpfs_N00N22s'])/_w)\
                    /(1+ratio)
        # noise bias correction for ellipticity
        e1  =   (e1+mm['fpfs_N00N22c']\
                /_w**2.)/(1+ratio)
        e2  =   (e2+mm['fpfs_N00N22s']\
                /_w**2.)/(1+ratio)
        # noise bias correction for flux, size
        s0  =   (s0+mm['fpfs_N00N00']\
                /_w**2.)/(1+ratio)
        s2  =   (s2+mm['fpfs_N00N20']\
                /_w**2.)/(1+ratio)
        s4  =   (s4+mm['fpfs_N00N40']\
                /_w**2.)/(1+ratio)

    # spin-2 properties
    out['fpfs_e1']  =   e1      # ellipticity
    out['fpfs_e2']  =   e2
    del e1,e2
    # spin-0 properties
    out['fpfs_s0']  =   s0      # flux
    out['fpfs_s2']  =   s2      # size2
    out['fpfs_s4']  =   s4      # size4
    out['fpfs_ee']  =   e1e1+e2e2# shape noise
    # response for ellipticity
    out['fpfs_R1E'] =   (s0-s4+2.*e1e1)/np.sqrt(2.)
    out['fpfs_R2E'] =   (s0-s4+2.*e2e2)/np.sqrt(2.)
    del s0,s2,s4,e1e1,e2e2
    # response for selection process (not response for selection function)
    out['fpfs_RS0']  =  -1.*eM22/np.sqrt(2.)
    out['fpfs_RS2']  =  -1.*eM42*np.sqrt(6.)/2.
    del eM22,eM42
    return out

class summary_stats():
    def __init__(self,mm,ell,use_sig=False,ratio=1.9):
        """
        Args:
            mm (ndarray):   FPFS moments
            ell (ndarray):  FPFS ellipticity
            use_sig (bool): whether use sigmoid (True) of truncated sine (False)
        """
        self.ratio = ratio
        self.use_sig= use_sig
        self.mm =   mm
        self.ell=   ell
        self.clear_outcomes()
        if 'fpfs_HR00' in self.ell.dtype.names:
            self.noirev=True
        else:
            self.noirev=False
        return

    def clear_outcomes(self):
        """clears the outcome of the class
        """
        self.nsel = 0
        self.ws =   np.ones(self.ell.shape)
        # bias
        self.ncor = 0
        self.corE1= 0.  # selection bias in e1
        self.corE2= 0.  # selection bias in e2
        self.corR1= 0.  # selection bias in R1E (response)
        self.corR2= 0.  # selection bias in R2E (response)
        # signal
        self.sumE1= 0.  # sum of e1
        self.sumE2= 0.  # sum of e2
        self.sumR1= 0.  # sum of R1E (response)
        self.sumR2= 0.  # sum of R2E (response)
        return

    def update_selection_weight(self,snms,cuts,cutsigs):
        """Updates the selection weight term with the current selection weight
        """
        if not isinstance(snms,list):
            if isinstance(snms,str) and isinstance(cuts,float) and isinstance(cutsigs,float):
                snms=[snms]
                cuts=[cuts]
                cutsigs=[cutsigs]
            else:
                raise TypeError('snms, cuts and cutsigs should be str, float, float')
        for selnm,cut,cutsig in zip(snms,cuts,cutsigs):
            if selnm=='detect':
                for iid in range(8):
                    self._update_selection_weight('det_v%d' %iid,cut,cutsig)
            elif selnm=='detect2':
                for iid in range(8):
                    self._update_selection_weight('det2_v%d' %iid,cut,cutsig)
            else:
                self._update_selection_weight(selnm,cut,cutsig)
        return

    def _update_selection_weight(self,selnm,cut,cutsig):
        """Updates the selection weight term with the current selection weight
        """
        if not isinstance(selnm,str):
            raise TypeError('selnm should be str')
        if not isinstance(cut,float):
            raise TypeError('cut should be float')
        if not isinstance(cutsig,float):
            raise TypeError('cutsig should be float')

        cut_final=cut
        if selnm=='M00':
            scol=self.mm['fpfs_M00']*self.ratio
        elif selnm=='M20':
            scol=-self.mm['fpfs_M20']
        elif selnm=='R2':
            # M00+M20>cut*M00 (M00>0., we have mag cut to ensure it)
            scol=self.mm['fpfs_M00']*(1.-cut)+self.mm['fpfs_M20']
            cut_final=0.
        elif selnm=='R2_upp':
            # M00+M20<cut*M00 (M00>0.)
            scol=self.mm['fpfs_M00']*(cut-1.)-self.mm['fpfs_M20']
            cut_final=0.
        elif 'det_' in selnm:
            vn=selnm.split('_')[-1]
            scol=self.mm['fpfs_%s'%vn]
        elif 'det2_' in selnm:
            vn=selnm.split('_')[-1]
            scol=self.mm['fpfs_%s'%vn]-self.mm['fpfs_M00']*cut
            cut_final=cutsig
        else:
            raise ValueError('Do not support selection vector name: %s' %selnm)
        # update weight
        ws  =   get_wsel_eff(scol,cut_final,cutsig,self.use_sig)
        self.ws=self.ws*ws
        # count the total number of selection cuts
        self.nsel=self.nsel+1
        return

    def update_selection_bias(self,snms,cuts,cutsigs):
        """Updates the selection bias correction term with the current
        selection weight
        """
        if not isinstance(snms,list):
            if isinstance(snms,str) and isinstance(cuts,float) and isinstance(cutsigs,float):
                snms=[snms]
                cuts=[cuts]
                cutsigs=[cutsigs]
            else:
                raise TypeError('snms, cuts and cutsigs should be (lists of) str, float, float')
        for selnm,cut,cutsig in zip(snms,cuts,cutsigs):
            if selnm=='detect':
                for iid in range(8):
                    self._update_selection_bias('det_v%d' %iid,cut,cutsig)
            if selnm=='detect2':
                for iid in range(8):
                    self._update_selection_bias('det2_v%d' %iid,cut,cutsig)
            else:
                self._update_selection_bias(selnm,cut,cutsig)
        assert self.nsel==self.ncor
        return

    def _update_selection_bias(self,selnm,cut,cutsig):
        """Updates the selection bias correction term with the current
        selection weight
        Args:
            selnm (str):    name of the selection variable ('M00', 'M20', 'R2')
            cut (float):    selection cut
            cutsig (float): width of the selection weight function (it is closer
                            to heavy step when cutsig is smaller)
        """
        if not isinstance(selnm,str):
            raise TypeError('selnm should be str')
        if not isinstance(cut,float):
            raise TypeError('cut should be float')
        if not isinstance(cutsig,float):
            raise TypeError('cutsig should be float')
        cut_final=cut
        if selnm=='M00':
            scol=self.mm['fpfs_M00']*self.ratio
            ccol1=self.ell['fpfs_RS0']*self.ratio
            ccol2=self.ell['fpfs_RS0']*self.ratio
            if self.noirev:
                dcol=self.ell['fpfs_HR00']*self.ratio
                ncol1=self.ell['fpfs_HE100']*self.ratio
                ncol2=self.ell['fpfs_HE200']*self.ratio
            else:
                dcol=None
                ncol1=None
                ncol2=None
        elif selnm=='M20':
            scol=-self.mm['fpfs_M20']
            ccol1=-self.ell['fpfs_RS2']
            ccol2=-self.ell['fpfs_RS2']
            if self.noirev:
                dcol=-self.ell['fpfs_HR20']
                ncol1=-self.ell['fpfs_HE120']
                ncol2=-self.ell['fpfs_HE220']
            else:
                dcol=None
                ncol1=None
                ncol2=None
        elif selnm=='R2' or selnm=='R2_upp':
            if '_upp' in selnm:
                fp=-1.
            else:
                fp=1.
            cut_final=0.
            scol=(self.mm['fpfs_M00']*(1.-cut)+self.mm['fpfs_M20'])*fp
            ccol1=(self.ell['fpfs_RS0']*(1.-cut)+self.ell['fpfs_RS2'])*fp
            ccol2=(self.ell['fpfs_RS0']*(1.-cut)+self.ell['fpfs_RS2'])*fp
            if self.noirev:
                dcol=(self.ell['fpfs_HR00']*(1.-cut)+self.ell['fpfs_HR20'])*fp
                ncol1=(self.ell['fpfs_HE100']*(1.-cut)+self.ell['fpfs_HE120'])*fp
                ncol2=(self.ell['fpfs_HE200']*(1.-cut)+self.ell['fpfs_HE220'])*fp
            else:
                dcol=None
                ncol1=None
                ncol2=None
        elif 'det_' in selnm:
            vn   =  selnm.split('_')[-1]
            scol =  self.mm['fpfs_%s' %vn]
            ccol1=  self.ell['fpfs_R1S%s' %vn]
            ccol2=  self.ell['fpfs_R2S%s' %vn]
            if self.noirev:
                dcol=self.ell['fpfs_HR%s' %vn]
                ncol1=self.ell['fpfs_HE1%s' %vn]
                ncol2=self.ell['fpfs_HE2%s' %vn]
            else:
                dcol=None
                ncol1=None
                ncol2=None
        elif 'det2_' in selnm:
            cut_final=cutsig
            vn  =   selnm.split('_')[-1]
            scol=   self.mm['fpfs_%s' %vn]-self.mm['fpfs_M00']*cut
            ccol1=  self.ell['fpfs_R1S%s' %vn]-self.ell['fpfs_RS0']*cut
            ccol2=  self.ell['fpfs_R2S%s' %vn]-self.ell['fpfs_RS0']*cut
            if self.noirev:
                dcol =  self.ell['fpfs_HR%s' %vn]-self.ell['fpfs_HR00']*cut
                ncol1=  self.ell['fpfs_HE1%s' %vn]-self.ell['fpfs_HE100']*cut
                ncol2=  self.ell['fpfs_HE2%s' %vn]-self.ell['fpfs_HE200']*cut
            else:
                dcol =  None
                ncol1=  None
                ncol2=  None
        else:
            raise ValueError('Do not support selection vector name: %s' %selnm)
        corSR1= get_wbias(scol,cut_final,cutsig,self.use_sig,self.ws,ccol1)
        corSR2= get_wbias(scol,cut_final,cutsig,self.use_sig,self.ws,ccol2)
        corNR = get_wbias(scol,cut_final,cutsig,self.use_sig,self.ws,dcol)
        corNE1= get_wbias(scol,cut_final,cutsig,self.use_sig,self.ws,ncol1)
        corNE2= get_wbias(scol,cut_final,cutsig,self.use_sig,self.ws,ncol2)
        self.corR1=self.corR1+corSR1+corNR
        self.corR2=self.corR2+corSR2+corNR
        self.corE1=self.corE1+corNE1
        self.corE2=self.corE2+corNE2
        self.ncor=self.ncor+1
        return

    def update_ellsum(self):
        """Updates the weighted sum of ellipticity and response with the currenct
        selection weight
        """
        self.sumE1=np.sum(self.ell['fpfs_e1']*self.ws)
        self.sumE2=np.sum(self.ell['fpfs_e2']*self.ws)
        self.sumR1=np.sum(self.ell['fpfs_R1E']*self.ws)
        self.sumR2=np.sum(self.ell['fpfs_R2E']*self.ws)
        return
