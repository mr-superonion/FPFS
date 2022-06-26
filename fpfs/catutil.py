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
def tsfunc(x,deriv=0,mu=0.,sigma=1.5):
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
        out = tsfunc(x,deriv=deriv,mu=cut,sigma=sigma)
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
        out['fpfs_HR00']=-(mm['fpfs_N00N00']-mm['fpfs_N00N40'])/_w/np.sqrt(2.)
        out['fpfs_HR20']=-(mm['fpfs_N00N20']-mm['fpfs_N20N40'])/_w/np.sqrt(2.)
        out['fpfs_HE100']=-(mm['fpfs_N00N22c'])/_w
        out['fpfs_HE200']=-(mm['fpfs_N00N22s'])/_w
        out['fpfs_HE120']=-(mm['fpfs_N20N22c'])/_w
        out['fpfs_HE220']=-(mm['fpfs_N20N22s'])/_w
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
            out['fpfs_HRv%d' %i]=-(mm['fpfs_N00V%d' %i]-mm['fpfs_N40V%d'%i])/_w/np.sqrt(2.)
            out['fpfs_HE1v%d'%i]=-(mm['fpfs_N22cV%d'%i])/_w
            out['fpfs_HE2v%d'%i]=-(mm['fpfs_N22sV%d'%i])/_w
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

def fpfsM2Err(moments,const=1.):
    """
    Estimate FPFS measurement errors from the fpfs
    moments and the moments covariances
    Args:
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

def fpfsM2E_old(moments,dets=None,const=1.,noirev=False):
    """
    Estimate FPFS ellipticities from fpfs moments

    Args:
        moments (ndarray):
            input FPFS moments
        dets (ndarray):
            detection array [default: None]
        const (float):
            the weight constant [default:1]
        noirev (bool):
            revise the second-order noise bias? [default: False]

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
    # q1      =   moments['fpfs_M42c']/_w #New
    # q2      =   moments['fpfs_M42s']/_w
    e1sq    =   e1*e1
    e2sq    =   e2*e2
    # e1q1    =   e1*q1   #New
    # e2q2    =   e2*q2
    # FPFS flux ratio
    s0      =   moments['fpfs_M00']/_w
    s2      =   moments['fpfs_M20']/_w
    s4      =   moments['fpfs_M40']/_w
    # # FPFS sel respose
    # e1sqS0  =   e1sq*s0
    # e2sqS0  =   e2sq*s0
    # e1sqS2  =   e1sq*s2 #New
    # e2sqS2  =   e2sq*s2

    if dets is not None:
        # shear response for detection
        dDict=  {}
        for i in range(8):
            types.append(('fpfs_e1v%dr1'%i,'<f8'))
            types.append(('fpfs_e2v%dr2'%i,'<f8'))
            dDict['fpfs_e1v%dr1' %i]=e1*dets['fpfs_v%dr1' %i]
            dDict['fpfs_e2v%dr2' %i]=e2*dets['fpfs_v%dr2' %i]
    else:
        dDict = None

    if noirev:
        ratio=  moments['fpfs_N00N00']/_w**2.
        if dDict is not None:
            # correction detection shear response for noise bias
            for i in range(8):
                dDict['fpfs_e1v%dr1'%i]=dDict['fpfs_e1v%dr1'%i]\
                    -1.*dets['fpfs_N22cV%dr1'%i]/_w\
                    +1.*e1*dets['fpfs_N00V%dr1'%i]/_w\
                    +1.*moments['fpfs_N00N22c']/_w**2.*dets['fpfs_v%dr1'%i]
                dDict['fpfs_e1v%dr1'%i]=dDict['fpfs_e1v%dr1'%i]/(1+ratio)
                dDict['fpfs_e2v%dr2'%i]=dDict['fpfs_e2v%dr2'%i]\
                    -1.*dets['fpfs_N22sV%dr2'%i]/_w\
                    +1.*e2*dets['fpfs_N00V%dr2'%i]/_w\
                    +1.*moments['fpfs_N00N22s']/_w**2.*dets['fpfs_v%dr2'%i]
                dDict['fpfs_e2v%dr2'%i]=dDict['fpfs_e2v%dr2'%i]/(1+ratio)

        # # shear response of flux selection
        # e1sqS0= (e1sqS0+3.*e1sq*moments['fpfs_N00N00']/_w**2.\
        #         -s0*moments['fpfs_N22cN22c']/_w**2.)/(1+6.*ratio)
        # e2sqS0= (e2sqS0+3.*e2sq*moments['fpfs_N00N00']/_w**2.\
        #         -s0*moments['fpfs_N22sN22s']/_w**2.)/(1+6.*ratio)
        # # shear response of resolution selection
        # e1sqS2= (e1sqS2+3.*e1sq*moments['fpfs_N00N20']/_w**2.\
        #         -s2*moments['fpfs_N22cN22c']/_w**2.)/(1+6.*ratio)
        # e2sqS2= (e2sqS2+3.*e2sq*moments['fpfs_N00N20']/_w**2.\
        #         -s2*moments['fpfs_N22sN22s']/_w**2.)/(1+6.*ratio)
        # e1q1=   (e1q1-moments['fpfs_N22cN42c']/_w**2.\
        #         +2.*e1*moments['fpfs_N00N42c']/_w**2.\
        #         +2.*q1*moments['fpfs_N00N22c']/_w**2.\
        #         )/(1.+3*ratio)
        # e2q2=   (e2q2-moments['fpfs_N22sN42s']/_w**2.\
        #         +2.*e2*moments['fpfs_N00N42s']/_w**2.\
        #         +2.*q2*moments['fpfs_N00N22s']/_w**2.\
        #         )/(1.+3*ratio)
        # # intrinsic shape dispersion
        # e1sq=   (e1sq-moments['fpfs_N22cN22c']/_w**2.\
        #         +4.*e1*moments['fpfs_N00N22c']/_w**2.)\
        #         /(1.+3*ratio)
        # e2sq=   (e2sq-moments['fpfs_N22sN22s']/_w**2.\
        #         +4.*e2*moments['fpfs_N00N22s']/_w**2.)\
        #         /(1.+3*ratio)

        # noise bias correction for ellipticity
        e1  =   (e1+moments['fpfs_N00N22c']\
                /_w**2.)/(1+ratio)
        e2  =   (e2+moments['fpfs_N00N22s']\
                /_w**2.)/(1+ratio)
        # q1  =   (q1+moments['fpfs_N00N42c']\
        #         /_w**2.)/(1+ratio)
        # q2  =   (q2+moments['fpfs_N00N42s']\
        #         /_w**2.)/(1+ratio)
        # noise bias correction for spin-0 observables
        s0  =   (s0+moments['fpfs_N00N00']\
                /_w**2.)/(1+ratio)
        s2  =   (s2+moments['fpfs_N00N20']\
                /_w**2.)/(1+ratio)
        s4  =   (s4+moments['fpfs_N00N40']\
                /_w**2.)/(1+ratio)

    # make the output ndarray
    out  =   np.array(np.zeros(moments.size),dtype=types)
    if dDict is not None:
        for i in range(8):
            out['fpfs_e1v%dr1'%i] = dDict['fpfs_e1v%dr1'%i]
            out['fpfs_e2v%dr2'%i] = dDict['fpfs_e2v%dr2'%i]
        del dDict

    # spin-2 properties
    out['fpfs_e1']  =   e1      # ellipticity
    out['fpfs_e2']  =   e2
    # out['fpfs_q1']  =   q1      # epsilon
    # out['fpfs_q2']  =   q2
    del e1,e2

    # spin-0 properties
    out['fpfs_s0']  =   s0      # flux
    out['fpfs_s2']  =   s2      # size2
    out['fpfs_s4']  =   s4      # size4
    eSq  =  e1sq+e2sq       # e_1^2+e_2^2
    RE   =  1./np.sqrt(2.)*(s0-s4+eSq) # shear response of e
    # Li et. al (2018) has a minus sign difference
    out['fpfs_RE']  =   RE
    out['fpfs_ee']  =   eSq     # shape noise
    del s0,s2,s4,RE,e1sq,e2sq

    # # for selection bias correction
    # eSqS0=  e1sqS0+e2sqS0   # (e_1^2+e_2^2)s_0 for selection bias correcton
    # del e1sqS0,e2sqS0
    # eqeq =  e1q1+e2q2
    # eSqS2=  e1sqS2+e2sqS2   # (e_1^2+e_2^2)s_2 for selection bias correcton
    # del e1q1,e2q2
    # out['fpfs_RS0'] =   (eSq-eSqS0)/np.sqrt(2.) # selection bias correction for flux
    # out['fpfs_RS2'] =   (eqeq*np.sqrt(3.)-eSqS2)/np.sqrt(2.) # selection bias correction for resolution
    return out

