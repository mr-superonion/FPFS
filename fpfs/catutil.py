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
import numpy as np
from .default import det_inds

det_inds=[(2,2),(1,2),(3,2),(2,1),(2,3)]
"""list: a list of pixel index, where (2,2) is the centroid
"""

def fpfsM2E(mm,dets=None,const=1.,noirev=False):
    """
    Estimate FPFS ellipticities from fpfs moments
    Args:
        mm (ndarray):
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
            flux, size and FPFS selection response)
    """
    # ellipticity, q-ellipticity, sizes, e^2, eq
    types   =   [('fpfs_e1','<f8'), ('fpfs_e2','<f8') , ('fpfs_ee','<f8'),\
                ('fpfs_s0','<f8') , ('fpfs_s2','<f8') , ('fpfs_s4','<f8'),\
                ('fpfs_RE','<f8') , ('fpfs_RS0','<f8'), ('fpfs_RS2','<f8')]

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
    ee      =   e1*e1+e2*e2
    eM22    =   e1*mm['fpfs_M22c']+e2*mm['fpfs_M22s']
    eM42    =   e1*mm['fpfs_M42c']+e2*mm['fpfs_M42s']

    if dets is not None:
        # shear response for peak detection process
        dDict=  {}
        for (j,i) in det_inds:
            types.append(('fpfs_RS0%d%d'%(j,i),'<f8'))
            dDict['fpfs_RS0%d%d' %(j,i)]=(e1*dets['pdet_v%d%dr1'%(j,i)]+e2*dets['pdet_v%d%dr2'%(j,i)])/2.
    else:
        dDict = None

    if noirev:
        ratio=  mm['fpfs_N00N00']/_w**2.
        if dDict is not None:
            # correction detection shear response for noise bias
            for (j,i) in det_inds:
                corr=(-1.*dets['pdet_N22cV%d%dr1'%(j,i)]/_w\
                    -1.*dets['pdet_N22sV%d%dr2'%(j,i)]/_w\
                    +1.*e1*dets['pdet_N00V%d%dr1'%(j,i)]/_w\
                    +1.*e2*dets['pdet_N00V%d%dr2'%(j,i)]/_w\
                    +1.*mm['fpfs_N00N22c']/_w**2.*dets['pdet_v%d%dr1'%(j,i)]\
                    +1.*mm['fpfs_N00N22s']/_w**2.*dets['pdet_v%d%dr2'%(j,i)]\
                    )/2.
                dDict['fpfs_RS0%d%d'%(j,i)]=(dDict['fpfs_RS0%d%d'%(j,i)]+corr)/(1+ratio)
        # intrinsic shape dispersion (not per component)
        ee      =   (ee-(mm['fpfs_N22cN22c']+mm['fpfs_N22sN22s'])/_w**2.\
                    +4.*(e1*mm['fpfs_N00N22c']+e2*mm['fpfs_N00N22s'])/_w**2.)\
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

    # make the output ndarray
    out  =   np.array(np.zeros(mm.size),dtype=types)
    # response for detection process (not response for deatection function)
    if dDict is not None:
        for (j,i) in det_inds:
            out['fpfs_RS0%d%d'%(j,i)] = dDict['fpfs_RS0%d%d'%(j,i)]
        del dDict
    # spin-2 properties
    out['fpfs_e1']  =   e1      # ellipticity
    out['fpfs_e2']  =   e2
    del e1,e2
    # spin-0 properties
    out['fpfs_s0']  =   s0      # flux
    out['fpfs_s2']  =   s2      # size2
    out['fpfs_s4']  =   s4      # size4
    out['fpfs_ee']  =   ee      # shape noise
    # response for ellipticity
    out['fpfs_RE']  =   (s0-s4+ee)/np.sqrt(2.)
    del s0,s2,s4,ee
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
        for (j,i) in det_inds:
            types.append(('fpfs_e1v%d%dr1'%(j,i),'<f8'))
            types.append(('fpfs_e2v%d%dr2'%(j,i),'<f8'))
            dDict['fpfs_e1v%d%dr1' %(j,i)]=e1*dets['pdet_v%d%dr1' %(j,i)]
            dDict['fpfs_e2v%d%dr2' %(j,i)]=e2*dets['pdet_v%d%dr2' %(j,i)]
    else:
        dDict = None

    if noirev:
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
        for (j,i) in det_inds:
            out['fpfs_e1v%d%dr1'%(j,i)] = dDict['fpfs_e1v%d%dr1'%(j,i)]
            out['fpfs_e2v%d%dr2'%(j,i)] = dDict['fpfs_e2v%d%dr2'%(j,i)]
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

