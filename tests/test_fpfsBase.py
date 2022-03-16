import gc
import fpfs
import numpy as np
import numpy.lib.recfunctions as rfn

def analyze_FPFS(rng:np.random.RandomState,input_shear:np.ndarray,num_gals:int,noi_stds,noi_psf=0.) -> tuple[np.ndarray,np.ndarray]:
    noi_stds=   np.array([noi_stds])
    if len(noi_stds.shape)==0:
        noi_stds=np.array([noi_stds])
    elif len(noi_stds.shape)>=2:
        raise ValueError('The input noi_stds should be float or 1d list')
    y       =   []
    y_err   =   []
    # initialize FPFS task with psf
    gal,psf =   fpfs.simutil.make_simple_sim(shear=[0.,0.],rng=rng,noise=0.)
    ngrid   =   psf.shape[0]
    rcut    =   16
    beg     =   ngrid//2-rcut
    end     =   beg+2*rcut
    psf     =   psf[beg:end,beg:end]
    gc.collect()
    num_tests=  noi_stds.size
    for i in range(num_tests):
        fpTask   =  fpfs.fpfsBase.fpfsTask(psf,noiFit=noi_stds[i]**2.,beta=0.75)
        results=[]
        for _ in range(num_gals):
            gal  =  fpfs.simutil.make_simple_sim(shear=input_shear,\
                    rng=rng,noise=noi_stds[i],psf_noise=noi_psf)[0]
            ngrid=  gal.shape[0]
            beg  =  ngrid//2-rcut
            end  =  beg+2*rcut
            gal  =  gal[beg:end,beg:end]
            modes=  fpTask.measure(gal)
            results.append(modes)
            del gal,modes
            gc.collect()
        mms =   rfn.stack_arrays(results,usemask=False)
        ells=   fpfs.fpfsBase.fpfsM2E(mms,const=2000,noirev=False)
        del mms,results
        resp=np.average(ells['fpfs_RE'])
        shear=np.average(ells['fpfs_e1'])/resp
        shear_err=np.std(ells['fpfs_e1'])/np.abs(resp)
        y.append(shear)
        y_err.append(shear_err)
    return np.array(y), np.array(y_err)

def test_noisy_gals(noi_std:float=0.) -> None:
    rng     =   np.random.RandomState(212)
    num_gals=   100
    shear,shear_err=analyze_FPFS(rng,np.array([0.03, 0.00]),num_gals,noi_std)
    #print(shear,shear_err)
    thres   =   max(2.*shear_err,1e-5)
    assert np.all(np.abs(shear-0.03)<thres)
    return

if __name__ == '__main__':
    test_noisy_gals(noi_std=0.)
    test_noisy_gals(noi_std=5e-4)
