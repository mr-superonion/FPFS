"""Needed to use parallelization"""
import os
import fpfs
import galsim
import numpy as np
import matplotlib.pylab as plt
from fpfs.simutil import LensTransform

flux = 40
scale = 0.1
nn = 100
sersic_gal = galsim.Sersic(n=1.5, half_light_radius=1.5, flux=flux, trunc=4)

from astropy.io import fits

def create_and_measure_F1G2(gamma1):
    # Fixing F1*G2 and F2*G1
    G1 = 0.0
    F2 = 0.0
    F1 = np.linspace(0.001,0.01,5)
    G2 = np.linspace(0.001,0.01,5)
    kappa = 0.0
    gamma1_measured = np.zeros((len(F1),len(G2)))
    gamma2 = 0.0
    for i in range(len(F1)):
        for j in range(len(G2)):
            lens = LensTransform(gamma1=gamma1, gamma2=gamma2, kappa=kappa,F1=F1[i],F2=F2,G1=G1,G2=G2[j])
            stamp = fpfs.simutil.Stamp(nn=nn, scale=scale)
            stamp.transform_grids(lens)
            gal_array3 = stamp.sample_galaxy(sersic_gal)
            psf_array = np.zeros(stamp.shape)
            psf_array[nn // 2, nn // 2] = 1
            coords = np.array([nn//2, nn//2])
            fpTask  =   fpfs.image.measure_source(psf_array, pix_scale = scale, sigma_arcsec=0.52)
            mms =  fpTask.measure(gal_array3, coords)
            mms = fpTask.get_results(mms)
            ells=   fpfs.catalog.fpfs_m2e(mms,const=20)
            resp1=np.average(ells['fpfs_R1E'])
            shear1=np.average(ells['fpfs_e1'])/resp1
            gamma1_measured[i,j] = shear1 
            del stamp
            del lens
    hdu1 = fits.PrimaryHDU(gamma1_measured)
    hdu1.writeto(f'data/Fix_F1G2/gamma1_measured_flexion_F1G2_{gamma1:.3f}.fits')
    print("file created")
    return  0      
    
    
def create_and_measure_full(gamma1):
    G1 = np.linspace(0.001,0.01,5)
    G2 = np.linspace(0.001,0.01,5)
    F1 = np.linspace(0.001,0.01,5)
    F2 = np.linspace(0.001,0.01,5)
    kappa=0.0
    gamma1_measured = np.zeros((len(F1),len(F2),len(G1),len(G2)))
    gamma2 = 0.0
    for i in range(len(F1)):
        for j in range(len(F2)):
            for k in range(len(G1)):
                for l in range(len(G2)):
                    #order is F1, F2, G1, G2
                    lens = LensTransform(gamma1=gamma1, gamma2=gamma2, kappa=kappa,F1=F1[i],F2=F2[j],G1=G1[k],G2=G2[l])
                    stamp = fpfs.simutil.Stamp(nn=64, scale=scale)
                    stamp.transform_grids(lens)
                    gal_array3 = stamp.sample_galaxy(sersic_gal)
                    psf_array = np.zeros(stamp.shape)
                    psf_array[nn // 2, nn // 2] = 1
                    coords = np.array([nn//2, nn//2])
                    fpTask  =   fpfs.image.measure_source(psf_array, pix_scale = scale, sigma_arcsec=0.52)
                    mms =  fpTask.measure(gal_array3, coords)
                    mms = fpTask.get_results(mms)
                    ells=   fpfs.catalog.fpfs_m2e(mms,const=20)
                    resp1=np.average(ells['fpfs_R1E'])
                    shear1=np.average(ells['fpfs_e1'])/resp1
                    gamma1_measured[i][j][k][l] = shear1
                    del stamp
                    del lens
    hdu1 = fits.PrimaryHDU(gamma1_measured)
    hdu1.writeto(f'data/gamma1_measured_flexion_{gamma1}.fits')
    print("file created")
    return  0

from multiprocessing import Pool
if __name__ == '__main__':
    with Pool(10) as p:
        p.map(create_and_measure_F1G2,np.linspace(0.0,0.02,21))
