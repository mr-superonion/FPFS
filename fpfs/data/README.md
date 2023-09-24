# Some input dataset used for unit tests

The [cat_used.fits](./cat_used.fits) file is the COSMOS HST 25.2 magnitude
limited galaxy sample. Please refer to [this
website](https://galsim-developers.github.io/GalSim/_build/html/real_gal.html#downloading-the-cosmos-catalog)
for the detailed information on this catalog.

The [correlation.fits](./correlation.fits) is the correlation funciton
simulated from 3rd order Lanczos kernel.

The [FPFS catalog](./fpfs-cut32-0000-g1-0000.fits) is the output of FPFS with
$sigma_as=0.45~\mathrm{arcsec}$

The [gamma1_measured_1] is a scan over np.linspace(0.01,0.05,40) with G1 = 0.003, G2 = 0.0025, F1 = 0.0007, F2 = 0.001