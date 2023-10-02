# Some input dataset used for unit tests

The [src_cosmos.fits](./src_cosmos.fits) file is the COSMOS HST 25.2 magnitude
limited galaxy sample. Please refer to [this
website](https://galsim-developers.github.io/GalSim/_build/html/real_gal.html#downloading-the-cosmos-catalog)
for the detailed information on this catalog.

The [correlation.fits](./correlation.fits) is the correlation funciton
simulated from 3rd order Lanczos kernel. This is used to simulate correlated
noise on images.

The [FPFS catalog](./fpfs-cut32-0000-g1-0000.fits) is the output of FPFS with
$sigma_as=0.45~\mathrm{arcsec}$ on HSC-like image simulations. This is used for
unit tests.
