# Examples

This directory gives some examples on simulating galaxy images with Galsim; and
galaxy detection and shape estimation with FPFS. Before running the sripts,
please source [this file](./install.sh) after modifying it.

## simulate noise
This [code](./noiSim.py) simulates pure noise with the third-order Lanczos
correlation function.

## simulate stamp-based galaxies
This [code](./cgcSimBasic.py) supports simulation of parametric galaxies (using
Sersic or bulge-disk model fitted to COSMOS galaxies) and galaxies simulated
with random knots (for very small galaxies). The galaxies are render into 64x64
postage stamps. Note the galaxies are isolated here.

## simulate galaxies in the universe
This [code](./cgcSimCosmo.py) simulates parametric galaxies (Sersic or
bulge-disk) randomly and uniformly distributed within a circle. Note the
galaxies are blended here. It supports simulations with different number
densities.

## process with FPFS detector and shear estimator
This [code](./processFPFS.py) detects galaxies from images and measure the FPFS
shapes for shear inference.

## estimate average shear
This [code](./meas_detect_mag.py) and this
[code](./meas_detect_r2.py) use FPFS shapes to estimate average shear
for a galaxy sample.
