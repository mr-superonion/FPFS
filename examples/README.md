# Examples

This directory gives some examples on image simulations with Galsim; detection
and shape estimation with FPFS.

## simulate noise
The [code](./noiSim.py) simulates noise with Lanczos correlation function.

## simulate stamp-based galaxies
The [code](./cgcSimBasic.py) supports simulation of parametric galaxies (using
Sersic (single or double) model fitted to COSMOS galaxies) and galaxies
simulated with random knots (for very small galaxies). The galaxies are render
into 64x64 postage stamps.

## simulate galaxies in the universe
The [code](./cgcSimCosmo.py) simulates parametric galaxies randomly distributed
in a big image. It supports simulations with different number densities.

## process with FPFS detector and shear estimator
The [code](./processFPFS.py) detects galaxies from images and measure the FPFS
shapes for shear inference.
