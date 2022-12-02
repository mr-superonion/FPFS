# Examples

This directory gives some examples on simulating galaxy images with
[Galsim](https://github.com/GalSim-developers/GalSim); and galaxy detection and
shape estimation with [FPFS](https://github.com/mr-superonion/FPFS).

## Image Simulations
### simulate noise
To simulate pure noise images with the third-order Lanczos correlation
function, users can use the following command to run the simulation:
```shell
fpfs_sim.py --config ./config_sim_noise.ini --minId 0 --maxId 1 --ncores 1
```
For flags enabling parallel computations, see [this
page](https://schwimmbad.readthedocs.io/en/latest/examples/index.html#selecting-a-pool-with-command-line-arguments).

### stamp-based galaxy image simulations
To simulate parametric galaxies (using Sersic or bulge-disk model fitted to
COSMOS galaxies) or galaxies simulated with random knots (for very small
galaxies). The galaxies are render into 64x64 postage stamps. Note, the
galaxies are isolated here. Users can use the following command to run the
simulation in parallel:
```shell
fpfs_sim.py --config ./config_sim_gal.ini --minId 0 --maxId 1 --ncores 1
```

### blended galaxies
To simulate parametric galaxies (Sersic or bulge-disk) randomly and uniformly
distributed within a circle. Note the galaxies are blended here. It supports
simulations with different number densities. Users can use the following
command to run the simulation in parallel:
```shell
fpfs_sim.py --config ./config_sim_galB.ini --minId 0 --maxId 1 --ncores 1
```

<img src="./simulation_isoblend.png" alt="sim_demo" width="800">

The example here rote each intrinsic galaxy four times by $i\times 45$ degree
where $i=0\dots3$ to remove any spin-4 anisotropy in the intrinsic galaxy
sample.

## Shear Estimation

### stamp-based simulations
First, run the FPFS shear estimator on simulated images to generate shear
catalogs:
```shell
fpfs_procsim.py --config ./config_procsim.ini --minId 0 --maxId 1 --ncores 1
```
Then get the summary statistics (average shear) from the shear catalog:
```shell
fpfs_summary.py --config ./config_procsim.ini --minId 0 --maxId 1 --ncores 1
```
The outputs are the multiplicative biases in different magnitude bins.

### blended galaxies

Again, from images to shear catalogs:
```shell
fpfs_procsim.py --config ./config_procsimB.ini --minId 0 --maxId 1 --ncores 1
```
Then get the summary statistics (average shear) from the shear catalog:
```shell
fpfs_summary.py --config ./config_procsimB.ini --minId 0 --maxId 1 --ncores 1
```
The outputs are the multiplicative biases and additive biases in different
magnitude bins. If you are interested in doing that (simulation, processing,
summary) repeatedly for different input shears, you are able to reproduce the
following plot

<img src="m_vs_gamma2.png" alt="mbias" width="400">

The following [notebook](./shear_perturbation.ipynb) can be used to make the
plots!
Note, the plot is from noiseless image simulation.
