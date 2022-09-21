# Examples

This directory gives some examples on simulating galaxy images with Galsim; and
galaxy detection and shape estimation with FPFS. Before running the sripts,
please source [this file](../bin/fpfs_config.sh) after modifying it.

## Image Simulations
### simulate noise
To simulate pure noise images with the third-order Lanczos correlation
function, users can use the following command to run the simulation:
```shell
fpfs_sim.py --config ./config_sim_noise.ini --minId 0 --maxId 1 --ncores 1
```
For flags enabling parallel computations, see [this
page](https://schwimmbad.readthedocs.io/en/latest/examples/index.html#selecting-a-pool-with-command-line-arguments).

### simulate stamp-based galaxies
To simulate parametric galaxies (using Sersic or bulge-disk model fitted to
COSMOS galaxies) or galaxies simulated with random knots (for very small
galaxies). The galaxies are render into 64x64 postage stamps. Note, the
galaxies are isolated here. Users can use the following command to run the
simulation in parallel:
```shell
fpfs_sim.py --config ./config_sim_gal.ini --minId 0 --maxId 1 --ncores 1
```

### simulate galaxies in the universe
To simulate parametric galaxies (Sersic or bulge-disk) randomly and uniformly
distributed within a circle. Note the galaxies are blended here. It supports
simulations with different number densities. Users can use the following
command to run the simulation in parallel:
```shell
fpfs_sim.py --config ./config_sim_galB.ini --minId 0 --maxId 1 --ncores 1
```
