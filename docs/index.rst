.. FPFS documentation master file, created by
   sphinx-quickstart on Wed Apr 27 02:50:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FPFS: A fast, accurate shear estimator
======================================
Fourier Power Function Shapelets (FPFS) is an innovative estimator for the
shear responses of galaxy shape, flux, and detection. Utilizing leading-order
perturbations of shear (a vector perturbation) and image noise (a tensor
perturbation), FPFS determines shear and noise responses for both measurements
and detections. Unlike traditional methods that distort each observed galaxy
repeatedly, FPFS employs analytical shear responses of select basis functions,
including Shapelets basis and peak basis. Remarkably efficient, FPFS can
process approximately 1,000 galaxies within a single CPU second. Testing under
simple simulations has proven its capability to maintain a multiplicative shear
estimation bias below 0.5%, even amidst blending challenges.

.. toctree::
    :maxdepth: 2

    Overview <overview.md>

    Examples <examples/README.md>

    fpfs


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
