# log

## 20201122
+ update ms
    - define $\frac{M_{22c}}{M_{20}}$ as ellipticity
    - define $\frac{M_{00}}{M_{20}}$ as FPFS flux ratio for selection

## 20201121
+ prepare to use higher order shaplets to construct shear estimator
    - avoid $e_{1,2}^2$ in the response
    - avoid $\frac{M_{00}}{M_{00}+C}$ in the response
    - Those terms boost the noise bias and require a large $C$.
+ prepare the galaxy image simulation
    - use BDK
    - use HSC S16A PSF and noise variance
    - 25000000 galaxies

