# uniCalib_Private



# Goal
Whether we can apply the calibration from i band to gr(i)zy band image of HSC survey? \\

# plan
sim:
    make different galaxy image simulations 
    rgc: COSMOS image 
    cgc: Sersic model galaxy image
        oneShear: two blended galaxies have the same shear (same redshift plan)
        twoShear: two blended galaxies have the different shear (different redshift plan)
proc:
    process the simulation with lsst pipeline and FPFS estimator

ana: 
    use machine learning (Gaussian Process..) to study the galaxy bias (multiplicative bias and fractional additive bias)


