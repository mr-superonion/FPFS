import numpy as np
import lsst.afw.image as afwImg

def getPositions(dist,angle,gamma1,gamma2):
    #Get the input positions of galaxy1 and galaxy2
    #with reference to the center of the postage stamp
    #(for a 2N*2N stampe, it is at N-0.5)
    dist    =   dist/2.
    xg1     =   -np.cos(angle)*dist
    yg1     =   -np.sin(angle)*dist
    xs1     =   (1+gamma1)*xg1+gamma2*yg1
    ys1     =   gamma2*xg1+(1-gamma1)*yg1
    xg2     =   np.cos(angle)*dist
    yg2     =   np.sin(angle)*dist
    xs2     =   (1+gamma1)*xg2+gamma2*yg2
    ys2     =   gamma2*xg2+(1-gamma1)*yg2
    return xs1,ys1,xs2,ys2


def magFromFlux(flux):
    dataCalib = afwImg.Calib()
    dataCalib.setFluxMag0(63095734448.0194)
    return dataCalib.getMagnitude(flux)
