# Copyright 20210905 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib
try:
    import lsst.geom as geom
    import lsst.afw.math as afwMath
    import lsst.afw.image as afwImg
    import lsst.afw.geom as afwGeom
    import lsst.meas.algorithms as meaAlg

    with_lsst = True
except ImportError as error:
    with_lsst = False

if with_lsst:

    def makeLsstExposure(galData, psfData, pixScale, variance):
        """Makes an LSST exposure object

        Args:
            galData (ndarray):  array of galaxy image
            psfData (ndarray):  array of PSF image
            pixScale (float):   pixel scale
            variance (float):   noise variance

        Returns:
            exposure:   LSST exposure object
        """
        if not with_lsst:
            raise ImportError("Do not have lsstpipe!")
        ny, nx = galData.shape
        exposure = afwImg.ExposureF(nx, ny)
        exposure.getMaskedImage().getImage().getArray()[:, :] = galData
        exposure.getMaskedImage().getVariance().getArray()[:, :] = variance
        # Set the PSF
        ngridPsf = psfData.shape[0]
        psfLsst = afwImg.ImageF(ngridPsf, ngridPsf)
        psfLsst.getArray()[:, :] = psfData
        psfLsst = psfLsst.convertD()
        kernel = afwMath.FixedKernel(psfLsst)
        kernelPSF = meaAlg.KernelPsf(kernel)
        exposure.setPsf(kernelPSF)
        # prepare the wcs
        # Rotation
        cdelt = pixScale * geom.arcseconds
        CD = afwGeom.makeCdMatrix(cdelt, geom.Angle(0.0))  # no rotation
        # wcs
        crval = geom.SpherePoint(
            geom.Angle(0.0, geom.degrees), geom.Angle(0.0, geom.degrees)
        )
        # crval   =   afwCoord.IcrsCoord(0.*afwGeom.degrees, 0.*afwGeom.degrees) # hscpipe6
        crpix = geom.Point2D(0.0, 0.0)
        dataWcs = afwGeom.makeSkyWcs(crpix, crval, CD)
        exposure.setWcs(dataWcs)
        # prepare the frc
        dataCalib = afwImg.makePhotoCalibFromCalibZeroPoint(63095734448.0194)
        exposure.setPhotoCalib(dataCalib)
        return exposure
