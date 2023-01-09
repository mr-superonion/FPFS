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
    import lsst.afw.math as afwmath
    import lsst.afw.image as afwimg
    import lsst.afw.geom as afwgeom
    import lsst.meas.algorithms as meas_alg

    with_lsst = True
except ImportError:
    with_lsst = False

if with_lsst:

    def make_lsst_exposure(galData, psfData, pixScale, variance):
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
        exposure = afwimg.ExposureF(nx, ny)
        exposure.getMaskedImage().getImage().getArray()[:, :] = galData
        exposure.getMaskedImage().getVariance().getArray()[:, :] = variance
        # Set the PSF
        ngrid_psf = psfData.shape[0]
        psf_lsst = afwimg.ImageF(ngrid_psf, ngrid_psf)
        psf_lsst.getArray()[:, :] = psfData
        psf_lsst = psf_lsst.convertD()
        kernel = afwmath.FixedKernel(psf_lsst)
        kernel_psf = meas_alg.KernelPsf(kernel)
        exposure.setPsf(kernel_psf)
        # prepare the wcs
        # Rotation
        cdelt = pixScale * geom.arcseconds
        cd_matrix = afwgeom.makeCdMatrix(cdelt, geom.Angle(0.0))  # no rotation
        # wcs
        crval = geom.SpherePoint(
            geom.Angle(0.0, geom.degrees), geom.Angle(0.0, geom.degrees)
        )
        # hscpipe6
        # crval   =   afwCoord.IcrsCoord(0.*afwgeom.degrees, 0.*afwgeom.degrees)
        crpix = geom.Point2D(0.0, 0.0)
        data_wcs = afwgeom.makeSkyWcs(crpix, crval, cd_matrix)
        exposure.setWcs(data_wcs)
        # prepare the frc
        data_calib = afwimg.makePhotoCalibFromCalibZeroPoint(63095734448.0194)
        exposure.setPhotoCalib(data_calib)
        return exposure
