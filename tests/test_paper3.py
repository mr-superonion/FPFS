import os
import fpfs
import galsim
import numpy as np
import astropy.io.fits as pyfits

""" This test checks the consistency with the result of paper: Li & Mandelbaum
to make sure we can always reproduce the published result for every version of
the code
"""

scale = 0.168  # HSC pixel size
seeing = 0.6  # HSC-like PSF
rcut = 32

psf_obj = galsim.Moffat(beta=3.5, fwhm=seeing, trunc=seeing * 4.0).shear(
    e1=0.02, e2=-0.02
)
psf_data = (
    psf_obj.shift(0.5 * scale, 0.5 * scale).drawImage(nx=64, ny=64, scale=scale).array
)

noise_fname = os.path.join(fpfs.__data_dir__, "noiPows3.npy")
noi_var = 7e-3  # about 2 times of HSC average
noise_pow = np.load(noise_fname, allow_pickle=True).item()["%s" % rcut] * noi_var * 100
cat_fname = os.path.join(fpfs.__data_dir__, "fpfs-cut32-0000-g1-0000.fits")
mms = pyfits.getdata(cat_fname)
colnames = list(mms.dtype.names)


def test_noise_cov():
    """Test the consistency between base functions of
    measure_noise_cov and the paper3's measurement
    """
    noise_task = fpfs.image.measure_noise_cov(
        psf_data,
        pix_scale=scale,
        sigma_arcsec=0.45,
        sigma_detect=0.45,
        nnord=4,
    )

    # Test whether the impt version is consistent with paper3
    cov_mat = noise_task.measure(noise_pow)
    cov_mat2 = fpfscov_to_imptcov(mms)
    for elem1, elem2 in zip(cov_mat.flatten(), cov_mat2.flatten()):
        if elem2 != 0.0:
            np.testing.assert_array_almost_equal(
                elem1,
                elem2,
            )
    mms2 = imptcov_to_fpfscov(cov_mat2)
    for cn in cov_names:
        np.testing.assert_almost_equal(mms2[cn], mms[cn][0])
    return


ncol = 31


def fpfscov_to_imptcov(data):
    """Converts FPFS noise Covariance elements into a covariance matrix of
    lensPT.

    Args:
        data (ndarray):     FPFS shapelet mode catalog
    Returns:
        out (ndarray):      Covariance matrix
    """
    # the colum names
    # M00 -> N00; v1 -> V1
    ll = [cn[5:].replace("M", "N").replace("v", "V") for cn in col_names]
    out = np.zeros((ncol, ncol))
    for i in range(ncol):
        for j in range(ncol):
            try:
                try:
                    cname = "fpfs_%s%s" % (ll[i], ll[j])
                    out[i, j] = data[cname][0]
                except (ValueError, KeyError):
                    cname = "fpfs_%s%s" % (ll[j], ll[i])
                    out[i, j] = data[cname][0]
            except (ValueError, KeyError):
                out[i, j] = 0.0
    return out


def imptcov_to_fpfscov(data):
    """Converts FPFS noise Covariance elements into a covariance matrix of
    lensPT.

    Args:
        data (ndarray):     impt covariance matrix
    Returns:
        out (ndarray):      FPFS covariance elements
    """
    # the colum names
    # M00 -> N00; v1 -> V1
    ll = [cn[5:].replace("M", "N").replace("v", "V") for cn in col_names]
    types = [(cn, "<f8") for cn in cov_names]
    out = np.zeros(1, dtype=types)
    for i in range(ncol):
        for j in range(i, ncol):
            cname = "fpfs_%s%s" % (ll[i], ll[j])
            if cname in cov_names:
                out[cname][0] = data[i, j]
    return out


col_names = [
    "fpfs_M00",
    "fpfs_M20",
    "fpfs_M22c",
    "fpfs_M22s",
    "fpfs_M40",
    "fpfs_M42c",
    "fpfs_M42s",
    "fpfs_v0",
    "fpfs_v1",
    "fpfs_v2",
    "fpfs_v3",
    "fpfs_v4",
    "fpfs_v5",
    "fpfs_v6",
    "fpfs_v7",
    "fpfs_v0r1",
    "fpfs_v1r1",
    "fpfs_v2r1",
    "fpfs_v3r1",
    "fpfs_v4r1",
    "fpfs_v5r1",
    "fpfs_v6r1",
    "fpfs_v7r1",
    "fpfs_v0r2",
    "fpfs_v1r2",
    "fpfs_v2r2",
    "fpfs_v3r2",
    "fpfs_v4r2",
    "fpfs_v5r2",
    "fpfs_v6r2",
    "fpfs_v7r2",
]

cov_names = [
    "fpfs_N00N00",
    "fpfs_N20N20",
    "fpfs_N22cN22c",
    "fpfs_N22sN22s",
    "fpfs_N40N40",
    "fpfs_N00N20",
    "fpfs_N00N22c",
    "fpfs_N00N22s",
    "fpfs_N00N40",
    "fpfs_N00N42c",
    "fpfs_N00N42s",
    "fpfs_N20N22c",
    "fpfs_N20N22s",
    "fpfs_N20N40",
    "fpfs_N22cN42c",
    "fpfs_N22sN42s",
    "fpfs_N00V0",
    "fpfs_N00V0r1",
    "fpfs_N00V0r2",
    "fpfs_N22cV0",
    "fpfs_N22sV0",
    "fpfs_N22cV0r1",
    "fpfs_N22sV0r2",
    "fpfs_N40V0",
    "fpfs_N00V1",
    "fpfs_N00V1r1",
    "fpfs_N00V1r2",
    "fpfs_N22cV1",
    "fpfs_N22sV1",
    "fpfs_N22cV1r1",
    "fpfs_N22sV1r2",
    "fpfs_N40V1",
    "fpfs_N00V2",
    "fpfs_N00V2r1",
    "fpfs_N00V2r2",
    "fpfs_N22cV2",
    "fpfs_N22sV2",
    "fpfs_N22cV2r1",
    "fpfs_N22sV2r2",
    "fpfs_N40V2",
    "fpfs_N00V3",
    "fpfs_N00V3r1",
    "fpfs_N00V3r2",
    "fpfs_N22cV3",
    "fpfs_N22sV3",
    "fpfs_N22cV3r1",
    "fpfs_N22sV3r2",
    "fpfs_N40V3",
    "fpfs_N00V4",
    "fpfs_N00V4r1",
    "fpfs_N00V4r2",
    "fpfs_N22cV4",
    "fpfs_N22sV4",
    "fpfs_N22cV4r1",
    "fpfs_N22sV4r2",
    "fpfs_N40V4",
    "fpfs_N00V5",
    "fpfs_N00V5r1",
    "fpfs_N00V5r2",
    "fpfs_N22cV5",
    "fpfs_N22sV5",
    "fpfs_N22cV5r1",
    "fpfs_N22sV5r2",
    "fpfs_N40V5",
    "fpfs_N00V6",
    "fpfs_N00V6r1",
    "fpfs_N00V6r2",
    "fpfs_N22cV6",
    "fpfs_N22sV6",
    "fpfs_N22cV6r1",
    "fpfs_N22sV6r2",
    "fpfs_N40V6",
    "fpfs_N00V7",
    "fpfs_N00V7r1",
    "fpfs_N00V7r2",
    "fpfs_N22cV7",
    "fpfs_N22sV7",
    "fpfs_N22cV7r1",
    "fpfs_N22sV7r2",
    "fpfs_N40V7",
]


if __name__ == "__main__":
    test_noise_cov()
