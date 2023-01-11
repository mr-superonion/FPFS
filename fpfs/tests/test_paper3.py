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
outLM = pyfits.getdata(cat_fname)
colnames = list(outLM.dtype.names)


def test_noise_cov():
    """Test the consistency between base functions of measure_source and
    measure_noise_cov
    """
    # Test whether the fpfs version is consistent with paper3
    source_task = fpfs.image.measure_source(
        psf_data,
        nnord=4,
        noise_ps=noise_pow,
        sigma_arcsec=0.45,
    )
    mms = source_task.measure(np.ones((64, 64)))
    for cn in colnames[33:]:
        np.testing.assert_array_almost_equal(
            outLM[0][cn],
            mms[0][cn],
        )
    noise_task = fpfs.image.measure_noise_cov(
        psf_data,
        nnord=4,
        sigma_arcsec=0.45,
    )

    # Test whether the impt version is consistent with paper3
    cov_mat = noise_task.measure(noise_pow)
    cov_mat2 = fpfs.catalog.fpfscov_to_imptcov(mms)
    for elem1, elem2 in zip(cov_mat.flatten(), cov_mat2.flatten()):
        if elem2 != 0.0:
            np.testing.assert_array_almost_equal(
                elem1,
                elem2,
            )
    return


if __name__ == "__main__":
    test_noise_cov()
