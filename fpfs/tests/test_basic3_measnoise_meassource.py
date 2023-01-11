import fpfs
import galsim
import numpy as np

""" This test checks the consistency between measure_source and
measure_noise_cov
"""

rcut = 32
scale = 0.168
psfInt = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(e1=0.02, e2=-0.02)
psf_data = (
    psfInt.shift(0.5 * scale, 0.5 * scale).drawImage(nx=64, ny=64, scale=scale).array
)
psf_data = psf_data[32 - rcut : 32 + rcut, 32 - rcut : 32 + rcut]
noise_ps = np.ones((rcut * 2, rcut * 2))
noise_ps[rcut - 2 : rcut + 2, rcut - 2 : rcut + 2] = 2.0


def test_basefunc_nord4():
    """Test the consistency between base functions of measure_source and
    measure_noise_cov
    """
    source_task = fpfs.image.measure_source(
        psf_data,
        nnord=4,
        noise_ps=noise_ps,
        sigma_arcsec=0.7,
    )
    noise_task = fpfs.image.measure_noise_cov(
        psf_data,
        nnord=4,
        sigma_arcsec=0.7,
    )
    np.testing.assert_array_almost_equal(
        np.max(np.abs(source_task.chi - noise_task.bfunc[0:7])),
        0.0,
    )
    psi_2 = noise_task.bfunc[7:]
    for i in range(3):
        np.testing.assert_array_almost_equal(
            np.max(np.abs(source_task.psi0[:, i, :, :] - psi_2[i * 8 : (i + 1) * 8])),
            0.0,
        )
    return


def test_cov_elements():
    """Test the consistency between covariance elements of measure_source and
    measure_noise_cov
    """
    noise_task = fpfs.image.measure_noise_cov(
        psf_data,
        nnord=4,
        sigma_arcsec=0.7,
    )
    cov_mat = noise_task.measure(noise_ps)

    source_task = fpfs.image.measure_source(
        psf_data,
        nnord=4,
        noise_ps=noise_ps,
        sigma_arcsec=0.7,
    )
    mms = source_task.measure(noise_ps)
    cov_mat2 = fpfs.catalog.fpfscov_to_imptcov(mms)
    for elem1, elem2 in zip(cov_mat.flatten(), cov_mat2.flatten()):
        if elem2 != 0.0:
            np.testing.assert_array_almost_equal(
                elem1,
                elem2,
            )
    return


if __name__ == "__main__":
    test_basefunc_nord4()
