import fpfs
import galsim
import numpy as np

rcut = 32
scale = 0.168
psfInt = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(e1=0.02, e2=-0.02)
psf_data = (
    psfInt.shift(0.5 * scale, 0.5 * scale).drawImage(nx=64, ny=64, scale=scale).array
)
psf_data = psf_data[32 - rcut : 32 + rcut, 32 - rcut : 32 + rcut]
noise_cov = np.ones((rcut * 2, rcut * 2))


def test_basefunc_nord4():
    source_task = fpfs.image.measure_source(
        psf_data,
        nnord=4,
        noiFit=noise_cov,
        sigma_arcsec=0.7,
    )
    noise_task = fpfs.image.measure_noise_cov(
        psf_data,
        nnord=4,
        noiFit=noise_cov,
        sigma_arcsec=0.7,
    )
    np.testing.assert_array_almost_equal(
        np.max(np.abs(source_task.Chi - noise_task.bfunc[0:7])),
        0.0,
    )
    psi_2 = noise_task.bfunc[7:]
    for i in range(3):
        np.testing.assert_array_almost_equal(
            np.max(np.abs(source_task.psi[:, i, :, :] - psi_2[i * 8 : (i + 1) * 8])),
            0.0,
        )
    return


if __name__ == "__main__":
    test_basefunc_nord4()
