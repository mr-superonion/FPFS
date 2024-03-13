import galsim
import jax.numpy as jnp
import numpy as np

import fpfs


def test_rotate():
    scale = 0.168
    ngrid = 64
    image = np.zeros((ngrid, ngrid))
    image[ngrid // 2 + 10, ngrid // 2 - 2] = 1.0
    image2 = fpfs.image.util.rotate90(image)
    np.testing.assert_approx_equal(image2[ngrid // 2 - 2, ngrid // 2 - 10], 1.0)

    psf_ini = np.zeros((ngrid, ngrid))
    psf_ini[ngrid // 2, ngrid // 2] = 1.0
    # test shear estimation
    task = fpfs.image.measure_source(
        psf_array=psf_ini,
        pix_scale=scale,
        sigma_arcsec=0.53,
        nord=4,
        det_nrot=4,
    )
    # linear observables
    mms = task.measure(image + image2).T
    e1 = mms[task.di["m22c"]] / (mms[task.di["m00"]] + mms[task.di["m40"]])
    e2 = mms[task.di["m22s"]] / (mms[task.di["m00"]] + mms[task.di["m40"]])
    np.testing.assert_almost_equal(e1[0], 0.0)
    np.testing.assert_almost_equal(e2[0], 0.0)
    return


def test_rotate2():
    scale = 0.168
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )
    ngrid = 64
    psf_data = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )

    psf_data2 = (
        psf_obj.rotate(90 * galsim.degrees)
        .shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )
    assert jnp.max(jnp.abs(fpfs.image.util.rotate90(psf_data) - psf_data2)) < 1e-7


if __name__ == "__main__":
    test_rotate()
