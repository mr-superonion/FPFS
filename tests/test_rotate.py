import fpfs
import numpy as np


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


if __name__ == "__main__":
    test_rotate()
