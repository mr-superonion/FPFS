import fpfs
import galsim
import numpy as np
import jax.numpy as jnp


def simulate_gal_psf(scale, ind0, rcut):
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )

    psf_data = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=64, ny=64, scale=scale)
        .array
    )
    psf_data = psf_data[32 - rcut : 32 + rcut, 32 - rcut : 32 + rcut]

    gname = "g1-0000"
    gal_data = fpfs.simutil.make_isolate_sim(
        gal_type="mixed",
        sim_method="fft",
        psf_obj=psf_obj,
        gname=gname,
        seed=ind0,
        ny=64,
        nx=256,
        scale=scale,
        do_shift=False,
    )[0]

    # force detection at center
    indx = np.arange(32, 256, 64)
    indy = np.arange(32, 64, 64)
    inds = np.meshgrid(indy, indx, indexing="ij")
    coords = np.vstack(inds).T
    return gal_data, psf_data, coords


def do_test(scale, ind0, rcut):
    gal_data, psf_data, coords = simulate_gal_psf(scale, ind0, rcut)
    # test shear estimation
    fpfs_task = fpfs.image.measure_source(
        psf_data,
        sigma_arcsec=0.6,
        sigma_detect=0.5,
        pix_scale=scale,
    )
    # linear observables
    mms = fpfs_task.measure(gal_data, coords)
    mms = fpfs_task.get_results(mms)
    # test detection
    p1 = 32 - rcut
    p2 = 64 * 2 - rcut
    psf_data2 = jnp.pad(psf_data, ((p1, p1), (p2, p2)))
    img_conv = fpfs.imgutil.convolve2gausspsf(
        gal_data,
        psf_data2,
        fpfs_task.sigmaf,
        fpfs_task.klim,
    )
    m00_sm = np.array([img_conv[tuple(cc)] for cc in coords])
    np.testing.assert_array_almost_equal(m00_sm, mms["fpfs_M00"] * scale**2.0)
    img_conv_det = fpfs.imgutil.convolve2gausspsf(
        gal_data,
        psf_data2,
        fpfs_task.sigmaf_det,
        fpfs_task.klim,
    )
    img_u = img_conv_det - jnp.roll(img_conv_det, shift=-1, axis=-1)
    v0_sm = np.array([img_u[tuple(cc)] for cc in coords])
    np.testing.assert_array_almost_equal(v0_sm, mms["fpfs_v0"] * scale**2.0)

    img_u = img_conv_det - jnp.roll(img_conv_det, shift=-1, axis=-2)
    v2_sm = np.array([img_u[tuple(cc)] for cc in coords])
    np.testing.assert_array_almost_equal(v2_sm, mms["fpfs_v2"] * scale**2.0)

    img_u = img_conv_det - jnp.roll(img_conv_det, shift=1, axis=-1)
    v4_sm = np.array([img_u[tuple(cc)] for cc in coords])
    np.testing.assert_array_almost_equal(v4_sm, mms["fpfs_v4"] * scale**2.0)

    img_u = img_conv_det - jnp.roll(img_conv_det, shift=1, axis=-2)
    v6_sm = np.array([img_u[tuple(cc)] for cc in coords])
    np.testing.assert_array_almost_equal(v6_sm, mms["fpfs_v6"] * scale**2.0)
    return


def test_detection():
    do_test(0.2, 1, 16)
    do_test(0.168, 2, 16)
    return


if __name__ == "__main__":
    test_detection()
