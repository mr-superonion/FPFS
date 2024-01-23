import jax
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
    gname = "g1-0"
    gal_data = fpfs.simulation.make_isolate_sim(
        gal_type="mixed",
        sim_method="fft",
        psf_obj=psf_obj,
        gname=gname,
        seed=ind0,
        ny=64,
        nx=256,
        scale=scale,
        do_shift=False,
        buff=0,
    )[0]

    # force detection at center
    indx = np.arange(32, 256, 64)
    indy = np.arange(32, 64, 64)
    inds = np.meshgrid(indy, indx, indexing="ij")
    coords = np.vstack(inds).T
    return gal_data, psf_data, coords


def do_test(scale, ind0, rcut):
    thres = 1e-5
    gal_data, psf_data, coords = simulate_gal_psf(scale, ind0, rcut)
    detect_nrot = 8
    detect_return_peak_modes = True

    # test shear estimation
    task = fpfs.image.measure_source(
        psf_data,
        pix_scale=scale,
        sigma_arcsec=0.55,
        sigma_detect=0.53,
        detect_nrot=detect_nrot,
        detect_return_peak_modes=detect_return_peak_modes,
    )
    # linear observables
    mms = task.measure(gal_data, coords)

    # new version
    cat_obj = fpfs.catalog.FpfsCatalog(
        m00_min=0.0,
        r2_min=0.0,
        v_min=0.0,
        sigma_m00=0.4,
        sigma_r2=0.8,
        sigma_v=0.2,
    )
    outcome = jnp.sum(
        jax.lax.map(jax.jit(cat_obj.measure_g1_noise_correct), mms), axis=0
    )
    shear1 = outcome[0] / outcome[1]
    assert np.all(np.abs(shear1 + 0.02) < thres)
    outcome = jnp.sum(
        jax.lax.map(jax.jit(cat_obj.measure_g2_noise_correct), mms), axis=0
    )
    shear2 = outcome[0] / outcome[1]
    assert np.all(np.abs(shear2) < 5e-5)

    # older version
    mms = task.get_results(mms)
    # non-linear observables
    ells = fpfs.catalog.m2e(mms, const=20)
    resp = np.average(ells["R1E"])
    shear = np.average(ells["e1"]) / resp
    assert np.all(np.abs(shear + 0.02) < thres)

    # test detection
    p1 = 32 - rcut
    p2 = 64 * 2 - rcut
    psf_data2 = jnp.pad(psf_data, ((p1, p1), (p2, p2)))
    coords2 = task.detect_sources(
        gal_data,
        psf_data2,
        thres=0.01,
        thres2=0.00,
        bound=4,
    )
    coords2 = task.get_results_detection(coords2)
    assert np.all(coords2["y"] == coords[:, 0])
    assert np.all(coords2["x"] == coords[:, 1])
    np.testing.assert_array_almost_equal(coords2["m00"], mms["m00"])
    np.testing.assert_array_almost_equal(coords2["m20"], mms["m20"])
    if detect_return_peak_modes:
        for _ in range(detect_nrot):
            np.testing.assert_array_almost_equal(coords2["v%d" % _], mms["v%d" % _])
    return


def test_hsc():
    print("Testing HSC-like image")
    do_test(0.168, 2, 16)
    do_test(0.168, 4, 32)
    return


if __name__ == "__main__":
    test_hsc()
