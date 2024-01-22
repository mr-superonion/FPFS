import fpfs
import galsim
import numpy as np
import jax.numpy as jnp

col_names_2 = (
    "m00",
    "m20",
    "m22c",
    "m22s",
    "m40",
    "m42c",
    "m42s",
    "v0",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v0r1",
    "v1r1",
    "v2r1",
    "v3r1",
    "v4r1",
    "v5r1",
    "v6r1",
    "v7r1",
    "v0r2",
    "v1r2",
    "v2r2",
    "v3r2",
    "v4r2",
    "v5r2",
    "v6r2",
    "v7r2",
)


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
    # test shear estimation
    task = fpfs.image.measure_source(psf_data, pix_scale=scale, sigma_arcsec=0.6)
    # linear observables
    mms = task.measure(gal_data, coords)
    mms = task.get_results(mms)
    assert mms.dtype.names == col_names_2
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

    np.testing.assert_array_almost_equal(coords2["m00"], mms["m00"])
    np.testing.assert_array_almost_equal(coords2["m20"], mms["m20"])
    return


def test_hsc():
    print("Testing HSC-like image")
    do_test(0.168, 2, 16)
    do_test(0.168, 4, 32)
    return


if __name__ == "__main__":
    test_hsc()
