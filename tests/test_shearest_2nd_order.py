import fpfs
import galsim
import numpy as np
import jax.numpy as jnp


def simulate_gal_psf(scale, ind0, rcut, gcomp="g1"):
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )
    ngrid = 64
    nrot = 12

    psf_data = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )
    psf_data = psf_data[
        ngrid // 2 - rcut : ngrid // 2 + rcut, ngrid // 2 - rcut : ngrid // 2 + rcut
    ]
    gname = "%s-0" % gcomp
    gal_data = fpfs.simulation.make_isolate_sim(
        gal_type="mixed",
        sim_method="fft",
        psf_obj=psf_obj,
        gname=gname,
        seed=ind0,
        ny=ngrid,
        nx=ngrid * nrot,
        scale=scale,
        do_shift=False,
        buff=0,
        nrot_per_gal=nrot,
    )[0]

    # force detection at center
    indx = np.arange(ngrid // 2, ngrid * nrot, ngrid)
    indy = np.arange(ngrid // 2, ngrid, ngrid)
    inds = np.meshgrid(indy, indx, indexing="ij")
    coords = np.vstack(inds).T
    return gal_data, psf_data, coords


def do_test(scale, ind0, rcut, gcomp):
    c_thres = 5e-5
    gal_data, psf_data, coords = simulate_gal_psf(scale, ind0, rcut, gcomp)
    nord = 4
    det_nrot = 4

    # test shear estimation
    task = fpfs.image.measure_source(
        psf_data,
        pix_scale=scale,
        sigma_arcsec=0.53,
        nord=nord,
        det_nrot=det_nrot,
    )
    # linear observables
    mms = task.measure(gal_data, coords)
    if gcomp == "g1":
        g1 = -0.02
        g2 = 0.0
    elif gcomp == "g2":
        g1 = 0.0
        g2 = -0.02
    else:
        raise ValueError("gcomp should be g1 or g2")

    # new version
    cat_obj = fpfs.catalog.fpfs_catalog(
        snr_min=0.0,
        r2_min=0.0,
        sigma_m00=0.4,
        sigma_r2=0.8,
        sigma_v=0.002,
        pthres=0.00,
        pratio=0.00,
        det_nrot=det_nrot,
    )
    print("Renoise")
    outcome = jnp.sum(cat_obj.measure_g1_renoise(mms), axis=0)
    shear1 = outcome[0] / outcome[1]
    bias1 = np.abs(shear1 - g1)
    print(bias1)
    assert np.all(bias1 < c_thres)
    outcome = jnp.sum(cat_obj.measure_g2_renoise(mms), axis=0)
    shear2 = outcome[0] / outcome[1]
    bias2 = np.abs(shear2 - g2)
    print(bias2)
    assert np.all(bias2 < c_thres)

    print("Denoise")
    outcome = jnp.sum(cat_obj.measure_g1_denoise(mms), axis=0)
    shear1 = outcome[0] / outcome[1]
    bias1 = np.abs(shear1 - g1)
    print(bias1)
    assert np.all(bias1 < c_thres)
    outcome = jnp.sum(cat_obj.measure_g2_denoise(mms), axis=0)
    shear2 = outcome[0] / outcome[1]
    bias2 = np.abs(shear2 - g2)
    print(bias2)
    assert np.all(bias2 < c_thres)

    # test detection
    p1 = gal_data.shape[0] // 2 - rcut
    p2 = gal_data.shape[1] // 2 - rcut
    psf_data2 = jnp.pad(psf_data, ((p1, p1), (p2, p2)))
    print("run detection")
    coords2 = task.detect_source(
        gal_data,
        psf_data2,
        cov_elem=np.eye(task.ncol),
        fthres=0.01,
        pthres=0.00,
        bound=4,
    )
    coords2 = task.get_results_detection(coords2)
    print("finish detection")
    return


def test_hsc():
    print("Testing HSC-like image")
    do_test(0.168, 42, 32, "g1")
    do_test(0.168, 42, 32, "g2")
    return


if __name__ == "__main__":
    test_hsc()
