import jax
import fpfs
import galsim
import numpy as np
import jax.numpy as jnp


def simulate_gal_psf(scale, ind0, rcut):
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
    gname = "g1-0"
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


def do_test(scale, ind0, rcut):
    c_thres = 3e-5
    m_thres = 3e-3
    gal_data, psf_data, coords = simulate_gal_psf(scale, ind0, rcut)
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
    print("run measurement")
    # linear observables
    mms = task.measure(gal_data, coords)
    print(mms.shape)
    print("finish measurement")

    # new version
    cat_obj = fpfs.catalog.fpfs_catalog(
        snr_min=0.0,
        r2_min=0.0,
        sigma_m00=0.4,
        sigma_r2=0.8,
        pthres=0.0,
        pratio=0.0,
        sigma_v=0.8,
        det_nrot=det_nrot,
    )
    print("run summary")
    outcome = jnp.sum(
        jax.lax.map(jax.jit(cat_obj.measure_g1_noise_correct), mms), axis=0
    )
    print("finish summary")
    shear1 = outcome[0] / outcome[1]
    mbias = np.abs(shear1 + 0.02) / 0.02
    print(mbias)
    assert np.all(mbias < m_thres)
    outcome = jnp.sum(
        jax.lax.map(jax.jit(cat_obj.measure_g2_noise_correct), mms), axis=0
    )
    shear2 = outcome[0] / outcome[1]
    cbias = np.abs(shear2)
    print(cbias)
    assert np.all(cbias < c_thres)

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
    for i in range(12, 40, 5):
        do_test(0.168, i, 16)
    # do_test(0.168, 41, 32)
    return


if __name__ == "__main__":
    test_hsc()
