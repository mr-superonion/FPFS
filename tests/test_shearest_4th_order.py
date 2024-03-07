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


def do_test(scale, ind0, rcut):
    c_thres = 3e-5
    m_thres = 3e-3
    gal_data, psf_data, coords = simulate_gal_psf(scale, ind0, rcut)
    nord = 6

    # test shear estimation
    task = fpfs.image.measure_source(
        psf_data,
        pix_scale=scale,
        sigma_arcsec=0.53,
        nord=nord,
    )
    print("run measurement")
    # linear observables
    mms = task.measure(gal_data, coords)
    print("finish measurement")

    # new version
    cat_obj = fpfs.catalog.fpfs4_catalog(
        snr_min=0.0,
        r2_min=0.0,
        sigma_m00=0.4,
        sigma_r2=0.8,
        sigma_v=0.002,
        pthres=0.00,
        pratio=0.00,
    )
    outcome = jnp.sum(cat_obj.measure_g1_denoise(mms), axis=0)
    shear1 = outcome[0] / outcome[1]
    mbias = np.abs(shear1 + 0.02) / 0.02
    print(mbias)
    assert np.all(mbias < m_thres)
    outcome = jnp.sum(cat_obj.measure_g2_denoise(mms), axis=0)
    shear2 = outcome[0] / outcome[1]
    cbias = np.abs(shear2)
    print(cbias)
    assert np.all(cbias < c_thres)
    return


def test_hsc():
    print("Testing HSC-like image")
    do_test(0.168, 12, 16)
    do_test(0.168, 45, 32)
    return


if __name__ == "__main__":
    test_hsc()
