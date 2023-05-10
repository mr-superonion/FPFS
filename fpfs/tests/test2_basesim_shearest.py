import fpfs
import galsim
import numpy as np
import jax.numpy as jnp


def simulate_gal_psf(scale, ind0, rcut):
    out_dir = "galaxy_basicCenter_psf60"
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )

    psf_data = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=64, ny=64, scale=scale)
        .array
    )
    psf_data = psf_data[32 - rcut : 32 + rcut, 32 - rcut : 32 + rcut]
    gal_data = fpfs.simutil.make_basic_sim(
        out_dir,
        psf_obj=psf_obj,
        gname="g1-0000",
        ind0=ind0,
        ny=64,
        nx=256,
        scale=scale,
        do_write=False,
        return_array=True,
    )

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
    fpfs_task = fpfs.image.measure_source(psf_data, sigma_arcsec=0.7)
    # linear observables
    mms = fpfs_task.measure(gal_data, coords)
    mms = fpfs_task.get_results(mms)
    # non-linear observables
    ells = fpfs.catalog.fpfs_m2e(mms, const=2000)
    resp = np.average(ells["fpfs_R1E"])
    shear = np.average(ells["fpfs_e1"]) / resp
    assert np.all(np.abs(shear + 0.02) < thres)
    # test detection
    p1 = 32 - rcut
    p2 = 64 * 2 - rcut
    psf_data2 = jnp.pad(psf_data, ((p1, p1), (p2, p2)))
    coords2 = fpfs.image.detect_sources(
        gal_data,
        psf_data2,
        gsigma=0.24,
        thres=0.01,
        thres2=0.00,
        klim=fpfs_task.klim,
    )
    assert np.all(coords2 == coords)
    return


def test_hsc():
    print("Testing HSC-like image")
    do_test(0.168, 2, 16)
    do_test(0.168, 4, 32)
    return


if __name__ == "__main__":
    test_hsc()
