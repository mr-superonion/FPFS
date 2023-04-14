import fpfs
import galsim
import numpy as np


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
    coords = np.array(
        np.zeros(inds[0].size),
        dtype=[("fpfs_y", "i4"), ("fpfs_x", "i4")],
    )
    coords["fpfs_y"] = np.ravel(inds[0])
    coords["fpfs_x"] = np.ravel(inds[1])
    image_list = [
        gal_data[
            cc["fpfs_y"] - rcut : cc["fpfs_y"] + rcut,
            cc["fpfs_x"] - rcut : cc["fpfs_x"] + rcut,
        ]
        for cc in coords
    ]
    return image_list, psf_data


def do_test(scale, ind0, rcut):
    thres = 1e-5
    image_list, psf_data = simulate_gal_psf(scale, ind0, rcut)
    fpfs_task = fpfs.image.measure_source(psf_data, sigma_arcsec=0.7)
    # linear observables
    mms = fpfs_task.measure(image_list)
    # non-linear observables
    ells = fpfs.catalog.fpfs_m2e(mms, const=2000)
    resp = np.average(ells["fpfs_R1E"])
    shear = np.average(ells["fpfs_e1"]) / resp
    assert np.all(np.abs(shear + 0.02) < thres)
    return


def test_hsc():
    print("Testing HSC-like image")
    do_test(0.168, 2, 16)
    return


if __name__ == "__main__":
    test_hsc()
