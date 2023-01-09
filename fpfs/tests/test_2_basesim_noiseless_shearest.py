import fpfs
import galsim
import numpy as np


def simulate_gal_psf(scale, Id0, rcut):
    outDir = "galaxy_basicCenter_psf60"
    psfInt = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(e1=0.02, e2=-0.02)
    psf_data = (
        psfInt.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=64, ny=64, scale=scale)
        .array
    )
    psf_data = psf_data[32 - rcut : 32 + rcut, 32 - rcut : 32 + rcut]
    gal_data = fpfs.simutil.make_basic_sim(
        outDir,
        psfInt=psfInt,
        gname="g1-0000",
        Id0=Id0,
        ny=64,
        nx=256,
        scale=scale,
        do_write=False,
        return_array=True,
    )

    # force detection at center
    indX = np.arange(32, 256, 64)
    indY = np.arange(32, 64, 64)
    inds = np.meshgrid(indY, indX, indexing="ij")
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


def do_test(scale, Id0, rcut):
    thres = 1e-5
    image_list, psf_data = simulate_gal_psf(scale, Id0, rcut)
    fpTask = fpfs.image.measure_source(psf_data, noiFit=0.0, sigma_arcsec=0.7)
    mms = fpTask.measure(image_list)
    ells = fpfs.catalog.fpfsM2E(mms, const=2000, noirev=False)
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
