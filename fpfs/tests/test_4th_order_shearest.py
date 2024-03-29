import fpfs
import galsim
import numpy as np

""" This test checks the high-order linear shear estimator
"""

col_names_4 = (
    "fpfs_M00",
    "fpfs_M20",
    "fpfs_M22c",
    "fpfs_M22s",
    "fpfs_M40",
    "fpfs_M42c",
    "fpfs_M42s",
    "fpfs_M60",
    "fpfs_v0",
    "fpfs_v1",
    "fpfs_v2",
    "fpfs_v3",
    "fpfs_v4",
    "fpfs_v5",
    "fpfs_v6",
    "fpfs_v7",
    "fpfs_v0r1",
    "fpfs_v1r1",
    "fpfs_v2r1",
    "fpfs_v3r1",
    "fpfs_v4r1",
    "fpfs_v5r1",
    "fpfs_v6r1",
    "fpfs_v7r1",
    "fpfs_v0r2",
    "fpfs_v1r2",
    "fpfs_v2r2",
    "fpfs_v3r2",
    "fpfs_v4r2",
    "fpfs_v5r2",
    "fpfs_v6r2",
    "fpfs_v7r2",
)


def simulate_gal_psf(scale, ind0, rcut, gname):
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )

    psf_data = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=64, ny=64, scale=scale)
        .array
    )
    psf_data = psf_data[32 - rcut : 32 + rcut, 32 - rcut : 32 + rcut]
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
    inds = np.meshgrid(indy, indx, indexing="ij")
    coords = np.vstack(inds).T
    return gal_data, psf_data, coords


def get_multiplicative_bias(scale, ind0, rcut):
    gnames = ["g1-0", "g1-1"]
    mm_names = {
        "g1-1": "p",
        "g1-0": "n",
    }
    mms = dict()
    for gname in gnames:
        gal, psf, coords = simulate_gal_psf(
            scale=scale, ind0=ind0, rcut=rcut, gname=gname
        )
        fpfs_task = fpfs.image.measure_source(
            psf, pix_scale=scale, sigma_arcsec=0.5, nnord=6,
        )
        out = fpfs_task.get_results(fpfs_task.measure(gal, coords))
        assert out.dtype.names == col_names_4
        mms[mm_names[gname]] = out
    num = np.average(mms["p"]["fpfs_M42c"] - mms["n"]["fpfs_M42c"])
    denom = (
        0.02
        * np.sqrt(6)
        / 2
        * np.average(
            mms["p"]["fpfs_M20"]
            - mms["p"]["fpfs_M60"]
            + mms["n"]["fpfs_M20"]
            - mms["n"]["fpfs_M60"]
        )
    )
    m_bias = num / denom - 1
    return m_bias


def do_test(scale, ind0, rcut):
    m_bias = get_multiplicative_bias(scale, ind0, rcut)
    assert np.all(np.abs(m_bias) < 0.003)
    return


def test_hsc():
    print("Testing HSC-like image")
    do_test(0.168, 2, 16)
    return


if __name__ == "__main__":
    test_hsc()
