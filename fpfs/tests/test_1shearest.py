import gc
import time
import fpfs
import numpy as np
import numpy.lib.recfunctions as rfn


def analyze_fpfs(rng, input_shear, num_gals, noi_stds, noi_psf=1e-9):
    noi_stds = np.array(noi_stds)
    if len(noi_stds.shape) == 0:
        noi_stds = np.array([noi_stds])
    elif len(noi_stds.shape) >= 2:
        raise ValueError("The input noi_stds should be float or 1d list")
    y = []
    y_err = []
    test_task = fpfs.simutil.sim_test(shear=input_shear, rng=rng)
    ngrid = test_task.psf.shape[0]
    rcut = 16
    beg = ngrid // 2 - rcut
    end = beg + 2 * rcut
    psf = test_task.psf[beg:end, beg:end]
    gc.collect()
    num_tests = noi_stds.size
    for i in range(num_tests):
        noii = noi_stds[i]
        # initialize fpfs task with psf and noise variance
        fpfs_task = fpfs.image.measure_source(
            psf,
            pix_scale=0.168,
            sigma_arcsec=0.59,
        )
        if noii <= 1e-10:
            print("noise level is too small; we only simulate one galaxy")
            num_tmp = 1
        else:
            num_tmp = num_gals
            print("test for noisy galaxies")
        start = time.time()
        results = []
        for _ in range(num_tmp):
            gal = test_task.make_image(noise=noi_stds[i], psf_noise=noi_psf)[0]
            ngrid = gal.shape[0]
            beg = ngrid // 2 - rcut
            end = beg + 2 * rcut
            gal = gal[beg:end, beg:end]
            modes = fpfs_task.measure(gal)
            modes = fpfs_task.get_results(modes)
            results.append(modes)
            del gal, modes, beg, end, ngrid
        end = time.time()
        print("%.5f seconds to process %d galaxies" % (end - start, num_tmp))
        mms = rfn.stack_arrays(results, usemask=False)
        ells = fpfs.catalog.fpfs_m2e(mms, const=2000)
        del mms, results
        resp = np.average(ells["fpfs_R1E"])
        shear = np.average(ells["fpfs_e1"]) / resp
        shear_err = np.std(ells["fpfs_e1"]) / np.abs(resp) / np.sqrt(num_gals)
        y.append(shear)
        y_err.append(shear_err)
    return np.array(y), np.array(y_err)


def test_noisy_gals(noi_std: float = 0.0):
    assert fpfs.catalog.ncol == len(
        fpfs.catalog.col_names
    ), "The number of output column does not match 'catalog.ncol'"
    rng = np.random.RandomState(212)
    num_gals = 10000
    shear, shear_err = analyze_fpfs(rng, np.array([0.03, 0.00]), num_gals, noi_std)
    thres = max(2.0 * shear_err, 1e-5)
    assert np.all(np.abs(shear - 0.03) < thres)
    return


if __name__ == "__main__":
    test_noisy_gals(noi_std=0.0)
