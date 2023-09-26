#!/usr/bin/env python
"""
simple example with ring test (rotating intrinsic galaxies)
"""
import os
import pickle
import schwimmbad
import numpy as np
from argparse import ArgumentParser
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.galaxies import (
    WLDeblendGalaxyCatalog,
)  # one of the galaxy catalog classes
from descwl_shear_sims.stars import StarCatalog  # star catalog class
from descwl_shear_sims.psfs import (
    make_ps_psf,
    make_fixed_psf,
)  # for making a power spectrum PSF
from descwl_shear_sims.sim import get_se_dim  # convert coadd dims to SE dims
import astropy.io.fits as pyfits


layout = "random_disk"
# layout = "hex"
rotate = False
dither = False
itest = 0

coadd_dim = 7200
# coadd_dim = 720
buff = 120

nrot = 2
g1_list = [-0.02, 0.02]
# one band one run
band_name = "g"
band_list = [band_name]
# band_list=['r', 'i', 'z']
rot_list = [np.pi / nrot * i for i in range(nrot)]
nshear = len(g1_list)

# img_root = "/hildafs/datasets/shared_phy200017p/LSST_like_GREAT3/"
img_root = "/lustre/work/xiangchong.li/work/FPFS2/sim_desc/"
# img_root = "/lustre/work/xiangchong.li/work/FPFS2/sim_desc_hex/"


def work(ifield=0):
    print("Simulating for field: %d" % ifield)
    rng = np.random.RandomState(ifield)

    if itest == 0:
        # basic test
        kargs = {
            "cosmic_rays": False,
            "bad_columns": False,
            "star_bleeds": False,
        }
        star_catalog = None
        psf = make_fixed_psf(psf_type="moffat")
        test_name = "basic"
    elif itest == 1:
        # spatial varying PSF
        kargs = {
            "cosmic_rays": False,
            "bad_columns": False,
            "star_bleeds": False,
        }
        star_catalog = None
        # this is the single epoch image sized used by the sim, we need
        # it for the power spectrum psf
        se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate, dither=dither)
        psf = make_ps_psf(rng=rng, dim=se_dim)
        test_name = "psf"
    elif itest == 2:
        # with star
        kargs = {
            "cosmic_rays": False,
            "bad_columns": False,
            "star_bleeds": False,
        }
        star_catalog = StarCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            density=(ifield % 1000) / 10 + 1,
            layout=layout,
        )
        # it for the power spectrum psf
        se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate, dither=dither)
        psf = make_ps_psf(rng=rng, dim=se_dim)
        test_name = "star"
    elif itest == 3:
        # with mask plane
        kargs = {
            "cosmic_rays": True,
            "bad_columns": True,
            "star_bleeds": True,
        }
        star_catalog = StarCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            density=(ifield % 1000) / 10 + 1,
            layout=layout,
        )
        # it for the power spectrum psf
        se_dim = get_se_dim(coadd_dim=coadd_dim, rotate=rotate, dither=dither)
        psf = make_ps_psf(rng=rng, dim=se_dim)
        test_name = "maskplane"
    else:
        raise ValueError("itest must be 0, 1, 2 or 3 !!!")

    img_dir = "%s/%s" % (img_root, test_name)
    os.makedirs(img_dir, exist_ok=True)

    # galaxy catalog; you can make your own
    galaxy_catalog = WLDeblendGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        layout=layout,
    )
    print("Simulation has galaxies: %d" % len(galaxy_catalog))
    for irot in range(nrot):
        for ishear in range(nshear):
            gal_fname = "%s/image-%05d_g1-%d_rot%d_%s.fits" % (
                img_dir,
                ifield,
                ishear,
                irot,
                band_name,
            )
            if os.path.isfile(gal_fname):
                print("Already has file: %s" % gal_fname)
                continue
            print("Start making file:  %s" % gal_fname)
            sim_data = make_sim(
                rng=rng,
                galaxy_catalog=galaxy_catalog,
                star_catalog=star_catalog,
                coadd_dim=coadd_dim,
                g1=g1_list[ishear],
                g2=0.00,
                psf=psf,
                dither=dither,
                rotate=rotate,
                bands=band_list,
                noise_factor=0.0,
                theta0=rot_list[irot],
                **kargs
            )
            # this is only for fixed PSF..
            psf_fname = "%s/PSF_%s.pkl" % (img_root, test_name)
            if irot == 0 and ishear == 0 and not os.path.isfile(psf_fname):
                psf_dim = sim_data["psf_dims"][0]
                se_wcs = sim_data["se_wcs"][0]
                with open(psf_fname, "wb") as f:
                    pickle.dump(
                        {"psf": psf, "psf_dim": psf_dim, "wcs": se_wcs},
                        f,
                    )
            mi = sim_data["band_data"][band_name][0].getMaskedImage()
            gdata = mi.getImage().getArray()
            pyfits.writeto(gal_fname, gdata)
            del mi, gdata
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="desc_simulate")
    parser.add_argument(
        "--minId",
        default=5000,
        type=int,
        help="minimum id number, e.g. 0",
    )
    parser.add_argument(
        "--maxId",
        default=10000,
        type=int,
        help="maximum id number, e.g. 4000",
    )
    #
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )
    cmd_args = parser.parse_args()
    min_id = cmd_args.minId
    max_id = cmd_args.maxId
    pool = schwimmbad.choose_pool(mpi=cmd_args.mpi, processes=cmd_args.n_cores)
    idlist = list(range(min_id, max_id))
    for r in pool.map(work, idlist):
        pass
    pool.close()
