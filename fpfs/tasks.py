import os
import gc
import json
import glob
import time
import logging

import fpfs
import galsim
import numpy as np
import astropy.io.fits as pyfits
from configparser import ConfigParser, ExtendedInterpolation

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)


class SimulationTask(object):
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        self.img_dir = cparser.get("files", "img_dir")
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir, exist_ok=True)
        self.sim_method = cparser.get("simulation", "sim_method", fallback="fft")
        self.gal_type = cparser.get("simulation", "gal_type", fallback="mixed").lower()
        self.nrot = cparser.getint("simulation", "nrot")
        self.buff = cparser.getint("simulation", "buff")
        self.image_nx = cparser.getint("simulation", "image_nx")
        self.image_ny = cparser.getint("simulation", "image_ny")
        self.band = cparser.get("simulation", "band")
        self.do_shift = cparser.getboolean("simulation", "do_shift", fallback=False)
        self.scale = cparser.getfloat("survey", "pixel_scale")
        self.max_hlr = cparser.getfloat(
            "simulation",
            "max_hlr",
            fallback=1.5e0,
        )
        self.min_hlr = cparser.getfloat(
            "simulation",
            "min_hlr",
            fallback=1e-2,
        )
        assert self.image_ny == self.image_nx, "'image_nx' must equals 'image_ny'!"
        self.psf_obj = None
        assert self.sim_method in ["fft", "mc"]
        assert self.gal_type in ["mixed", "sersic", "bulgedisk", "debug"]
        # PSF
        seeing = cparser.getfloat("survey", "psf_fwhm", fallback=4.0 * self.scale)
        logging.info("Using modelled Moffat PSF with seeing %.2f arcsec. " % seeing)
        psffname = os.path.join(self.img_dir, "psf-%d.fits" % (seeing * 100))
        psf_beta = cparser.getfloat("survey", "psf_moffat_beta")
        psf_trunc = cparser.getfloat("survey", "psf_trunc_ratio")
        psf_e1 = cparser.getfloat("survey", "psf_e1")
        psf_e2 = cparser.getfloat("survey", "psf_e2")
        no_pixel = cparser.getboolean("survey", "no_pixel")
        if no_pixel:
            self.draw_method = "no_pixel"
        else:
            self.draw_method = "fft"
        if psf_trunc < 1.0:
            self.psf_obj = galsim.Moffat(
                beta=psf_beta,
                fwhm=seeing,
            ).shear(e1=psf_e1, e2=psf_e2)
        else:
            trunc = seeing * psf_trunc
            self.psf_obj = galsim.Moffat(beta=psf_beta, fwhm=seeing, trunc=trunc).shear(
                e1=psf_e1, e2=psf_e2
            )
        # write psf image
        psf_image = self.psf_obj.shift(
            0.5 * self.scale,
            0.5 * self.scale,
        ).drawImage(nx=64, ny=64, scale=self.scale, method=self.draw_method)
        psf_image.write(psffname)
        # Shear
        self.gver = cparser.get("distortion", "g_version")
        zlist = json.loads(cparser.get("distortion", "shear_z_list"))
        self.gname_list = ["%s-%s" % (self.gver, i1) for i1 in zlist]
        logging.info(
            "We will test the following constant shear distortion setups %s. "
            % self.gname_list
        )
        self.shear_value = cparser.getfloat("distortion", "shear_value")
        self.rot_list = [np.pi / self.nrot * i for i in range(self.nrot)]
        return

    def run(self, ifield):
        logging.info("start ID: %d" % (ifield))
        for gn in self.gname_list:
            # do basic stamp-like image simulation
            nfiles = len(
                glob.glob(
                    "%s/image-%05d_%s_rot*_%s.fits"
                    % (
                        self.img_dir,
                        ifield,
                        gn,
                        self.band,
                    )
                )
            )
            if nfiles == self.nrot:
                logging.info("We already have all the output files for %s" % gn)
                continue
            sim_img = fpfs.simutil.make_isolate_sim(
                sim_method="fft",  # we use FFT method to render galaxy images
                psf_obj=self.psf_obj,
                gname=gn,
                seed=ifield,
                ny=self.image_ny,
                nx=self.image_nx,
                scale=self.scale,
                do_shift=self.do_shift,
                shear_value=self.shear_value,
                nrot_per_gal=1,
                min_hlr=self.min_hlr,  # set the minimum hlr to 0
                max_hlr=self.max_hlr,  # set maximum hlr (sersic fit)
                rot_field=self.rot_list,
                gal_type=self.gal_type,
                buff=self.buff,
                draw_method=self.draw_method,
            )
            for irot in range(self.nrot):
                gal_fname = "%s/image-%05d_%s_rot%d_%s.fits" % (
                    self.img_dir,
                    ifield,
                    gn,
                    irot,
                    self.band,
                )
                fpfs.io.save_image(gal_fname, sim_img[irot])
            gc.collect()
        logging.info("finish processing field ID: %d" % (ifield))
        return

    def clear(self, ifield):
        for gn in self.gname_list:
            for irot in range(self.nrot):
                gal_fname = "%s/image-%05d_%s_rot%d_%s.fits" % (
                    self.img_dir,
                    ifield,
                    gn,
                    irot,
                    self.band,
                )
                if os.path.isfile(gal_fname):
                    os.remove(gal_fname)
        logging.info("Cleaning results for field ID: %s" % (ifield))
        return

    def load_outcomes(self, ifield):
        outcomes = {}
        for gn in self.gname_list:
            for irot in range(self.nrot):
                gal_fname = "%s/image-%05d_%s_rot%d_%s.fits" % (
                    self.img_dir,
                    ifield,
                    gn,
                    irot,
                    self.band,
                )
                if os.path.isfile(gal_fname):
                    image = pyfits.getdata(gal_fname)
                    outcomes.update({"%s_rot%s" % (gn, irot): image})
        return outcomes


def get_random_seed_from_fname(fname, band, nrot=2):
    band_map = {
        "g": 0,
        "r": 1,
        "i": 2,
        "z": 3,
        "a": 4,
    }
    fid = int(fname.split("image-")[-1].split("_")[0]) + 212
    rid = int(fname.split("rot")[1][0])
    bid = band_map[band]
    nband = len(band_map.items())
    return (fid * nrot + rid) * nband + bid


def get_sim_fnames(directory, ftype, min_id, max_id, gver, nshear, nrot, band):
    """Generate filename for simulations
    Args:
        ftype (str):    file type ('src' for source, and 'image' for exposure
        min_id (int):   minimum id
        max_id (int):   maximum id
        gver (str):     shear version "g1", "g2" or "g1_g2"
        nshear (int):   number of shear
        nrot (int):     number of rotations
        band (str):     'grizy' or 'a'
    Returns:
        out (list):     a list of file name
    """
    out = [
        os.path.join(
            directory,
            "%s-%05d_%s-%d_rot%d_%s.fits" % (ftype, fid, gver, gid, rid, band),
        )
        for fid in range(min_id, max_id)
        for gid in range(nshear)
        for rid in range(nrot)
    ]
    return out


class ProcessSimulationTask(object):
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        self.nrot = cparser.getint("simulation", "nrot")
        zlist = json.loads(cparser.get("distortion", "shear_z_list"))
        self.gver = cparser.get("distortion", "g_version")
        self.gname_list = ["%s-%s" % (self.gver, i1) for i1 in zlist]
        self.nshear = len(zlist)
        # TODO: change this type
        # print(type(zlist[0]))
        #
        self.img_dir = cparser.get("files", "img_dir")
        self.cat_dir = cparser.get("files", "cat_dir")
        self.psf_file_name = cparser.get("files", "psf_file_name")
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find input images directory!")
        logging.info("The input directory for galaxy images is %s. " % self.img_dir)
        if not os.path.isdir(self.cat_dir):
            os.makedirs(self.cat_dir, exist_ok=True)
        logging.info("The output directory for shear catalogs is %s. " % self.cat_dir)

        # setup FPFS task
        self.sigma_as = cparser.getfloat("FPFS", "sigma_as")
        self.sigma_det = cparser.getfloat("FPFS", "sigma_det")
        self.rcut = cparser.getint("FPFS", "rcut", fallback=32)
        self.psf_rcut = cparser.getint("FPFS", "psf_rcut", fallback=22)
        self.psf_rcut = min(self.psf_rcut, self.rcut)
        self.nnord = cparser.getint("FPFS", "nnord", fallback=4)
        if self.nnord not in [4, 6]:
            raise ValueError(
                "Only support for nnord= 4 or nnord=6, but your input\
                    is nnord=%d"
                % self.nnord
            )

        # setup survey parameters
        self.nstd_f = cparser.getfloat("survey", "noise_std")
        self.ncov_fname = cparser.get(
            "FPFS",
            "ncov_fname",
            fallback="",
        )
        if len(self.ncov_fname) == 0 or not os.path.isfile(self.ncov_fname):
            # estimate and write the noise covariance
            self.ncov_fname = os.path.join(self.cat_dir, "cov_matrix.fits")
        self.magz = cparser.getfloat("survey", "mag_zero")
        self.band = cparser.get("survey", "band")
        self.scale = cparser.getfloat("survey", "pixel_scale")
        ngrid = 2 * self.rcut
        # By default, we use uncorrelated noise
        # TODO: enable correlated noise here
        self.noise_pow = np.ones((ngrid, ngrid)) * self.nstd_f**2.0 * ngrid**2.0
        return

    def get_image_fnames(self, ifield):
        refs = get_sim_fnames(
            self.img_dir,
            "image",
            ifield,
            ifield + 1,
            gver=self.gver,
            nshear=self.nshear,
            nrot=self.nrot,
            band=self.band,
        )
        for rr in refs:
            assert os.path.isfile(rr), "file %s does not exist" % rr
        return refs

    def prepare_noise_psf(self, fname):
        exposure = pyfits.getdata(fname)
        self.image_nx = exposure.shape[1]
        psf_array = pyfits.getdata(self.psf_file_name)
        fpfs.imgutil.truncate_square(psf_array, self.psf_rcut)
        npad = (self.image_nx - psf_array.shape[0]) // 2
        psf_array2 = np.pad(psf_array, (npad, npad), mode="constant")
        if not os.path.isfile(self.ncov_fname):
            # FPFS noise cov task
            noise_task = fpfs.image.measure_noise_cov(
                psf_array,
                sigma_arcsec=self.sigma_as,
                sigma_detect=self.sigma_det,
                nnord=self.nnord,
                pix_scale=self.scale,
            )
            cov_elem = np.array(noise_task.measure(self.noise_pow))
            pyfits.writeto(self.ncov_fname, cov_elem, overwrite=True)
        else:
            cov_elem = pyfits.getdata(self.ncov_fname)
        assert np.all(
            np.diagonal(cov_elem) > 1e-10
        ), "The covariance matrix is incorrect"
        return psf_array, psf_array2, cov_elem

    def prepare_image(self, fname):
        gal_array = np.zeros((self.image_nx, self.image_nx))
        logging.info("processing %s band" % self.band)
        gal_array = pyfits.getdata(fname)
        if self.nstd_f > 1e-10:
            # noise
            seed = get_random_seed_from_fname(fname, self.band)
            rng = np.random.RandomState(seed)
            logging.info("Using noisy setup with std: %.5f" % self.nstd_f)
            logging.info("The random seed is %d" % seed)
            gal_array = gal_array + rng.normal(
                scale=self.nstd_f,
                size=gal_array.shape,
            )
        else:
            logging.info("Using noiseless setup")
        return gal_array

    def process_image(self, gal_array, psf_array, psf_array2, cov_elem):
        # measurement task
        meas_task = fpfs.image.measure_source(
            psf_array,
            sigma_arcsec=self.sigma_as,
            sigma_detect=self.sigma_det,
            nnord=self.nnord,
            pix_scale=self.scale,
        )

        std_modes = np.sqrt(np.diagonal(cov_elem))
        idm00 = fpfs.catalog.indexes["m00"]
        idv0 = fpfs.catalog.indexes["v0"]
        # Temp fix for 4th order estimator
        if self.nnord == 6:
            idv0 += 1
        thres = 9.5 * std_modes[idm00] * self.scale**2.0
        thres2 = -1.5 * std_modes[idv0] * self.scale**2.0
        coords = meas_task.detect_sources(
            img_data=gal_array,
            psf_data=psf_array2,
            thres=thres,
            thres2=thres2,
            bound=self.rcut + 5,
        )
        logging.info("pre-selected number of sources: %d" % len(coords))
        out = meas_task.measure(gal_array, coords)
        out = meas_task.get_results(out)
        sel = (out["fpfs_M00"] + out["fpfs_M20"]) > 0.0
        out = out[sel]
        logging.info("final number of sources: %d" % len(out))
        coords = coords[sel]
        coords = np.rec.fromarrays(coords.T, dtype=[("fpfs_y", "i4"), ("fpfs_x", "i4")])
        return out, coords

    def run(self, ifield):
        fnames = self.get_image_fnames(ifield=ifield)
        for ff in fnames:
            self.run_one_file(ff)
        return

    def run_one_file(self, fname):
        logging.info(f"Compressing image: {fname}")
        out_fname = os.path.join(self.cat_dir, fname.split("/")[-1])
        out_fname = out_fname.replace("image-", "src-")

        det_fname = os.path.join(self.cat_dir, fname.split("/")[-1])
        det_fname = det_fname.replace("image-", "det-")
        if os.path.isfile(out_fname) and os.path.isfile(det_fname):
            logging.info("Already has measurement for simulation: %s." % fname)
            return
        psf_array, psf_array2, cov_elem = self.prepare_noise_psf(fname)
        gal_array = self.prepare_image(fname)
        start_time = time.time()
        cat, det = self.process_image(gal_array, psf_array, psf_array2, cov_elem)
        del gal_array, psf_array, psf_array2, cov_elem
        # Stop the timer
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        # Print the elapsed time
        logging.info(f"Elapsed time: {elapsed_time} seconds")
        fpfs.io.save_catalog(det_fname, det, dtype="position", nnord=str(self.nnord))
        fpfs.io.save_catalog(out_fname, cat, dtype="shape", nnord=str(self.nnord))
        # jax.clear_backends()
        gc.collect()
        return

    def load_outcomes(self, ifield, data_type="shape"):
        if data_type == "shape":
            dtp = "src"
        elif data_type == "detection":
            dtp = "det"
        else:
            raise ValueError("We do not support data type: %s" % data_type)
        outcomes = {}
        for gn in self.gname_list:
            for irot in range(self.nrot):
                fn = "%s/%s-%05d_%s_rot%d_%s.fits" % (
                    self.cat_dir,
                    dtp,
                    ifield,
                    gn,
                    irot,
                    self.band,
                )
                if os.path.isfile(fn):
                    data = pyfits.getdata(fn)
                    outcomes.update({"%s_rot%s" % (gn, irot): data})
        return outcomes

    def clear(self, ifield):
        outcomes = {}
        for gn in self.gname_list:
            for irot in range(self.nrot):
                fn1 = "%s/%s-%05d_%s_rot%d_%s.fits" % (
                    self.cat_dir,
                    "src",
                    ifield,
                    gn,
                    irot,
                    self.band,
                )
                if os.path.isfile(fn1):
                    os.remove(fn1)
                fn2 = "%s/%s-%05d_%s_rot%d_%s.fits" % (
                    self.cat_dir,
                    "det",
                    ifield,
                    gn,
                    irot,
                    self.band,
                )
                if os.path.isfile(fn2):
                    os.remove(fn2)
        return outcomes
