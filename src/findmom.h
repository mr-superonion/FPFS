
namespace galsim {
namespace hsm {

    struct HSMParams {
    /*
     * @brief Parameters that determine how the moments/shape estimation routines make
     *        speed/accuracy tradeoff decisions.
     *
     * @param nsig_rg            A parameter used to optimize convolutions by cutting off the galaxy
     *                           profile.  In the first step of the re-Gaussianization method of PSF
     *                           correction, a Gaussian approximation to the pre-seeing galaxy is
     *                           calculated. If re-Gaussianization is called with the flag 0x4 (as
     *                           is the default), then this approximation is cut off at nsig_rg
     *                           sigma to save computation time in convolutions.
     * @param nsig_rg2           A parameter used to optimize convolutions by cutting off the PSF
     *                           residual profile.  In the re-Gaussianization method of PSF
     *                           correction, a "PSF residual" (the difference between the true PSF
     *                           and its best-fit Gaussian approximation) is constructed. If
     *                           re-Gaussianization is called with the flag 0x8 (as is the default),
     *                           then this PSF residual is cut off at nsig_rg2 sigma to save
     *                           computation time in convolutions.
     * @param max_moment_nsig2   A parameter for optimizing calculations of adaptive moments by
     *                           cutting off profiles. This parameter is used to decide how many
     *                           sigma^2 into the Gaussian adaptive moment to extend the moment
     *                           calculation, with the weight being defined as 0 beyond this point.
     *                           i.e., if max_moment_nsig2 is set to 25, then the Gaussian is
     *                           extended to (r^2/sigma^2)=25, with proper accounting for elliptical
     *                           geometry.  If this parameter is set to some very large number, then
     *                           the weight is never set to zero and the exponential function is
     *                           always called. Note: GalSim script devel/modules/test_mom_timing.py
     *                           was used to choose a value of 25 as being optimal, in that for the
     *                           cases that were tested, the speedups were typically factors of
     *                           several, but the results of moments and shear estimation were
     *                           changed by <10^-5.  Not all possible cases were checked, and so for
     *                           use of this code for unusual cases, we recommend that users check
     *                           that this value does not affect accuracy, and/or set it to some
     *                           large value to completely disable this optimization.
     * @param regauss_too_small  A parameter for how strictly the re-Gaussianization code treats
     *                           small galaxies. If this parameter is 1, then the re-Gaussianization
     *                           code does not impose a cut on the apparent resolution before trying
     *                           to measure the PSF-corrected shape of the galaxy; if 0, then it is
     *                           stricter.  Using the default value of 1 prevents the
     *                           re-Gaussianization PSF correction from completely failing at the
     *                           beginning, before trying to do PSF correction, due to the crudest
     *                           possible PSF correction (Gaussian approximation) suggesting that
     *                           the galaxy is very small.  This could happen for some usable
     *                           galaxies particularly when they have very non-Gaussian surface
     *                           brightness profiles -- for example, if there's a prominent bulge
     *                           that the adaptive moments attempt to fit, ignoring a more
     *                           extended disk.  Setting a value of 1 is useful for keeping galaxies
     *                           that would have failed for that reason.  If they later turn out to
     *                           be too small to really use, this will be reflected in the final
     *                           estimate of the resolution factor, and they can be rejected after
     *                           the fact.
     * @param adapt_order        The order to which circular adaptive moments should be calculated
     *                           for KSB method. This parameter only affects calculations using the
     *                           KSB method of PSF correction.  Warning: deviating from default
     *                           value of 2 results in code running more slowly, and results have
     *                           not been significantly tested.
     * @param convergence_threshold Accuracy (in x0, y0, and sigma as a fraction of sigma) when
     *                           calculating adaptive moments.
     * @param max_mom2_iter      Maximum number of iterations to use when calculating adaptive
     *                           moments.  This should be sufficient in nearly all situations, with
     *                           the possible exception being very flattened profiles.
     * @param num_iter_default   Number of iterations to report in the output ShapeData structure
     *                           when code fails to converge within max_mom2_iter iterations.
     * @param bound_correct_wt   Maximum shift in centroids and sigma between iterations for
     *                           adaptive moments.
     * @param max_amoment        Maximum value for adaptive second moments before throwing
     *                           exception.  Very large objects might require this value to be
     *                           increased.
     * @param max_ashift         Maximum allowed x / y centroid shift (units: pixels) between
     *                           successive iterations for adaptive moments before throwing
     *                           exception.
     * @param ksb_moments_max    Use moments up to ksb_moments_max order for KSB method of PSF
     *                           correction.
     * @param ksb_sig_weight     The width of the weight function (in pixels) to use for the KSB
     *                           method.  Normally, this is derived from the measured moments of the
     *                           galaxy image; this keyword overrides this calculation.  Can be
     *                           combined with ksb_sig_factor.
     * @param ksb_sig_factor     Factor by which to multiply the weight function width for the KSB
     *                           method (default: 1.0).  Can be combined with ksb_sig_weight.
     * @param failed_moments     Value to report for ellipticities and resolution factor if shape
     *                           measurement fails.
     */
        HSMParams(double _nsig_rg,
                  double _nsig_rg2,
                  double _max_moment_nsig2,
                  int _regauss_too_small,
                  int _adapt_order,
                  double _convergence_threshold,
                  long _max_mom2_iter,
                  long _num_iter_default,
                  double _bound_correct_wt,
                  double _max_amoment,
                  double _max_ashift,
                  int _ksb_moments_max,
                  double _ksb_sig_weight,
                  double _ksb_sig_factor,
                  double _failed_moments) :
            nsig_rg(_nsig_rg),
            nsig_rg2(_nsig_rg2),
            max_moment_nsig2(_max_moment_nsig2),
            regauss_too_small(_regauss_too_small),
            adapt_order(_adapt_order),
            convergence_threshold(_convergence_threshold),
            max_mom2_iter(_max_mom2_iter),
            num_iter_default(_num_iter_default),
            bound_correct_wt(_bound_correct_wt),
            max_amoment(_max_amoment),
            max_ashift(_max_ashift),
            ksb_moments_max(_ksb_moments_max),
            ksb_sig_weight(_ksb_sig_weight),
            ksb_sig_factor(_ksb_sig_factor),
            failed_moments(_failed_moments)
        {}

        /**
         * A reasonable set of default values
         */
        HSMParams() :
            nsig_rg(3.0),
            nsig_rg2(3.6),
            max_moment_nsig2(25.0),
            regauss_too_small(1),
            adapt_order(2),
            convergence_threshold(1.e-6),
            max_mom2_iter(400),
            num_iter_default(-1),
            bound_correct_wt(0.25),
            max_amoment(8000.),
            max_ashift(15.),
            ksb_moments_max(4),
            ksb_sig_weight(0.0),
            ksb_sig_factor(1.0),
            failed_moments(-1000.)
            {}

        // These are all public.  So you access them just as member values.
        double nsig_rg;
        double nsig_rg2;
        double max_moment_nsig2;
        int regauss_too_small;
        int adapt_order;
        double convergence_threshold;
        long max_mom2_iter;
        long num_iter_default;
        double bound_correct_wt;
        double max_amoment;
        double max_ashift;
        int ksb_moments_max;
        double ksb_sig_weight;
        double ksb_sig_factor;
        double failed_moments;
    };

// clang doesn't like the mmgr new macro in this next line.
#ifdef MEM_TEST
#ifdef __clang__
#if __has_warning("-Wpredefined-identifier-outside-function")
#pragma GCC diagnostic ignored "-Wpredefined-identifier-outside-function"
#endif
#endif
#endif

    // All code between the @cond and @endcond is excluded from Doxygen documentation
    //! @cond

    /**
     * @brief Exception class thrown by the adaptive moment and shape measurement routines in the
     * hsm namespace
     */
    class HSMError : public std::runtime_error {
    public:
        HSMError(const std::string& m) : std::runtime_error(m) {}
    };

    //! @endcond

    /**
     * @brief Characterizes the shape and other parameters of objects.
     *
     * Describe the hsm shape-related parameters of some object (usually galaxy) before and after
     * PSF correction.  All ellipticities are defined as (1-q^2)/(1+q^2), with the 1st component
     * aligned with the pixel grid and the 2nd aligned at 45 degrees with respect to it.  There are
     * two choices for measurement type: 'e' = Bernstein & Jarvis (2002) ellipticity (or
     * distortion), 'g' = shear estimator = shear*responsivity.  The sigma is defined based on the
     * observed moments M_xx, M_xy, and M_yy as sigma = (Mxx Myy - M_xy^2)^(1/4) =
     * [ det(M) ]^(1/4). */
    struct ObjectData
    {
        // Make sure everything starts with 0's.
        ObjectData() :
            x0(0.), y0(0.), sigma(0.), flux(0.), e1(0.), e2(0.), responsivity(0.),
            meas_type('\0'), resolution(0.) {}

        double x0; ///< x centroid position within the postage stamp, in units of pixels
        double y0; ///< y centroid position within the postage stamp, in units of pixels
        double sigma; ///< size parameter
        double flux; ///< total flux
        double e1; ///< first ellipticity component
        double e2; ///< second ellipticity component
        double responsivity; ///< responsivity of ellipticity estimator
        char meas_type; ///< type of measurement (see function description)
        double resolution; ///< resolution factor (0=unresolved, 1=resolved)
    };

    /**
     * @brief Struct containing information about the shape of an object.
     *
     * This representation of an object shape contains information about observed shapes and shape
     * estimators after PSF correction.  It also contains information about what PSF correction was
     * used; if no PSF correction was carried out and only the observed moments were measured, the
     * PSF correction method will be 'None'.  Note that observed shapes are bounded to lie in the
     * range |e| < 1 or |g| < 1, so they can be represented using a Shear object.  In contrast,
     * the PSF-corrected distortions and shears are not bounded at a maximum of 1 since they are
     * shear estimators, and placing such a bound would bias the mean.  Thus, the corrected results
     * are not represented using Shear objects, since it may not be possible to make a meaningful
     * per-object conversion from distortion to shear (e.g., if |e|>1).
     */
    struct ShapeData
    {
        /// @brief galsim::Bounds object describing the image of the object
        Bounds<int> image_bounds;

        // Now put information about moments measurement

        /// @brief Status after measuring adaptive moments; -1 indicates no attempt to measure them
        int moments_status;

        /// @brief The observed shape e1,e2
        float observed_e1, observed_e2;

        /// @brief Size sigma = (det M)^(1/4) from the adaptive moments, in units of pixels; -1 if
        /// not measured
        float moments_sigma;

        /// @brief Total image intensity for best-fit elliptical Gaussian from adaptive moments;
        /// note that this is not flux, since flux = (total image intensity)*(pixel scale)^2
        float moments_amp;

        /// @brief Centroid of best-fit elliptical Gaussian
        Position<double> moments_centroid;

        /// @brief The weighted radial fourth moment of the image
        double moments_rho4;

        /// @brief Number of iterations needed to get adaptive moments; 0 if not measured
        int moments_n_iter;

        // Then put information about PSF correction

        /// @brief Status after carrying out PSF correction; -1 indicates no attempt to do so
        int correction_status;

        /// @brief Estimated e1 after correcting for effects of the PSF, for methods that return a
        /// distortion.  Default value -10 if no correction carried out.
        float corrected_e1;

        /// @brief Estimated e2 after correcting for effects of the PSF, for methods that return a
        /// distortion.  Default value -10 if no correction carried out.
        float corrected_e2;

        /// @brief Estimated g1 after correcting for effects of the PSF, for methods that return a
        /// shear.  Default value -10 if no correction carried out.
        float corrected_g1;

        /// @brief Estimated g2 after correcting for effects of the PSF, for methods that return a
        /// shear.  Default value -10 if no correction carried out.
        float corrected_g2;

        /// @brief 'e' for PSF correction methods that return a distortion, 'g' for methods that
        /// return a shear.  "None" if PSF correction was not done.
        std::string meas_type;

        /// @brief Shape measurement uncertainty sigma_gamma (not sigma_e) per component
        float corrected_shape_err;

        /// @brief String indicating PSF-correction method; "None" if PSF correction was not done.
        std::string correction_method;

        /// @brief Resolution factor R_2; 0 indicates object is consistent with a PSF, 1 indicates
        /// perfect resolution; default -1
        float resolution_factor;

        /// @brief PSF size sigma = (det M)^(1/4) from the adaptive moments, in units of pixels;
        /// default -1.
        float psf_sigma;

        /// @brief PSF shape from the adaptive moments
        float psf_e1, psf_e2;

        /// @brief A string containing any error messages from the attempted measurements, to
        /// facilitate proper error handling in both C++ and python
        std::string error_message;

        /// @brief Constructor, setting defaults
        // NB: To avoid errors associated with the infamous Mac dynamic string bug, we initialize
        // all strings with a non-empty string.  Including error_message, which otherwise would
        // make more sense to initialize with "".
        // cf. http://stackoverflow.com/questions/4697859/mac-os-x-and-static-boost-libs-stdstring-fail
        ShapeData() : image_bounds(galsim::Bounds<int>()), moments_status(-1),
            observed_e1(0.), observed_e2(0.), moments_sigma(-1.), moments_amp(-1.),
            moments_centroid(galsim::Position<double>(0.,0.)), moments_rho4(-1.), moments_n_iter(0),
            correction_status(-1), corrected_e1(-10.), corrected_e2(-10.), corrected_g1(-10.),
            corrected_g2(-10.), meas_type("None"), corrected_shape_err(-1.),
            correction_method("None"), resolution_factor(-1.),
            psf_sigma(-1.0), psf_e1(0.), psf_e2(0.), error_message("None")
        {}
    };

    /**
     * @brief Measure the adaptive moments of an object.
     *
     * This function iteratively computes the adaptive moments of an image, and tells the user the
     * results plus other diagnostic information.  The key result is the best-fit elliptical
     * Gaussian to the object, which is computed by initially guessing a circular Gaussian that is
     * used as a weight function, computing the weighted moments, recomputing the moments using the
     * result of the previous step as the weight function, and so on until the moments that are
     * measured are the same as those used for the weight function.
     * @param[in] data      The Image for the object being measured.
     * @param[out] A        The amplitude of the best-fit elliptical Gaussian (total image intensity
     *                      for the Gaussian is 2A).
     * @param[out] x0       The x centroid of the best-fit elliptical Gaussian.
     * @param[out] y0       The y centroid of the best-fit elliptical Gaussian.
     * @param[out] Mxx      The xx component of the moment matrix.
     * @param[out] Mxy      The xy component of the moment matrix.
     * @param[out] Myy      The yy component of the moment matrix.
     * @param[out] rho4     The weighted radial fourth moment.
     * @param[in] convergence_threshold   The required level of accuracy.
     * @param[out] num_iter The number of iterations needed to converge.
     * @param[in] hsmparams Optional argument to specify parameters to be used for shape
     *                      measurement routines, as an HSMParams object.
     */
    void find_ellipmom_2(
        ConstImageView<double> data, double& A, double& x0, double& y0,
        double& Mxx, double& Mxy, double& Myy, double& rho4, double convergence_threshold,
        int& num_iter, const HSMParams& hsmparams);

}
}
#endif
