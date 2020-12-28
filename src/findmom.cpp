/* -*- c++ -*-
 * Copyright (c) 2012-2019 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

/****************************************************************
Copyright (c) 2003-2014 by Christopher Hirata (hirata.10@osu.edu),
Rachel Mandelbaum (rmandelb@andrew.cmu.edu), and Uros Seljak
(useljak@berkeley.edu)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

This software is made available to you on an ``as is'' basis with no
representations or warranties, express or implied, including but not
limited to any warranty of performance, merchantability, fitness for a
particular purpose, commercial utility, non-infringement or title.
Neither the authors nor the organizations providing the support under
which the work was developed will be liable to you or any third party
with respect to any claim arising from your further development of the
software or any products related to or derived from the software, or for
lost profits, business interruption, or indirect special or consequential
damages of any kind.
*******************************************************************/

#include <cstring>
#include <string>
#include "findmom.h"

namespace FShapelets {
    /* find_ellipmom_1
     * *** FINDS ELLIPTICAL GAUSSIAN MOMENTS OF AN IMAGE ***
     *
     * Returns the parameters:
     * A = int f(x,y) w(x,y)
     * B_i = int (r_i-r0_i) f(r) w(r)
     * C_ij = int (r_i-r0_i) (r_j-r0_j) f(r) w(r)
     * rho4 = int rho^4 f(r) w(r)
     *
     * where w(r) = exp(-rho^2/2), rho^2 = (x-x0) * M^{-1} * (y-y0),
     * M = adaptive covariance matrix, and note that the weight may be set to zero for rho^2 >
     * hsmparams.max_moment_nsig2 if that parameter is defined.
     *
     * Arguments:
     *   data: the input image (ImageView format)
     *   x0: weight centroid (x coordinate)
     *   y0: weight centroid (y coordinate)
     *   Mxx: xx element of adaptive covariance
     *   Mxy: xy element of adaptive covariance
     *   Myy: yy element of adaptive covariance
     * > A: amplitude
     * > Bx: weighted centroid displacement (x)
     * > By: weighted centroid displacement (y)
     * > Cxx: weighted covariance (xx)
     * > Cxy: weighted covariance (xy)
     * > Cyy: weighted covariance (yy)
     * > rho4w: weighted radial fourth moment
     */

    void find_ellipmom_1(
        ConstImageView<double> data, double x0, double y0, double Mxx,
        double Mxy, double Myy, double& A, double& Bx, double& By, double& Cxx,
        double& Cxy, double& Cyy, double& rho4w, const HSMParams& hsmparams)
    {
        long xmin = data.getXMin();
        long xmax = data.getXMax();
        long ymin = data.getYMin();
        long ymax = data.getYMax();
        dbg<<"Entering find_ellipmom_1 with Mxx, Myy, Mxy: "<<Mxx<<" "<<Myy<<" "<<Mxy<<std::endl;
        dbg<<"e1,e2 = "<<(Mxx-Myy)/(Mxx+Myy)<<" "<<2.*Mxy/(Mxx+Myy)<<std::endl;
        dbg<<"x0, y0: "<<x0<<" "<<y0<<std::endl;
        dbg<<"xmin, xmax: "<<xmin<<" "<<xmax<<std::endl;

        /* Compute M^{-1} for use in computing weights */
        double detM = Mxx * Myy - Mxy * Mxy;
        if (detM<=0 || Mxx<=0 || Myy<=0) {
            throw HSMError("Error: non positive definite adaptive moments!\n");
        }
        double Minv_xx    =  Myy/detM;
        double TwoMinv_xy = -Mxy/detM * 2.0;
        double Minv_yy    =  Mxx/detM;
        double Inv2Minv_xx = 0.5/Minv_xx; // Will be useful later...

        /* Generate Minv_xx__x_x0__x_x0 array */
        VectorXd Minv_xx__x_x0__x_x0(xmax-xmin+1);
        for(int x=xmin;x<=xmax;x++) Minv_xx__x_x0__x_x0[x-xmin] = Minv_xx*(x-x0)*(x-x0);

        /* Now let's initialize the outputs and then sum
         * over all the pixels
         */
        A = Bx = By = Cxx = Cxy = Cyy = rho4w = 0.;

        // rho2 = Minv_xx(x-x0)^2 + 2Minv_xy(x-x0)(y-y0) + Minv_yy(y-y0)^2
        // The minimum/maximum y that have a solution rho2 = max_moment_nsig2 is at:
        //   2*Minv_xx*(x-x0) + 2Minv_xy(y-y0) = 0
        // rho2 = Minv_xx (Minv_xy(y-y0)/Minv_xx)^2 - 2Minv_xy(Minv_xy(y-y0)/Minv_xx)(y-y0)
        //           + Minv_yy(y-y0)^2
        //      = (Minv_xy^2/Minv_xx - 2Minv_xy^2/Minv_xx + Minv_yy) (y-y0)^2
        //      = (Minv_xx Minv_yy - Minv_xy^2)/Minv_xx (y-y0)^2
        //      = (1/detM) / Minv_xx (y-y0)^2
        //      = (1/Myy) (y-y0)^2
        double y2 = sqrt(hsmparams.max_moment_nsig2 * Myy);  // This still needs the +y0 bit.
        double y1 = -y2 + y0;
        y2 += y0;  // ok, now it's right.
        int iy1 = int(ceil(y1));
        int iy2 = int(floor(y2));
        if (iy1 < ymin) iy1 = ymin;
        if (iy2 > ymax) iy2 = ymax;
        dbg<<"y1,y2 = "<<y1<<','<<y2<<std::endl;
        dbg<<"iy1,iy2 = "<<iy1<<','<<iy2<<std::endl;
        if (iy1 > iy2) {
            throw HSMError("Bounds don't make sense");
        }

        //
        /* Use these pointers to speed up referencing arrays */
        for(int y=iy1;y<=iy2;y++) {
            double y_y0 = y-y0;
            double TwoMinv_xy__y_y0 = TwoMinv_xy * y_y0;
            double Minv_yy__y_y0__y_y0 = Minv_yy * y_y0 * y_y0;

            // Now for a particular value of y, we want to find the min/max x that satisfy
            // rho2 < max_moment_nsig2.
            //
            // 0 = Minv_xx(x-x0)^2 + 2Minv_xy(x-x0)(y-y0) + Minv_yy(y-y0)^2 - max_moment_nsig2
            // Simple quadratic formula:
            double a = Minv_xx;
            double b = TwoMinv_xy__y_y0;
            double c = Minv_yy__y_y0__y_y0 - hsmparams.max_moment_nsig2;
            double d = b*b-4.*a*c;
            if (d < 0.)
                throw HSMError("Failure in finding min/max x for some y!");
            double sqrtd = sqrt(d);
            double inv2a = Inv2Minv_xx;
            double x1 = inv2a*(-b - sqrtd) + x0;
            double x2 = inv2a*(-b + sqrtd) + x0;
            int ix1 = int(ceil(x1));
            int ix2 = int(floor(x2));
            if (ix1 < xmin) ix1 = xmin;
            if (ix2 > xmax) ix2 = xmax;
            if (ix1 > ix2) continue;  // rare, but it can happen after the ceil and floor.

            const double* imageptr = data.getPtr(ix1,y);
            const int step = data.getStep();
            double x_x0 = ix1 - x0;
#ifdef USE_TMV
            const double* mxxptr = Minv_xx__x_x0__x_x0.cptr() + ix1-xmin;
#else
            const double* mxxptr = Minv_xx__x_x0__x_x0.data() + ix1-xmin;
#endif
            for(int x=ix1;x<=ix2;++x,x_x0+=1.,imageptr+=step) {
                /* Compute displacement from weight centroid, then
                 * get elliptical radius and weight.
                 */
                double rho2 = Minv_yy__y_y0__y_y0 + TwoMinv_xy__y_y0*x_x0 + *mxxptr++;
                xdbg<<"Using pixel: "<<x<<" "<<y<<" with value "<<*(imageptr)<<" rho2 "<<rho2<<" x_x0 "<<x_x0<<" y_y0 "<<y_y0<<std::endl;
                xassert(rho2 < hsmparams.max_moment_nsig2 + 1.e-8); // allow some numerical error.

                double intensity = std::exp(-0.5 * rho2) * (*imageptr);

                /* Now do the addition */
                double intensity__x_x0 = intensity * x_x0;
                double intensity__y_y0 = intensity * y_y0;
                A    += intensity;
                Bx   += intensity__x_x0;
                By   += intensity__y_y0;
                Cxx  += intensity__x_x0 * x_x0;
                Cxy  += intensity__x_x0 * y_y0;
                Cyy  += intensity__y_y0 * y_y0;
                rho4w+= intensity * rho2 * rho2;
            }
        }
        dbg<<"Exiting find_ellipmom_1 with results: "<<A<<" "<<Bx<<" "<<By<<" "<<Cxx<<" "<<Cyy<<" "<<Cxy<<" "<<rho4w<<std::endl;
    }

    /* find_ellipmom_2
     * *** COMPUTES ADAPTIVE ELLIPTICAL MOMENTS OF AN IMAGE ***
     *
     * Finds the best-fit Gaussian:
     *
     * f ~ A / (pi*sqrt det M) * exp( - (r-r0) * M^-1 * (r-r0) )
     *
     * The fourth moment rho4 is also returned.
     * Note that the total image intensity for the Gaussian is 2A.
     *
     * Arguments:
     *   data: ImageView structure containing the image
     * > A: adaptive amplitude
     * > x0: adaptive centroid (x)
     * > y0: adaptive centroid (y)
     * > Mxx: adaptive covariance (xx)
     * > Mxy: adaptive covariance (xy)
     * > Myy: adaptive covariance (yy)
     * > rho4: rho4 moment
     *   convergence_threshold: required accuracy
     * > num_iter: number of iterations required to converge
     */

    void find_ellipmom_2(
        ConstImageView<double> data, double& A, double& x0, double& y0,
        double& Mxx, double& Mxy, double& Myy, double& rho4, double convergence_threshold,
        int& num_iter, const HSMParams& hsmparams)
    {

        double convergence_factor = 1.0;
        double Amp,Bx,By,Cxx,Cxy,Cyy;
        double semi_a2, semi_b2, two_psi;
        double dx, dy, dxx, dxy, dyy;
        double shiftscale, shiftscale0=0.;
        double x00 = x0;
        double y00 = y0;

        num_iter = 0;

#ifdef N_CHECKVAL
        if (convergence_threshold <= 0 || convergence_threshold >= convergence_factor) {
            throw HSMError("Error: convergence_threshold out of range in find_ellipmom_2.\n");
        }
#endif

        /*
         * Set Amp = -1000 as initial value just in case the while() block below is never triggered;
         * in this case we have at least *something* defined to divide by, and for which the output
         * will fairly clearly be junk.
         */
        Amp = -1000.;

        /* Iterate until we converge */
        while(convergence_factor > convergence_threshold) {

            /* Get moments */
            find_ellipmom_1(data, x0, y0, Mxx, Mxy, Myy, Amp, Bx, By, Cxx, Cxy, Cyy, rho4, hsmparams);

            /* Compute configuration of the weight function */
            two_psi = std::atan2( 2* Mxy, Mxx-Myy );
            semi_a2 = 0.5 * ((Mxx+Myy) + (Mxx-Myy)*std::cos(two_psi)) + Mxy*std::sin(two_psi);
            semi_b2 = Mxx + Myy - semi_a2;

            if (semi_b2 <= 0) {
                throw HSMError("Error: non positive-definite weight in find_ellipmom_2.\n");
            }

            shiftscale = std::sqrt(semi_b2);
            if (num_iter == 0) shiftscale0 = shiftscale;

            /* Now compute changes to x0, etc. */
            dx = 2. * Bx / (Amp * shiftscale);
            dy = 2. * By / (Amp * shiftscale);
            dxx = 4. * (Cxx/Amp - 0.5*Mxx) / semi_b2;
            dxy = 4. * (Cxy/Amp - 0.5*Mxy) / semi_b2;
            dyy = 4. * (Cyy/Amp - 0.5*Myy) / semi_b2;

            if (dx     >  hsmparams.bound_correct_wt) dx     =  hsmparams.bound_correct_wt;
            if (dx     < -hsmparams.bound_correct_wt) dx     = -hsmparams.bound_correct_wt;
            if (dy     >  hsmparams.bound_correct_wt) dy     =  hsmparams.bound_correct_wt;
            if (dy     < -hsmparams.bound_correct_wt) dy     = -hsmparams.bound_correct_wt;
            if (dxx    >  hsmparams.bound_correct_wt) dxx    =  hsmparams.bound_correct_wt;
            if (dxx    < -hsmparams.bound_correct_wt) dxx    = -hsmparams.bound_correct_wt;
            if (dxy    >  hsmparams.bound_correct_wt) dxy    =  hsmparams.bound_correct_wt;
            if (dxy    < -hsmparams.bound_correct_wt) dxy    = -hsmparams.bound_correct_wt;
            if (dyy    >  hsmparams.bound_correct_wt) dyy    =  hsmparams.bound_correct_wt;
            if (dyy    < -hsmparams.bound_correct_wt) dyy    = -hsmparams.bound_correct_wt;

            /* Convergence tests */
            convergence_factor = std::abs(dx)>std::abs(dy)? std::abs(dx): std::abs(dy);
            convergence_factor *= convergence_factor;
            if (std::abs(dxx)>convergence_factor) convergence_factor = std::abs(dxx);
            if (std::abs(dxy)>convergence_factor) convergence_factor = std::abs(dxy);
            if (std::abs(dyy)>convergence_factor) convergence_factor = std::abs(dyy);
            convergence_factor = std::sqrt(convergence_factor);
            if (shiftscale<shiftscale0) convergence_factor *= shiftscale0/shiftscale;

            /* Now update moments */
            x0 += dx * shiftscale;
            y0 += dy * shiftscale;
            Mxx += dxx * semi_b2;
            Mxy += dxy * semi_b2;
            Myy += dyy * semi_b2;

            /* If the moments have gotten too large, or the centroid is out of range,
             * report a failure */
            if (std::abs(Mxx)>hsmparams.max_amoment || std::abs(Mxy)>hsmparams.max_amoment
                || std::abs(Myy)>hsmparams.max_amoment
                || std::abs(x0-x00)>hsmparams.max_ashift
                || std::abs(y0-y00)>hsmparams.max_ashift) {
                throw HSMError("Error: adaptive moment failed\n");
            }

            if (++num_iter > hsmparams.max_mom2_iter) {
                throw HSMError("Error: too many iterations in adaptive moments\n");
            }

            if (math::isNan(convergence_factor) || math::isNan(Mxx) ||
                math::isNan(Myy) || math::isNan(Mxy) ||
                math::isNan(x0) || math::isNan(y0)) {
                throw HSMError("Error: NaN in calculation of adaptive moments\n");
            }
        }

        /* Re-normalize rho4 */
        A = Amp;
        rho4 /= Amp;
    }
}
