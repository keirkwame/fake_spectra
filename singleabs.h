#ifndef SINGLE_ABSORBER_H
#define SINGLE_ABSORBER_H

#include <cmath>
#include "Faddeeva.h"

#define NGRID 8

inline double sph_kernel(const double q)
{
    if (q >= 1)
        return 0;
    if(q<0.5)
        return 1-6*q*q+6*q*q*q;
    else
        return 2*pow(1.- q,3);
}


/* Compute the Voigt profile, which is the real part of the 
 * Faddeeva (complex probability) function of the variable 
 * w = u + i a
 * So that F(w) = H(a,u) + i J(a,u) = exp(-w^2) erfc(-iw)
 * Arguments:
 * T0 = vdiff/btherm
 * aa: voigt_fac/btherm
 * (note btherm is sqrt(2k T / M))
 */
inline double profile(const double uu, const double aa)
{
    std::complex<double> ww ( uu , aa);
    std::complex<double> result = Faddeeva::w(ww);
    return result.real();
}

class SingleAbsorber
{
    public:

        /*This class evaluates the convolution integral for the optical depth.
         * This is:
         *
         * tau = int_{bin} dv' int_{sph kernel support} dv n(v) Phi(v-v')
         * where n is the local density and Phi is the broadening function.
         *
         * Arguments:
         * btherm: thermal b parameter: sqrt(2kT/m)
         * vel: velocity of particle parallel to sightline
         * vdr2: impact parameter from particle center to sightline in velocity units (km/s)
         * vsmooth: smoothing length in velocity units (km/s)
         * aa: voigt_fac/btherm the parameter for the Voigt profile
         */
        SingleAbsorber(double bth_i, double vdr2_i, double vsm_i, double aa_i):
            btherm(bth_i), vdr2(vdr2_i), vsmooth(vsm_i), aa(aa_i),
            vhigh(sqrt(vsmooth*vsmooth-vdr2))
        {};

        /*Find the mean optical depth in the bin, by averaging over the optical depth
         * at different points within it. The relevant scale here is the thermal broadening, b,
         * and we only need to sample once every delta v = b. If b > bin width, no sub-sampling is necessary.
         *
         * In practice b ~ 0.13 sqrt(T/Z) (where Z is the atomic mass of the species)
         * For iron at 10^4 K this gives b ~ 2, and so 1 km/s bins are fine
         * and no subsampling is needed (but we do it a litle just to be sure).
         * The minimal b we should encounter is iron at T = 2000K,
         * which gives b ~ 0.8 and 1 km/s is still probably ok, however we subsample a little anyway.
         * Arguments:
         * vlow, vhigh: integration limits, which should be v_bin - v_particle
         */
        double tau_kern_outer(const double vlow, const double vhigh)
        {
            //If the bin width is less than half
            //the thermal broadening scale,
            //no need to subsample
            if ((vhigh - vlow) < btherm/2.)
            {
              return tau_kern_inner((vhigh+vlow)/2.);
            }
            const int npoints = 2*ceil((vhigh - vlow)/(btherm/2.)/2)+1.;
            double total = tau_kern_inner(vlow)/2.;
            //Note that the number of points does not need to be NGRID here,
            //but it is for now.
            const double deltav=(vhigh-vlow)/(npoints-1);
            for(int i=1; i< npoints-1; ++i)
            {
                const double vv = i*deltav+vlow;
                total += tau_kern_inner(vv);
            }
            total += tau_kern_inner(vhigh)/2.;
            return total/(npoints-1);
        }

    private:
        /* This gives us the optical depth at velocity vouter, from a convolution of the density and
         * broadening function.
         * This is:
         *
         * tau(v') = int_{sph kernel support} dv n(v) Phi(v-v')
         * where n is the local density and Phi is the broadening function.
         *
         * (Strictly speaking internally we use v'' = v - v_part and v''' = v' - v_part)
         * Arguments:
         * vdr2: impact parameter from particle center to sightline in velocity units (km/s)
         * vouter: velocity value for the outer integral. This should be v_bin - v_particle.
         * btherm: thermal b parameter: sqrt(2kT/m)
         * vsmooth: smoothing length in velocity units (km/s)
         */
        inline double tau_kern_inner(const double vouter)
        {
            //Integration region goes from -vhigh to vhigh, where vhigh is precomputed kernel support
            const double deltav=2.*vhigh/NGRID;
            //Because we are integrating over the whole sph kernel,
            //the first and last terms will have q = 1, sph_kernel = 0, so don't need to compute them.
            double total = 0;
            for(int i=1; i<NGRID; ++i)
            {
                const double vv = i*deltav-vhigh;
                const double q = sqrt(vdr2+vv*vv)/vsmooth;
                //The difference between this velocity bin and the particle velocity
                const double vdiff = vv - vouter;
                const double T0 = vdiff/btherm;
                total+=sph_kernel(q)*profile(T0,aa);
            }
            return 32./3.*deltav*total;
        }

        const double btherm;
        const double vdr2;
        const double vsmooth;
        const double aa;
        const double vhigh;
};

double sph_kern_frac(double zlow, double zhigh, double smooth, double dr2, double zrange);
#endif
