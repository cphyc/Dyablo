// molecules.h
#pragma once
#include <Kokkos_Core.hpp>

namespace PRISM {

KOKKOS_INLINE_FUNCTION
double alpha_H2_prim(double T, double xe, double H2_cosmic_ray_ionization_rate, double G0, double xHI, double xHII){
    // H- channel for H2 formation
    double formation_rate = 0.0;

    // Primordial channel
    double logT = log10(T);
    double lnTe = log(T*8.621738E-5); // K -> eV

    // ! Creation and destruction channels of H- included with updated rates from Glover et al. 2010
    // ! H + e- -> H- + gamma
    double k1 = pow(10.0,-17.845 + 0.762*logT + 0.1523*pow(logT, 2)-0.03274*pow(logT, 3));
    if (T >= 6000.0) {
        k1 = pow(10.0,-16.420 + 0.1998*pow(logT, 2)-5.447E-3*pow(logT, 4)+4.0415E-5*pow(logT, 6));
    }

    // ! H- + H -> H2 + e
    double k2 = 4.0E-9*pow(FMAX(T,300.0),-0.17);

    // ! H- + H+ -> H + H
    double k5 = 2.4E-6/sqrt(T)*(1.0 + T/20000.);

    // ! H- + CR --> H + e-
    double k_hm_cr = 1.28E-13 * (H2_cosmic_ray_ionization_rate / 1.E-16);

    // ! H- + gamma --> H + e-
    double k_hm_gamma = 5.9E-9 * G0;

    // ! H- + e -> H + e + e
    double k13 = -1.801849334E1 + 2.36085220E0*lnTe - 2.82744300E-1*pow(lnTe,2.) \
                 +1.62331664E-2*pow(lnTe,3.)-3.36501203E-2*pow(lnTe,4.)+1.17832978E-2*pow(lnTe,5.) \
                 -1.65619470E-3*pow(lnTe,6.)+1.06827520E-4*pow(lnTe,7.)-2.63128581E-6*pow(lnTe,8.);
    k13 = exp(k13);

    // ! H- + H --> H + H + e-
    // ! I think this reaction was broken in glover so I took the results from
    // ! https://www.aanda.org/articles/aa/pdf/2016/02/aa27262-15.pdf Table A1
    // ! Harley added the fudge factor for continuity
    double k14 = 1.357772745525155 * 2.5634E-15 * pow(exp(lnTe),1.78186); // Note that T must be in eV for this reaction to make sense
    if (T > 1160.0) {
        k14 = -3.388464953E1 + 1.13944933E0*lnTe - 1.4210135E-1*pow(lnTe,2.) \
              + 8.4644554E-3*pow(lnTe,3.) - 1.4328641E-3*pow(lnTe,4.) + 2.0122503E-4*pow(lnTe,5.) \
              + 8.6639632E-5*pow(lnTe,6.) - 2.5850097E-5*pow(lnTe,7.) + 2.4555012E-6*pow(lnTe,8.) \
              - 8.0683825E-8*pow(lnTe,9.);
        k14 = exp(k14);
    }

    // ! H- + H+ --> H2+ + e- --> H + H (via recombinative dissociation)
    double k15 = 6.9E-9 * pow(T,-0.35);
    if (T > 8000.0) {
        k15 = 9.6E-7 * pow(T,-0.90);
    }

    formation_rate += k1*k2*xe/(k2 + k5*xHII + k_hm_cr + k_hm_gamma + k13*xe + k14*xHI + k15*xHII); 

    return formation_rate;
}

KOKKOS_INLINE_FUNCTION
double alpha_H2_dust(double T, double dust_to_gas_mass_ratio_over_mw){
    // Formation on dust
    double clumping_factor = 1.0;
    double T2 = T / 100.0;
    double formation_rate = dust_to_gas_mass_ratio_over_mw * (3.5E-17) * clumping_factor * sqrt(FMIN(T2,1e2));

    return formation_rate;
}

KOKKOS_INLINE_FUNCTION
double alpha_H2(double T, double dust_to_gas_mass_ratio_over_mw, double xe, double H2_cosmic_ray_ionization_rate, double G0, double xHI, double xHII, double nH){
    // Creation rate of molecular hydrogen
    // We consider both the primordial channel (via H-) as well
    // as formation on dust

    double formation_rate = 0.0;

    // Formation rate on dust. Consider only HI
    formation_rate += alpha_H2_dust(T, dust_to_gas_mass_ratio_over_mw) * xHI * nH;

    // Primordial H- channel
    formation_rate += alpha_H2_prim(T, xe, H2_cosmic_ray_ionization_rate, G0, xHI, xHII) * xHI * nH;

    return formation_rate;
}

KOKKOS_INLINE_FUNCTION
double beta_H2_krome(double T, double nH, double ne, double nH2, double nHe){
    //   ! From bovino 2016
    //   ! https://www.aanda.org/articles/aa/pdf/2016/06/aa28158-16.pdf

    double destruction_rate = 0.0;

    // H2 + H --> H + H + H (k18)
    destruction_rate += ((6.67E-12 * sqrt(T) * exp(-1.0 * (1.0 + (63593.0/T)))) * nH);

    // H2 + H2 --> H2 + H + H (k19)
    destruction_rate += ((5.996E-30 * pow(T,4.1881) * pow(1.0 + (6.761 * T),-5.6881) * exp(-54657.4/T)) * nH2);

    // H2 + e- --> H + H + e- (k22)
    destruction_rate += (4.38E-10 * pow(T,0.35) * exp(-102000.0/T) * ne);

    // H2 + He --> H + H + He
    destruction_rate += (pow(10.0,-27.029 + (3.801 * log10(T)) - (29487.0/T)) * nHe);

    return destruction_rate;
}

KOKKOS_INLINE_FUNCTION
double alpha_CO(double G0, double xi_cr_H2, double nCII, double nH2, double xO, double n){

    //! see glover 2012
    double k0 = 5.E-16; //! cm^3 s^-1
    double k1 = 5.E-10; //! Rate coefficient for the formation of CO from O + CHx

    //! Assuming CO formation is modulated by CH2+, https://home.strw.leidenuniv.nl/~ewine/photo/display_ch2+_65e31a07e69e64dbd64d37801983018f.html
    double gammaCHx_cr = 8.88E-15 * (xi_cr_H2 / 1E-16); //! Cosmic rays
    double gammaCHx = (1.41E-10 * G0) + gammaCHx_cr;

    double beta = k1 * xO/(k1*xO + gammaCHx/n);
    double formation_rate = k0 * nCII * nH2 * beta;

    return formation_rate;
}

KOKKOS_INLINE_FUNCTION
double beta_CO(double G0, double xi_cr_H2){
    //! CO destruction
    double gammaCO = 2.43E-10 * G0;
    double gammaCO_cr = 4.62E-15 * (xi_cr_H2 / 1E-16);

    double destruction_rate = gammaCO + gammaCO_cr;

    return destruction_rate;
}

} // namespace PRISM
