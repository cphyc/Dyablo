// molecules.h
#pragma once
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION
double alpha_H2_prim(double T, double xe, double H2_cosmic_ray_ionization_rate, double G0, double xHI, double xHII){
    // H- channel for H2 formation
    double formation_rate = 0.0;

    // Primordial channel
    double logT = log10(T);
    double lnTe = log(T*8.621738E-5); // K -> eV

    // ! Creation and destruction channels of H- included with updated rates from Glover et al. 2010
    // ! H + e- -> H- + gamma
    double k1 = pow(10.0,-17.845 + 0.762*logT + 0.1523*pow(logT,2.)-0.03274*pow(logT,3.));
    if (T >= 6000.0) {
        k1 = pow(10.0,-16.420 + 0.1998*pow(logT,2.)-5.447E-3*pow(logT,4.)+4.0415E-5*pow(logT,6.));
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
    double k14 = 2.5634E-15 * pow(exp(lnTe),1.78186); // Note that T must be in eV for this reaction to make sense
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
    double formation_rate = dust_to_gas_mass_ratio_over_mw * (3.5E-17) * clumping_factor * sqrt(FMIN(T2,5));

    return formation_rate;
}

KOKKOS_INLINE_FUNCTION
double alpha_H2(double T, double dust_to_gas_mass_ratio_over_mw, double xe, double H2_cosmic_ray_ionization_rate, double G0, double xHI, double xHII, double nH){
    // Creation rate of molecular hydrogen
    // We consider both the primordial channel (via H-) as well
    // as formation on dust

    double formation_rate = 0.0;

    // Formation rate on dust. Consider only HI
    formation_rate += alpha_H2_dust(T, dust_to_gas_mass_ratio_over_mw) * xHI;

    // Primordial H- channel
    formation_rate += alpha_H2_prim(T, xe, H2_cosmic_ray_ionization_rate, G0, xHI, xHII) * xHI * nH;

    // Artificially down-weight creation at high T to prevent excess compute???
    formation_rate *= exp(-1.0 * pow(T / 1E5,2));

    return formation_rate;
}

KOKKOS_INLINE_FUNCTION
double beta_H2_umist(double T, double nH, double ne, double nH2){
    // H2 destrubtion from umist
    // http://udfa.ajmarkwick.net/index.php?species=4

    double destruction_rate = 0.0;

    //H2 + H2 --> H2 + H + H
    destruction_rate += 1.00e-8 * pow(T/300,0.00) * exp(-84100.00/T) * nH2;
    //H2 + e- --> H + H + e-
    destruction_rate += 3.22e-9 * pow(T/300,0.35) * exp(-102000.00/T) * ne;
    //H2 + H --> H + H + H
    destruction_rate += 4.67e-7 * pow(T/300,-1.00) * exp(-55000.00/T) * nH;

    return destruction_rate;
}

KOKKOS_INLINE_FUNCTION
double beta_H2(double T, double nH, double xHI, double xH2, double xHe, double ne, double nHI, double nH2, double nHeI){
    // ! Returns the collisional dissociation rates of H2 for four different
    // ! reactions [cm3s-1] from Glover & Abel (2008)
    // ! http://mnras.oxfordjournals.org/content/388/4/1627.full.pdf 

    double T4 = T / 1E4;

    // ! Critical number densities.  See eqns 15, 16, and 17
    double ncrH =  pow(10.0,3.0 - 0.416*log10(T4) - 0.327*log10(T4)*log10(T4));
    double ncrH2 = pow(10.0,4.845 - 1.3*log10(T4) + 1.62*log10(T4)*log10(T4));
    double ncrHe = pow(10.0,5.0792*(1.0 - 1.23E-5*(T - 2000.0)));

    // ! 1/ncr.  see eqn 14
    double invncr = (xHI/ncrH) + (xH2/ncrH2) + (xHe/ncrHe);

    // ! prefactors for LTE and NLTE collision rates  see eqn 13
    double LTEfac = (nH*invncr)/(1.0 + (nH*invncr));
    double NLTEfac = 1.0/(1.0 + (nH*invncr));
    if (ne > nHI) {
        LTEfac = 1.0;
        NLTEfac = 0.0;
    }

    // !reaction rates from the appendix:
    // !H2 + e- --> H + H + e-
    double k8 = 3.73E-9 * pow(T,0.1121) * exp(-99430.0/T); // ! Glover et al. (2010)

    // !H2 + H --> H + H + H
    double k9 = (6.67E-12)*sqrt(T)*exp(-1.0*(1.0 + (63593.0/T)));
    double k9L = (3.52E-9)*exp(-43900.0/T);

    // !H2 + H2 --> H2 + H + H
    double k10 = ( (5.996E-30*pow(T,4.1881)) / pow(1.0 + 6.761E-6*T,5.6881)) * exp(-54657.4/T);
    double k10L = (1.3E-9)*exp(-53300.0/T);

    // !H2 + He --> H + H + He
    double k11 = pow(10.0,-27.029 + (3.801*log10(T)) - (29487.0/T));
    double k11L = pow(10.0,-2.729 - (1.75*log10(T)) - (23474.0/T));

    k8   = FMAX(k8, 1E-40);
    k9   = FMAX(k9, 1E-40);
    k10  = FMAX(k10, 1E-40);
    k11  = FMAX(k11, 1E-40);
    k9L  = FMAX(k9L, 1E-40);
    k10L = FMAX(k10L, 1E-40);
    k11L = FMAX(k11L, 1E-40);

    // !Log of all the rates
    double lk8 = log10(k8);
    double lk9 = (LTEfac*log10(k9L)) + (NLTEfac*log10(k9));
    double lk10 = (LTEfac*log10(k10L)) + (NLTEfac*log10(k10));
    double lk11 = (LTEfac*log10(k11L)) + (NLTEfac*log10(k11));

    double destruction_rate = (ne*pow(10.0,lk8)) + (nHI*pow(10.0,lk9)) + (nH2*pow(10.0,lk10)) + (nHeI*pow(10.0,lk11));
    destruction_rate = FMAX(destruction_rate, 1E-40);

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
