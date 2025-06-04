// collisional_ionization.h
#pragma once

#include "types.hpp"
#include <Kokkos_Core.hpp>

template<int N>
inline void copy1D (double src[N], Kokkos::View<double[N]> &data_d) {
    data_d = Kokkos::View<double[N]>("data_d", N);
    auto data_h = Kokkos::create_mirror_view(data_d);
    for (int i = 0; i < N; i++)
        data_h(i) = src[i];

    Kokkos::deep_copy(data_d, data_h);
}

inline void init_collisional_ionization(TabulatedData& tabData) {
    // Define the collisional ionization data
    // "carbon"
    double dE_carbon[] = { 11.3e0, 24.4e0, 47.9e0, 64.5e0, 392.1e0, 490.0e0 };
    double A_carbon[]  = { 0.685e-7, 0.186e-7, 0.635e-8, 0.150e-8, 0.299e-9, 0.123e-9 };
    double X_carbon[]  = { 0.193e0, 0.286e0, 0.427e0, 0.416e0, 0.666e0, 0.620e0 };
    double K_carbon[]  = { 0.25e0, 0.24e0, 0.21e0, 0.13e0, 0.02e0, 0.16e0 };
    double P_carbon[]  = { 0., 1., 1., 1., 1., 1. };
    copy1D(dE_carbon, tabData.dE_carbon);
    copy1D(A_carbon, tabData.A_carbon);
    copy1D(X_carbon, tabData.X_carbon);
    copy1D(K_carbon, tabData.K_carbon);
    copy1D(P_carbon, tabData.P_carbon);

    // "oxygen"
    double dE_oxygen[] = { 13.6e0, 35.1e0, 54.9e0, 77.4e0, 113.9e0, 138.1e0, 739.3e0, 871.5e0 };
    double A_oxygen[]  = { 0.359e-7, 0.139e-7, 0.931e-8, 0.102e-7, 0.219e-8, 0.195e-8, 0.212e-9, 0.521e-10 };
    double X_oxygen[]  = { 0.073e0, 0.212e0, 0.270e0, 0.614e0, 0.630e0, 0.360e0, 0.396e0, 0.629e0 };
    double K_oxygen[]  = { 0.34e0, 0.22e0, 0.27e0, 0.27e0, 0.17e0, 0.54e0, 0.35e0, 0.16e0 };
    double P_oxygen[]  = { 0., 1., 1., 0., 1., 0., 0., 1. };
    copy1D(dE_oxygen, tabData.dE_oxygen);
    copy1D(A_oxygen, tabData.A_oxygen);
    copy1D(X_oxygen, tabData.X_oxygen);
    copy1D(K_oxygen, tabData.K_oxygen);
    copy1D(P_oxygen, tabData.P_oxygen);

    // "nitrogen"
    double dE_nitrogen[] = { 14.5e0, 29.6e0, 47.5e0, 77.5e0, 97.9e0, 552.1e0, 667.0e0 };
    double A_nitrogen[]  = { 0.482e-7, 0.298e-7, 0.810e-8, 0.371e-8, 0.151e-8, 0.371e-9, 0.777e-10 };
    double X_nitrogen[]  = { 0.0652e0, 0.310e0, 0.350e0, 0.549e0, 0.0167e0, 0.546e0, 0.624e0 };
    double K_nitrogen[]  = { 0.42e0, 0.30e0, 0.24e0, 0.18e0, 0.74e0, 0.29e0, 0.16e0 };
    double P_nitrogen[]  = { 0., 0., 1., 1., 0., 0., 1. };
    copy1D(dE_nitrogen, tabData.dE_nitrogen);
    copy1D(A_nitrogen, tabData.A_nitrogen);
    copy1D(X_nitrogen, tabData.X_nitrogen);
    copy1D(K_nitrogen, tabData.K_nitrogen);
    copy1D(P_nitrogen, tabData.P_nitrogen);

    // "neon"
    double dE_neon[] = { 21.6e0, 41.0e0, 63.5e0, 97.1e0, 126.2e0, 157.9e0, 207.3e0, 239.1e0, 1196.0e0, 1360.6e0 };
    double A_neon[] = { 0.150e-7, 0.198e-7, 0.703e-8, 0.424e-8, 0.279e-8, 0.345e-8, 0.956e-9, 0.473e-9, 0.392e-10, 0.277e-10 };
    double X_neon[] = { 0.0329e0, 0.295e0, 0.0677e0, 0.0482e0, 0.305e0, 0.581e0, 0.749e0, 0.992e0, 0.262e0, 0.661e0 };
    double K_neon[] = { 0.43e0, 0.20e0, 0.39e0, 0.58e0, 0.25e0, 0.28e0, 0.14e0, 0.04e0, 0.20e0, 0.13e0 };
    double P_neon[] = { 1., 0., 1., 1., 1., 0., 1., 1., 1., 1. };
    copy1D(dE_neon, tabData.dE_neon);
    copy1D(A_neon, tabData.A_neon);
    copy1D(X_neon, tabData.X_neon);
    copy1D(K_neon, tabData.K_neon);
    copy1D(P_neon, tabData.P_neon);
        
    // "magnesium": {
    double dE_magnesium[] = { 7.6e0, 15.2e0, 80.1e0, 109.3e0, 141.3e0, 186.5e0, 224.9e0, 266.0e0, 328.2e0, 367.5e0, 1761.8e0, 1962.7e0 };
    double A_magnesium[] = { 0.621e-6, 0.192e-7, 0.556e-8, 0.435e-8, 0.710e-8, 0.170e-8, 0.122e-8, 0.220e-8, 0.486e-9, 0.235e-9, 0.206e-10, 0.175e-10 };
    double X_magnesium[] = { 0.592e0, 0.0027e0, 0.107e0, 0.159e0, 0.658e0, 0.242e0, 0.343e0, 0.897e0, 0.751e0, 1.030e0, 0.196e0, 0.835e0 };
    double K_magnesium[] = { 0.39e0, 0.85e0, 0.30e0, 0.31e0, 0.25e0, 0.28e0, 0.23e0, 0.22e0, 0.14e0, 0.10e0, 0.25e0, 0.11e0 };
    double P_magnesium[] = { 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1. };
    copy1D(dE_magnesium, tabData.dE_magnesium);
    copy1D(A_magnesium, tabData.A_magnesium);
    copy1D(X_magnesium, tabData.X_magnesium);
    copy1D(K_magnesium, tabData.K_magnesium);
    copy1D(P_magnesium, tabData.P_magnesium);

    // "silicon":
    double dE_silicon[] = { 8.2e0, 16.4e0, 33.5e0, 54.0e0, 166.8e0, 205.3e0, 246.5e0, 303.5e0, 351.1e0, 401.4e0, 476.4e0, 523.5e0, 2437.7e0, 2673.2e0 };
    double A_silicon[] = { 0.188e-6, 0.643e-7, 0.201e-7, 0.494e-8, 0.176e-8, 0.174e-8, 0.123e-8, 0.827e-9, 0.601e-9, 0.465e-9, 0.263e-9, 0.118e-9, 0.336e-10, 0.119e-10 };
    double X_silicon[] = { 0.376e0, 0.632e0, 0.473e0, 0.172e0, 0.102e0, 0.180e0, 0.518e0, 0.239e0, 0.305e0, 0.666e0, 0.666e0, 0.734e0, 0.336e0, 0.989e0 };
    double K_silicon[] = { 0.25e0, 0.20e0, 0.22e0, 0.23e0, 0.31e0, 0.29e0, 0.07e0, 0.28e0, 0.25e0, 0.04e0, 0.16e0, 0.16e0, 0.37e0, 0.08e0 };
    double P_silicon[] = { 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1. };
    copy1D(dE_silicon, tabData.dE_silicon);
    copy1D(A_silicon, tabData.A_silicon);
    copy1D(X_silicon, tabData.X_silicon);
    copy1D(K_silicon, tabData.K_silicon);
    copy1D(P_silicon, tabData.P_silicon);
        
    // "sulfur"
    double dE_sulfur[] = { 10.4e0, 23.3e0, 34.8e0, 47.3e0, 72.6e0, 88.1e0, 280.9e0, 328.2e0, 379.1e0, 447.1e0, 504.8e0, 564.7e0, 651.6e0, 707.2e0, 3223.9e0, 3494.2e0 };
    double A_sulfur[] = { 0.549e-7, 0.681e-7, 0.214e-7, 0.166e-7, 0.612e-8, 0.133e-8, 0.493e-8, 0.873e-9, 0.135e-8, 0.459e-9, 0.349e-9, 0.523e-9, 0.259e-9, 0.750e-10, 0.267e-10, 0.632e-11 };
    double X_sulfur[] = { 0.100e0, 0.693e0, 0.353e0, 1.030e0, 0.580e0, 0.0688e0, 1.130e0, 0.193e0, 0.431e0, 0.242e0, 0.305e0, 0.428e0, 0.854e0, 0.734e0, 0.572e0, 0.585e0 };
    double K_sulfur[] = { 0.25e0, 0.21e0, 0.24e0, 0.14e0, 0.19e0, 0.35e0, 0.16e0, 0.28e0, 0.32e0, 0.28e0, 0.25e0, 0.35e0, 0.12e0, 0.16e0, 0.28e0, 0.17e0 };
    double P_sulfur[] = { 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1. };
    copy1D(dE_sulfur, tabData.dE_sulfur);
    copy1D(A_sulfur, tabData.A_sulfur);
    copy1D(X_sulfur, tabData.X_sulfur);
    copy1D(K_sulfur, tabData.K_sulfur);
    copy1D(P_sulfur, tabData.P_sulfur);
        
    // "iron"
    double dE_iron[] = { 7.9e0, 16.2e0, 30.6e0, 54.8e0, 75.0e0, 99.0e0, 125.0e0, 151.1e0, 233.6e0, 262.1e0, 290.0e0, 331.0e0, 361.0e0, 392.0e0, 457.0e0, 489.3e0, 1262.0e0, 1360.0e0, 1470.0e0, 1582.0e0, 1690.0e0, 1800.0e0, 1960.0e0, 2046.0e0, 8828.0e0, 9277.7e0 };
    double A_iron[] = { 0.252e-6, 0.221e-7, 0.410e-7, 0.353e-7, 0.104e-7, 0.123e-7, 0.947e-8, 0.471e-8, 0.302e-8, 0.234e-8, 0.176e-8, 0.114e-8, 0.866e-9, 0.661e-9, 0.441e-9, 0.118e-9, 0.361e-9, 0.245e-9, 0.187e-9, 0.133e-9, 0.784e-10, 0.890e-10, 0.229e-10, 0.112e-10, 0.246e-11, 0.979e-12 };
    double X_iron[] = { 0.701e0, 0.033e0, 0.366e0, 0.243e0, 0.285e0, 0.411e0, 0.458e0, 0.280e0, 0.697e0, 0.764e0, 0.805e0, 0.773e0, 0.805e0, 0.762e0, 0.698e0, 0.211e0, 1.160e0, 0.978e0, 0.988e0, 1.030e0, 0.848e0, 1.200e0, 0.936e0, 0.034e0, 1.020e0, 0.664e0 };
    double K_iron[] = { 0.25e0, 0.45e0, 0.17e0, 0.39e0, 0.17e0, 0.21e0, 0.21e0, 0.28e0, 0.15e0, 0.14e0, 0.14e0, 0.15e0, 0.14e0, 0.14e0, 0.16e0, 0.15e0, 0.09e0, 0.13e0, 0.14e0, 0.12e0, 0.14e0, 0.35e0, 0.12e0, 0.81e0, 0.02e0, 0.14e0 };
    double P_iron[] = { 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1. };
    copy1D(dE_iron, tabData.dE_iron);
    copy1D(A_iron, tabData.A_iron);
    copy1D(X_iron, tabData.X_iron);
    copy1D(K_iron, tabData.K_iron);
    copy1D(P_iron, tabData.P_iron);
};

KOKKOS_INLINE_FUNCTION
double coll_ion(double T, double dE, double A, double X, double K, double P){
    // Eqn 1 of Voronov 1997
    double U = dE / (T * 8.61732814974056E-05); // #! K --> eV
    return A * (1. + P * sqrt(U)) * pow(U,K) * exp(-U) / (X + U);
}

KOKKOS_INLINE_FUNCTION
double collisional_ionization(double T, int ion, int element_idx, const TabulatedData& tabData) {
    /*
    collisional ionization rate
    */

    double col = 0.0;

    switch (element_idx) {
        case 1: // hydrogen
            {
            // #! Collisional ionization rate [cm3 s-1] of HI (Maselli&'03)-------------
            double T5 = T / 1E5;
            double f = 1.0 + sqrt(T5);
            col = 5.85E-11 * (sqrt(T) / f) * exp(-157809.1/T);
            break;
            }
        case 2: // helium
            {
            double T5 = T / 1E5;
            double f = 1.0 + sqrt(T5);
            switch (ion) {
                case 0: // HeI --> HeII
                    col = 2.38E-11 * (sqrt(T) / f) * exp(-285335.4/T);
                    break;
                case 1: // HeII --> HeIII
                    col = 5.68E-12 * (sqrt(T) / f) * exp(-631515.0/T);
                    break;
            }
            break;
            }
        case 6: // carbon
            col = coll_ion(T, tabData.dE_carbon(ion), tabData.A_carbon(ion), tabData.X_carbon(ion), tabData.K_carbon(ion), tabData.P_carbon(ion));
            break;
        case 7: // nitrogen
            col = coll_ion(T, tabData.dE_nitrogen(ion), tabData.A_nitrogen(ion), tabData.X_nitrogen(ion), tabData.K_nitrogen(ion), tabData.P_nitrogen(ion));
            break;
        case 8: // oxygen
            col = coll_ion(T, tabData.dE_oxygen(ion), tabData.A_oxygen(ion), tabData.X_oxygen(ion), tabData.K_oxygen(ion), tabData.P_oxygen(ion));
            break;
        case 10: // neon
            col = coll_ion(T, tabData.dE_neon(ion), tabData.A_neon(ion), tabData.X_neon(ion), tabData.K_neon(ion), tabData.P_neon(ion));
            break;
        case 12: // magnesium
            col = coll_ion(T, tabData.dE_magnesium(ion), tabData.A_magnesium(ion), tabData.X_magnesium(ion), tabData.K_magnesium(ion), tabData.P_magnesium(ion));
            break;
        case 14: // silicon
            col = coll_ion(T, tabData.dE_silicon(ion), tabData.A_silicon(ion), tabData.X_silicon(ion), tabData.K_silicon(ion), tabData.P_silicon(ion));
            break;
        case 16: // sulfur
            col = coll_ion(T, tabData.dE_sulfur(ion), tabData.A_sulfur(ion), tabData.X_sulfur(ion), tabData.K_sulfur(ion), tabData.P_sulfur(ion));
            break;
        case 26: // iron
            col = coll_ion(T, tabData.dE_iron(ion), tabData.A_iron(ion), tabData.X_iron(ion), tabData.K_iron(ion), tabData.P_iron(ion));
            break;
    }

    return col;
}