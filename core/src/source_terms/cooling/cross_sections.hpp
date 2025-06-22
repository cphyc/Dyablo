#pragma once

// cross_sections.c
#include <stdio.h>
#include <math.h>
#include "constants.hpp"

namespace PRISM {

// Number of points for the cross section and energy integrals

inline PRISM::CrossSection_C initialize_cross_sections(){

    PRISM::CrossSection_C out;

    // Initialize cross sections from Verner 1996
    printf("Initializing cross section data\n");

    // Hydrogen
    double hydrogen_E_th[] = { 1.360E1 };
    double hydrogen_E_max[] = { 5.000E4 };
    double hydrogen_E_0[] = { 4.298E-1 };
    double hydrogen_sig_0[] = { 5.475E4 };
    double hydrogen_y_a[] = { 3.288E1 };
    double hydrogen_P[] = { 2.963E0 };
    double hydrogen_y_w[] = { 0.E0 };
    double hydrogen_y_0[] = { 0.E0 };
    double hydrogen_y_1[] = { 0.E0 };

    // Copy the data into the module
    for (int j = 0; j < 1; j++) {
        out.E_th[1][j] = hydrogen_E_th[j];
        out.E_max[1][j] = hydrogen_E_max[j];
        out.E_0[1][j] = hydrogen_E_0[j];
        out.sig_0[1][j] = hydrogen_sig_0[j];
        out.y_a[1][j] = hydrogen_y_a[j];
        out.P[1][j] = hydrogen_P[j];
        out.y_w[1][j] = hydrogen_y_w[j];
        out.y_0[1][j] = hydrogen_y_0[j];
        out.y_1[1][j] = hydrogen_y_1[j];
    }

    // Helium
    double helium_E_th[] = { 2.459E1, 5.442E1 };
    double helium_E_max[] = { 5.000E4, 5.000E4 };
    double helium_E_0[] = { 1.361E1, 1.720E0 };
    double helium_sig_0[] = { 0.492E2, 1.369E4 };
    double helium_y_a[] = { 1.469E0, 3.288E1 };
    double helium_P[] = { 3.188E0, 2.963E0 };
    double helium_y_w[] = { 2.039E0, 0.E0 };
    double helium_y_0[] = { 4.434E-1, 0.E0 };
    double helium_y_1[] = { 2.136E0, 0.E0 };

    // Copy the data into the module
    for (int j = 0; j < 2; j++) {
        out.E_th[2][j] = helium_E_th[j];
        out.E_max[2][j] = helium_E_max[j];
        out.E_0[2][j] = helium_E_0[j];
        out.sig_0[2][j] = helium_sig_0[j];
        out.y_a[2][j] = helium_y_a[j];
        out.P[2][j] = helium_P[j];
        out.y_w[2][j] = helium_y_w[j];
        out.y_0[2][j] = helium_y_0[j];
        out.y_1[2][j] = helium_y_1[j];
    }

    // Carbon
    double carbon_E_th[] = { 1.126E1, 2.438E1, 4.789E1, 6.449E1, 3.921E2, 4.900E2 };
    double carbon_E_max[] = { 2.910E2, 3.076E2, 3.289E2, 3.522E2, 5.000E4, 5.000E4 };
    double carbon_E_0[] = { 2.144E0, 4.058E-1, 4.614E0, 3.506E0, 4.624E1, 1.548E1 };
    double carbon_sig_0[] = { 5.027E2, 8.709E0, 1.539E4, 1.068E2, 2.344E2, 1.521E3 };
    double carbon_y_a[] = { 6.216E1, 1.261E2, 1.737E0, 1.436E1, 2.183E1, 3.288E1 };
    double carbon_P[] = { 5.101E0, 8.578E0, 1.593E1, 7.457E0, 2.581E0, 2.963E0 };
    double carbon_y_w[] = { 9.157E-2, 2.093E0, 5.922E0, 0.E0, 0.E0, 0.E0 };
    double carbon_y_0[] = { 1.133E0, 4.929E1, 4.378E-3, 0.E0, 0.E0, 0.E0 };
    double carbon_y_1[] = { 1.607E0, 3.234E0, 2.528E-2, 0.E0, 0.E0, 0.E0 };

    // Copy the data into the module
    for (int j = 0; j < 6; j++) {
        out.E_th[6][j] = carbon_E_th[j];
        out.E_max[6][j] = carbon_E_max[j];
        out.E_0[6][j] = carbon_E_0[j];
        out.sig_0[6][j] = carbon_sig_0[j];
        out.y_a[6][j] = carbon_y_a[j];
        out.P[6][j] = carbon_P[j];
        out.y_w[6][j] = carbon_y_w[j];
        out.y_0[6][j] = carbon_y_0[j];
        out.y_1[6][j] = carbon_y_1[j];
    }

    // Nitrogen
    double nitrogen_E_th[] = { 1.453E1, 2.960E1, 4.745E1, 7.747E1, 9.789E1, 5.521E2, 6.671E2 };
    double nitrogen_E_max[] = { 4.048E2, 4.236E2, 4.473E2, 5.753E2, 5.043E2, 5.0E4, 5.0E4 };
    double nitrogen_E_0[] = { 4.034E0, 6.128E-2, 2.420E-1, 5.494E0, 4.471E0, 6.943E1, 2.108E1 };
    double nitrogen_sig_0[] = { 8.235E2, 1.944E0, 9.375E-1, 1.690E4, 8.376E1, 1.519E2, 1.117E3 };
    double nitrogen_y_a[] = { 8.033E1, 8.163E2, 2.788E2, 1.714E0, 3.297E1, 2.627E1, 3.288E1 };
    double nitrogen_P[] = { 3.928E0, 8.773E0, 9.156E0, 1.706E1, 6.003E0, 2.315E0, 2.963E0 };
    double nitrogen_y_w[] = { 9.097E-2, 1.043E1, 1.850E0, 7.904E0, 0.E0, 0.E0, 0.E0 };
    double nitrogen_y_0[] = { 8.598E-1, 4.280E2, 1.877E2, 6.415E-3, 0.E0, 0.E0, 0.E0 };
    double nitrogen_y_1[] = { 2.325E0, 2.030E1, 3.999E0, 1.937E-2, 0.E0, 0.E0, 0.E0 };

    // Copy the data into the module
    for (int j = 0; j < 7; j++) {
        out.E_th[7][j] = nitrogen_E_th[j];
        out.E_max[7][j] = nitrogen_E_max[j];
        out.E_0[7][j] = nitrogen_E_0[j];
        out.sig_0[7][j] = nitrogen_sig_0[j];
        out.y_a[7][j] = nitrogen_y_a[j];
        out.P[7][j] = nitrogen_P[j];
        out.y_w[7][j] = nitrogen_y_w[j];
        out.y_0[7][j] = nitrogen_y_0[j];
        out.y_1[7][j] = nitrogen_y_1[j];
    }

    // Oxygen
    double oxygen_E_th[] = { 1.362E1, 3.512E1, 5.494E1, 7.741E1, 1.139E2, 1.381E2, 7.393E2, 8.714E2 };
    double oxygen_E_max[] = { 5.380E2, 5.581E2, 5.840E2, 6.144E2, 6.491E2, 6.837E2, 5.000E4, 5.000E4 };
    double oxygen_E_0[] = { 1.240E0, 1.386E0, 1.723E-1, 2.044E-1, 2.854E0, 7.824E0, 8.709E0, 2.754E1 };
    double oxygen_sig_0[] = { 1.745E3, 5.967E1, 6.753E2, 8.659E-1, 1.642E4, 6.864E1, 1.329E2, 8.554E2 };
    double oxygen_y_a[] = { 3.784E0, 3.175E1, 3.852E2, 4.931E2, 1.792E0, 3.210E1, 2.535E1, 3.288E1 };
    double oxygen_P[] = { 1.764E1, 8.943E0, 6.822E0, 8.785E0, 2.647E1, 5.495E0, 2.336E0, 2.963E0 };
    double oxygen_y_w[] = { 7.589E-2, 1.934E-2, 1.191E-1, 3.143E0, 2.836E1, 0.E0, 0.E0, 0.E0 };
    double oxygen_y_0[] = { 8.698E0, 2.131E1, 3.839E-3, 3.328E2, 3.036E-2, 0E0, 0E0, 0E0 };
    double oxygen_y_1[] = { 1.271E-1, 1.503E-2, 4.569E-1, 4.285E1, 5.554E-2, 0E0, 0E0, 0E0 };

    // Copy the data into the module
    for (int j = 0; j < 8; j++) {
        out.E_th[8][j] = oxygen_E_th[j];
        out.E_max[8][j] = oxygen_E_max[j];
        out.E_0[8][j] = oxygen_E_0[j];
        out.sig_0[8][j] = oxygen_sig_0[j];
        out.y_a[8][j] = oxygen_y_a[j];
        out.P[8][j] = oxygen_P[j];
        out.y_w[8][j] = oxygen_y_w[j];
        out.y_0[8][j] = oxygen_y_0[j];
        out.y_1[8][j] = oxygen_y_1[j];
    }

    // Neon
    double neon_E_th[] = { 2.156E1, 4.096E1, 6.346E1, 9.712E1, 1.262E2, 1.579E2, 2.073E2, 2.391E2, 1.196E3, 1.362E3 };
    double neon_E_max[] = { 8.701E2, 8.831E2, 9.131E2, 9.480E2, 9.873E2, 1.031E3, 1.078E3, 1.125E3, 5.000E4, 5.000E4 };
    double neon_E_0[] = { 4.870E0, 1.247E1, 7.753E-1, 5.566E0, 1.248E0, 1.499E0, 4.888E0, 1.003E1, 1.586E2, 4.304E1 };
    double neon_sig_0[] = { 4.287E3, 1.583E3, 5.708E0, 1.685E3, 2.430E0, 9.854E-1, 1.198E4, 5.631E1, 6.695E1, 5.475E2 };
    double neon_y_a[] = { 5.798E0, 3.935E0, 6.725E1, 6.409E2, 1.066E2, 1.350E2, 1.788E0, 3.628E1, 3.352E1, 3.288E1 };
    double neon_P[] = { 8.355E0, 7.810E0, 1.005E1, 3.056E0, 8.999E0, 8.836E0, 2.550E1, 5.585E0, 2.002E0, 2.963E0 };
    double neon_y_w[] = { 2.434E-1, 6.558E-2, 4.633E-1, 8.290E-3, 6.855E-1, 1.656E0, 2.811E1, 0.000E0, 0.000E0, 0.000E0 };
    double neon_y_0[] = { 4.236E-2, 1.520E0, 7.654E1, 5.149E0, 9.169E1, 1.042E2, 2.536E-2, 0.000E0, 0.000E0, 0.000E0 };
    double neon_y_1[] = { 5.873E0, 1.084E-1, 2.023E0, 6.687E0, 3.702E-1, 1.435E0, 4.417E-2, 0.000E0, 0.000E0, 0.000E0 };

    // Copy the data into the module
    for (int j = 0; j < 10; j++) {
        out.E_th[10][j] = neon_E_th[j];
        out.E_max[10][j] = neon_E_max[j];
        out.E_0[10][j] = neon_E_0[j];
        out.sig_0[10][j] = neon_sig_0[j];
        out.y_a[10][j] = neon_y_a[j];
        out.P[10][j] = neon_P[j];
        out.y_w[10][j] = neon_y_w[j];
        out.y_0[10][j] = neon_y_0[j];
        out.y_1[10][j] = neon_y_1[j];
    }

    // Magnesium
    double magnesium_E_th[] = { 7.646E0, 1.504E1, 8.014E1, 1.093E2, 1.413E2, 1.865E2, 2.249E2, 2.660E2, 3.282E2, 3.675E2, 1.762E3, 1.963E3 };
    double magnesium_E_max[] = { 5.490E1, 6.569E1, 1.317E3, 1.356E3, 1.400E3, 1.449E3, 1.503E3, 1.558E3, 1.618E3, 1.675E3, 5.0E4, 5.0E4 };
    double magnesium_E_0[] = { 1.197E1, 8.139E0, 1.086E1, 2.912E1, 9.762E-1, 1.711E0, 3.570E0, 4.884E-1, 3.482E1, 1.452E1, 2.042E2, 6.203E1 };
    double magnesium_sig_0[] = { 1.372E8, 3.278E0, 5.377E2, 1.394E3, 1.728E0, 2.185E0, 3.104E0, 6.344E-2, 9.008E2, 4.427E1, 6.140E1, 3.802E2 };
    double magnesium_y_a[] = { 2.228E-1, 4.241E7, 9.779E0, 2.895E0, 9.184E1, 9.350E1, 6.060E1, 5.085E2, 1.823E0, 3.826E1, 2.778E1, 3.288E1 };
    double magnesium_P[] = { 1.574E1, 3.610E0, 7.117E0, 6.487E0, 1.006E1, 9.202E0, 8.857E0, 9.385E0, 1.444E1, 5.460E0, 2.161E0, 2.963E0 };
    double magnesium_y_w[] = { 2.805E-1, 0.E0, 2.604E0, 4.326E-2, 8.090E-1, 6.325E-1, 1.422E0, 6.666E-1, 2.751E0, 0.E0, 0.E0, 0.E0 };
    double magnesium_y_0[] = { 0.E0, 0.E0, 4.860E0, 9.402E-1, 1.276E2, 1.007E2, 5.452E1, 5.348E2, 5.444E0, 0.E0, 0.E0, 0.E0 };
    double magnesium_y_1[] = { 0.E0, 0.E0, 3.722E0, 1.135E-1, 3.979E0, 1.729E0, 2.078E0, 3.997E-3, 7.918E-2, 0.E0, 0.E0, 0.E0 };

    // Copy the data into the module
    for (int j = 0; j < 12; j++) {
        out.E_th[12][j] = magnesium_E_th[j];
        out.E_max[12][j] = magnesium_E_max[j];
        out.E_0[12][j] = magnesium_E_0[j];
        out.sig_0[12][j] = magnesium_sig_0[j];
        out.y_a[12][j] = magnesium_y_a[j];
        out.P[12][j] = magnesium_P[j];
        out.y_w[12][j] = magnesium_y_w[j];
        out.y_0[12][j] = magnesium_y_0[j];
        out.y_1[12][j] = magnesium_y_1[j];
    }

    // Silicon
    double silicon_E_th[] = { 8.152E0, 1.635E1, 3.349E1, 4.514E1, 1.668E2, 2.051E2, 2.465E2, 3.032E2, 3.511E2, 4.014E2, 4.761E2, 5.235E2, 2.438E3, 2.673E3 };
    double silicon_E_max[] = { 1.060E2, 1.186E2, 1.311E2, 1.466E2, 1.887E3, 1.946E3, 2.001E3, 2.058E3, 2.125E3, 2.194E3, 2.268E3, 2.336E3, 5.0E4, 5.0E4 };
    double silicon_E_0[] = { 2.317E1, 2.556E0, 1.659E-1, 1.288E1, 7.761E-1, 6.305E1, 3.277E-1, 7.655E-1, 3.343E-1, 8.787E-1, 1.205E1, 3.560E1, 2.752E2, 8.447E1 };
    double silicon_sig_0[] = { 2.506E1, 4.140E0, 5.790E-4, 6.083E0, 8.863E-1, 7.293E1, 6.680E-2, 3.477E-1, 1.465E-1, 1.950E-1, 1.992E4, 2.539E1, 4.754E1, 2.793E2 };
    double silicon_y_a[] = { 2.057E1, 1.337E1, 1.474E2, 1.356E6, 1.541E2, 1.558E2, 4.132E1, 3.733E2, 1.404E3, 7.461E2, 1.582E0, 3.307E1, 2.848E1, 3.288E1 };
    double silicon_P[] = { 3.546E0, 1.191E1, 1.336E1, 3.353E0, 9.980E0, 2.400E0, 1.606E1, 8.986E0, 8.503E0, 8.302E0, 2.425E1, 4.728E0, 2.135E0, 2.963E0 };
    double silicon_y_w[] = { 2.837E-1, 1.570E0, 8.626E-1, 0.E0, 1.303E0, 2.989E-3, 3.280E0, 1.476E-3, 1.646E0, 4.489E-1, 2.392E1, 0.E0, 0.E0, 0.E0 };
    double silicon_y_0[] = { 1.672E-5, 6.634E0, 9.613E1, 0.E0, 2.009E2, 1.115E0, 1.149E-2, 3.850E2, 1.036E3, 4.528E2, 1.990E-2, 0.E0, 0.E0, 0.E0 };
    double silicon_y_1[] = { 4.207E-1, 1.272E-1, 6.442E-1, 0.E0, 4.537E0, 8.051E-2, 6.396E-1, 8.999E-2, 2.936E-1, 1.015E0, 1.007E-2, 0.E0, 0.E0, 0.E0 };

    // Copy the data into the module
    for (int j = 0; j < 14; j++) {
        out.E_th[14][j] = silicon_E_th[j];
        out.E_max[14][j] = silicon_E_max[j];
        out.E_0[14][j] = silicon_E_0[j];
        out.sig_0[14][j] = silicon_sig_0[j];
        out.y_a[14][j] = silicon_y_a[j];
        out.P[14][j] = silicon_P[j];
        out.y_w[14][j] = silicon_y_w[j];
        out.y_0[14][j] = silicon_y_0[j];
        out.y_1[14][j] = silicon_y_1[j];
    }

    // Sulfur
    double sulfur_E_th[] = { 10.36E0, 23.33E0, 34.83E0, 47.31E0, 72.68E0, 88.05E0, 280.9E0, 328.2E0, 379.1E0, 447.1E0, 504.8E0, 564.7E0, 651.7E0, 707.2E0, 3224.E0, 3494.E0 };
    double sulfur_E_max[] = { 170.E0, 184.6E0, 199.5E0, 216.4E0, 235.E0, 255.7E0, 2569.E0, 2641.E0, 2705.E0, 2782.E0, 2859.E0, 2941.E0, 3029.E0, 3107.E0, 50000.E0, 50000.E0 };
    double sulfur_E_0[] = { 1.808E1, 8.787E0, 2.027E0, 2.173E0, 1.713E-1, 1.413E1, 3.757E-1, 1.462E1, 1.526E-1, 1.040E1, 6.485E0, 2.443E0, 1.474E1, 3.310E1, 4.390E2, 1.104E2 };
    double sulfur_sig_0[] = { 4.564E4, 3.136E2, 6.666E0, 2.606E0, 5.072E-4, 9.139E0, 5.703E-1, 3.161E1, 9.646E3, 5.364E1, 1.275E1, 3.490E-1, 2.294E4, 2.555E1, 2.453E1, 2.139E2 };
    double sulfur_y_a[] = { 1.000E0, 3.442E0, 5.454E1, 6.641E1, 1.986E2, 1.656E3, 1.460E2, 1.611E1, 1.438E3, 3.641E1, 6.583E1, 5.411E2, 1.529E0, 3.821E1, 4.405E1, 3.288E1 };
    double sulfur_P[] = { 13.61E0, 12.81E0, 8.611E0, 8.655E0, 13.07E0, 3.626E0, 11.35E0, 8.642E0, 5.977E0, 7.09E0, 7.692E0, 7.769E0, 25.68E0, 5.037E0, 1.765E0, 2.963E0 };
    double sulfur_y_w[] = { 6.385E-1, 7.354E-1, 4.109E0, 1.863E0, 7.880E-1, 0.000E0, 1.503E0, 1.153E-3, 1.492E0, 2.310E0, 1.678E0, 7.033E-1, 2.738E1, 0.000E0, 0.000E0, 0.000E0 };
    double sulfur_y_0[] = { 9.935E-1, 2.782E0, 1.568E1, 1.975E1, 9.424E1, 0.000E0, 2.222E2, 1.869E1, 1.615E-3, 1.775E1, 3.426E1, 2.279E2, 2.203E-2, 0.000E0, 0.000E0, 0.000E0 };
    double sulfur_y_1[] = { 0.2486E0, 0.1788E0, 9.421E0, 3.361E0, 0.6265E0, 0.E0, 4.606E0, 0.3037E0, 0.4049E0, 1.663E0, 0.137E0, 1.172E0, 0.01073E0, 0.E0, 0.E0, 0.E0 };

    // Copy the data into the module
    for (int j = 0; j < 16; j++) {
        out.E_th[16][j] = sulfur_E_th[j];
        out.E_max[16][j] = sulfur_E_max[j];
        out.E_0[16][j] = sulfur_E_0[j];
        out.sig_0[16][j] = sulfur_sig_0[j];
        out.y_a[16][j] = sulfur_y_a[j];
        out.P[16][j] = sulfur_P[j];
        out.y_w[16][j] = sulfur_y_w[j];
        out.y_0[16][j] = sulfur_y_0[j];
        out.y_1[16][j] = sulfur_y_1[j];
    }

    // Iron
    double iron_E_th[] = { 7.902E0, 1.619E1, 3.065E1, 5.480E1, 7.501E1, 9.906E1, 1.250E2, 1.511E2, 2.336E2, 2.621E2, 2.902E2, 3.308E2, 3.610E2, 3.922E2, 4.570E2, 4.893E2, 1.262E3, 1.358E3, 1.456E3, 1.582E3, 1.689E3, 1.799E3, 1.950E3, 2.046E3, 8.829E3, 9.278E3 };
    double iron_E_max[] = { 66.E0, 76.17E0, 87.05E0, 106.7E0, 128.8E0, 152.7E0, 178.3E0, 205.5E0, 921.1E0, 959.E0, 998.3E0, 1039.E0, 1081.E0, 1125.E0, 1181.E0, 1216.E0, 7651.E0, 7769.E0, 7918.E0, 8041.E0, 8184.E0, 8350.E0, 8484.E0, 8638.E0, 50000.E0, 50000.E0 };
    double iron_E_0[] = { 5.461E-2, 1.761E-1, 1.698E-1, 2.544E1, 7.256E-1, 2.656E0, 5.059E0, 7.098E-2, 6.741E0, 6.886E1, 8.284E0, 6.295E0, 1.317E-1, 8.509E-1, 5.555E-2, 2.873E1, 3.444E-1, 3.190E1, 7.519E-4, 2.011E1, 9.243E0, 9.713E0, 4.575E1, 7.326E1, 1.057E3, 2.932E2 };
    double iron_sig_0[] = { 3.062E-1, 4.365E3, 6.107E0, 3.653E2, 1.523E-3, 5.259E-1, 2.420E4, 1.979E1, 2.687E1, 6.470E1, 3.281E0, 1.738E0, 2.791E-3, 1.454E-1, 2.108E2, 1.207E1, 1.452E0, 2.388E0, 6.066E-5, 4.455E-1, 1.098E1, 7.204E-2, 2.580E4, 1.276E1, 1.195E1, 8.099E1 };
    double iron_y_a[] = { 2.671E7, 6.298E3, 1.555E3, 8.913E0, 3.736E1, 1.450E1, 4.850E4, 1.745E4, 1.807E2, 2.062E1, 5.360E1, 1.130E2, 2.487E3, 1.239E3, 2.045E4, 5.150E2, 3.960E2, 2.186E1, 1.606E6, 4.236E1, 7.637E1, 1.853E2, 1.358E0, 4.914E1, 5.769E1, 3.288E1 };
    double iron_P[] = { 7.923E0, 5.204E0, 8.055E0, 6.538E0, 17.67E0, 16.32E0, 2.374E0, 6.75E0, 6.29E0, 4.111E0, 8.571E0, 8.037E0, 9.791E0, 8.066E0, 6.033E0, 3.846E0, 10.13E0, 9.589E0, 8.813E0, 9.724E0, 7.962E0, 8.843E0, 26.04E0, 4.941E0, 1.718E0, 2.963E0 };
    double iron_y_w[] = { 2.069E1, 1.141E1, 8.698E0, 5.602E-1, 5.064E1, 1.558E1, 2.516E-3, 2.158E2, 2.387E-4, 2.778E-4, 3.279E-1, 3.096E-1, 6.938E-1, 4.937E-1, 1.885E-3, 0.000E0, 1.264E0, 2.902E-2, 4.398E0, 2.757E0, 1.748E0, 9.551E-3, 2.723E1, 0.000E0, 0.000E0, 0.000E0 };
    double iron_y_0[] = { 1.382E2, 9.272E1, 1.760E2, 0.000E0, 8.871E1, 3.361E1, 4.546E-1, 2.542E3, 2.494E1, 1.190E-5, 2.971E1, 4.671E1, 2.170E3, 4.505E2, 2.706E-4, 0.000E0, 2.891E1, 3.805E1, 1.915E6, 6.847E1, 4.446E1, 1.702E2, 3.582E-2, 0.000E0, 0.000E0, 0.000E0 };
    double iron_y_1[] = { 2.481E-1, 1.075E2, 1.847E1, 0.000E0, 5.280E-2, 3.743E-3, 2.683E1, 4.672E2, 8.251E0, 6.570E-3, 5.220E-1, 1.425E-1, 6.852E-3, 2.504E0, 1.628E0, 0.000E0, 3.404E0, 4.805E-1, 3.140E1, 3.989E0, 3.512E0, 4.263E0, 8.712E-3, 0.000E0, 0.000E0, 0.000E0 };

    // Copy the data into the module
    for (int j = 0; j < 26; j++) {
        out.E_th[26][j] = iron_E_th[j];
        out.E_max[26][j] = iron_E_max[j];
        out.E_0[26][j] = iron_E_0[j];
        out.sig_0[26][j] = iron_sig_0[j];
        out.y_a[26][j] = iron_y_a[j];
        out.P[26][j] = iron_P[j];
        out.y_w[26][j] = iron_y_w[j];
        out.y_0[26][j] = iron_y_0[j];
        out.y_1[26][j] = iron_y_1[j];
    }

    return out;
}

inline double get_cross_section(double lambda, int element, int ion, const PRISM::CrossSection_C &verner_cross_sections) {

    // Initialize cross section to 0
   double cross_sec = 0.0;

   // Convert lambda into ev
   double E = H_PLANCK * C_CGS/(lambda*1E-8) / EV_2_ERG; // photon energy in ev

   // Deal with molecular hydrogen separately
   if ((element == 1) & (ion == 2)){
       if ((E>11.20001) & (E<13.59)){
           cross_sec = 2.47E-18;
       }
       if ((E>=13.59) & (E<=15.21)){
           cross_sec = 0.0;
       }
       if ((E>15.21) & (E<=15.45)){
           cross_sec = 0.09E-18;
       }
       if ((E>15.70) & (E<=15.95)){
           cross_sec = 1.15E-18;
       }
       if ((E>15.95) & (E<=16.20)){
           cross_sec = 3.00E-18;
       }
       if ((E>16.20) & (E<=16.40)){
           cross_sec = 5.00E-18;
       }
       if ((E>16.40) & (E<=16.65)){
           cross_sec = 6.75E-18;
       }
       if ((E>16.65) & (E<=16.85)){
           cross_sec = 8.00E-18;
       }
       if ((E>16.85) & (E<=17.00)){
           cross_sec = 9.00E-18;
       }
       if ((E>17.00) & (E<=17.20)){
           cross_sec = 9.50E-18;
       }
       if ((E>17.20) & (E<=17.65)){
           cross_sec = 9.80E-18;
       }
       if ((E>17.65) & (E<=18.10)){
           cross_sec = 10.10E-18;
       }
       if (E>18.10){
           cross_sec = 10.10E-18 * pow(18.10/E, 3);
       }

       return cross_sec;
   }

   double x = (E / verner_cross_sections.E_0[element][ion]) - verner_cross_sections.y_0[element][ion];
   double y = sqrt( (x*x) + pow(verner_cross_sections.y_1[element][ion], 2) );

   double F = pow(x - 1.0, 2);
   F = F + pow(verner_cross_sections.y_w[element][ion], 2);
   F = F * pow(y, (0.5*verner_cross_sections.P[element][ion] - 5.5));
   F = F * pow((1.0 + sqrt(y/verner_cross_sections.y_a[element][ion])), (-1.0*verner_cross_sections.P[element][ion]));

   cross_sec = verner_cross_sections.sig_0[element][ion] * F * 1E-18;
   if (E < verner_cross_sections.E_th[element][ion]){
      cross_sec = 0.0;
   }

   if (E > verner_cross_sections.E_max[element][ion]){
      cross_sec = 0.0;
   }

   return cross_sec;
}


inline double blackbody_nu(double T, double nu){
    // Blackbody function B_lam
    // T --> temeprature [K]
    // ni --> frequency [Hz] 

    // now compute B_lam
    double B_nu = 2.0 * H_PLANCK * pow(nu, 3) / pow(C_CGS, 2);
    B_nu = B_nu * (1.0 / (exp(H_PLANCK * nu / (KB * T)) - 1.0));

    return B_nu;
}

inline double sigma_N_num(double nu, double T, int element, int ion, const PRISM::CrossSection_C &verner_cross_sections){
    return get_cross_section((C_CGS / nu)*1e8, element, ion, verner_cross_sections) * blackbody_nu(T, nu) / (H_PLANCK * nu);
}

inline double sigma_N_den(double nu, double T, int element, int ion){
    return blackbody_nu(T, nu) / (H_PLANCK * nu);
}

template<typename T>
inline double simpson_rule(const T func, const double a, const double b) {
    constexpr int n = N_INTEGRATION_POINTS;
    double h = (b - a) / n;
    double sum = func(a) + func(b);

    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2 * func(x);
        } else {
            sum += 4 * func(x);
        }
    }

    return sum * h / 3.0;
}

inline double sigma_N(double E0, double E1, double T, int element, int ion, const PRISM::CrossSection_C &verner_cross_sections){
    // Photon number-weighted cross section
    // E0 lower energy of the bin in eV
    // E1 upper energy of the bin in eV

    double a = (E0 * EV_2_ERG) / H_PLANCK;
    double b = (E1 * EV_2_ERG) / H_PLANCK;

    // Simpson's rule for sigma * B / hnu
    double numerator = simpson_rule([&](double x) { return sigma_N_num(x, T, element, ion, verner_cross_sections); }, a, b);
    // Simpson's rule for B / hnu
    double denominator = simpson_rule([&](double x) { return sigma_N_den(x, T, element, ion); }, a, b);

    return numerator / denominator;
}

inline double sigma_E_num(double nu, double T, int element, int ion, const CrossSection_C &verner_cross_sections){
    return get_cross_section((C_CGS / nu)*1e8, element, ion, verner_cross_sections) * blackbody_nu(T, nu);
}

inline double sigma_E_den(double nu, double T, int element, int ion){
    return blackbody_nu(T, nu) / (H_PLANCK * nu);
}

inline double sigma_E(double E0, double E1, double T, int element, int ion, const CrossSection_C &verner_cross_sections){
    // Photon energy-weighted cross section
    // E0 lower energy of the bin in eV
    // E1 upper energy of the bin in eV

    double a = (E0 * EV_2_ERG) / H_PLANCK;
    double b = (E1 * EV_2_ERG) / H_PLANCK;

    // Simpson's rule for sigma * B / hnu
    double numerator = simpson_rule([&](double x) { return sigma_E_num(x, T, element, ion, verner_cross_sections); }, a, b);

    // Simpson's rule for B / hnu
    double denominator = simpson_rule([&](double x) { return sigma_E_den(x, T, element, ion); }, a, b);

    return numerator / denominator;
}

inline double e_bar_num(double nu, double T){
    return blackbody_nu(T, nu);
}

inline double e_bar_den(double nu, double T){
    return blackbody_nu(T, nu) / (H_PLANCK * nu);
}

inline double e_bar(double E0, double E1, double T){
    // Mean energy in the bin
    // E0 lower energy of the bin in eV
    // E1 upper energy of the bin in eV

    double a = (E0 * EV_2_ERG) / H_PLANCK;
    double b = (E1 * EV_2_ERG) / H_PLANCK;

    // Simpson's rule for sigma * B / hnu
    double numerator = simpson_rule([&](double x) { return e_bar_num(x, T); }, a, b);
    // Simpson's rule for B / hnu
    double denominator = simpson_rule([&](double x) { return e_bar_den(x, T); }, a, b);

    return (numerator / denominator) / EV_2_ERG;
}

using _cs_t = std::array<std::array<std::array<std::array<double, 3>, MAX_N_GROUPS>, MAX_ELEMENTS>, MAX_ELEMENTS>;

inline _cs_t update_cross_sections(
        const double T,
        const CrossSection_C& verner_cross_sections,
        const std::array<double, MAX_N_GROUPS> &group_E_min,
        const std::array<double, MAX_N_GROUPS> &group_E_max,
        const int N_groups
) {
    // Updates the cross sections for each element and group
    // TODO(code): update C_CGS with the reduced speed of light if needed
    _cs_t cs_ph = {};

    printf("Updating cross sections to a %e K blackbody\n",T);

    double group_energy;

    for (int i = 0; i < N_groups; i++){ // Loop over photon groups

        // First get the group energy in eV
        group_energy = e_bar(group_E_min[i], group_E_max[i], T);

        for (int j = 0; j < MAX_ELEMENTS; j++){ // Loop over elements
            // Initialize everything to 0 first
            for (int k = 0; k < MAX_ELEMENTS; k++){
                cs_ph[j][k][i][0] = 0.0;
                cs_ph[j][k][i][1] = 0.0;
                cs_ph[j][k][i][2] = 0.0;
            }

            // Now calculate cross sections
            for (int k = 0; k < j; k++){ // Loop over ions
                // Update number-weighted cross section
                cs_ph[j][k][i][0] = sigma_N(group_E_min[i], group_E_max[i], T, j, k, verner_cross_sections) * C_CGS; // cm^3 / s

                // Update energy-weighted cross section
                cs_ph[j][k][i][1] = sigma_E(group_E_min[i], group_E_max[i], T, j, k, verner_cross_sections) * C_CGS; // cm^3 / s

                // Update photoheating rates
                cs_ph[j][k][i][2] = (cs_ph[j][k][i][1] * group_energy) - (cs_ph[j][k][i][0] * verner_cross_sections.E_th[j][k]);
                cs_ph[j][k][i][2] = fmax(cs_ph[j][k][i][2],0.0) * EV_2_ERG; // Don't let photoheating go below 0.0
            }
        } // End loop over elements

        // Deal with H2 separately
        cs_ph[1][2][i][0] = sigma_N(group_E_min[i], group_E_max[i], T, 1, 2, verner_cross_sections); // cm^3 / s
        cs_ph[1][2][i][1] = sigma_E(group_E_min[i], group_E_max[i], T, 1, 2, verner_cross_sections); // cm^3 / s
        cs_ph[1][2][i][2] = (cs_ph[1][2][i][1] * group_energy) - (cs_ph[1][2][i][0] * verner_cross_sections.E_th[1][2]);
        cs_ph[1][2][i][2] = fmax(cs_ph[1][2][i][2],0.0) * EV_2_ERG; // Don't let photoheating go below 0.0


    } // End loop over photon groups

    return cs_ph;
}

} // namespace PRISM
