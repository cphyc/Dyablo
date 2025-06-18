#pragma once
#include <Kokkos_Core.hpp>

#define MAX_ELEMENTS 27
#define MAX_N_GROUPS 2
#define N_INTEGRATION_POINTS 10000

using Array2D = std::array<std::array<double, MAX_ELEMENTS>, MAX_ELEMENTS>;

typedef struct
{
    int atomic_number;
    int n_ions;
    int n_mol;
    double atomic_mass;
    double z_solar;
    double G0_photo_rate;
    double depletion;
} Element;

typedef struct {
    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS]> E_th;
    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS]> E_max;
    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS]> E_0;
    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS]> sig_0;
    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS]> y_a;
    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS]> P;
    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS]> y_w;
    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS]> y_0;
    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS]> y_1;
} CrossSection;


typedef struct
{
    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS][2]> HM12_UVB_z {"HM12_UVB_z"};
    Kokkos::View<double[MAX_ELEMENTS]> G0_heating_rates {"G0_heating_rates"};
    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS]> cosmic_ray_ionization_rates {"cosmic_ray_ionization_rates"};
    Kokkos::View<double[MAX_ELEMENTS]> cosmic_ray_ionization_rates_induced_UV {"cosmic_ray_ionization_rates_induced_UV"};
    Kokkos::View<double[MAX_ELEMENTS]> cosmic_ray_ionization_rates_induced_UV_heat {"cosmic_ray_ionization_rates_induced_UV_heat"};
    Kokkos::View<double[MAX_ELEMENTS][7]> dust_rec_coefs {"dust_rec_coefs"};
    Kokkos::View<double[7][3][31]> CTIon {"CTIon"};
    Kokkos::View<double[6][4][31]> CTRecomb {"CTRecomb"};
    Kokkos::View<double[MAX_ELEMENTS][160][8]> fs_cool_tab {"fs_cool_tab"};

    Kokkos::View<double[121][MAX_ELEMENTS][MAX_ELEMENTS]> high_t_cooling_rates {"high_t_cooling_rates"};
    Kokkos::View<bool[MAX_ELEMENTS][MAX_ELEMENTS]> high_t_cooling_rates_tflag {"high_t_cooling_rates_tflag"};

    Kokkos::View<double[7][6]> RR_rates_carbon {"RR_rates_carbon"};
    Kokkos::View<double[7][9]> DR_rates_c_carbon {"DR_rates_c_carbon"};
    Kokkos::View<double[7][9]> DR_rates_e_carbon {"DR_rates_e_carbon"};
    Kokkos::View<double[8][6]> RR_rates_nitrogen {"RR_rates_nitrogen"};
    Kokkos::View<double[8][9]> DR_rates_c_nitrogen {"DR_rates_c_nitrogen"};
    Kokkos::View<double[8][9]> DR_rates_e_nitrogen {"DR_rates_e_nitrogen"};
    Kokkos::View<double[9][6]> RR_rates_oxygen {"RR_rates_oxygen"};
    Kokkos::View<double[9][9]> DR_rates_c_oxygen {"DR_rates_c_oxygen"};
    Kokkos::View<double[9][9]> DR_rates_e_oxygen {"DR_rates_e_oxygen"};
    Kokkos::View<double[11][6]> RR_rates_neon {"RR_rates_neon"};
    Kokkos::View<double[11][9]> DR_rates_c_neon {"DR_rates_c_neon"};
    Kokkos::View<double[11][9]> DR_rates_e_neon {"DR_rates_e_neon"};
    Kokkos::View<double[13][6]> RR_rates_magnesium {"RR_rates_magnesium"};
    Kokkos::View<double[13][9]> DR_rates_c_magnesium {"DR_rates_c_magnesium"};
    Kokkos::View<double[13][9]> DR_rates_e_magnesium {"DR_rates_e_magnesium"};
    Kokkos::View<double[15][6]> RR_rates_silicon {"RR_rates_silicon"};
    Kokkos::View<double[15][9]> DR_rates_c_silicon {"DR_rates_c_silicon"};
    Kokkos::View<double[15][9]> DR_rates_e_silicon {"DR_rates_e_silicon"};
    Kokkos::View<double[17][6]> RR_rates_sulfur {"RR_rates_sulfur"};
    Kokkos::View<double[17][9]> DR_rates_c_sulfur {"DR_rates_c_sulfur"};
    Kokkos::View<double[17][9]> DR_rates_e_sulfur {"DR_rates_e_sulfur"};
    Kokkos::View<double[17][2]> RR_rates_alt_sulfur {"RR_rates_alt_sulfur"};
    Kokkos::View<double[17][4]> DR_rates_alt_sulfur {"DR_rates_alt_sulfur"};
    Kokkos::View<double[27][6]> RR_rates_iron {"RR_rates_iron"};
    Kokkos::View<double[27][9]> DR_rates_c_iron {"DR_rates_c_iron"};
    Kokkos::View<double[27][9]> DR_rates_e_iron {"DR_rates_e_iron"};
    Kokkos::View<double[27][2]> RR_rates_alt_iron {"RR_rates_alt_iron"};
    Kokkos::View<double[27][8]> DR_rates_alt_iron {"DR_rates_alt_iron"};

    Kokkos::View<double[6]> dE_carbon {"dE_carbon"};
    Kokkos::View<double[6]> A_carbon {"A_carbon"};
    Kokkos::View<double[6]> X_carbon {"X_carbon"};
    Kokkos::View<double[6]> K_carbon {"K_carbon"};
    Kokkos::View<double[6]> P_carbon {"P_carbon"};

    Kokkos::View<double[7]> dE_nitrogen {"dE_nitrogen"};
    Kokkos::View<double[7]> A_nitrogen {"A_nitrogen"};
    Kokkos::View<double[7]> X_nitrogen {"X_nitrogen"};
    Kokkos::View<double[7]> K_nitrogen {"K_nitrogen"};
    Kokkos::View<double[7]> P_nitrogen {"P_nitrogen"};

    Kokkos::View<double[8]> dE_oxygen {"dE_oxygen"};
    Kokkos::View<double[8]> A_oxygen {"A_oxygen"};
    Kokkos::View<double[8]> X_oxygen {"X_oxygen"};
    Kokkos::View<double[8]> K_oxygen {"K_oxygen"};
    Kokkos::View<double[8]> P_oxygen {"P_oxygen"};

    Kokkos::View<double[10]> dE_neon {"dE_neon"};
    Kokkos::View<double[10]> A_neon {"A_neon"};
    Kokkos::View<double[10]> X_neon {"X_neon"};
    Kokkos::View<double[10]> K_neon {"K_neon"};
    Kokkos::View<double[10]> P_neon {"P_neon"};

    Kokkos::View<double[12]> dE_magnesium {"dE_magnesium"};
    Kokkos::View<double[12]> A_magnesium {"A_magnesium"};
    Kokkos::View<double[12]> X_magnesium {"X_magnesium"};
    Kokkos::View<double[12]> K_magnesium {"K_magnesium"};
    Kokkos::View<double[12]> P_magnesium {"P_magnesium"};

    Kokkos::View<double[14]> dE_silicon {"dE_silicon"};
    Kokkos::View<double[14]> A_silicon {"A_silicon"};
    Kokkos::View<double[14]> X_silicon {"X_silicon"};
    Kokkos::View<double[14]> K_silicon {"K_silicon"};
    Kokkos::View<double[14]> P_silicon {"P_silicon"};

    Kokkos::View<double[16]> dE_sulfur {"dE_sulfur"};
    Kokkos::View<double[16]> A_sulfur {"A_sulfur"};
    Kokkos::View<double[16]> X_sulfur {"X_sulfur"};
    Kokkos::View<double[16]> K_sulfur {"K_sulfur"};
    Kokkos::View<double[16]> P_sulfur {"P_sulfur"};

    Kokkos::View<double[26]> dE_iron {"dE_iron"};
    Kokkos::View<double[26]> A_iron {"A_iron"};
    Kokkos::View<double[26]> X_iron {"X_iron"};
    Kokkos::View<double[26]> K_iron {"K_iron"};
    Kokkos::View<double[26]> P_iron {"P_iron"};

    Kokkos::View<double[MAX_ELEMENTS][MAX_ELEMENTS][MAX_N_GROUPS][3]> cs_ph {"cs_ph"};

    CrossSection verner_cross_sections;

    Kokkos::View<double*> group_E_min;
    Kokkos::View<double*> group_E_max;

} TabulatedData;
