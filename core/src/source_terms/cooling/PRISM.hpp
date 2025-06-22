#pragma once

#include "types.hpp"
#include "recombination.hpp"
#include "collisional_ionization.hpp"
#include "cosmic_ray_ionization.hpp"
#include "photoionization_UVB.hpp"
#include "charge_transfer.hpp"
#include "dust_recombination.hpp"
#include "molecules.hpp"
#include "cooling.hpp"
#include "cross_sections.hpp"
#include <Kokkos_Core.hpp>

#define MAX_ELEMENTS 27
namespace PRISM {

const real_t MIN_XION = 1E-20;   // Minimum ion fraction
const real_t X_PCT_RULE = 1E-1;  // Value required for convergence
const real_t X_FM = 1e-7;        // Damping value for X_PCT_RULE
const real_t T_MIN = 2.725;
const real_t T_MAX = 1E9;

// Structs for element properties
template<bool include_H2, bool include_CO>
KOKKOS_INLINE_FUNCTION
void initialize_elements(Element *elements, const std::array<int, MAX_ELEMENTS> &nions)
{
    // Initialize
    for (int i = 0; i < MAX_ELEMENTS; i++)
    {
        elements[i].atomic_number = -1;
        elements[i].n_ions = -1;
        elements[i].atomic_mass = -1.0;
        elements[i].z_solar = -1.0;
        elements[i].G0_photo_rate = 0.0;
        elements[i].n_mol = 0;
        elements[i].depletion = 1.0;
    }

    // Element 1: Hydrogen
    if (nions[1] > 0) {
        elements[1].atomic_number = (nions[1] > 0) ? 1 : -1;
        elements[1].atomic_mass = 1.008;
        elements[1].z_solar = 1.0;
        elements[1].G0_photo_rate = 0.0; // No subionizing PI
        // elements[1].n_ions = elements[1].atomic_number + 1;
        elements[1].n_ions = nions[1];
        elements[1].n_mol = include_H2 ? 1 : 0;
        elements[1].depletion = 1.0;
    }

    // Element 2: Helium
    if (nions[2] > 0) {
        elements[2].atomic_number = (nions[2] > 0) ? 2 : -1;
        elements[2].atomic_mass = 4.0026;
        elements[2].z_solar = 8.51E-02;
        elements[2].G0_photo_rate = 0.0; // No subionizing PI
        // // elements[2].n_ions = elements[2].atomic_number + 1;
        elements[2].n_ions = nions[2];
        elements[2].depletion = 1.0;
    }

    // Element 6: Carbon
    if (nions[6] > 0) {
        elements[6].atomic_number = (nions[6] > 0) ? 6 : -1;
        elements[6].atomic_mass = 12.0107;
        elements[6].z_solar = 2.69E-04;
        elements[6].G0_photo_rate = 3.39E-10;
        // elements[6].n_ions = elements[6].atomic_number + 1;
        elements[6].n_ions = nions[6];
        elements[6].n_mol = include_CO ? 1 : 0; // Include CO if requested
        elements[6].depletion = 0.5;
    }

    // Element 7: Nitrogen
    if (nions[7] > 0) {
        elements[7].atomic_number = (nions[7] > 0) ? 7 : -1;
        elements[7].atomic_mass = 14.0067;
        elements[7].z_solar = 6.76E-05;
        elements[7].G0_photo_rate = 0.0; // No subionizing PI
        // elements[7].n_ions = elements[7].atomic_number + 1;
        elements[7].n_ions = nions[7];
        elements[7].depletion = 0.6;
    }

    // Element 8: Oxygen
    if (nions[8] > 0) {
        elements[8].atomic_number = (nions[8] > 0) ? 8 : -1;
        elements[8].atomic_mass = 15.9994;
        elements[8].z_solar = 4.90E-04;
        elements[8].G0_photo_rate = 0.0; // No subionizing PI
        // elements[8].n_ions = elements[8].atomic_number + 1;
        elements[8].n_ions = nions[8];
        elements[8].depletion = 0.73;
    }

    // Element 10: Neon
    if (nions[10] > 0) {
        elements[10].atomic_number = (nions[10] > 0) ? 10 : -1;
        elements[10].atomic_mass = 20.1797;
        elements[10].z_solar = 8.51E-05;
        elements[10].G0_photo_rate = 0.0; // No subionizing PI
        // elements[10].n_ions = elements[10].atomic_number + 1;
        elements[10].n_ions = nions[10];
        elements[10].depletion = 1.0;
    }

    // Element 12: Magnesium
    if (nions[12] > 0) {
        elements[12].atomic_number = (nions[12] > 0) ? 12 : -1;
        elements[12].atomic_mass = 24.305;
        elements[12].z_solar = 3.98E-05;
        elements[12].G0_photo_rate = 6.59E-11;
        // elements[12].n_ions = elements[12].atomic_number + 1;
        elements[12].n_ions = nions[12];
        elements[12].depletion = 0.16;
    }

    // Element 14: Silicon
    if (nions[14] > 0) {
        elements[14].atomic_number = (nions[14] > 0) ? 14 : -1;
        elements[14].atomic_mass = 28.0855;
        elements[14].z_solar = 3.24E-05;
        elements[14].G0_photo_rate = 4.47E-09;
        // elements[14].n_ions = elements[14].atomic_number + 1;
        elements[14].n_ions = nions[14];
        elements[14].depletion = 0.1;
    }

    // Element 16: Sulfur
    if (nions[16] > 0) {
        elements[16].atomic_number = (nions[16] > 0) ? 16 : -1;
        elements[16].atomic_mass = 32.065;
        elements[16].z_solar = 1.32E-05;
        elements[16].G0_photo_rate = 1.13E-09;
        // elements[16].n_ions = elements[16].atomic_number + 1;
        elements[16].n_ions = nions[16];
        elements[16].depletion = 1.0;
    }

    // Element 26: Iron
    if (nions[26] > 0) {
        elements[26].atomic_number = (nions[26] > 0) ? 26 : -1;
        elements[26].atomic_mass = 55.854;
        elements[26].z_solar = 3.16E-05;
        elements[26].G0_photo_rate = 4.71E-10;
        // elements[26].n_ions = elements[26].atomic_number + 1;
        elements[26].n_ions = nions[26];
        elements[26].depletion = 0.01;
    }
}

typedef struct
{
    real_t n_element;
    real_t ion_fracs[MAX_ELEMENTS];
    real_t ion_fracs_new[MAX_ELEMENTS];
} ParticleIonData;

inline real_t get_dust_mass_and_depletion(
    Element *elements, real_t metallicity)
{
    // Parameters for the RR14 dust-to-gas mass ratio
    real_t a = 2.21;
    real_t aH = 1.00;
    real_t b = 0.96;
    real_t aL = 3.10;
    real_t xt = 8.10;
    real_t xs = 8.69;
    real_t x, y;

    x = FMAX(metallicity, 5.0); //! Mild extrapolation

    if (metallicity > xt) {
        y = a + aH * (xs - x);
    } else {
        y = b + aL * (xs - x);
    }

    real_t G2D = pow(10.0,y);
    real_t G2D_sol = pow(10.0,a);

    real_t y_ratio = FMAX(FMIN(G2D_sol / G2D, 1.0), 0.0);

    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;
        elements[i].depletion = 1.0 - ((1.0 - elements[i].depletion) * y_ratio);
    }

    return y_ratio;
}

// Structs for element properties
KOKKOS_INLINE_FUNCTION
void initialize_ion_fracs(Element *elements, ParticleIonData *n_and_ion_fracs, real_t hdens, real_t metallicity, int itype)
{
    // Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;

        // Only scale metals by metallicity
        if (i < 3){
            n_and_ion_fracs[i].n_element = hdens * elements[i].z_solar * elements[i].depletion;
        } else {
            n_and_ion_fracs[i].n_element = hdens * elements[i].z_solar * metallicity * elements[i].depletion;
        }
        // n_and_ion_fracs[i].ion_fracs = (real_t *)malloc((elements[i].n_ions + elements[i].n_mol) * sizeof(real_t));
        // n_and_ion_fracs[i].ion_fracs_new = (real_t *)malloc((elements[i].n_ions + elements[i].n_mol) * sizeof(real_t));

        switch (itype)
        {
        case 1:
            // fully ionized
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++) {
                n_and_ion_fracs[i].ion_fracs[j] = MIN_XION;
                n_and_ion_fracs[i].ion_fracs_new[j] = MIN_XION;
            }
            n_and_ion_fracs[i].ion_fracs[elements[i].n_ions-1] = 1.0;
            n_and_ion_fracs[i].ion_fracs_new[elements[i].n_ions-1] = 1.0;
            break;
        case 2:
            // fully neutral
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++) {
                n_and_ion_fracs[i].ion_fracs[j] = MIN_XION;
                n_and_ion_fracs[i].ion_fracs_new[j] = MIN_XION;
            }
            n_and_ion_fracs[i].ion_fracs[0] = 1.0;
            n_and_ion_fracs[i].ion_fracs_new[0] = 1.0;
            break;
        case 3:
            // fully molecular or neutral
            for (int j = 0; j < elements[i].n_ions + elements[i].n_mol; j++) {
                n_and_ion_fracs[i].ion_fracs[j] = MIN_XION;
                n_and_ion_fracs[i].ion_fracs_new[j] = MIN_XION;
            }
            if (elements[i].n_mol > 0) {
                n_and_ion_fracs[i].ion_fracs[elements[i].n_ions + elements[i].n_mol - 1] = 1.0;
                n_and_ion_fracs[i].ion_fracs_new[elements[i].n_ions + elements[i].n_mol -1] = 1.0;
            } else {
            n_and_ion_fracs[i].ion_fracs[0] = 1.0;
            n_and_ion_fracs[i].ion_fracs_new[0] = 1.0;
            }
            break;
        case 4:
            // Equally spread
            for (int j = 0; j < elements[i].n_ions; ++j) {
                n_and_ion_fracs[i].ion_fracs[j] = 1.0 / elements[i].n_ions;
                n_and_ion_fracs[i].ion_fracs_new[j] = 1.0 / elements[i].n_ions;
            }
            break;
        default:
            printf("YOU NEED TO SELECT AN APPROPRIATE CASE\n");
            break;
        }
    }
}

template<bool use_new>
KOKKOS_INLINE_FUNCTION
real_t get_ne(const Element *elements, const ParticleIonData *n_and_ion_fracs)
{
    /*
    Returns the electron number density by looping over all elements
    and summing their contributions
    */
    real_t ne = 0E0;

    //Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;

        //Get the number of ions
        int n_ions = elements[i].n_ions;

        //Loop over all ions
        //Start loop at 1, no electrons in the ground state
        for (int j = 1; j < n_ions; j++) {
            if constexpr (use_new){
                ne += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs_new[j] * (real_t)j;
            } else {
                ne += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs[j] * (real_t)j;
            }
        } //End loop ion fracs
    } //End loop elements

    return ne;
}

template<bool include_H2, bool use_new>
KOKKOS_INLINE_FUNCTION
real_t get_mu(const real_t ne, const Element elements[MAX_ELEMENTS], const ParticleIonData n_and_ion_fracs[MAX_ELEMENTS]) {
    /*
    Calculated the mean molecular weight
    */
    real_t m_bar = 0.0;
    real_t n_hat = 0.0;

    // Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;

        // Get the number of ions
        int n_ions = elements[i].n_ions;

        // Loop over all ions
        for (int j = 0; j < n_ions; j++) {
            if constexpr (use_new){
                m_bar += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs_new[j] * elements[i].atomic_mass;
                n_hat += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs_new[j];
            } else {
                m_bar += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs[j] * elements[i].atomic_mass;
                n_hat += n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs[j];
            }
        } // End loop over ions
    } // End loop over elements

    // Include electrons
    n_hat += ne;

    // Include contribution from H2
    if constexpr (include_H2) {
        if constexpr (use_new){
            m_bar += (n_and_ion_fracs[1].n_element * n_and_ion_fracs[1].ion_fracs_new[2] * elements[1].atomic_mass);
            n_hat += (0.5 * n_and_ion_fracs[1].n_element * n_and_ion_fracs[1].ion_fracs_new[2]);
        } else {
            m_bar += (n_and_ion_fracs[1].n_element * n_and_ion_fracs[1].ion_fracs[2] * elements[1].atomic_mass);
            n_hat += (0.5 * n_and_ion_fracs[1].n_element * n_and_ion_fracs[1].ion_fracs[2]);
        }
    }

    real_t mu = m_bar / n_hat;
    return mu;
}

KOKKOS_INLINE_FUNCTION
real_t get_rho(const Element elements[MAX_ELEMENTS], const ParticleIonData n_and_ion_fracs[MAX_ELEMENTS]) {

    real_t rho = 0.0;

    // Loop over all elements
    for (int i = 1; i < MAX_ELEMENTS; i++) {
        if (elements[i].atomic_number < 1) continue;

        rho += n_and_ion_fracs[i].n_element * elements[i].atomic_mass / elements[i].depletion; // amu/cm^3
    }

    return rho;
}

KOKKOS_INLINE_FUNCTION
void reduce_xion(real_t *ion_fracs, int size)
{
    /*
    Makes sure that the ion fractions sum to one and that
    the ion fractions don't go below some minimum value
    */
    real_t total_ion_frac = 0.0;

    // Loop over all ions and get the total
    for (int i = 0; i < size; i++) {
        total_ion_frac += ion_fracs[i];
    }
    // Loop again to normalize
    for (int i = 0; i < size; i++) {
        ion_fracs[i] *= (1.0 / total_ion_frac);
    }
}


// void copyArray(real_t src[], real_t dest[], int size)
// {
//     for (int i = 0; i < size; i++)
//     {
//         dest[i] = src[i];
//     }
// }

template <
    bool constant_temperature,
    bool ramses_rt_T_scheme,
    bool rosenbrock_T_scheme,
    bool include_H2,
    bool include_CO,
    bool rt_advect
>
KOKKOS_INLINE_FUNCTION
std::tuple<int, real_t> subcycle_chemistry(
        const Element elements[MAX_ELEMENTS],
        ParticleIonData n_and_ion_fracs[MAX_ELEMENTS],
        real_t T2,
        const real_t aexp,
        const real_t total_dt,
        const real_t UV_background_G0,
        const real_t primary_cosmic_ray_ionization_rate,
        const real_t dust_to_gas_mass_ratio_over_mw,
        real_t N_phot[MAX_N_GROUPS],
        real_t F_phot[MAX_N_GROUPS][3],
        const int N_groups,
        const int conv_its,
        const int max_its,
        const TabulatedData& tabData
) {
    static_assert(constant_temperature || (ramses_rt_T_scheme ^ rosenbrock_T_scheme),
                  "Exactly one temperature scheme must be selected");
    /*
    Computes chemical equilibrium

    if total_dt [s] is <=0 --> run to equilibrium otherwise stop at a fixed time
    */

    //Parameters
    real_t ddt = 10000.0 * 365.25 * 60. * 60.;
    // If we are not running to convergence, try to take one single step
    if (total_dt > 0) {
        ddt = total_dt;
    }

    // Convergence variables
    int success_counter      = 0;
    int total_iterations     = 0;
    bool model_converged     = false;
    bool x_percent_rule      = true;
    bool compute_atoms       = true;
    bool compute_molecules   = true;
    bool compute_temperature = true;
    real_t total_time        = 0.E0;
    real_t residual;
    real_t ne_residual;
    real_t max_residual_all;

    //Physical variables
    real_t mu; //Mean molecular weight
    real_t ne; //Electron number density
    real_t ne_initial; //Electron number density for convergence
    real_t cr; //Creation rate
    real_t de; //Destruction rate
    real_t xe; //Electron fraction (n_electron/nH)
    real_t phi_s; // Secondary electron CR factor
    real_t Tmu_old = T2; // T/mu
    real_t Tmu_new = T2; // T/mu
    real_t N_phot_new[MAX_N_GROUPS]; // Holds the new photon states
    real_t F_phot_new[MAX_N_GROUPS][3];

    int n_ions; //Number of ions --> reset per element

    // Self-shielding
    real_t ss_factor = exp(-1.0 * n_and_ion_fracs[0].n_element/1E-2);

    // Cooling
    real_t cooling_rate = 0.0;
    real_t cooling_rate_prime = 0.0;

    // Compute effective rho
    real_t loc_rho = get_rho(elements, n_and_ion_fracs);

    //# infinite loop
    while (1) {  // Infinite loop: TODO(code): put a max loop counter here and then call an error function
        total_iterations += 1;
        compute_atoms = true;
        compute_molecules = true;

        //Loop over elements and set the ion_fracs new to
        //the old ion fracs
        for (int i = 1; i < MAX_ELEMENTS; i++) {
            if (elements[i].atomic_number < 1) continue;

            int size = elements[i].n_ions + elements[i].n_mol;
            for (int j = 0; j < size; j++) {
                n_and_ion_fracs[i].ion_fracs_new[j] = n_and_ion_fracs[i].ion_fracs[j];
            }
        } // End element loop

        // First get the electron number density
        ne = get_ne<false>(elements, n_and_ion_fracs);
        // Update the electron fraction
        xe = ne / n_and_ion_fracs[1].n_element;
        // Measure mu
        mu = get_mu<include_H2, true>(ne, elements, n_and_ion_fracs);
        // Get TK
        real_t TK = Tmu_old * mu;
        if constexpr (constant_temperature) {
            TK = Tmu_old;
        }

        // Measure phi_s for secondary CR ionization
        phi_s = secondary_cr_rates(xe);
        // Now calculate the total CR ionization rate
        real_t total_cosmic_ray_ionization_rate = primary_cosmic_ray_ionization_rate * (1.0 + phi_s);
        real_t H2_cosmic_ray_ionization_rate = 2.0 * primary_cosmic_ray_ionization_rate * (1.0 + phi_s);

        // Set the cosmic ray scale factor -> only needed for secondary UVs
        real_t cosmic_ray_scale_factor = H2_cosmic_ray_ionization_rate / 1E-16;

        // Initialize the X percent rule
        x_percent_rule = true;

        // Convergence
        max_residual_all = -999.;

        // Make sure the timestep isn't larger than how long we
        // need to evolve for, otherwise shorten it
        if (total_dt > 0.0) {
            if ((total_time+ddt) > total_dt){
                ddt = total_dt - total_time;
            }
        }

        /////////////////////////
        //      Radiation      //
        /////////////////////////
        if constexpr (rt_advect) {
            real_t phAbs = 0.0;

            // First get the opacity
            for (int k = 0; k < N_groups; k++) { // Loop over photon groups

                phAbs = 0.0; // zero out absorption

                for (int i = 1; i < MAX_ELEMENTS; i++) { // Loop over all elements
                    if (elements[i].atomic_number < 1) continue;

                    // Get the number of ions
                    n_ions = elements[i].n_ions;

                    for (int j = 0; j < n_ions - 1; j++) { // Loop over the number of ions --> no absorption by final ion
                        phAbs += (n_and_ion_fracs[i].n_element * n_and_ion_fracs[i].ion_fracs_new[j] * tabData.cs_ph(i, j, k, 0));
                    } // End loop over ions
                } // End loop over elements

                // Deal with molecular hydrogen separately if needed
                if (include_H2) {
                    // TODO(code): for better accuracy need to include self-shielding here
                    phAbs += (0.5 * n_and_ion_fracs[1].n_element * n_and_ion_fracs[1].ion_fracs_new[1] * tabData.cs_ph(1, 2, k, 0));
                }

                // Next deplete the photons from the group
                N_phot_new[k] = N_phot[k] / (1.0 + ddt * phAbs);
                F_phot_new[k][0] = F_phot[k][0] / (1.0 + ddt * phAbs);
                F_phot_new[k][1] = F_phot[k][1] / (1.0 + ddt * phAbs);
                F_phot_new[k][2] = F_phot[k][2] / (1.0 + ddt * phAbs);

                // Measure the residual
                residual = FABS((N_phot_new[k] - N_phot[k]) / (N_phot[k] + 1E-13));
                max_residual_all = FMAX(max_residual_all, residual);

                if (residual > X_PCT_RULE){
                    // printf("Broken photons %e %e %e %e %e %d %e %e %e %e %e\n",Tmu_new,Tmu_old,residual,mu,ddt,total_iterations,element_ion_fractions[1][0],element_ion_fractions[1][1],element_ion_fractions[1][2],cooling_rate,cooling_rate_prime);
                    // Reset the success counter
                    success_counter = 0;
                    // Declare the x-percent rule was violated
                    x_percent_rule = false;
                    // Don't compute temperature
                    compute_temperature = false;
                    // Dont compute molecules
                    compute_molecules = false;
                    // Don't compute atoms
                    compute_atoms = false;
                }
            } // End loop over photon groups
        }

        /////////////////////////
        //     Temperature     //
        /////////////////////////
        if constexpr (!constant_temperature) if (compute_temperature){
            Array1D element_number_densities {};
            iArray1D element_number_ions {};
            Array2D element_ion_fractions {};

            // Loop over all elements
            for (int i = 1; i < MAX_ELEMENTS; i++) {
                element_number_densities[i] = n_and_ion_fracs[i].n_element;
                element_number_ions[i] = elements[i].n_ions + elements[i].n_mol;
                for (int j = 0; j < element_number_ions[i]; j++) {
                    element_ion_fractions[i][j] = n_and_ion_fracs[i].ion_fracs_new[j];
                }
            }

            cooling_rate = all_cooling<include_H2, rt_advect>(
                                    TK, ne, aexp,
                                    element_number_densities,
                                    element_number_ions,
                                    element_ion_fractions,
                                    UV_background_G0,
                                    dust_to_gas_mass_ratio_over_mw,
                                    xe,
                                    primary_cosmic_ray_ionization_rate,
                                    H2_cosmic_ray_ionization_rate,
                                    ss_factor,
                                    N_phot, 
                                    N_groups,
                                    tabData);

            cooling_rate_prime = all_cooling<include_H2, rt_advect>(
                                    1.001 * TK, ne, aexp,
                                    element_number_densities,
                                    element_number_ions,
                                    element_ion_fractions,
                                    UV_background_G0,
                                    dust_to_gas_mass_ratio_over_mw,
                                    xe,
                                    primary_cosmic_ray_ionization_rate,
                                    H2_cosmic_ray_ionization_rate,
                                    ss_factor,
                                    N_phot,
                                    N_groups,
                                    tabData);

            cooling_rate_prime = (cooling_rate_prime - cooling_rate) / ((1.001*TK) - TK);

            real_t X_nHkb = 1.0 / (1.5 * loc_rho * 1.380649e-16);

            cooling_rate *= X_nHkb;
            cooling_rate_prime *= (-X_nHkb) * mu;

            real_t dUU = FABS(FMAX(T_MIN, Tmu_old + cooling_rate * ddt) - Tmu_old);
            Tmu_new = FMAX(T_MIN,Tmu_old + cooling_rate * ddt/(1.0 - cooling_rate_prime * ddt));

            // ! New T2 value
            TK = Tmu_new * mu;

            residual = FMAX(dUU, FABS(Tmu_new-Tmu_old)) / (Tmu_old+T_MIN);

            // Convergence
            // X% rule

            // Max residual over all elements
            max_residual_all = FMAX(max_residual_all,residual);

            if (residual > X_PCT_RULE){
                // printf("Broken temperature %e %e %e %e %e %d %e %e %e %e %e\n",Tmu_new,Tmu_old,residual,mu,ddt,total_iterations,element_ion_fractions[1][0],element_ion_fractions[1][1],element_ion_fractions[1][2],cooling_rate,cooling_rate_prime);
                // Reset the success counter
                success_counter = 0;
                // Declare the x-percent rule was violated
                x_percent_rule = false;
                // Dont compute molecules
                compute_molecules = false;
                // Don't compute atoms
                compute_atoms = false;
            }
        }

        /////////////////////////
        //      Molecules      //
        /////////////////////////
        real_t alpha_H2_loc = 0.0;
        real_t beta_H2_loc = 0.0;
        real_t cr_H2 = 0.E0;
        real_t de_H2 = 0.E0;
        if (compute_molecules) {

            // Molecular Hydrogen
            if constexpr (include_H2) {

                // H2 fraction --> Note that we actually store 2 * xH2
                real_t xH2 = n_and_ion_fracs[1].ion_fracs_new[2]/2.0;

                // Creation //

                // Contains formation on dust and via the primordial channel
                // TODO(code): update G0 to be the total G0
                alpha_H2_loc = alpha_H2(TK, dust_to_gas_mass_ratio_over_mw, xe, H2_cosmic_ray_ionization_rate,
                                        UV_background_G0, n_and_ion_fracs[1].ion_fracs_new[0], n_and_ion_fracs[1].ion_fracs_new[1],
                                        n_and_ion_fracs[1].n_element); //! [cm3 s-1]

                // We can only create H2 if there is any HI and HII
                cr_H2 += alpha_H2_loc;

                // Destruction //

                // Collisional destruction
                beta_H2_loc = beta_H2_krome(TK,
                                        n_and_ion_fracs[1].ion_fracs_new[0] * n_and_ion_fracs[1].n_element,
                                        ne,
                                        xH2 * n_and_ion_fracs[1].n_element,
                                        n_and_ion_fracs[2].ion_fracs_new[0] * n_and_ion_fracs[2].n_element);
                de_H2 += beta_H2_loc;

                // Photodissociation
                de_H2 += UV_background_G0 * 5.68E-11;

                // Cosmic ray destruction
                de_H2 += H2_cosmic_ray_ionization_rate;

                // Update xH2
                xH2 = (cr_H2*ddt + xH2)/(1.+de_H2*ddt);

                // Store in the struct
                n_and_ion_fracs[1].ion_fracs_new[2] = 2.0 * FMIN(FMAX(xH2,MIN_XION),0.5);

                // Convergence
                // X% rule
                residual = FABS(n_and_ion_fracs[1].ion_fracs_new[2] - n_and_ion_fracs[1].ion_fracs[2]) / (n_and_ion_fracs[1].ion_fracs[2] + X_FM);

                // Max residual over all elements
                max_residual_all = FMAX(max_residual_all,residual);

                if (residual > X_PCT_RULE){
                    // printf("Broken molecules %e %e %e %e\n",n_and_ion_fracs[1].ion_fracs_new[2],n_and_ion_fracs[1].ion_fracs[2],residual,ddt);
                    // Reset the success counter
                    success_counter = 0;
                    // Declare the x-percent rule was violated
                    x_percent_rule = false;
                    // Break the loop over ions
                    compute_atoms = false;
                }
            }
            // CO
            // TODO(code)

            //  Update mu and T
            mu = get_mu<include_H2, true>(ne, elements, n_and_ion_fracs);
            TK = Tmu_new * mu;
        }

        /////////////////////////
        //        Atoms        //
        /////////////////////////
        // Update the HI number density in case it has changed e.g. due to charge exchange
        real_t dust_effective_number_density = n_and_ion_fracs[1].n_element * dust_to_gas_mass_ratio_over_mw;

        if (compute_atoms) {
            // Loop over all elements
            for (int i = 1; i < MAX_ELEMENTS; i++) {
                if (elements[i].atomic_number < 1) continue;

                // Get the number of ions
                n_ions = elements[i].n_ions;

                // Loop over the number of ions
                for (int j = 0; j < n_ions; j++) {

                    /////////////////////////
                    //       Creation      //
                    /////////////////////////
                    cr = 0.E0;

                    // Account for molecular hydrogen
                    if constexpr (include_H2) if ((i == 1) & (j == 0)) { // select only HI
                        // Note: no factor of 2 needed since is 2*xH2
                        cr += de_H2 * n_and_ion_fracs[1].ion_fracs_new[2];
                    }

                    // Recombinations of the more excited ionization state
                    if (j < (n_ions - 1)) cr += recombination(TK, j+1, i, tabData) * ne * n_and_ion_fracs[i].ion_fracs_new[j+1];

                    // Collisional ionization of the less excited state
                    if (j > 0) cr += collisional_ionization(TK, j-1, i, tabData) * ne * n_and_ion_fracs[i].ion_fracs_new[j-1];

                    // Photoionization of the less excited state
                    if (j > 0) cr += tabData.HM12_UVB_z(i, j-1, 0) * ss_factor * n_and_ion_fracs[i].ion_fracs_new[j-1];

                    // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
                    if (j == 1) cr += UV_background_G0 * elements[i].G0_photo_rate * n_and_ion_fracs[i].ion_fracs_new[j-1];

                    // Cosmic ray ionization of the less excited state
                    if (j > 0) cr += tabData.cosmic_ray_ionization_rates(i, j-1) * total_cosmic_ray_ionization_rate * n_and_ion_fracs[i].ion_fracs_new[j-1];

                    // Cosmic ray ionization of the less excited state from induced UV
                    if (j == 1) cr += tabData.cosmic_ray_ionization_rates_induced_UV(i) * cosmic_ray_scale_factor * n_and_ion_fracs[i].ion_fracs_new[j-1];

                    // Recombination on dust for the more excited state
                    if (j < (n_ions - 1)) cr += dust_recombination_rates(j+1, i, TK, UV_background_G0, ne, tabData.dust_rec_coefs) * dust_effective_number_density * n_and_ion_fracs[i].ion_fracs_new[j+1];

                    // Photoionization of the less excited state from local radiation field
                    if (rt_advect) {
                        if (j > 0) {
                            for (int k = 0; k < N_groups; k++) { // Loop over photon groups
                                cr += N_phot_new[k] * tabData.cs_ph(i, j-1, k, 0) * n_and_ion_fracs[i].ion_fracs_new[j-1];
                            } // End loop over photon groups
                        }
                    }

                    /////////////////////////
                    //     Destruction     //
                    /////////////////////////
                    de = 0.E0;

                    // Account for molecular hydrogen
                    if ((i == 1) & (j == 0) & include_H2) { // select only HI
                        de += alpha_H2_loc;
                    }

                    // Collisional ionization
                    if (j < (n_ions - 1)) de += collisional_ionization(TK, j, i, tabData) * ne;

                    // Photoionization
                    if (j < (n_ions - 1)) de += tabData.HM12_UVB_z(i, j, 0) * ss_factor;

                    // Photoionization by sub-ionizing ISRF --> only impacts lowest ionization states
                    if (j == 0) de += UV_background_G0 * elements[i].G0_photo_rate;

                    // Recombination
                    if (j > 0) de += recombination(TK, j, i, tabData) * ne;

                    // Cosmic ray ionization
                    if (j < (n_ions - 1)) de += tabData.cosmic_ray_ionization_rates(i, j) * total_cosmic_ray_ionization_rate;

                    // Cosmic ray ionization from induced UV
                    if (j == 0) de += tabData.cosmic_ray_ionization_rates_induced_UV(i) * cosmic_ray_scale_factor;

                    // Recombination on dust
                    if (j > 0) de += dust_recombination_rates(j, i, TK, UV_background_G0, ne, tabData.dust_rec_coefs) * dust_effective_number_density;

                    // Photoionization from local radiation field
                    if (rt_advect) {
                        if (j < (n_ions - 1)) {
                            for (int k = 0; k < N_groups; k++) { // Loop over photon groups
                                de += N_phot_new[k] * tabData.cs_ph(i, j, k, 0);
                            } // End loop over photon groups
                        }
                    }

                    /////////////////////////
                    //   Charge Transfer   //
                    /////////////////////////
                    // Note, this was split off due to cross species
                    // coupling. Saves us an extra double loop
                    // Charge exchange
                    if (i == 1) {  // If element is hydrogen
                        // Loop over all other elements
                        for (int ii = 2; ii < MAX_ELEMENTS; ii++){
                            if (elements[ii].atomic_number < 1) continue;

                            // Loop over all other ionization states
                            for (int jj = 0; jj < elements[ii].n_ions; jj++) {
                                real_t paired_ion_number_density = n_and_ion_fracs[ii].ion_fracs_new[jj] * n_and_ion_fracs[ii].n_element;
                                if (j == 0) { // H
                                    cr += charge_transfer_ionization(jj,ii,TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j+1] * paired_ion_number_density; //! Example:  O + H+ => O+ + H
                                    de += charge_transfer_recombination(jj,ii,TK, tabData) * paired_ion_number_density; //! Example:  O+ + H => O + H+
                                } else { // H+
                                    de += charge_transfer_ionization(jj,ii,TK, tabData) * paired_ion_number_density; //! Example:  O + H+ => O+ + H
                                    cr += charge_transfer_recombination(jj,ii,TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j-1] * paired_ion_number_density; //! Example:  O+ + H => O + H+
                                }
                            } // end loop over ionization states
                        } // end loop over other elements
                    } else { // All other elements
                        real_t HI_number_density = n_and_ion_fracs[1].ion_fracs_new[0] * n_and_ion_fracs[1].n_element;
                        real_t HII_number_density = n_and_ion_fracs[1].ion_fracs_new[1] * n_and_ion_fracs[1].n_element;

                        if (j > 0) {
                            // Ionization from less excited state
                            cr += charge_transfer_ionization(j-1,i,TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j-1] * HII_number_density; //! Example:  O + H+ => O+ + H

                            // Charge exchange recombination
                            de += charge_transfer_recombination(j,i,TK, tabData) * HI_number_density; //! Example:  O+ + H => O + H+
                        }

                        if (j < (n_ions - 1)) {
                            // Charge exchange ionization
                            de += charge_transfer_ionization(j,i,TK, tabData) * HII_number_density; //! Example:  O + H+ => O+ + H

                            // Charge exchange recombination from the more excited state
                            cr += charge_transfer_recombination(j+1,i,TK, tabData) * n_and_ion_fracs[i].ion_fracs_new[j+1] * HI_number_density; //! Example:  O+ + H => O + H+
                        }
                    }

                    /////////////////////////
                    //       Update        //
                    /////////////////////////
                    // The update
                    n_and_ion_fracs[i].ion_fracs_new[j] = (cr*ddt + n_and_ion_fracs[i].ion_fracs_new[j])/(1.0 + de*ddt);
                    n_and_ion_fracs[i].ion_fracs_new[j] = FMIN(FMAX(n_and_ion_fracs[i].ion_fracs_new[j],MIN_XION),1.0);

                    // Get the new electron fraction
                    ne_initial = ne;
                    ne = get_ne<true>(elements,n_and_ion_fracs);
                    xe = ne / n_and_ion_fracs[1].n_element;
                    phi_s = secondary_cr_rates(xe);
                    total_cosmic_ray_ionization_rate = primary_cosmic_ray_ionization_rate * (1.0 + phi_s);

                    // Update mu and T --> only H and He (others don't matter)
                    if (i < 3) {
                        //  Update mu and T
                        mu = get_mu<include_H2, true>(ne, elements, n_and_ion_fracs);
                        TK = Tmu_new * mu;
                    }

                    /////////////////////////
                    //     Convergence     //
                    /////////////////////////
                    // X% rule
                    residual = FABS(n_and_ion_fracs[i].ion_fracs_new[j] - n_and_ion_fracs[i].ion_fracs[j]) / (n_and_ion_fracs[i].ion_fracs[j] + X_FM);

                    // electron density residual
                    ne_residual = FABS(ne - ne_initial) / (ne + X_FM);
                    residual = FMAX(residual,ne_residual);

                    // Max residual over all elements
                    max_residual_all = FMAX(max_residual_all,residual);

                    // Check if X percent rule is broken
                    if (residual > X_PCT_RULE){
                        // printf("Broken element %d %d, %e %e %e %e\n",i,j,n_and_ion_fracs[i].ion_fracs_new[j],n_and_ion_fracs[i].ion_fracs[j],residual,ddt);
                        // Reset the success counter
                        success_counter = 0;
                        // Declare the x-percent rule was violated
                        x_percent_rule = false;
                        // Break the loop over ions
                        break;
                    }
                } // End ion loop

                // Make sure ions sum to 1
                int size = elements[i].n_ions + elements[i].n_mol;
                reduce_xion(n_and_ion_fracs[i].ion_fracs_new, size);

                //If the X% rule is violated restart with shorter timestep
                if (!x_percent_rule){
                    // Reset the success counter
                    success_counter = 0;
                    // Break out of the element loop
                    break;
                }

            } // End element loop
        } // End compute atoms

        /////////////////////////
        //   More Convergence  //
        /////////////////////////

        // Set new dt
        max_residual_all /= (X_PCT_RULE);
        if (!x_percent_rule) {
            // If we broke the 10% rule, divide timestep by 2.0
            ddt /= 2;
        } else {

            // Update the total time
            total_time += ddt;

            // TODO(code): allow for choice of aggressive vs conservative timestep
            ddt = 0.9 * ddt / pow(0.07+max_residual_all,0.3);
            // ddt = 0.50 * ddt / pow(0.01 + max_residual_all,0.5);
        }
        // TODO(code): Make this a parameter for the timestep limiter
        // ddt = FMIN(ddt,1E11);
        ddt = FMIN(ddt,1E12 * FMIN(TK/100.0,1.0) * FMIN((1.0/n_and_ion_fracs[1].n_element),1.0) * FMIN(sqrt(1.0/UV_background_G0),1.0));

        // If all elements completed
        // Loop over elements and ions and
        // set old values to new values
        if (x_percent_rule) {
            if (!constant_temperature) {
                // Update temperature
                Tmu_old = FMIN(FMAX(Tmu_new,T_MIN),T_MAX);
            }
            // Update atomic data
            for (int i = 1; i < MAX_ELEMENTS; i++) {
                if (elements[i].atomic_number < 1) continue;

                int size = elements[i].n_ions + elements[i].n_mol;
                for (int j = 0; j < size; j++) {
                    n_and_ion_fracs[i].ion_fracs[j] = n_and_ion_fracs[i].ion_fracs_new[j];
                } // End ion loop
            } // End element loop

            // Update the RT quantities if needed
            if (rt_advect) {
                for (int k = 0; k < N_groups; k++) { // Loop over photon groups
                    // Update the photon properties
                    N_phot[k] = FMAX(N_phot_new[k],1E-20);
                    F_phot[k][0] = F_phot_new[k][0];
                    F_phot[k][1] = F_phot_new[k][1];
                    F_phot[k][2] = F_phot_new[k][2];
                    // Reduce the flux if needed
                    real_t fred = SQRT(SQR(F_phot[k][0]) + SQR(F_phot[k][1]) + SQR(F_phot[k][2])) / (C_CGS * N_phot[k]);
                    if (fred > 1.0) {
                        F_phot[k][0] /= fred;
                        F_phot[k][1] /= fred;
                        F_phot[k][2] /= fred;
                    }
                } // End loop over photon groups
            }

            // Update the success counter
            success_counter += 1;

            // Check if we are converged
            if (success_counter > conv_its) {
                model_converged = true;
            }

            // Break the infinite loop if the timestep has been reached
            if (total_dt > 0.0) {
                if (FABS(total_time-total_dt)/total_dt < 1e-6){
                    model_converged = true;
                }
            }
        }

        // Check if we are converged
        if (total_iterations > max_its) {
            printf("Too many iterations %d\n",total_iterations);
            model_converged = true;
        }

        // Finish the calculation if model converged
        if (model_converged) break; //break from infinite loop

    } // End infinite loop

    return std::make_tuple(total_iterations, Tmu_old);
}

inline void parseIonInputs(
    const std::vector<std::string>& ions,
    std::array<int, MAX_ELEMENTS>& nions,
    std::array<int, MAX_ELEMENTS>& elems2passive,
    std::array<int, MAX_ELEMENTS>& ions2passive,
    std::map<std::string, int>& ion_counts
) {
    for (auto ion : ions) {
        // Trim whitespace
        ion.erase(ion.find_last_not_of(" \n\r\t") + 1);
        ion.erase(0, ion.find_first_not_of(" \n\r\t"));

        std::string element_name;
        if (ion == "H2") {
          // Special case for H2, we don't want to count it as an ion
          element_name = "H";
        } else if (ion == "CO") {
            // Special case for CO, we don't want to count it as an ion
            element_name = "C"; // TODO: Add to O?
        } else {
            element_name = ion.substr(0, ion.find_first_of("_"));
        }

        // Insert if missing
        if (ion_counts.find(element_name) == ion_counts.end()) {
          // If the element is not already in the map, initialize to 0
          ion_counts[element_name] = 0;
        }

        // Increment
        ion_counts[element_name]++;
    }

    // DYABLO_ASSERT_HOST_RELEASE(ion_counts.size() + ions.size() <= n_passive_scalars, "More ions than passive scalars");

    // Create mask of elements
    int iions = ion_counts.size(), ielems = 0;
    auto check_set = [&](
        const std::string& elem_name,
        std::array<int, MAX_ELEMENTS>& nions,
        std::array<int, MAX_ELEMENTS>& elems2passive,
        std::array<int, MAX_ELEMENTS>& ions2passive,
        const int index
    ) {
        if (ion_counts.find(elem_name) == ion_counts.end()) return;

        int nions_this_element = ion_counts.at(elem_name);
        nions[index] = nions_this_element;
        ions2passive[index] = iions;
        elems2passive[index] = ielems;
        iions += nions_this_element;
        ielems++;

        // std::cout << "Element: " << elem_name
        //           << ", Index: " << index
        //           << ", nions: " << nions_this_element
        //           << ", ions2passive: " << ions2passive[index]
        //           << ", elems2passive: " << elems2passive[index]
        //           << std::endl;
    };

    for (auto i = 0; i < MAX_ELEMENTS; ++i) {
        nions[i] = 0;
        ions2passive[i] = -1;
    }

    check_set( "H", nions, elems2passive, ions2passive,  1);
    check_set("He", nions, elems2passive, ions2passive,  2);
    check_set( "C", nions, elems2passive, ions2passive,  6);
    check_set( "N", nions, elems2passive, ions2passive,  7);
    check_set( "O", nions, elems2passive, ions2passive,  8);
    check_set("Ne", nions, elems2passive, ions2passive, 10);
    check_set("Mg", nions, elems2passive, ions2passive, 12);
    check_set("Si", nions, elems2passive, ions2passive, 14);
    check_set( "S", nions, elems2passive, ions2passive, 16);
    check_set("Fe", nions, elems2passive, ions2passive, 26);

}
}