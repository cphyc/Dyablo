// charge_transfer.h
#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace PRISM {


using _CTRecomb_t = std::array<std::array<std::array<double, 31>, 4>, 6>;
using _CTIon_t = std::array<std::array<std::array<double, 31>, 3>, 7>;

inline std::tuple<
    _CTRecomb_t,
    _CTIon_t
> load_ct_rates(const std::string path){
    _CTRecomb_t CTRecomb {};
    _CTIon_t CTIon {};
    // Load the charge transfer ionization
    // and recombination rates from file

    FILE *file;

    // Ionization
    file = fopen((path + "/ct_ionization.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open charge transfer ionization file");

    // Reading data from the file into the 3D array
    for (int i = 3; i < 31; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 7; k++) {
                fscanf(file, "%lf", &CTIon[k][j][i]);
            }
        }
    }
    // Close the file
    fclose(file);


    // Ionization
    file = fopen((path + "/ct_recombination.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open charge transfer ionization file");

    // Reading data from the file into the 3D array
    for (int i = 2; i < 31; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 6; k++) {
                fscanf(file, "%lf", &CTRecomb[k][j][i]);
            }
        }
    }
    // Close the file
    fclose(file);

    return std::make_tuple(
        CTRecomb,
        CTIon
    );
}

KOKKOS_INLINE_FUNCTION
double charge_transfer_recombination(int ion, int nelem, double T, const TabulatedData& tabData){
    // ion is stage of ionization, 2 for the ion going to the atom
    // nelem is atomic number of element, 2 up to 30
    // Example:  O+ + H => O + H+ is HCTRecom(2,8,1e4)
    // Note that temperature is in linear scale

    double ct_recomb = 0.0;

    if (ion == 0)
    {
        return ct_recomb;
    }

    int ipIon = ion - 1; 
    //  use statistical charge transfer for ion > 4
    if( ion > 4 ) 
    {
        ct_recomb = 1.92E-9 * (ipIon+1);
        return ct_recomb;
    }
    
    // Make sure te is between temp. boundaries; set constant outside of range
    double tused = 0.0;
    tused = FMIN(FMAX(T,tabData.CTRecomb(4, ipIon, nelem)),tabData.CTRecomb(5, ipIon, nelem));
    tused *= 1E-4;

    // The interpolation equation
    ct_recomb = tabData.CTRecomb(0, ipIon, nelem) * 1E-9 * pow(tused,tabData.CTRecomb(1, ipIon, nelem)) * (1. + tabData.CTRecomb(2, ipIon, nelem) * exp(tabData.CTRecomb(3, ipIon, nelem)*tused) );

    return ct_recomb;
}

KOKKOS_INLINE_FUNCTION
double charge_transfer_ionization(int ion, int nelem, double T, const TabulatedData& tabData){
    // ion is stage of ionization, 1 for atom
    // nelem is atomic number of element, 2 up to 30
    // Example:  O + H+ => O+ + H is HCTIon(1,8,1e4)
    // Note that temperature is in linear scale

    double ct_ion = 0.0;

    int ipIon = ion;
    if( ipIon > 1 )
    {
        ct_ion = 0.0;
        return ct_ion;
    }

    // ! Make sure te is between temp. boundaries; set constant outside of range
    double tused = 0.0;
    tused = FMIN(FMAX(T, tabData.CTIon(4, ipIon, nelem)), tabData.CTIon(5, ipIon, nelem));
    tused *= 1E-4;
    tused = FMAX(tused,1E-10); //! harley added to prevent zero temperature

    // ! the interpolation equation
    ct_ion = tabData.CTIon(0, ipIon, nelem) * 1E-9 * pow(tused,tabData.CTIon(1, ipIon, nelem)) * (1. + tabData.CTIon(2, ipIon, nelem) * exp(tabData.CTIon(3, ipIon, nelem)*tused) ) * exp(-1.0 * tabData.CTIon(6, ipIon, nelem)/tused);

    return ct_ion;
}

} // namespace PRISM
