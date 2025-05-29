// charge_transfer.h
#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"


double CTRecomb[6][4][31];
double CTIon[7][3][31];

void load_ct_rates(){
    // Load the charge transfer ionization
    // and recombination rates from file

    FILE *file;

    // Ionization
    file = fopen("./data/charge_transfer/ct_ionization.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open charge transfer ionization file\n");
    }

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

    // Zero out helium
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 7; k++) {
            CTIon[k][j][2] = 0.0;
        }
    }

    // Ionization
    file = fopen("./data/charge_transfer/ct_recombination.dat", "r");
    if (file == NULL) {
        printf("Error: Could not open charge transfer ionization file\n");
    }

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

}

KOKKOS_FUNCTION
double charge_transfer_recombination(int ion, int nelem, double T, const TabulatedData& tabData){
    // ion is stage of ionization, 2 for the ion going to the atom
    // nelem is atomic number of element, 2 up to 30
    // Example:  O+ + H => O + H+ is HCTRecom(2,8,1e4)
    // Note that temperature is in linear scale

    double ct_recomb = 0.0;

    // No charge transfer above 10^5 K
    if (T > 1E5)
    {
        return ct_recomb;
    }

    if (ion == 0)
    {
        return ct_recomb;
    }

    // Deal with helium separately
    // He+ + H --> He + H+
    // if (nelem == 2 && ion == 1)
    // {
    //     ct_recomb = 1.20E-15 * pow(T/300.0,0.25);
    //     return ct_recomb;
    // }

    // deal with oxygen separately 
    if (nelem == 8) 
    {            
        if (ion == 1)
        {
            if (T < 10.0)
            {
                ct_recomb = 3.744E-10;
                return ct_recomb;
            }
            double a_op = 2.3344302E-10;
            double b_op = 2.3651505E-10;
            double c_op = -1.3146803E-10;
            double d_op = 2.9979994E-11;
            double e_op = -2.8577012E-12;
            double f_op = 1.1963502E-13;
            double logT = log(T);
            ct_recomb = ((((f_op*logT + e_op)*logT + d_op)*logT + c_op)*logT + b_op)*logT + a_op;
            return ct_recomb; 
        } else if (ion == 2)
        {
            if (T <= 1500.0) 
            {
                ct_recomb = 0.5337E-9 * pow(T/100.0,-0.076);
            } else {
                ct_recomb = 0.4344E-9 + (0.6340E-9 * pow(log10(T/1500.0),2.06));

            } 
            return ct_recomb; 
        }
    }

    // Deal with nitrogen separately
    if (nelem == 7 && ion == 2)
    {
		/* N+2 + H -> N+ + H+ */
		if( T <= 1500. )
		{
			ct_recomb = 0.8692e-9*pow( (T/1500.) ,0.17);
		}
		else if( T <= 20000. )
		{
			ct_recomb = 0.9703e-9*pow( (T/10000.) ,0.058);
		}
		else
		{
			ct_recomb = 1.0101e-9 + 1.4589e-9*pow( log10(T/20000.) ,2.06 );
		}
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
    tused = fmin(fmax(T,tabData.CTRecomb(4, ipIon, nelem)),tabData.CTRecomb(5, ipIon, nelem));
    tused *= 1E-4;

    // The interpolation equation
    ct_recomb = tabData.CTRecomb(0, ipIon, nelem) * 1E-9 * pow(tused,tabData.CTRecomb(1, ipIon, nelem)) * (1. + tabData.CTRecomb(2, ipIon, nelem) * exp(tabData.CTRecomb(3, ipIon, nelem)*tused) );

    return ct_recomb;
}

KOKKOS_FUNCTION
double charge_transfer_ionization(int ion, int nelem, double T, const TabulatedData& tabData){
    // ion is stage of ionization, 1 for atom
    // nelem is atomic number of element, 2 up to 30
    // Example:  O + H+ => O+ + H is HCTIon(1,8,1e4)
    // Note that temperature is in linear scale

    double ct_ion = 0.0;

    // No charge transfer above 10^5 K
    if (T > 1E5)
    {
        ct_ion = 0.0;
        return ct_ion;
    }

    // Deal with helium separately
    // He + H+ --> He+ + H
    if (nelem == 2 && ion == 1)
    {
        // ! This particular rate seems to get very small at low te
        if (T < 3E3)
        {
            ct_ion = 0.0;
            return ct_ion;
        }

        if (T < 1E4)
        { 
            ct_ion = 1.26E-9 * pow(T,-0.75) * exp(-1.275E5/T);
        } else {
            ct_ion = 4E-37 * pow(T,4.74);
        }
        return ct_ion;
    }

    // Deal with oxygen separately
    if (nelem == 8 && ion == 0)
    {
        if (T <= 10.0) 
        {
            ct_ion = 4.749E-20;
        } else if (T > 10.0 && T <= 190.0)
        {
            double a = -21.134531;
            double b = -242.06831;
            double c = 84.761441;
            ct_ion = exp(a + (b/T) + (c/(T*T)));
        } else if (T > 190.0 && T <= 200.0)
        {
            ct_ion = 2.18733E-12*(T-190.0) + 1.85823E-10;
        } else
        { 
            double a_o = -7.6767404E-14;
            double b_o = -3.7282001E-13;
            double c_o = -1.488594E-12;
            double d_o = -3.6606214E-12; 
            double e_o = 2.0699463E-12;
            double f_o = -2.6139493E-13;
            double g_o = 1.1580844E-14;
            double logT = log(T);
            ct_ion = (((((g_o*logT + f_o)*logT + e_o)*logT + d_o)*logT + c_o)*logT + b_o)*logT + a_o;
        }

        return ct_ion;
    }

    // Deal with iron separately
    // Fe + H+ -> Fe+ + H
    if (nelem == 26 && ion == 0)
    {
        ct_ion = 1.e-14;
        return ct_ion;
    }

    // Deal with sulfur separately
    // S + H+ -> S+ + H
    if (nelem == 26 && ion == 0)
    {
        ct_ion = 5.4e-9;
        return ct_ion;
    }

    // Deal with magnesium separately
    // Mg + H+ -> Mg+ + H
    if (nelem == 12 && ion == 0)
    {
        ct_ion = 9.76e-12*pow((T/1e4),3.14)*(1. + 55.54*exp(1.12*T/1e4));
        return ct_ion;
    }

    // Deal with Silicon separately
    // Si + H+ -> Si+ + H
    if (nelem == 14 && ion == 0)
    {
        ct_ion = 0.92e-12*pow((T/1e4),1.15)*(1. + 0.80*exp(0.24*T/1e4));
        return ct_ion;
    }

    int ipIon = ion;
    if( ipIon > 1 )
    {
        ct_ion = 0.0;
        return ct_ion;
    }

    // ! Make sure te is between temp. boundaries; set constant outside of range
    double tused = 0.0;
    tused = fmin(fmax(T, tabData.CTIon(4, ipIon, nelem)), tabData.CTIon(5, ipIon, nelem));
    tused *= 1E-4;
    tused = fmax(tused,1E-10); //! harley added to prevent zero temperature

    // ! the interpolation equation
    ct_ion = tabData.CTIon(0, ipIon, nelem) * 1E-9 * pow(tused,tabData.CTIon(1, ipIon, nelem)) * (1. + tabData.CTIon(2, ipIon, nelem) * exp(tabData.CTIon(3, ipIon, nelem)*tused) ) * exp(-1.0 * tabData.CTIon(6, ipIon, nelem)/tused);

    return ct_ion;
}