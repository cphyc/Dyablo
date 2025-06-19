// photoionization_UVB.h
#pragma once

const int N_UVB_POINTS = 60;

namespace PRISM {

// Array that holds the UVB for each ion at a given redshift
// double HM12_UVB_z[27][27][2];

struct UVB_data {
    // Initialize the UVB arrays
    double redshifts[N_UVB_POINTS];
    double hydrogen[N_UVB_POINTS][1][2];
    double helium[N_UVB_POINTS][2][2];
    double carbon[N_UVB_POINTS][6][2];
    double nitrogen[N_UVB_POINTS][7][2];
    double oxygen[N_UVB_POINTS][8][2];
    double neon[N_UVB_POINTS][10][2];
    double magnesium[N_UVB_POINTS][12][2];
    double silicon[N_UVB_POINTS][14][2];
    double sulfur[N_UVB_POINTS][16][2];
    double iron[N_UVB_POINTS][26][2];
} ;

inline struct UVB_data load_UVB_data(const std::string path) {
    struct UVB_data HM12_UVB_data;
    // Load in all of the UV background tables
    //TODO(code): can probably abstract this into a function...
    FILE *file;

    printf("Initializing UV background data\n");

    // Redshifts
    file = fopen((path + "/redshifts.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Redshifts UVB file\n");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        fscanf(file, "%lf", &HM12_UVB_data.redshifts[i]);
    }
    // Close the file
    fclose(file);

    // Hydrogen
    file = fopen((path + "/hydrogen_pi.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Hydrogen UVB file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 1; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.hydrogen[i][j][0]);
        }
    }
    // Close the file
    fclose(file);

    file = fopen((path + "/hydrogen_ph.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Hydrogen UVB heat file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 1; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.hydrogen[i][j][1]);
        }
    }
    // Close the file
    fclose(file);

    // Helium
    file = fopen((path + "/helium_pi.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Helium UVB file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 2; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.helium[i][j][0]);
        }
    }
    // Close the file
    fclose(file);

    file = fopen((path + "/helium_ph.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Helium UVB heat file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 2; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.helium[i][j][1]);
        }
    }
    // Close the file
    fclose(file);

    // Carbon
    file = fopen((path + "/carbon_pi.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Carbon UVB file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 6; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.carbon[i][j][0]);
        }
    }
    // Close the file
    fclose(file);

    file = fopen((path + "/carbon_ph.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Carbon UVB heat file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 6; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.carbon[i][j][1]);
        }
    }
    // Close the file
    fclose(file);

    // Nitrogen
    file = fopen((path + "/nitrogen_pi.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Nitrogen UVB file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 7; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.nitrogen[i][j][0]);
        }
    }
    // Close the file
    fclose(file);

    file = fopen((path + "/nitrogen_ph.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Nitrogen UVB heat file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 7; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.nitrogen[i][j][1]);
        }
    }
    // Close the file
    fclose(file);

    // Oxygen
    file = fopen((path + "/oxygen_pi.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Oxygen UVB file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 8; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.oxygen[i][j][0]);
        }
    }
    // Close the file
    fclose(file);

    file = fopen((path + "/oxygen_ph.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Oxygen UVB heat file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 8; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.oxygen[i][j][1]);
        }
    }
    // Close the file
    fclose(file);

    // Neon
    file = fopen((path + "/neon_pi.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Neon UVB file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 10; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.neon[i][j][0]);
        }
    }
    // Close the file
    fclose(file);

    file = fopen((path + "/neon_ph.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Neon UVB heat file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 10; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.neon[i][j][1]);
        }
    }
    // Close the file
    fclose(file);

    // Magnesium
    file = fopen((path + "/magnesium_pi.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Magnesium UVB file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 12; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.magnesium[i][j][0]);
        }
    }
    // Close the file
    fclose(file);

    file = fopen((path + "/magnesium_ph.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Magnesium UVB heat file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 12; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.magnesium[i][j][1]);
        }
    }
    // Close the file
    fclose(file);

    // Silicon
    file = fopen((path + "/silicon_pi.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Silicon UVB file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 14; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.silicon[i][j][0]);
        }
    }
    // Close the file
    fclose(file);

    file = fopen((path + "/silicon_ph.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Silicon UVB heat file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 14; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.silicon[i][j][1]);
        }
    }
    // Close the file
    fclose(file);

    // Sulfur
    file = fopen((path + "/sulfur_pi.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Sulfur UVB file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 16; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.sulfur[i][j][0]);
        }
    }
    // Close the file
    fclose(file);

    file = fopen((path + "/sulfur_ph.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Sulfur UVB heat file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 16; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.sulfur[i][j][1]);
        }
    }
    // Close the file
    fclose(file);

    // Iron
    file = fopen((path + "/iron_pi.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Iron UVB file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 26; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.iron[i][j][0]);
        }
    }
    // Close the file
    fclose(file);

    file = fopen((path + "/iron_ph.dat").c_str(), "r");
    DYABLO_ASSERT_HOST_RELEASE(file != NULL, "Error: Could not open Iron UVB heat file");

    // Reading data from the file into the 2D array
    for (int i = 0; i < N_UVB_POINTS; i++) {
        for (int j = 0; j < 26; j++) {
            fscanf(file, "%lf", &HM12_UVB_data.iron[i][j][1]);
        }
    }
    // Close the file
    fclose(file);

    return HM12_UVB_data;
}


using UVB_table_t = std::array<std::array<std::array<double, 2>, 27>, 27>;
inline UVB_table_t update_UVB(const double redshift, const struct UVB_data& HM12_UVB_data) {
    /* 
    interpolates the UVB for all ions
    to a particular redshift
    */
    UVB_table_t HM12_UVB_z;

    double scale_low;
    double scale_high;

    // No interpolation beyond redshift bounds --> set the UVB to zero
    if (redshift < HM12_UVB_data.redshifts[0] || redshift >= HM12_UVB_data.redshifts[N_UVB_POINTS-1]){
        for (int i = 0; i < 27; i++) {
            for (int j = 0; j < 27; j++) {
                for (int k = 0; k < 2; k++) {
                    HM12_UVB_z[i][j][k]  = 0.0;
                }
            }
        }
    } else {
        // Loop through the array to find the interval in which x_interp lies
        for (int i = 0; i < N_UVB_POINTS; i++) {
            if (redshift >= HM12_UVB_data.redshifts[i] && redshift < HM12_UVB_data.redshifts[i+1]) {
                
                // Perform linear interpolation
                scale_high = (redshift - HM12_UVB_data.redshifts[i]) / (HM12_UVB_data.redshifts[i+1] - HM12_UVB_data.redshifts[i]);
                scale_low  = 1.0 - scale_high;

                for (int j = 0; j < 27; j++) {
                    if (j<1){
                        HM12_UVB_z[1][j][0]  = (scale_low * HM12_UVB_data.hydrogen[i][j][0])  + (scale_high * HM12_UVB_data.hydrogen[i+1][j][0]);
                        HM12_UVB_z[1][j][1]  = (scale_low * HM12_UVB_data.hydrogen[i][j][1])  + (scale_high * HM12_UVB_data.hydrogen[i+1][j][1]);
                    }
                    if (j<2){
                        HM12_UVB_z[2][j][0]  = (scale_low * HM12_UVB_data.helium[i][j][0])    + (scale_high * HM12_UVB_data.helium[i+1][j][0]);
                        HM12_UVB_z[2][j][1]  = (scale_low * HM12_UVB_data.helium[i][j][1])    + (scale_high * HM12_UVB_data.helium[i+1][j][1]);
                    }
                    if (j<6){  
                        HM12_UVB_z[6][j][0]  = (scale_low * HM12_UVB_data.carbon[i][j][0])    + (scale_high * HM12_UVB_data.carbon[i+1][j][0]);
                        HM12_UVB_z[6][j][1]  = (scale_low * HM12_UVB_data.carbon[i][j][1])    + (scale_high * HM12_UVB_data.carbon[i+1][j][1]);
                    }
                    if (j<7){
                        HM12_UVB_z[7][j][0]  = (scale_low * HM12_UVB_data.nitrogen[i][j][0])  + (scale_high * HM12_UVB_data.nitrogen[i+1][j][0]);
                        HM12_UVB_z[7][j][1]  = (scale_low * HM12_UVB_data.nitrogen[i][j][1])  + (scale_high * HM12_UVB_data.nitrogen[i+1][j][1]);
                    }
                    if (j<8){
                        HM12_UVB_z[8][j][0]  = (scale_low * HM12_UVB_data.oxygen[i][j][0])    + (scale_high * HM12_UVB_data.oxygen[i+1][j][0]);
                        HM12_UVB_z[8][j][1]  = (scale_low * HM12_UVB_data.oxygen[i][j][1])    + (scale_high * HM12_UVB_data.oxygen[i+1][j][1]);
                    }
                    if (j<10){
                        HM12_UVB_z[10][j][0] = (scale_low * HM12_UVB_data.neon[i][j][0])      + (scale_high * HM12_UVB_data.neon[i+1][j][0]);
                        HM12_UVB_z[10][j][1] = (scale_low * HM12_UVB_data.neon[i][j][1])      + (scale_high * HM12_UVB_data.neon[i+1][j][1]);
                    }
                    if (j<12){
                        HM12_UVB_z[12][j][0] = (scale_low * HM12_UVB_data.magnesium[i][j][0]) + (scale_high * HM12_UVB_data.magnesium[i+1][j][0]);
                        HM12_UVB_z[12][j][1] = (scale_low * HM12_UVB_data.magnesium[i][j][1]) + (scale_high * HM12_UVB_data.magnesium[i+1][j][1]);
                    }
                    if (j<14){
                        HM12_UVB_z[14][j][0] = (scale_low * HM12_UVB_data.silicon[i][j][0])   + (scale_high * HM12_UVB_data.silicon[i+1][j][0]);
                        HM12_UVB_z[14][j][1] = (scale_low * HM12_UVB_data.silicon[i][j][1])   + (scale_high * HM12_UVB_data.silicon[i+1][j][1]);
                    }
                    if (j<16){
                        HM12_UVB_z[16][j][0] = (scale_low * HM12_UVB_data.sulfur[i][j][0])    + (scale_high * HM12_UVB_data.sulfur[i+1][j][0]);
                        HM12_UVB_z[16][j][1] = (scale_low * HM12_UVB_data.sulfur[i][j][1])    + (scale_high * HM12_UVB_data.sulfur[i+1][j][1]);
                    }
                    if (j<26){
                        HM12_UVB_z[26][j][0]  = (scale_low * HM12_UVB_data.iron[i][j][0])      + (scale_high * HM12_UVB_data.iron[i+1][j][0]);
                        HM12_UVB_z[26][j][1]  = (scale_low * HM12_UVB_data.iron[i][j][1])      + (scale_high * HM12_UVB_data.iron[i+1][j][1]);
                    }
                }                
            }
        }
    }

    return HM12_UVB_z;
}

} // namespace PRISM