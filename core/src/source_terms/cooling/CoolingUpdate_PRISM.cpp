#include <math.h>

#include "states/State_hydro.h"
#include "../SourceUpdate_base.h"
#include "utils/units/Units.h"

#include "PRISM.hpp"


namespace dyablo {

/**
 * @brief Simple density-metallicity dependent cooling function.
 */

class CoolingUpdate_PRISM : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;
  real_t gamma0;
  real_t smallr;
  real_t smallc;
  real_t smallp;

public:
  CoolingUpdate_PRISM(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  : foreach_cell(foreach_cell),
    timers(timers),
    gamma0(configMap.getValue<real_t>("hydro", "gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro", "smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro", "smallc", 1e-10)),
    smallp(smallc * smallc / gamma0),
    cosmo_run(configMap.getValue<bool>("cosmology", "active", false)),
    UVB_table_path(configMap.getValue<std::string>("cooling", "UVB_tables")),
    ct_rates_path(configMap.getValue<std::string>("cooling", "charge_transfer_tables")),
    high_temperature_metal_cooling_path(configMap.getValue<std::string>("cooling", "high_temperature_metal_cooling_tables")),
    fine_structure_path(configMap.getValue<std::string>("cooling", "fine_structure_tables")),
    n_passive_scalars( configMap.getValue<int>("run", "n_passive_scalars", 0) ),
    HM12_UVB_data(PRISM::load_UVB_data(UVB_table_path)),
    ions( configMap.getValue<std::vector<std::string>>("cooling", "ions" ) ),
    T_blackbody( configMap.getValue<real_t>("cooling", "T_blackbody", 1e4) ),
    unit_time( configMap.getValue<real_t>("units", "time", 1.0) ),
    unit_density( configMap.getValue<real_t>("units", "density", 1.0) ),
    unit_length( configMap.getValue<real_t>("units", "length", 1.0) )
  {

    PRISM::parseIonInputs(ions, this->nions, this->elems2passive, this->ions2passive, this->ion_counts);

    // Initialize cosmic ray rates
    const auto& [cosmic_ray_ionization_rates, cosmic_ray_ionization_rates_induced_UV, cosmic_ray_ionization_rates_induced_UV_heat] = PRISM::initialize_cr_rates();

    // Initialize the charge transfer rates
    const auto& [CTRecomb, CTIon] = PRISM::load_ct_rates(ct_rates_path);

    // Initialize high temperature cooling tables
    const auto& [high_t_cooling_temp, high_t_cooling_rates, high_t_cooling_rates_tflag] = PRISM::initialize_high_temperature_metal_cooling(high_temperature_metal_cooling_path);

    // Initialize the low temperature cooling tables
    const auto& fs_cool_tab = PRISM::init_fine_structure_tables(fine_structure_path);

    // Get hard-coded rates
    const auto& dust_rec_coefs = PRISM::get_dust_rec_coefs();
    const auto& G0_heating_rates = PRISM::get_G0_heating_rates();

    // HM-related data
    PRISM::copy_data_1D(G0_heating_rates, tabData.G0_heating_rates);

    // Initialize rates
    init_recombination_rates(tabData);
    init_collisional_ionization(tabData);

    // Initialize photon groups
    // TODO: This should be configurable
    group_E_min = {};
    group_E_max = {};
    group_E_min[0] = 13.6; // eV
    group_E_max[0] = 500.0; // eV

    // Cosmic ray and dust-related data
    PRISM::copy_data_2D(cosmic_ray_ionization_rates, tabData.cosmic_ray_ionization_rates);
    PRISM::copy_data_1D(cosmic_ray_ionization_rates_induced_UV, tabData.cosmic_ray_ionization_rates_induced_UV);
    PRISM::copy_data_1D(cosmic_ray_ionization_rates_induced_UV_heat, tabData.cosmic_ray_ionization_rates_induced_UV_heat);
    PRISM::copy_data_2D(dust_rec_coefs, tabData.dust_rec_coefs);

    // Charge transfer
    PRISM::copy_data_3D(CTRecomb, tabData.CTRecomb);
    PRISM::copy_data_3D(CTIon, tabData.CTIon);

    // Fine structure cooling
    PRISM::copy_data_3D(fs_cool_tab, tabData.fs_cool_tab);

    // High-T cooling rates
    PRISM::copy_data_1D(high_t_cooling_temp, tabData.high_t_cooling_temp);
    PRISM::copy_data_3D(high_t_cooling_rates, tabData.high_t_cooling_rates);
    PRISM::copy_data_2D(high_t_cooling_rates_tflag, tabData.high_t_cooling_rates_tflag);
  }

  ~CoolingUpdate_PRISM() {}

  void update( UserData &U,
               ScalarSimulationData& scalar_data
               )
  {
    enum VarIndex_rad {
      Ie_rad, Ifx_rad, Ify_rad, Ifz_rad,       // radiation field
    };
    timers.get("Cooling PRISM").start();
    // ---------------------------------------------------
    // MODULE CONFIGURATOIN
    constexpr bool include_H2 = false;               // Whether to include molecular hydrogen
    constexpr bool include_CO = false;               // Whether to include CO
    constexpr bool ramses_rt_T_scheme = true;        // Whether to use the ramses-rt temperature update
    constexpr bool rosenbrock_T_scheme = false;      // Whether to use the rosenbrocat temperature update
    constexpr bool constant_temperature = false;     // Whether to compute at constant temperature
    // constexpr real_t metallicity = 0.1;              // Some assumed metallicity

    // ---------------------------------------------------
    // UPDATE TIME-DEPENDENT QUANTITIES
    const real_t aexp = cosmo_run ? scalar_data.get<real_t>("aexp") : 1;
    const real_t redshift = 1e0/aexp - 1e0;

    const real_t dt = scalar_data.get<real_t>("dt");

    // Update UVB
    const auto& HM12_UVB_z = PRISM::update_UVB(redshift, this->HM12_UVB_data); // Interpolate to the correct redshift
    PRISM::copy_data_3D(HM12_UVB_z, tabData.HM12_UVB_z);

    // Update cross sections
    // NB: for a blackbody, this could be done once at initialization.
    //     But we keep it here for flexibility.
    const auto& cross_sections = PRISM::initialize_cross_sections();
    const auto& cs_ph = PRISM::update_cross_sections(T_blackbody, cross_sections, group_E_min, group_E_max);

    // Photo-heating cross sections
    PRISM::copy_data_4D(cs_ph, tabData.cs_ph);

    // ---------------------------------------------------
    // GET ACCESSORS
    // Hydro fields accessors
    const UserData::FieldAccessor Uin = U.getAccessor({
        {"rho", ConsHydroState::VarIndex::Irho},
        {"e_tot", ConsHydroState::VarIndex::Ie_tot},
        {"rho_vx", ConsHydroState::VarIndex::Irho_vx},
        {"rho_vy", ConsHydroState::VarIndex::Irho_vy},
        {"rho_vz", ConsHydroState::VarIndex::Irho_vz}
    });
    UserData::FieldAccessor Uout = U.getAccessor({
        {"rho_next", ConsHydroState::VarIndex::Irho},
        {"e_tot_next", ConsHydroState::VarIndex::Ie_tot},
        {"rho_vx_next", ConsHydroState::VarIndex::Irho_vx},
        {"rho_vy_next", ConsHydroState::VarIndex::Irho_vy},
        {"rho_vz_next", ConsHydroState::VarIndex::Irho_vz},
    });

    // RT fields accessors
    const UserData::FieldAccessor Uin_rad = U.getAccessor({
        {"e_rad", VarIndex_rad::Ie_rad},
        {"fx_rad", VarIndex_rad::Ifx_rad},
        {"fy_rad", VarIndex_rad::Ify_rad},
        {"fz_rad", VarIndex_rad::Ifz_rad}
    });

    // Ion abundances and ionization fractions accessors
    std::vector<UserData::FieldAccessor::FieldInfo> passive_in, passive_out;
    for (auto ipassive = 0; ipassive < int(ion_counts.size() + ions.size()); ++ipassive) {
        std::ostringstream oss;
        oss << "passive_scalar_" << ipassive;
        passive_in.push_back({oss.str(), ipassive});

        oss << "_next";
        passive_out.push_back({oss.str(), ipassive});
    }
    const UserData::FieldAccessor Uin_passive = U.getAccessor( passive_in );
    UserData::FieldAccessor Uout_passive = U.getAccessor( passive_out );

    // ---------------------------------------------------
    // GET CONSTANTS
    const real_t unit_time = this->unit_time;  // code -> [s]
    const real_t unit_density = this->unit_density / (Units::PROTON_MASS * 1e6); // code -> [kg/m^3] -> [mp/cm^3]
    const real_t unit_T = SQR(this->unit_length / this->unit_time) * Units::PROTON_MASS / Units::KBOLTZ; // code -> [K]

    const real_t dt_s = dt * unit_time;
    const real_t gamma0 = this->gamma0;
    const auto& nions = this->nions;
    const auto& elems2passive = this->elems2passive;
    const auto& ions2passive = this->ions2passive;
    const auto& tabData = this->tabData;

    // ---------------------------------------------------
    // THERMOCHEMISTRY STEP
    int Ncell = 0, Nstep_tot = 0;
    foreach_cell.reduce_cell( "Cooling::update", Uout.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell, int& Nstep_tot, int& Ncell) {
        dyablo::ConsHydroState u;
        getConservativeState<3>(Uin, iCell, u);
        dyablo::PrimHydroState q = consToPrim<3>(u, gamma0);

        // Compute temperature
        const real_t Tmu = q.p / q.rho * (gamma0 - 1) * unit_T;

        // Extract ion data
        PRISM::Element elements_loc[MAX_ELEMENTS];
        PRISM::ParticleIonData n_and_ion_fracs_loc[MAX_ELEMENTS];
        // Reset values
        for (auto i = 0; i < MAX_ELEMENTS; ++i) {
          n_and_ion_fracs_loc[i].n_element = 0.0;
          for (auto j = 0; j < MAX_ELEMENTS; ++j) {
            n_and_ion_fracs_loc[i].ion_fracs[j] = 0.0;
            n_and_ion_fracs_loc[i].ion_fracs_new[j] = 0.0;
          }
        }

        // Initialize the elements
        PRISM::initialize_elements<include_H2, include_CO>(elements_loc, nions);

        // Get nH
        real_t nH = q.rho * unit_density * (1.0 - Units::YHE) / elements_loc[1].atomic_mass;
        n_and_ion_fracs_loc[1].n_element = nH;

        // Get other species abundances
        for (auto i = 2; i < MAX_ELEMENTS; ++i) {
          if (elements_loc[i].atomic_number < 0) continue;
          n_and_ion_fracs_loc[i].n_element = nH * Uin_passive.at(iCell, elems2passive[i]);
        }

        // Get ion fractions (incl. H)
        for (auto i = 1; i < MAX_ELEMENTS; ++i) {
          if (elements_loc[i].atomic_number < 0) continue;
          real_t xtot = 0.0;
          for (auto j = 0; j < elements_loc[i].n_ions + elements_loc[i].n_mol; ++j) {
              n_and_ion_fracs_loc[i].ion_fracs[j] = Uin_passive.at_ivar(iCell, ions2passive[i] + j);
              xtot += n_and_ion_fracs_loc[i].ion_fracs[j];
          }
          // Normalize ion fractions
          for (auto j = 0; j < elements_loc[i].n_ions + elements_loc[i].n_mol; ++j) {
              n_and_ion_fracs_loc[i].ion_fracs[j] /= xtot;
          }
        }

        constexpr double UV_background_G0 = 1E-6;
        constexpr double generic_cosmic_ray_ionization_rate = 1E-25;
        constexpr double dust_to_gas_mass_ratio_over_mw = 0.04;
        real_t N_phot[MAX_N_GROUPS] = {0};
        real_t F_phot[MAX_N_GROUPS][2] = {{0, 0}};

        const auto& [total_iterations, Tout] = PRISM::subcycle_chemistry<
          constant_temperature, ramses_rt_T_scheme, rosenbrock_T_scheme, include_H2, include_CO,
          true
        >(elements_loc, n_and_ion_fracs_loc, Tmu, aexp, dt_s,
          UV_background_G0, generic_cosmic_ray_ionization_rate, dust_to_gas_mass_ratio_over_mw,
          N_phot, F_phot, 1,
          20000, 100000,
          tabData);

        // Store new temperature
        q.p = q.rho * Tout / (gamma0 - 1) / unit_T;

        u = primToCons<3>(q, gamma0);
        Uout.at(iCell, dyablo::ConsHydroState::Ie_tot) = u.e_tot;
        // Copy passive scalars
        for (auto i = 1; i < MAX_ELEMENTS; ++i) {
          if (elements_loc[i].atomic_number < 0) continue;
          for (auto j = 0; j < elements_loc[i].n_ions + elements_loc[i].n_mol; ++j) {
              Uout_passive.at(iCell, ions2passive[i] + j) = n_and_ion_fracs_loc[i].ion_fracs_new[j];
          }
        }

        printf("T = %e, rho = %e, xHI = %e, xHII = %e, iterations = %d\n",
               Tout, q.rho, Uout_passive.at(iCell, ions2passive[1]), Uout_passive.at(iCell, ions2passive[1] + 1), total_iterations);

        Nstep_tot += total_iterations;
        Ncell++;

    }, Nstep_tot, Ncell);

    std::cout << "Converged in " << real_t(Nstep_tot) / Ncell << " iterations on average." << std::endl;

    timers.get("Cooling PRISM").stop();

  }

  PRISM::TabulatedData tabData;
  bool cosmo_run;

  std::string UVB_table_path;
  std::string ct_rates_path;
  std::string high_temperature_metal_cooling_path;
  std::string fine_structure_path;

  int n_passive_scalars;

  // UV background data
  PRISM::UVB_table_t HM12_UVB;
  PRISM::UVB_data HM12_UVB_data;

  // Information about network
  std::vector<std::string> ions;
  std::map<std::string, int> ion_counts;
  std::array<int, MAX_ELEMENTS> nions;
  std::array<int, MAX_ELEMENTS> elems2passive;
  std::array<int, MAX_ELEMENTS> ions2passive;

  // Photon groups
  std::array<double, MAX_N_GROUPS> group_E_min;
  std::array<double, MAX_N_GROUPS> group_E_max;

  // Blackbody temperature
  real_t T_blackbody;

  // Units
  real_t unit_time;
  real_t unit_density;
  real_t unit_length;

};

} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory,
                  dyablo::CoolingUpdate_PRISM,
                  "CoolingUpdate_PRISM");