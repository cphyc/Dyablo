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
    n_passive_scalars( configMap.getValue<int>("run", "n_passive_scalars", 0) )
  {

    PRISM::parseIonInputs(ions, this->nions, this->elems2passive, this->ions2passive, this->ion_counts); 

    // Initialize the UV background data
    load_UVB_data(UVB_table_path);      // First load the UVB data from files

    // Initialize cosmic ray rates
    initialize_cr_rates();

    // Initialize the charge transfer rates
    load_ct_rates(ct_rates_path);

    // Initialize high temperature cooling tables
    initialize_high_temperature_metal_cooling(high_temperature_metal_cooling_path);

    // Initialize the low temperature cooling tables
    init_fine_structure_tables(fine_structure_path);

    // HM-related data
    PRISM::copy_data_1D(G0_heating_rates, tabData.G0_heating_rates);

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
    PRISM::copy_data_3D(high_t_cooling_rates, tabData.high_t_cooling_rates);
    PRISM::copy_data_2D(high_t_cooling_rates_tflag, tabData.high_t_cooling_rates_tflag);

    // Initialize rates
    init_recombination_rates(tabData);
    init_collisional_ionization(tabData);
  }

  ~CoolingUpdate_PRISM() {}

  void update( UserData &U,
               ScalarSimulationData& scalar_data
               )
  {
    constexpr bool include_H2 = false;               // Whether to include molecular hydrogen
    constexpr bool include_CO = false;               // Whether to include CO
    constexpr bool ramses_rt_T_scheme = true;        // Whether to use the ramses-rt temperature update
    constexpr bool rosenbrock_T_scheme = false;      // Whether to use the rosenbrocat temperature update
    constexpr bool constant_temperature = false;     // Whether to compute at constant temperature
    constexpr real_t metallicity = 0.1;              // Some assumed metallicity

    const real_t aexp = cosmo_run ? scalar_data.get<real_t>("aexp") : 1;
    const real_t redshift = 1e0/aexp - 1e0;

    // Update UVB
    update_UVB(redshift); // Interpolate to the correct redshift
    PRISM::copy_data_3D(HM12_UVB_z, tabData.HM12_UVB_z);

    std::vector<dyablo::UserData_fields::FieldAccessor_FieldInfo> in_fields = {
        {"rho", dyablo::ConsHydroState::Irho},
        {"e_tot", dyablo::ConsHydroState::Ie_tot},
        {"rho_vx", dyablo::ConsHydroState::Irho_vx},
        {"rho_vy", dyablo::ConsHydroState::Irho_vy},
        {"rho_vz", dyablo::ConsHydroState::Irho_vz}
    };
    std::vector<dyablo::UserData_fields::FieldAccessor_FieldInfo> out_fields = {
        {"rho_next", dyablo::ConsHydroState::Irho},
        {"e_tot_next", dyablo::ConsHydroState::Ie_tot},
        {"rho_vx_next", dyablo::ConsHydroState::Irho_vx},
        {"rho_vy_next", dyablo::ConsHydroState::Irho_vy},
        {"rho_vz_next", dyablo::ConsHydroState::Irho_vz},
    };
    // Push back the ion abundances + element mass fractions
    for (auto ipassive = 0; ipassive < ion_counts.size() + ions.size(); ++ipassive) {
        std::ostringstream oss;
        oss << "passive_scalar_" << ipassive;
        in_fields.push_back({oss.str(), dyablo::ConsHydroState::Irho_vz + ipassive + 1});

        oss << "_next";
        out_fields.push_back({oss.str(), dyablo::ConsHydroState::Irho_vz + ipassive + 1});
    }

    const Kokkos::Array<int, MAX_ELEMENTS> elems2passive = this->elems2passive;
    const Kokkos::Array<int, MAX_ELEMENTS> ions2passive = this->ions2passive;

    const UserData::FieldAccessor Uin = U.getAccessor( in_fields );
    UserData::FieldAccessor Uout = U.getAccessor( out_fields );

    foreach_cell.foreach_cell( "Cooling::update", Uout.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell) {
        dyablo::ConsHydroState u;
        getConservativeState<3>(Uin, iCell, u);
        const dyablo::PrimHydroState q = consToPrim<3>(u, gamma0);

        // Compute temperature
        // TODO: fix units!
        const real_t Tmu = q.p / q.rho / Units::KBOLTZ * Units::PROTON_MASS;

        // Extract ion data
        Element elements_loc[MAX_ELEMENTS];
        PRISM::ParticleIonData n_and_ion_fracs_loc[MAX_ELEMENTS];

        // Initialize the elements
        // TODO: read from state
        PRISM::initialize_elements<include_H2, include_CO>(elements_loc, nions);

        // Initialize the ion fractions
        // TODO: read from state, q.rho in [cm^-3]
        // initialize_ion_fracs(elements_loc, n_and_ion_fracs_loc, q.rho, metallicity, 4);
        {
            for (auto i = 1; i < MAX_ELEMENTS; ++i) {
                if (elements_loc[i].atomic_number < 0) continue;
                // FIXME: we assume rho to be in [mp/cm**3 already]
                n_and_ion_fracs_loc[i].n_element = (
                    q.rho * Uin.at_ivar(iCell, dyablo::ConsHydroState::Irho_vz + 1 + elems2passive[i])
                    / (elements_loc[i].atomic_mass)
                );
                for (auto j = 0; j < elements_loc[i].n_ions + elements_loc[i].n_mol; ++j) {
                    n_and_ion_fracs_loc[i].ion_fracs[j] = Uin.at_ivar(iCell, dyablo::ConsHydroState::Irho_vz + 1 + ions2passive[i] + j);
                }
            }
        }

        const real_t Tout = PRISM::get_chemical_eqm<constant_temperature, ramses_rt_T_scheme, rosenbrock_T_scheme, include_H2, include_CO>(
            elements_loc, n_and_ion_fracs_loc, Tmu, aexp, -1.0, 0.59, 1e-10, 0.04, tabData);

        // std::cout << "rho =" << q.rho
        //           << " Tin = " << Tmu << " Tout = " << Tout
        //           << " xHI  = " << n_and_ion_fracs_loc[1].ion_fracs_new[0]
        //           << " xHII = " << n_and_ion_fracs_loc[1].ion_fracs_new[1]
        //           << std::endl;

    });

  }

  TabulatedData tabData;
  bool cosmo_run;

  std::string UVB_table_path;
  std::string ct_rates_path;
  std::string high_temperature_metal_cooling_path;
  std::string fine_structure_path;

  std::vector<std::string> ions;
  std::map<std::string, int> ion_counts;
  Kokkos::Array<int, MAX_ELEMENTS> nions;
  Kokkos::Array<int, MAX_ELEMENTS> elems2passive;
  Kokkos::Array<int, MAX_ELEMENTS> ions2passive;
  int n_passive_scalars;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory,
                  dyablo::CoolingUpdate_PRISM,
                  "CoolingUpdate_PRISM");