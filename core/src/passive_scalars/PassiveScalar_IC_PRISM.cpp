#include "PassiveScalar_IC_base.h"
#include "../source_terms/cooling/PRISM.hpp"

namespace dyablo{

struct PassiveScalar_IC_PRISM : public PassiveScalar_IC {
  ForeachCell &foreach_cell;

  int n_passive_scalars;

  const std::vector<real_t> initial_abundances;
  const std::vector<real_t> initial_ionization;
  const std::vector<std::string> ions;

  std::map<std::string, int> ion_counts;
  Kokkos::Array<int, MAX_ELEMENTS> nions;
  Kokkos::Array<int, MAX_ELEMENTS> ions2passive;
  Kokkos::Array<int, MAX_ELEMENTS> elems2passive;

  PassiveScalar_IC_PRISM(  ConfigMap& configMap,
                                      ForeachCell& foreach_cell,  
                                      Timers& timers ) : 
    foreach_cell(foreach_cell),
    n_passive_scalars(configMap.getValue<int>("run", "n_passive_scalars", 0)),
    initial_abundances(configMap.getValue<std::vector<real_t>>("cooling", "initial_abundances")),
    initial_ionization(configMap.getValue<std::vector<real_t>>("cooling", "initial_ionization")),
    ions(configMap.getValue<std::vector<std::string>>("cooling", "ions"))
    {
      PRISM::parseIonInputs(ions, this->nions, 
        this->elems2passive, this->ions2passive, this->ion_counts);

      DYABLO_ASSERT_HOST_RELEASE( n_passive_scalars >= int(initial_abundances.size() + initial_ionization.size()),
        "Not enough passive scalars for initial abundances and ionization. "
        "n_passive_scalars = " << n_passive_scalars << ", "
        "initial_abundances.size() = " << initial_abundances.size() << ", "
        "initial_ionization.size() = " << initial_ionization.size() );
      DYABLO_ASSERT_HOST_DEBUG( ions.size() == initial_ionization.size(),
        "Number of ions must match number of initial ionization values. "
        "ions.size() = " << ions.size() << ", "
        "initial_ionization.size() = " << initial_ionization.size() );

      for (auto i = 0; i < int(initial_abundances.size()); ++i) {
        DYABLO_ASSERT_HOST_DEBUG( initial_abundances[i] >= 0.0 && initial_abundances[i] <= 1.0,
          "Initial abundance must be between 0 and 1. "
          "initial_abundances[" << i << "] = " << initial_abundances[i] );
      }
      for (auto i = 0; i < int(initial_ionization.size()); ++i) {
        DYABLO_ASSERT_HOST_DEBUG( initial_ionization[i] >= 0.0 && initial_ionization[i] <= 1.0,
          "Initial ionization must be between 0 and 1. "
          "initial_ionization[" << i << "] = " << initial_ionization[i] );
      }
  }

  void init( UserData &U, int passive_scalar_id ) {
    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
    std::ostringstream oss;
    oss << "passive_scalar_" << passive_scalar_id;
    const UserData::FieldAccessor Uout = U.getAccessor( {{oss.str(), 0}} );

    real_t v;
    if (passive_scalar_id < int(ion_counts.size())) {
      v = initial_abundances[passive_scalar_id];
    } else {
      v = initial_ionization[passive_scalar_id - ion_counts.size()];
    }
    foreach_cell.foreach_cell( "PassiveScalar_IC_PRISM::fill_U", U.getShape(),
                KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell_U )
    {
      Uout.at(iCell_U, 0) = v;
    });
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::PassiveScalar_IC_Factory, 
                 dyablo::PassiveScalar_IC_PRISM,
                 "PRISM");

