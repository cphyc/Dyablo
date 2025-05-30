#pragma once

#include <type_traits>

#include "HyperbolicUpdate_base.h"
#include "mpi/GhostCommunicator_partial_blocks.h"
#include "foreach_cell/ForeachCell_utils.h"

namespace dyablo {
namespace{
using CellIndex     = ForeachCell::CellIndex;
using FieldAccessor = UserData::FieldAccessor;
using offset_t      = typename CellIndex::offset_t;
using PatchArray = ForeachCell::CellArray_patch;

}// namespace
}// namespace dyablo

namespace dyablo {

namespace impl{
namespace {
template <typename T, typename = int, typename=int>
struct HasDensityAndPressure : std::false_type { };

template <typename T>
struct HasDensityAndPressure <T, decltype((void) T::rho, 0), decltype((void) T::p, 0)> : std::true_type { };
}
}

/**
 * @brief Noop - deactivates hydro
 *
 * @tparam State the type of state to treat
 */
template<typename Policy>
class Hyperbolic_noop : public HyperbolicUpdate {
  static_assert( is_HyperbolicPolicy_v<Policy>,
  "Policy must be wrapped in HyperbolicPolicy_base");

public:
  using PrimState = typename Policy::PrimState;
  using ConsState = typename Policy::ConsState;

public:
  Hyperbolic_noop(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers)
  : foreach_cell(foreach_cell),
    policy_params(Policy::getParams(configMap)),
    n_passive_scalars( configMap.getValue<int>("run", "n_passive_scalars", 0) )
 {};

  /**
   * @brief Solves hydro for one step using the euler method
   *
   * @param U the input/output global array
   * @param scalar_data input scalar data
   */
  void update( UserData& U, ScalarSimulationData& scalar_data)
  {
    const Policy policy( this->policy_params, scalar_data );
        ForeachCell& foreach_cell = this->foreach_cell;


    FieldAccessor Uin = policy.getUin(U);
    FieldAccessor Uout = policy.getUout(U);

    int n_passive_scalars = this->n_passive_scalars;
    std::vector<UserData::FieldAccessor::FieldInfo> passive_scalars_ids;
    for (int i=0; i < n_passive_scalars; ++i) {
      std::ostringstream oss;
      oss << "passive_scalar_" << i;
      passive_scalars_ids.push_back({oss.str(), i});
    }

    FieldAccessor passive_scalars_in, passive_scalars_out;
    if (n_passive_scalars > 0) {
      passive_scalars_in = U.getAccessor(passive_scalars_ids);
      for (auto &s: passive_scalars_ids)
        s.name += "_next";
      passive_scalars_out = U.getAccessor(passive_scalars_ids);
    }

    // Copy old -> new
    foreach_cell.foreach_cell("HyperbolicUpdate:copy", Uout.getShape(),
      CELL_LAMBDA( const CellIndex& iCell ) 
    {
      ConsState u;
      // Copy primitive states
      u = policy.getConsState(Uin, iCell);
      policy.setConsState(Uout, iCell, u);
      
      // Copy passive scalars
      for (auto i = 0; i < n_passive_scalars; ++i) {
        passive_scalars_out.at(iCell, i) = passive_scalars_in.at(iCell, i);
      }
    }
    );

  }

private:
  ForeachCell& foreach_cell;

  typename Policy::Params policy_params;

  int n_passive_scalars;
};

} // namespace dyablo
