#pragma once

#include <type_traits>

#include "HyperbolicUpdate_base.h"
#include "mpi/GhostCommunicator_partial_blocks.h"
#include "foreach_cell/ForeachCell_utils.h"

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
          Timers& timers) {};

  /**
   * @brief Solves hydro for one step using the euler method
   * 
   * @param U the input/output global array
   * @param scalar_data input scalar data
   */
  void update( UserData& U, ScalarSimulationData& scalar_data)
  {
    
  }

private:
};

} // namespace dyablo
