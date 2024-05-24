#pragma once

#include "hydro/FiniteVolumePolicy_tools.h"

namespace dyablo{

template< typename LegacyState_t >
class FiniteVolumePolicy_Slope_minmod
{
public:
  using PrimState = typename LegacyState_t::PrimState;
  using ConsState = typename LegacyState_t::ConsState;

  FiniteVolumePolicy_Slope_minmod( ConfigMap& configMap )
  {}

  static std::string name() {return "minmod";}

  KOKKOS_INLINE_FUNCTION
  PrimState compute_slope( PrimState qL, PrimState qC, PrimState qR, real_t dL, real_t dR) const
  {
    auto dqp = (qR - qC) / dR;
    auto dqm = (qC - qL) / dL;

    PrimState slope{};
    state_foreach_var([](real_t& res, real_t dvp, real_t dvm) {
      if (dvp * dvm <= 0.0)
        res = 0.0;
      else
        res = fabs(dvp) > fabs(dvm) ? dvm : dvp;
    }, slope, dqp, dqm);

    return slope;
  }

};

template< typename State_t >
class FiniteVolumePolicy_Slope_superbee
{
public:
  using PrimState = typename State_t::PrimState;
  using ConsState = typename State_t::ConsState;

  FiniteVolumePolicy_Slope_superbee( ConfigMap& configMap )
  {}

  static std::string name() {return "superbee";}

  KOKKOS_INLINE_FUNCTION
  PrimState compute_slope( PrimState qL, PrimState qC, PrimState qR, real_t dL, real_t dR) const
  {
    auto dqp = (qR - qC) / dR;
    auto dqm = (qC - qL) / dL;

    PrimState slope{};
    state_foreach_var([](real_t& res, real_t dvp, real_t dvm) {
      if( dvp * dvm <= 0 )
        res = 0.0;
      else
      {
        real_t g1 = (fabs(2*dvp) < fabs(  dvm) ? 2*dvp :   dvm);
        real_t g2 = (fabs(  dvp) < fabs(2*dvm) ?   dvp : 2*dvm);
        if( g1 * g2 <= 0 )
          res = 0;
        else
          res = fabs(g1) < fabs(g2) ? g2 : g1;
      }
        
    }, slope, dqp, dqm);

    return slope;
  }
};

/***
 * Dynamic Implementation for FiniteVolumePolicy_Slope
 * Merge multiple implementations of FiniteVolumePolicy_Slope and 
 * choose which version to use at runtime
 * @tparam LegacyState_t::PrimState input/output state used for compute_slope
 * @tparam Ts... FiniteVolumePolicy_Slope types to choose from
 ***/
template< typename LegacyState_t, typename... Ts >
class FiniteVolumePolicy_Slope_dynamic_impl
{
  std::tuple<Ts...> slopes;
  size_t selected_slope;
public:
  using PrimState = typename LegacyState_t::PrimState;

  FiniteVolumePolicy_Slope_dynamic_impl( ConfigMap& configMap )
  : slopes(Ts(configMap)...)
  {
    std::string slope_limiter = configMap.getValue<std::string>("hydro", "slope_limiter", "minmod");

    this->selected_slope = tuple_find_name( slope_limiter, slopes );
  }

public:
  KOKKOS_INLINE_FUNCTION
  PrimState compute_slope( PrimState qL, PrimState qC, PrimState qR, real_t dL, real_t dR) const
  {
    return tuple_apply_nth( selected_slope, [&](const auto& s){return s.compute_slope( qL, qC, qR, dL, dR );}, slopes );
  }
};

template<typename LegacyState_t>
using FiniteVolumePolicy_Slope_dynamic = FiniteVolumePolicy_Slope_dynamic_impl< LegacyState_t,
  FiniteVolumePolicy_Slope_minmod<LegacyState_t>,
  FiniteVolumePolicy_Slope_superbee<LegacyState_t>
>;

} //namespace dyablo