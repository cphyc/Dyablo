#pragma once

#include "utils/misc/dyablo_variant.h"

namespace dyablo{

template< typename LegacyState_t >
class HyperbolicPolicy_Slope_minmod
{
public:
  using PrimState = typename LegacyState_t::PrimState;
  using ConsState = typename LegacyState_t::ConsState;

  static std::string name() {return "minmod";}

  struct Params {};

  static Params getParams( const ConfigMap& )
  {
    return {};
  }

  HyperbolicPolicy_Slope_minmod(const Params& Params)
  {}

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
class HyperbolicPolicy_Slope_superbee
{
public:
  using PrimState = typename State_t::PrimState;
  using ConsState = typename State_t::ConsState;

  static std::string name() {return "superbee";}

  struct Params {};

  static Params getParams( const ConfigMap& )
  {
    return {};
  }

  HyperbolicPolicy_Slope_superbee(const Params& Params)
  {}

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

namespace Slope_dynamic_impl{
namespace {

template <typename var_out, typename T0, typename... Ts>
var_out slope_params_from_name_aux( const std::string& name, const ConfigMap& configMap )
{
  if( T0::name() == name )
    return T0::getParams(configMap);
  else
  {
    DYABLO_ASSERT_HOST_RELEASE( sizeof...(Ts) > 0, "Could not find slope '" << name << "' for HyperbolicPolicy_Slope_dynamic" );
    if constexpr (sizeof...(Ts) > 0)
      return slope_params_from_name_aux<var_out, Ts...>( name, configMap );
  }
}

template< typename... Ts>
dyablo_variant<typename Ts::Params...> slope_params_from_name( const std::string& name, const ConfigMap& configMap )
{
  return slope_params_from_name_aux<dyablo_variant<typename Ts::Params...>, Ts...>(name, configMap);
}

template< typename var_in, typename var_out, typename T0, typename... Ts>
var_out slope_from_params_aux( const var_in& params )
{
  using Params_t = typename T0::Params;
  if( dyablo_variant_holds_alternative<Params_t>(params) )
    return T0( dyablo_variant_get<Params_t>(params) );
  else
  {
    DYABLO_ASSERT_HOST_RELEASE( sizeof...(Ts) > 0, "Internal error : invalid slope parameters for HyperbolicPolicy_Slope_dynamic" );
    if constexpr (sizeof...(Ts) > 0)
      return slope_from_params_aux<var_in, var_out, Ts...>( params );
  }
}

template< typename... Ts>
dyablo_variant<Ts...> slope_from_params( const dyablo_variant<typename Ts::Params...>& params )
{
  return slope_from_params_aux<dyablo_variant<typename Ts::Params...>, dyablo_variant<Ts...>, Ts...>(params);
}

} // namespace
} // namespace Slope_dynamic_impl

/***
 * Dynamic Implementation for HyperbolicPolicy_Slope
 * Merge multiple implementations of HyperbolicPolicy_Slope and 
 * choose which version to use at runtime
 * @tparam LegacyState_t::PrimState input/output state used for compute_slope
 * @tparam Ts... HyperbolicPolicy_Slope types to choose from
 ***/
template< typename LegacyState_t, typename... Ts >
class HyperbolicPolicy_Slope_dynamic_impl
{
  dyablo_variant<Ts...> slope;
public:
  using PrimState = typename LegacyState_t::PrimState;

  struct Params{
    dyablo_variant<typename Ts::Params...> params;
  };

  static Params getParams(ConfigMap& configMap)
  {
    std::string slope_limiter = configMap.getValue<std::string>("hydro", "slope_limiter", "minmod");
    
    return {
      .params = Slope_dynamic_impl::slope_params_from_name<Ts...>(slope_limiter, configMap)
    };
  }

  HyperbolicPolicy_Slope_dynamic_impl(const Params& params)
  : slope(Slope_dynamic_impl::slope_from_params<Ts...>(params.params))
  {}

public:
  KOKKOS_INLINE_FUNCTION
  PrimState compute_slope( PrimState qL, PrimState qC, PrimState qR, real_t dL, real_t dR) const
  {
    return dyablo_variant_visit<PrimState>( 
      [&](const auto& s) -> PrimState
    {
      return s.compute_slope( qL, qC, qR, dL, dR );
    }, this->slope);
  }
};

template<typename LegacyState_t>
using HyperbolicPolicy_Slope_dynamic = HyperbolicPolicy_Slope_dynamic_impl< LegacyState_t,
  HyperbolicPolicy_Slope_minmod<LegacyState_t>,
  HyperbolicPolicy_Slope_superbee<LegacyState_t>
>;

} //namespace dyablo