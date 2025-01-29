#pragma once

#include "HyperbolicPolicy_base.h"
#include "utils/misc/dyablo_variant.h"

namespace dyablo{

template< typename State_t >
class HyperbolicPolicy_BoundaryConditions_PeriodicOnly
{
private:
  using HyperbolicPolicy_State = State_t;
  using CellIndex     = ForeachCell::CellIndex;
  using CellMetaData  = ForeachCell::CellMetaData;
  using offset_t      = CellIndex::offset_t;
public:
  using PrimState = typename HyperbolicPolicy_State::PrimState;
  using ConsState = typename HyperbolicPolicy_State::ConsState;

  struct Params{};

  static Params getParams( ConfigMap& configMap )
  {
    DYABLO_ASSERT_HOST_RELEASE( 
       configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_PERIODIC) == BC_PERIODIC 
    && configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_PERIODIC) == BC_PERIODIC 
    && configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_PERIODIC) == BC_PERIODIC 
    && configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_PERIODIC) == BC_PERIODIC 
    && configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_PERIODIC) == BC_PERIODIC 
    && configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_PERIODIC) == BC_PERIODIC ,
    "Boundary conditions are 'periodic only' but other boundary conditions are set in .ini" );
    return {};
  }

  HyperbolicPolicy_BoundaryConditions_PeriodicOnly( const Params& params, const ScalarSimulationData& scalar_data )
  {}

  template < typename Array_t, typename Policy_t>
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Policy_t      &policy, 
                                   const Array_t       &U, 
                                   const CellIndex     &iCell_boundary, 
                                   const CellMetaData  &metadata) const 
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( false, "BoundaryConditions_PeriodicOnly::getBoundaryValue_impl should not be called" );
    return {};
  }

  template < typename Array_t, typename Policy_t>
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux( const Policy_t      &policy, 
                                  const Array_t       &U, 
                                  const CellIndex     &iCell_boundary, 
                                  const PrimState     &q_in_reconstructed,
                                  const CellMetaData  &metadata) const 
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( false, "BoundaryConditions_PeriodicOnly::getBoundaryFlux_impl should not be called" );
    return {};
  }
};

namespace BoundaryConditions_dynamic_impl{
namespace {

template <typename var_out, typename T0, typename... Ts>
var_out bc_params_from_name_aux( const std::string& name, ConfigMap& configMap )
{
  if( T0::name() == name )
    return T0::getParams(configMap);
  else
  {
    DYABLO_ASSERT_HOST_RELEASE( sizeof...(Ts) > 0, "Could not find boundary conditions '" << name << "' for HyperbolicPolicy_BoundaryConditions_dynamic" );
    if constexpr (sizeof...(Ts) > 0)
      return bc_params_from_name_aux<var_out, Ts...>( name, configMap );
  }
}

template< typename... Ts>
dyablo_variant<typename Ts::Params...> bc_params_from_name( const std::string& name, ConfigMap& configMap )
{
  return bc_params_from_name_aux<dyablo_variant<typename Ts::Params...>, Ts...>(name, configMap);
}

template< typename var_in, typename var_out, typename T0, typename... Ts>
var_out bc_from_params_aux( const var_in& params, const ScalarSimulationData& scalar_data )
{
  using Params_t = typename T0::Params;
  if( dyablo_variant_holds_alternative<Params_t>(params) )
    return T0( dyablo_variant_get<Params_t>(params), scalar_data );
  else
  {
    DYABLO_ASSERT_HOST_RELEASE( sizeof...(Ts) > 0, "Internal error : invalid boundary conditions parameters for HyperbolicPolicy_BoundaryConditions_dynamic" );
    if constexpr (sizeof...(Ts) > 0)
      return bc_from_params_aux<var_in, var_out, Ts...>( params, scalar_data );
  }
}

template< typename... Ts>
dyablo_variant<Ts...> bc_from_params( const dyablo_variant<typename Ts::Params...>& params, const ScalarSimulationData& scalar_data )
{
  return bc_from_params_aux<dyablo_variant<typename Ts::Params...>, dyablo_variant<Ts...>, Ts...>(params, scalar_data);
}

} // namespace
} // namespace BoundaryConditions_dynamic_impl


template< typename State_t, typename... Ts >
class HyperbolicPolicy_BoundaryConditions_dynamic
{
private:
  using HyperbolicPolicy_State = State_t;
  using CellIndex     = ForeachCell::CellIndex;
  using CellMetaData  = ForeachCell::CellMetaData;
  using offset_t      = CellIndex::offset_t;

  dyablo_variant<Ts...> boundary_conditions;

public:
  using PrimState = typename HyperbolicPolicy_State::PrimState;
  using ConsState = typename HyperbolicPolicy_State::ConsState;

  // static std::string name() {
  //   return std::string("dynamic : ")
  //         + (( (std::string(" ") + Ts::name()) + ...));
  // }

  struct Params{
    dyablo_variant<typename Ts::Params...> params;
  };

  static Params getParams( ConfigMap& configMap )
  {
    std::string bc_name = configMap.getValue<std::string>("hydro", "boundary_conditions", "default");

    return {
      .params = BoundaryConditions_dynamic_impl::bc_params_from_name<Ts...>(bc_name, configMap)
    };
  }

  HyperbolicPolicy_BoundaryConditions_dynamic( const Params& params, const ScalarSimulationData& scalar_data )
  : boundary_conditions(BoundaryConditions_dynamic_impl::bc_from_params<Ts...>(params.params, scalar_data))
  {}

  template < typename Array_t, typename Policy_t>
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Policy_t      &policy, 
                              const Array_t       &U, 
                              const CellIndex     &iCell_boundary, 
                              const CellMetaData  &metadata) const 
  {
    return dyablo_variant_visit<ConsState>( 
      [&](const auto& bc) -> ConsState
    {
      return bc.getBoundaryValue(policy, U, iCell_boundary, metadata);
    }, this->boundary_conditions);
  }

  template < typename Array_t, typename Policy_t>
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux( const Policy_t      &policy, 
                                  const Array_t       &U, 
                                  const CellIndex     &iCell_boundary, 
                                  const PrimState     &q_in_reconstructed,
                                  const CellMetaData  &metadata) const 
  {
    return dyablo_variant_visit<ConsState>( 
      [&](const auto& bc) -> ConsState
    {
      return bc.getBoundaryFlux(policy, U, iCell_boundary, q_in_reconstructed, metadata);
    }, this->boundary_conditions);
  }
};

} //namespace dyablo