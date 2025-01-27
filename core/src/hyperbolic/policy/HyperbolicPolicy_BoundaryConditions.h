#pragma once

#include "HyperbolicPolicy_base.h"

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

} //namespace dyablo