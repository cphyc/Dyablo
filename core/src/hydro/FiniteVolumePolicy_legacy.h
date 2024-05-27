#pragma once

#include "RiemannSolvers.h"
#include "foreach_cell/ForeachCell.h"
#include "hydro/FiniteVolumePolicy_Slope.h"

namespace dyablo{

namespace{
using CellIndex     = ForeachCell::CellIndex;
using CellMetaData     = ForeachCell::CellMetaData;
using FieldAccessor = UserData::FieldAccessor;
using offset_t      = typename CellIndex::offset_t;

}// namespace

template< typename State_t >
class FiniteVolumePolicy_State_legacy
{
private:
  int ndim;
  real_t gamma0;
public:
  using PrimState = typename State_t::PrimState;
  using ConsState = typename State_t::ConsState;

  FiniteVolumePolicy_State_legacy( ConfigMap& configMap )
  : ndim(configMap.getValue<int>("mesh", "ndim", 3)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4))
  {}

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getConsState( const Array_t& U, const CellIndex& iCell ) const
  {
    ConsState u;
    if(ndim==3)
      getConservativeState<3>(U, iCell, u);
    else
      getConservativeState<2>(U, iCell, u);
    return u;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    if(ndim==3)
      setConservativeState<3>(U, iCell, u);
    else
      setConservativeState<2>(U, iCell, u);
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void atomic_addConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    if(ndim==3)
      atomic_add_ConservativeState<3>( U, iCell, u );
    else
      atomic_add_ConservativeState<2>( U, iCell, u );
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  PrimState getPrimState( const Array_t& Q, const CellIndex& iCell ) const
  {
    PrimState q;
    if(ndim==3)
      getPrimitiveState<3>(Q, iCell, q);
    else
      getPrimitiveState<2>(Q, iCell, q);
    return q;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setPrimState( const Array_t& Q, const CellIndex& iCell, const PrimState& q ) const
  {
    if(ndim==3)
      setPrimitiveState<3>(Q, iCell, q);
    else
      setPrimitiveState<2>(Q, iCell, q);
  }

  KOKKOS_INLINE_FUNCTION
  PrimState consToPrim( const ConsState& u ) const
  {
    if(ndim==3)
      return dyablo::consToPrim<3>(u, gamma0);
    else
      return dyablo::consToPrim<2>(u, gamma0);
  }

  KOKKOS_INLINE_FUNCTION
  ConsState primToCons( const PrimState& q ) const
  {
    if(ndim==3)
      return dyablo::primToCons<3>(q, gamma0);
    else
      return dyablo::primToCons<2>(q, gamma0);
  }
};

template< typename LegacyState_t >
class FiniteVolumePolicy_RiemannSolver_legacy
{
public:
  RiemannParams rparams;
public:
  using PrimState = typename LegacyState_t::PrimState;
  using ConsState = typename LegacyState_t::ConsState;

  FiniteVolumePolicy_RiemannSolver_legacy( ConfigMap& configMap )
  : rparams(configMap)
  {}

  KOKKOS_INLINE_FUNCTION
  ConsState riemann_solver( PrimState qL, PrimState qR, ComponentIndex3D dir ) const
  {
    qL = swapComponents(qL, dir);
    qR = swapComponents(qR, dir);
    ConsState flux = riemann_hydro(qL, qR, rparams);
    flux = swapComponents(flux, dir);
    return flux;
  }
};

template< typename LegacyState_t >
class FiniteVolumePolicy_BoundaryConditions_value_euler
{
  public:
  FiniteVolumePolicy_BoundaryConditions_value_euler( ConfigMap& configMap )
  {}

};

template< typename FiniteVolumePolicy_State_t,
          typename FiniteVolumePolicy_RiemannSolver_t,
          typename FiniteVolumePolicy_BoundaryConditions_t,
          typename FiniteVolumePolicy_Slope_t > 
class FiniteVolumePolicy_impl : 
  public FiniteVolumePolicy_State_t,
  public FiniteVolumePolicy_RiemannSolver_t,
  public FiniteVolumePolicy_BoundaryConditions_t,
  public FiniteVolumePolicy_Slope_t
{
public:
  using PrimState = typename FiniteVolumePolicy_State_t::PrimState;
  using ConsState = typename FiniteVolumePolicy_State_t::ConsState;
  using CellIndex = ForeachCell::CellIndex;

  static_assert( std::is_same_v< typename FiniteVolumePolicy_RiemannSolver_t::PrimState
                               , PrimState >, "RiemannSolver State type mismatch" );
  static_assert( std::is_same_v< typename FiniteVolumePolicy_RiemannSolver_t::ConsState
                               , ConsState >, "RiemannSolver State type mismatch" );

  FiniteVolumePolicy_impl( ConfigMap& configMap )
  : FiniteVolumePolicy_State_t(configMap),
    FiniteVolumePolicy_RiemannSolver_t(configMap),
    FiniteVolumePolicy_BoundaryConditions_t(configMap),
    FiniteVolumePolicy_Slope_t(configMap),
    bc_min{
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING)
    },
    bc_max{
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING)
    }
  {}

  FieldAccessor getUin( UserData& U ) const
  {
    return U.getAccessor( ConsState::getFieldsInfo() );
  }

  FieldAccessor getUout( UserData& U ) const
  {
    auto fields_info_next = ConsState::getFieldsInfo();
    for( auto& p : fields_info_next )
      p.name += "_next";
    return U.getAccessor( fields_info_next );
  }

  using FiniteVolumePolicy_State_t::getConsState;
  using FiniteVolumePolicy_State_t::setConsState;
  using FiniteVolumePolicy_State_t::atomic_addConsState;
  using FiniteVolumePolicy_State_t::getPrimState;
  using FiniteVolumePolicy_State_t::setPrimState;
  using FiniteVolumePolicy_State_t::consToPrim;
  using FiniteVolumePolicy_State_t::primToCons;

  using FiniteVolumePolicy_RiemannSolver_t::riemann_solver;
  
  Kokkos::Array<BoundaryConditionType, 3> bc_min, bc_max;

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata) const 
  {
    const FiniteVolumePolicy_impl& policy = *this;

    CellIndex iCell_inside;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_inside, offset);

    auto sign = [](int x){return (x>0)-(x<0);};

    CellIndex::offset_t symmetric_offset {
      (int16_t)(-offset[IX] + sign(offset[IX])), 
      (int16_t)(-offset[IY] + sign(offset[IY])), 
      (int16_t)(-offset[IZ] + sign(offset[IZ]))
    }; 

    CellIndex iCell_sym = iCell_inside.getNeighbor(symmetric_offset);
    ConsState u_sym = policy.getConsState( U, iCell_sym );    
    ConsState res = u_sym;

    if ( (offset[IX] > 0 && bc_max[IX] == BC_REFLECTING)
      || (offset[IX] < 0 && bc_min[IX] == BC_REFLECTING) )
    {
        res.rho_u = -u_sym.rho_u;
    }
    if ( (offset[IY] > 0 && bc_max[IY] == BC_REFLECTING)
      || (offset[IY] < 0 && bc_min[IY] == BC_REFLECTING) )
    {
        res.rho_v = -u_sym.rho_v;
    }
    if ( (offset[IZ] > 0 && bc_max[IZ] == BC_REFLECTING)
      || (offset[IZ] < 0 && bc_min[IZ] == BC_REFLECTING) )
    {
        res.rho_w = -u_sym.rho_w;
    }

    return res;
  }
  
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux( const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata) const
  {
    CellIndex iCell_ref;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_ref, offset);

    const FiniteVolumePolicy_impl& policy = *this;

    ConsState u_in = policy.getConsState( U, iCell_ref );
    PrimState q_in = policy.consToPrim(u_in);

    
    bool dir_IX = offset[IX] == -1 || offset[IX] == 1;
    bool dir_IY = offset[IY] == -1 || offset[IY] == 1;
    bool dir_IZ = offset[IZ] == -1 || offset[IZ] == 1;
    DYABLO_ASSERT_KOKKOS_DEBUG( (int)dir_IX + (int)dir_IY + (int)dir_IZ == 1
                              , "offset is not compatible with getBoundaryFlux" );

    ComponentIndex3D dir = IZ;
    if( dir_IX )
      dir = IX;
    else if( dir_IY )
      dir = IY;
    else if( dir_IZ )
      dir = IZ;
    else
      DYABLO_ASSERT_KOKKOS_DEBUG(false, "Internal error! Should not happen");

    bool reflecting = (offset[dir] > 0 && bc_max[dir] == BC_REFLECTING)
                  ||  (offset[dir] < 0 && bc_min[dir] == BC_REFLECTING);
    bool absorbing  = (offset[dir] > 0 && bc_max[dir] == BC_ABSORBING)
                  ||  (offset[dir] < 0 && bc_min[dir] == BC_ABSORBING);
    
    // TODO : don't break encapsulation here
    real_t gamma0 = FiniteVolumePolicy_RiemannSolver_t::rparams.gamma0;
    real_t smallr = FiniteVolumePolicy_RiemannSolver_t::rparams.smallr;
    real_t smallp = FiniteVolumePolicy_RiemannSolver_t::rparams.smallp;
    real_t smallc = FiniteVolumePolicy_RiemannSolver_t::rparams.smallc;
    
    real_t r_in = q_in.rho;
    real_t p_in = q_in.p;
    real_t v_in[3] = {q_in.u, q_in.v, q_in.w};
    real_t v_normal = v_in[dir];

    // Left variables
    real_t rl = fmax(r_in, smallr);
    real_t pl = fmax(p_in, rl*smallp);   

    // Right variables
    real_t rr = fmax(r_in, smallr);
    real_t pr = fmax(p_in, rr*smallp);

    real_t ul=0, ur=0;
    if( reflecting )
    {
       ul = offset[dir] * v_normal;
       ur = - ul;
    }
    else if( absorbing )
    {
       ul = v_normal;
       ur = v_normal;
    }
    
    real_t ptotl = pl;
    real_t ptotr = pr;
      
    // Find the largest eigenvalues in the normal direction to the interface
    real_t cfastl = SQRT(fmax(gamma0*pl/rl,smallc*smallc));
    real_t cfastr = SQRT(fmax(gamma0*pr/rr,smallc*smallc));

    // Compute HLL wave speed
    real_t SL = fmin(ul,ur) - fmax(cfastl,cfastr);
    real_t SR = fmax(ul,ur) + fmax(cfastl,cfastr);

    // Compute lagrangian sound speed
    real_t rcl = rl*(ul-SL);
    real_t rcr = rr*(SR-ur);
      
    // Compute acoustic star state
    real_t ptotstar = (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);

    ConsHydroState flux_out {};
    if( reflecting )
    {
      flux_out.rho_u = ((dir==IX) ? ptotstar : 0);
      flux_out.rho_v = ((dir==IY) ? ptotstar : 0);
      flux_out.rho_w = ((dir==IZ) ? ptotstar : 0);
    }
    else if( absorbing )
    {
      real_t f_rho = r_in*v_normal;

      flux_out.rho = f_rho;
      flux_out.rho_u = f_rho*q_in.u + ((dir==IX) ? ptotstar : 0);
      flux_out.rho_v = f_rho*q_in.v + ((dir==IY) ? ptotstar : 0);
      flux_out.rho_w = f_rho*q_in.w + ((dir==IZ) ? ptotstar : 0);
      flux_out.e_tot = (ptotstar + u_in.e_tot) * v_normal;
    }

    return flux_out;
  }

  using FiniteVolumePolicy_Slope_t::compute_slope;
};

template<typename LegacyState_t >
using FiniteVolumePolicy_legacy = FiniteVolumePolicy_impl<
  FiniteVolumePolicy_State_legacy<LegacyState_t>,
  FiniteVolumePolicy_RiemannSolver_legacy<LegacyState_t>,
  FiniteVolumePolicy_BoundaryConditions_value_euler<LegacyState_t>,
  FiniteVolumePolicy_Slope_dynamic<LegacyState_t>
  >;


} //namespace dyablo