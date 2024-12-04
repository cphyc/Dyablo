#pragma once

#include "hydro/FiniteVolumePolicy_base.h"
#include "hydro/FiniteVolumePolicy_tools.h"

namespace dyablo{

template< typename LegacyState_t >
class FiniteVolumePolicy_BoundaryConditions_Hydro
{
private:
  using FiniteVolumePolicy_State = LegacyState_t;
  using CellIndex     = ForeachCell::CellIndex;
  using CellMetaData  = ForeachCell::CellMetaData;
  using offset_t      = CellIndex::offset_t;

  Kokkos::Array<BoundaryConditionType, 3> bc_min, bc_max;
  struct Rparams {
    real_t gamma0;
    real_t smallr;
    real_t smallp;
    real_t smallc;
  } rparams;

public:
  using PrimState = typename FiniteVolumePolicy_State::PrimState;
  using ConsState = typename FiniteVolumePolicy_State::ConsState;

  static std::string name() {return "default";}

  FiniteVolumePolicy_BoundaryConditions_Hydro( ConfigMap& configMap )
  : bc_min{
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING)
    },
    bc_max{
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING)
    },
    rparams( 
    {
      configMap.getValue<real_t>("hydro", "gamma0", 1.4),
      configMap.getValue<real_t>("hydro", "smallr", 1e-10),
      configMap.getValue<real_t>("hydro", "smallp", 1e-10),
      configMap.getValue<real_t>("hydro", "smallc", 1e-10)
    })
  {}

  template < typename Array_t, typename Policy_t, typename ScalarData_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue_impl( const Policy_t      &policy, 
                                   const Array_t       &U, 
                                   const CellIndex     &iCell_boundary, 
                                   const CellMetaData  &metadata, 
                                   const ScalarData_t  &scalar_data) const 
  {
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

  template < typename Array_t, typename Policy_t, typename ScalarData_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux_impl( const Policy_t      &policy, 
                                  const Array_t       &U, 
                                  const CellIndex     &iCell_boundary, 
                                  const CellMetaData  &metadata, 
                                  const ScalarData_t  &scalar_data) const 
  {
    CellIndex iCell_ref;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_ref, offset);

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

    real_t v_in[3] = {q_in.u, q_in.v, q_in.w};
    real_t v_normal = v_in[dir];

    ConsState flux_out {};
    /**
     * In the reflecting case, the values in the "ghosts" are supposed to be reflecting the
     * ones inside the domain, hence reconstruction yields u_norm = 0 at the boundary, simplifying
     * the calculation of the flux to only the pressure gradient term in the flux.
     */
    if( reflecting )
    {
      flux_out.rho_u = ((dir==IX) ? q_in.p : 0);
      flux_out.rho_v = ((dir==IY) ? q_in.p : 0);
      flux_out.rho_w = ((dir==IZ) ? q_in.p : 0);
    }
    /**
     * In the absorbing case, the values in the ghosts are supposed to be interpolated from the ones 
     * inside the domain to provide a null gradient through the boundary. Hence we can take the
     * reconstructed value at the boundary as the riemann-problem solution.
     */
    else if( absorbing )
    {
      real_t f_rho = q_in.rho*v_normal;

      flux_out.rho = f_rho;
      flux_out.rho_u = f_rho*q_in.u + ((dir==IX) ? q_in.p : 0);
      flux_out.rho_v = f_rho*q_in.v + ((dir==IY) ? q_in.p : 0);
      flux_out.rho_w = f_rho*q_in.w + ((dir==IZ) ? q_in.p : 0);
      flux_out.e_tot = (q_in.p + u_in.e_tot) * v_normal;
    }

    return flux_out;
  }
};


template<typename LegacyState_t>
class FiniteVolumePolicy_BoundaryConditions_DoubleMach
{
private:
  using FiniteVolumePolicy_State = LegacyState_t;
  using CellIndex     = ForeachCell::CellIndex;
  using CellMetaData  = ForeachCell::CellMetaData;
  using offset_t      = CellIndex::offset_t;

  struct Rparams {
    real_t gamma0;
    real_t x0;
    real_t alpha;
    real_t post_rho, post_vel, post_p;
    real_t pre_rho, pre_vel, pre_p;
  } rparams;

public:
  using PrimState = typename FiniteVolumePolicy_State::PrimState;
  using ConsState = typename FiniteVolumePolicy_State::ConsState;

  static std::string name() {return "double_mach";}

  FiniteVolumePolicy_BoundaryConditions_DoubleMach( ConfigMap& configMap )
  : 
    rparams( 
    {
      configMap.getValue<real_t>("hydro", "gamma0", 1.4),
      configMap.getValue<real_t>("double_mach", "x0", 0.666666667),
      configMap.getValue<real_t>("double_mach", "alpha", 1.0471975511965976),
      configMap.getValue<real_t>("double_mach", "rho_left", 8.0),
      configMap.getValue<real_t>("double_mach", "vel_left", 8.25),
      configMap.getValue<real_t>("double_mach", "p_left", 116.5),
      configMap.getValue<real_t>("double_mach", "rho_right", 1.4),
      configMap.getValue<real_t>("double_mach", "vel_right", 0.0),
      configMap.getValue<real_t>("double_mach", "p_right", 1.0)

    })
  {}
  
  template < typename Array_t, typename Policy_t, typename ScalarData_t  >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux_impl( const Policy_t      &policy, 
                                  const Array_t       &U, 
                                  const CellIndex     &iCell_boundary, 
                                  const CellMetaData  &metadata, 
                                  const ScalarData_t  &scalar_data) const 
{
    ConsState out_val = getBoundaryValue_impl(policy, U, iCell_boundary, metadata, scalar_data, true);
    CellIndex iCell_inside;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_inside, offset);
    auto in_val = policy.getConsState(U, iCell_inside);

    bool left = (offset[IX] < 0 || offset[IY] < 0); 
    ComponentIndex3D dir = (offset[IX] != 0 ? IX : IY);

    PrimState qL, qR;
    if (left) {
      qL = policy.consToPrim(out_val);
      qR = policy.consToPrim(in_val);
    }
    else {
      qL = policy.consToPrim(in_val);
      qR = policy.consToPrim(out_val);
    }

    return policy.riemann_solver(qL, qR, dir);
  }

  template < typename Array_t, typename Policy_t, typename ScalarData_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue_impl( const Policy_t      &policy, 
                                   const Array_t       &U, 
                                   const CellIndex     &iCell_boundary, 
                                   const CellMetaData  &metadata, 
                                   const ScalarData_t  &scalar_data,
                                   bool for_flux=false) const 
  {
    CellIndex iCell_inside;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_inside, offset);
    ConsState res;
    auto sign = [](int x){return (x>0)-(x<0);};

    CellIndex::offset_t symmetric_offset {
      (int16_t)(-offset[IX] + sign(offset[IX])), 
      (int16_t)(-offset[IY] + sign(offset[IY])), 
      (int16_t)(-offset[IZ] + sign(offset[IZ]))
    }; 

    // X right boundary -> Absorbing
    if ( offset[IX] > 0 ) {
      CellIndex iCell_sym = iCell_inside.getNeighbor(symmetric_offset);
      res = policy.getConsState( U, iCell_sym );    
    }
    // X left boundary -> Post-shock state
    else if ( offset[IX] < 0 ) {
      PrimState q;
      q.rho = rparams.post_rho;
      q.u   = rparams.post_vel*sin(rparams.alpha);
      q.v   = -rparams.post_vel*cos(rparams.alpha);
      q.p   = rparams.post_p;
      res = policy.primToCons(q);
    }
    // Y left boundary -> Postshock state before x0 else Reflection
    else if (offset[IY] < 0) {
      real_t x = metadata.getCellCenter(iCell_inside)[IX];
      if (x < rparams.x0) {
        PrimState q;
        q.rho = rparams.post_rho;
        q.u   = rparams.post_vel*sin(rparams.alpha);
        q.v   = -rparams.post_vel*cos(rparams.alpha);
        q.p   = rparams.post_p;
        res = policy.primToCons(q);
      }
      else {    
        CellIndex iCell_sym = iCell_inside.getNeighbor(symmetric_offset);
        ConsState u_sym = policy.getConsState( U, iCell_sym );    
        res = u_sym;
        res.rho_v *= -1.0;
      }
    }
    // Y right boundary -> Pre/post shock state depending on time 
    else if (offset[IY] > 0) {
      real_t xs = 10.0*scalar_data.sim_time/sin(rparams.alpha) + 1.0/6.0 + 1.0/tan(rparams.alpha);
      real_t x = metadata.getCellCenter(iCell_inside)[IX];
      PrimState q;

      // Post-shock
      if (x < xs) {  
        q.rho = rparams.post_rho;
        q.u   = rparams.post_vel*sin(rparams.alpha);
        q.v   = -rparams.post_vel*cos(rparams.alpha);
        q.p   = rparams.post_p;
        res = policy.primToCons(q);
      }
      // Pre-shock
      else {
        q.rho = rparams.pre_rho;
        q.u   = rparams.pre_vel;
        q.v   = rparams.pre_vel;
        q.p   = rparams.pre_p;
        res = policy.primToCons(q);
      }
    }

    return res;
  }
};

/***
 * Dynamic Implementation for FiniteVolume_BoundaryConditions
 * Merge multiple implementations of FiniteVolume_BoundaryConditions and 
 * choose which version to use at runtime
 * @tparam LegacyState_t input/output state used for the boundaries values
 * @tparam Ts... FiniteVolume_BoundaryConditions types to choose from
 ***/
template< typename LegacyState_t, typename... Ts >
class FiniteVolumePolicy_BoundaryConditions_dynamic_impl
{
  std::tuple<Ts...> boundary_conditions;
  size_t selected_bc;

  using CellIndex = ForeachCell::CellIndex;
  using CellMetaData  = ForeachCell::CellMetaData;
public:
  using ConsState = typename LegacyState_t::ConsState;

  FiniteVolumePolicy_BoundaryConditions_dynamic_impl( ConfigMap& configMap )
  : boundary_conditions(Ts(configMap)...)
  {
    std::string bc_name = configMap.getValue<std::string>("hydro", "boundary_conditions_class", "default");

    this->selected_bc = tuple_find_name( bc_name, boundary_conditions );
  }

public:
  template < typename Array_t, typename Policy_t, typename ScalarData_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux( const Policy_t& policy, const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata, const ScalarData_t &scalar_data ) const
  {
    return tuple_apply_nth( selected_bc, [&](const auto& s){return s.getBoundaryFlux_impl(policy,U,iCell_boundary,metadata,scalar_data);},boundary_conditions );
  }
  
  template < typename Array_t, typename Policy_t, typename ScalarData_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Policy_t& policy, const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata, const ScalarData_t &scalar_data ) const 
  {
    return tuple_apply_nth( selected_bc, [&](const auto& s){return s.getBoundaryValue_impl(policy,U,iCell_boundary,metadata,scalar_data);},boundary_conditions );
  }
};

template<typename LegacyState_t>
using FiniteVolumePolicy_BoundaryConditions_dynamic = FiniteVolumePolicy_BoundaryConditions_dynamic_impl< LegacyState_t,
  FiniteVolumePolicy_BoundaryConditions_Hydro<LegacyState_t>,
  FiniteVolumePolicy_BoundaryConditions_DoubleMach<LegacyState_t>
>;

} //namespace dyablo