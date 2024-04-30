#include "HydroUpdate_base.h"
#include "RiemannSolvers.h"
#include "HydroUpdate_utils.h"

#include "boundary_conditions/BoundaryConditions.h"
#include "mpi/GhostCommunicator_partial_blocks.h"

namespace dyablo {
namespace{
using CellIndex     = ForeachCell::CellIndex;
using FieldAccessor = UserData::FieldAccessor;
using offset_t      = typename CellIndex::offset_t;

enum VarIndex_gravity {IGX, IGY, IGZ};

}// namespace
}// namespace dyablo

namespace dyablo {

template< typename State_t >
class FiniteVolumePolicy_State_legacy
{
private:
  real_t gamma0;
  constexpr static int ndim = 3;
public:
  using PrimState = typename State_t::PrimState;
  using ConsState = typename State_t::ConsState;

  FiniteVolumePolicy_State_legacy( ConfigMap& configMap )
  : //ndim(configMap.getValue<int>("mesh", "ndim", 3)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4))
  {}

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getConsState( const Array_t& U, const CellIndex& iCell ) const
  {
    ConsState u;
    getConservativeState<ndim>(U, iCell, u); 
    return u;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    setConservativeState<ndim>(U, iCell, u);
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void atomic_addConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    atomic_add_ConservativeState<ndim>( U, iCell, u );
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  PrimState getPrimState( const Array_t& Q, const CellIndex& iCell ) const
  {
    PrimState q;
    getPrimitiveState<ndim>(Q, iCell, q);
    return q;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setPrimState( const Array_t& Q, const CellIndex& iCell, const PrimState& q ) const
  {
    setPrimitiveState<ndim>(Q, iCell, q);
  }

  KOKKOS_INLINE_FUNCTION
  PrimState consToPrim( const ConsState& u ) const
  {
    return dyablo::consToPrim<ndim>(u, gamma0);
  }

  KOKKOS_INLINE_FUNCTION
  ConsState primToCons( const PrimState& q ) const
  {
    return dyablo::primToCons<ndim>(q, gamma0);
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
private:
  constexpr static int ndim = 3;
  BoundaryConditions boundary_conditions;
public:
  using PrimState = typename LegacyState_t::PrimState;
  using ConsState = typename LegacyState_t::ConsState;

  FiniteVolumePolicy_BoundaryConditions_value_euler( ConfigMap& configMap )
  : boundary_conditions(configMap)
  {}

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata) const
  {
    return boundary_conditions.template getBoundaryValue<ndim, LegacyState_t>(U, iCell_boundary, metadata );
  }
};

template< typename LegacyState_t >
class FiniteVolumePolicy_Slope_minmod
{
public:
  using PrimState = typename LegacyState_t::PrimState;
  using ConsState = typename LegacyState_t::ConsState;

  FiniteVolumePolicy_Slope_minmod( ConfigMap& configMap )
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
  constexpr static int ndim = 3;
  using PrimState = typename FiniteVolumePolicy_State_t::PrimState;
  using ConsState = typename FiniteVolumePolicy_State_t::ConsState;
  using CellIndex = ForeachCell::CellIndex;

  static_assert( std::is_same_v< typename FiniteVolumePolicy_RiemannSolver_t::PrimState
                               , PrimState >, "RiemannSolver State type mismatch" );
  static_assert( std::is_same_v< typename FiniteVolumePolicy_RiemannSolver_t::ConsState
                               , ConsState >, "RiemannSolver State type mismatch" );
  static_assert( std::is_same_v< typename FiniteVolumePolicy_BoundaryConditions_t::PrimState
                               , PrimState >, "BoundaryConditions State type mismatch" );
  static_assert( std::is_same_v< typename FiniteVolumePolicy_BoundaryConditions_t::ConsState
                               , ConsState >, "BoundaryConditions State type mismatch" );

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

template< typename Policy, typename Array_t >
void clean_negative_primitive_values(const Policy& policy, const ForeachCell& foreach_cell, const Array_t& U, double smallr, double smallp)
{
  using PrimState = typename Policy::PrimState;
  using ConsState = typename Policy::ConsState;

  int negative_p_count=0;
  int negative_rho_count=0;

  foreach_cell.reduce_cell( "clean_negative_values", U.getShape(),
    KOKKOS_LAMBDA(  const ForeachCell::CellIndex& iCell, 
                    int& negative_p_count, 
                    int& negative_rho_count )
  {
    ConsState u = policy.getConsState(U, iCell);
    PrimState q = policy.consToPrim(u);
    if( q.rho < 0.0 || q.p < 0.0 )
    {
      if (q.rho < 0.0) {
        negative_rho_count++;
        q.rho = smallr;
      }
      if (q.p < 0.0) {
        negative_p_count++;
        q.p   = smallp;
      }
      ConsState u = policy.primToCons(q);
      policy.setConsState(U, iCell, u);
    }    
  }, negative_p_count, negative_rho_count);

  if( negative_rho_count > 0 )
    printf("WARNING ! Negative density detected (x%d) !!!\n", negative_rho_count);
  if( negative_p_count > 0 )
    printf("WARNING ! Negative pressure detected (x%d) !!!\n", negative_p_count);

}

/**
 * @brief Euler update algorithm
 * 
 * @tparam State the type of state to treat
 */
template<typename Policy>
class HydroUpdate_euler_no_patch : public HydroUpdate {
public:
  using PrimState = typename Policy::PrimState;
  using ConsState = typename Policy::ConsState;

public:
  HydroUpdate_euler_no_patch(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    timers(timers),
    policy(configMap),
    gravity_type(configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE)),
    well_balanced(configMap.getValue<bool>("hydro", "well_balanced", false)),
    smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
    smallp( configMap.getValue<real_t>("hydro","smallp", 1e-10) )
  {
    if (gravity_type & GRAVITY_CONSTANT) {
      gx = configMap.getValue<real_t>("gravity", "gx", 0.0);
      gy = configMap.getValue<real_t>("gravity", "gy", 0.0);
      gz = configMap.getValue<real_t>("gravity", "gz", 0.0);
    } 

    DYABLO_ASSERT_HOST_RELEASE((std::is_same_v<PrimState, PrimHydroState> || !well_balanced), 
                               "Well balanced scheme not implemented for MHD solvers yet !");
  }

  ~HydroUpdate_euler_no_patch() {}

  /**
   * @brief Solves hydro for one step using the euler method
   * 
   * @param U the input/output global array
   * @param scalar_data input scalar data
   */
  void update( UserData& U, ScalarSimulationData& scalar_data)
  {
    real_t dt = scalar_data.get<real_t>("dt");
    constexpr int ndim = Policy::ndim;

    const Policy& policy = this->policy; 
    Timers& timers = this->timers; 
    ForeachCell& foreach_cell = this->foreach_cell;

    FieldAccessor Uin = policy.getUin(U);
    FieldAccessor Uout = policy.getUout(U);
    
    timers.get("HydroUpdate_euler").start();

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    // Initializing output array 
    // TODO : remove this and copy Uin->Uout in timeloop or field creation logic
    foreach_cell.foreach_cell( "HydroUpdate_euler_no_patch::init",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell) 
    {
      ConsState uC = policy.getConsState(Uin, iCell);
      policy.setConsState(Uout, iCell, uC);
    });

    // Setting the ghosts to 0 to accumulate fluxes
    foreach_cell.foreach_ghost_cell( "HydroUpdate_euler_no_patch::resetting_ghosts",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell) 
    {
      ConsState empty_state{};
      policy.setConsState(Uout, iCell, empty_state);
    });

    // Iterate over cells
    foreach_cell.foreach_cell( "HydroUpdate_euler_no_patch::update",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell)
    {
      // Return Slope at position iCell
      auto get_slope = [&](const CellIndex &iCell, ComponentIndex3D dir) {        

        auto get_neighbor_prim_value = [&]( const CellIndex& iCell_n, const CellIndex::offset_t& off )
        {
          ConsState u {};
          // Getting left value
          int level_diff = iCell_n.level_diff();
          if (iCell_n.is_boundary())
            u = policy.getBoundaryValue(Uin, iCell_n, cellmetadata);
          else if (level_diff < 0) {
            int subcell_count = 
            foreach_smaller_neighbor<ndim>(iCell_n, off, Uin.getShape(),
              [&](const CellIndex& iCell_neigh) {
                ConsState uloc = policy.getConsState(Uin, iCell_neigh);
                u += uloc;
              });
            u /= subcell_count;
          }
          else
            u = policy.getConsState(Uin, iCell_n);

          return policy.consToPrim(u);
        };

        ConsState uC = policy.getConsState(Uin, iCell);
        const PrimState qC = policy.consToPrim( uC );
        offset_t off_m{}; off_m[dir] = -1;
        CellIndex iCell_L = iCell.getNeighbor_ghost(off_m, Uout.getShape());
        const PrimState qL = get_neighbor_prim_value(iCell_L, off_m);
        offset_t off_p{}; off_p[dir] =  1;
        CellIndex iCell_R = iCell.getNeighbor_ghost(off_p, Uout.getShape());
        const PrimState qR = get_neighbor_prim_value(iCell_R, off_p);    

        // Getting the length right and left
        constexpr real_t sizes[] = {0.75, 1.0, 1.5};
        const real_t dL = sizes[iCell_L.level_diff()+1];
        const real_t dR = sizes[iCell_R.level_diff()+1];  

        // Computing minmod slope for the direction
        PrimState slope = policy.compute_slope( qL, qC, qR, dL, dR);
        return slope;
      }; // get_slope


      auto process_dir = [&](const CellIndex &iCell, ComponentIndex3D dir) {
         // Getting centered value and slope
        ConsState uC = policy.getConsState( Uin, iCell );
        PrimState qC0 = policy.consToPrim(uC);
        PrimState slope_C = get_slope(iCell, dir);
        real_t size_C = cellmetadata.getCellSize(iCell)[dir];

        constexpr real_t dim_fac = (ndim == 2 ? 0.5 : 0.25);

        // Compute left side flux
        ConsState fluxL {};
        {
          offset_t off_m{}; 
          off_m[dir] = -1;
          const CellIndex iCell_m = iCell.getNeighbor_ghost(off_m, Uin.getShape());
          if( iCell_m.is_boundary() )
          {
            fluxL = policy.getBoundaryFlux(Uin, iCell_m, cellmetadata);
          }
          else
          {  
            int Ldiff = iCell_m.level_diff();
            if (Ldiff >= 0) 
            {       
              ConsState uL = policy.getConsState( Uin, iCell_m );
              PrimState qL0 = policy.consToPrim(uL);
              PrimState slope_L = get_slope(iCell_m, dir);
              real_t size_L = cellmetadata.getCellSize(iCell_m)[dir];

              // Reconstructing
              PrimState qL = qL0 + 0.5 * slope_L;
              PrimState qC = qC0 - 0.5 * slope_C;

              // Solving
              fluxL = policy.riemann_solver(qL, qC, dir);
              
              // Adding flux to the neighbor if it is bigger
              if (Ldiff == 1) 
              {
                ConsState du_n = fluxL * - dim_fac * dt / size_L;
                policy.atomic_addConsState(Uout, iCell_m, du_n);
              }
            } // If smaller we skip
          }
        }

        // Compute right side flux
        ConsState fluxR {};
        {      
          offset_t off_p{}; 
          off_p[dir] = 1;
          const CellIndex iCell_p = iCell.getNeighbor_ghost(off_p, Uin.getShape());
          if( iCell_p.is_boundary() )
          {
            fluxR = policy.getBoundaryFlux(Uin, iCell_p, cellmetadata);
          }
          else
          {
            int Rdiff = iCell_p.level_diff();
            if (Rdiff >= 0) 
            {
              ConsState uR = policy.getConsState( Uin, iCell_p );
              PrimState qR0 = policy.consToPrim(uR);
              PrimState slope_R = get_slope(iCell_p, dir);
              real_t size_R = cellmetadata.getCellSize(iCell_p)[dir];

              // Reconstructing
              PrimState qC = qC0 + 0.5 * slope_C;
              PrimState qR = qR0 - 0.5 * slope_R;

              // Solving
              fluxR = policy.riemann_solver(qC, qR, dir);

              // Adding flux to the neighbor if it is bigger
              if (Rdiff == 1)
              {
                ConsState du_n = fluxR * dim_fac * dt / size_R;
                policy.atomic_addConsState(Uout, iCell_p, du_n);
              }          
            }
          }
        } 

        ConsState du = (fluxL-fluxR) * dt / size_C;
        return du;
      };

      ConsState du{};
      du += process_dir(iCell, IX);
      du += process_dir(iCell, IY);
      if (ndim == 3)
        du += process_dir(iCell, IZ);
      policy.atomic_addConsState(Uout, iCell, du);
      
    });

    // Reducing the ghosts to accumulate the flux in the data arrays 
    GhostCommunicator_partial_blocks ghost_comm ( foreach_cell.get_amr_mesh().getMesh() );
    ghost_comm.reduce_ghosts( Uout );

    clean_negative_primitive_values(policy, foreach_cell, Uout, smallr, smallp);

    timers.get("HydroUpdate_euler").stop();
  }

private:
  ForeachCell& foreach_cell;
  
  Timers& timers;  
  Policy policy;

  GravityType gravity_type;
  real_t gx, gy, gz;
  bool well_balanced;
  real_t smallr, smallp;
};


template<typename LegacyState_t >
using FiniteVolumePolicy_legacy = FiniteVolumePolicy_impl<
  FiniteVolumePolicy_State_legacy<LegacyState_t>,
  FiniteVolumePolicy_RiemannSolver_legacy<LegacyState_t>,
  FiniteVolumePolicy_BoundaryConditions_value_euler<LegacyState_t>,
  FiniteVolumePolicy_Slope_minmod<LegacyState_t>
  >;


template<int ndim>
class FiniteVolumePolicy_hydro 
  : public FiniteVolumePolicy_legacy<dyablo::HydroState>
{
public:
  using FiniteVolumePolicy_legacy<dyablo::HydroState>::FiniteVolumePolicy_legacy;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::HydroUpdateFactory, 
                  dyablo::HydroUpdate_euler_no_patch<dyablo::FiniteVolumePolicy_hydro<3>>, 
                  "HydroUpdate_euler_no_patch")

