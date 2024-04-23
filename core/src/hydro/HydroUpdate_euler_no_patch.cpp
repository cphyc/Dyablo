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
    if( this->ndim == 2 )
      setConservativeState<2>(U, iCell, u);
    else if( this->ndim == 3 )
      setConservativeState<3>(U, iCell, u);

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
private:
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
class FiniteVolumePolicy_BoundaryConditions_legacy
{
private:
  constexpr static int ndim = 3;
  BoundaryConditions boundary_conditions;
public:
  using PrimState = typename LegacyState_t::PrimState;
  using ConsState = typename LegacyState_t::ConsState;

  FiniteVolumePolicy_BoundaryConditions_legacy( ConfigMap& configMap )
  : boundary_conditions(configMap)
  {}

  KOKKOS_INLINE_FUNCTION
  bool boundaryMode_overrideFlux() const
  {
    return false;
  }
  
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata) const
  {
    return boundary_conditions.template getBoundaryValue<ndim, LegacyState_t>(U, iCell_boundary, metadata );
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux( const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata) const
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( boundaryMode_overrideFlux(), "called getBoundaryFlux() but Boundary mode is not flux");
    return ConsState{};
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
    FiniteVolumePolicy_Slope_t(configMap)
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
  
  using FiniteVolumePolicy_BoundaryConditions_t::boundaryMode_overrideFlux;
  using FiniteVolumePolicy_BoundaryConditions_t::getBoundaryValue;
  using FiniteVolumePolicy_BoundaryConditions_t::getBoundaryFlux;

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
          if( iCell_m.is_boundary() && policy.boundaryMode_overrideFlux() )
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
          if( iCell_p.is_boundary() && policy.boundaryMode_overrideFlux() )
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
  FiniteVolumePolicy_BoundaryConditions_legacy<LegacyState_t>,
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

