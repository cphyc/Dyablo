#pragma once

#include "HyperbolicUpdate_base.h"
#include "HyperbolicUpdate_utils.h"
#include "mpi/GhostCommunicator_partial_blocks.h"

namespace dyablo {
namespace{
using CellIndex     = ForeachCell::CellIndex;
using FieldAccessor = UserData::FieldAccessor;
using offset_t      = typename CellIndex::offset_t;
using PatchArray = ForeachCell::CellArray_patch;

}// namespace
}// namespace dyablo

namespace dyablo {

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
 * @brief Euler update algorithm with WENO reconstruction
 * 
 * @tparam State the type of state to treat
 */
template<typename Policy>
class Hyperbolic_euler_WENO : public HyperbolicUpdate {
  static_assert( is_HyperbolicPolicy_v<Policy>,
  "Policy must be wrapped in HyperbolicPolicy_base");

public:
  using PrimState = typename Policy::PrimState;
  using ConsState = typename Policy::ConsState;

public:
  Hyperbolic_euler_WENO(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    timers(timers),
    policy(configMap),
    ndim(configMap.getValue<int>("mesh", "ndim", 3)),
    smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
    smallp( configMap.getValue<real_t>("hydro","smallp", 1e-10) )
  { }

  /**
   * @brief Solves hydro for one step using the euler method
   * 
   * @param U the input/output global array
   * @param scalar_data input scalar data
   */
  void update( UserData& U, ScalarSimulationData& scalar_data)
  {
    real_t dt = scalar_data.get<real_t>("dt");
    int ndim = this->ndim;

    const Policy& policy = this->policy; 
    Timers& timers = this->timers; 
    ForeachCell& foreach_cell = this->foreach_cell;

    FieldAccessor Uin = policy.getUin(U);
    FieldAccessor Uout = policy.getUout(U);
    
    timers.get("HyperbolicUpdate_euler").start();

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
    auto policy_scalar_data = policy.getScalarData( scalar_data );

    // Initializing output array 
    // TODO : remove this and copy Uin->Uout in timeloop or field creation logic
    foreach_cell.foreach_cell( "Hyperbolic_euler_WENO::init",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell) 
    {
      ConsState uC = policy.getConsState(Uin, iCell);
      policy.setConsState(Uout, iCell, uC);
    });

    // Setting the ghosts to 0 to accumulate fluxes
    foreach_cell.foreach_ghost_cell( "Hyperbolic_euler_WENO::resetting_ghosts",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell) 
    {
      ConsState empty_state{};
      policy.setConsState(Uout, iCell, empty_state);
    });

    auto fm_prim = PrimState::getFieldManager().get_id2index();
    
    // 3 ghosts for CWENO-4
    enum WenoVars {
      IWEIGHT
    };
    FieldManager weno_vars {{IWEIGHT}};

    int nb_ghosts = 3;
    PatchArray::Ref Qpatch_ = foreach_cell.reserve_patch_tmp("Qpatch", nb_ghosts, nb_ghosts, (ndim == 3)?nb_ghosts:0, fm_prim, State_traits<PrimState>::nvars);
    PatchArray::Ref Wpatch_ = foreach_cell.reserve_patch_tmp("weights", nb_ghosts-2, nb_ghosts-2, (ndim == 3)?nb_ghosts-2:0, weno_vars.get_id2index(), 1);

    foreach_cell.foreach_patch( "Hyperbolic_euler_WENO::update", 
      PATCH_LAMBDA( const ForeachCell::Patch& patch )
    {
      PatchArray Qpatch = patch.allocate_tmp(Qpatch_);
      PatchArray Wpatch = patch.allocate_tmp(Wpatch_);

      // 1. Computing weights
      patch.foreach_cell( Wpatch,
        CELL_LAMBDA( const CellIndex &iCell_W ) 
      {

      });

      patch.foreach_cell( Qpatch, 
        CELL_LAMBDA( const CellIndex& iCell_Qpatch )
      {
       
      });

    });
      

    // Reducing the ghosts to accumulate the flux in the data arrays 
    int ghost_count = 1;
    GhostCommunicator_partial_blocks ghost_comm ( 
      foreach_cell.get_amr_mesh().getMesh(),
      Uout.getShape(),
      ghost_count );
    ghost_comm.reduce_ghosts( Uout );

    clean_negative_primitive_values(policy, foreach_cell, Uout, smallr, smallp);

    timers.get("HyperbolicUpdate_euler").stop();
  }

private:
  ForeachCell& foreach_cell;
  
  Timers& timers;  
  Policy policy;

  int ndim;
  real_t smallr, smallp;
};

} // namespace dyablo

