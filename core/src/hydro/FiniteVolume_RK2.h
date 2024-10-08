#pragma once

#include "HydroUpdate_base.h"
#include "HydroUpdate_utils.h"
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
 * @brief Strong Stability Preserving RK2 algorithm
 * 
 * @tparam The type of policy to apply
 */
template<typename Policy>
class FiniteVolume_RK2 : public HydroUpdate {
  static_assert( is_FiniteVolumePolicy_v<Policy>,
  "Policy must be wrapped in FiniteVolumePolicy_base");

public:
  using PrimState = typename Policy::PrimState;
  using ConsState = typename Policy::ConsState;

public:
  FiniteVolume_RK2(
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
   * @brief Solves hydro using the SSP-RK2 scheme
   * 
   * @param U the input/output global array
   * @param scalar_data input scalar data
   */
  void update( UserData& U, ScalarSimulationData& scalar_data ) 
  {
    real_t dt = scalar_data.get<real_t>("dt");

    timers.get("HydroUpdate_RK2").start();
    FieldAccessor Uin = policy.getUin(U);
    FieldAccessor Uout = policy.getUout(U);

    const Policy& policy = this->policy; 
    auto fm_cons = Policy::ConsState::getFieldManager();
    auto Ustar = foreach_cell.allocate_ghosted_array("U*", fm_cons);

    // Performing two steps
    update_once(Uin, Ustar, dt, true);
    update_once(Ustar, Uout, dt, false);

    // And correcting
    foreach_cell.foreach_ghost_cell( "FiniteVolume_RK2::resetting_ghosts",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell) 
    {
      auto u0 = policy.getConsState(Uin,  iCell);
      auto u1 = policy.getConsState(Uout, iCell);
      
      policy.setConsState(Uout, iCell, 0.5*(u0+u1));
    });

    // Cleaning all negative values out of the solution
    clean_negative_primitive_values(policy, foreach_cell, Uout, smallr, smallp);

    timers.get("HydroUpdate_RK2").stop();

  }
  /**
   * @brief Solves hydro for one step using the RK2 method
   * 
   * @param Uin the input array
   * @param Uout the output array
   * @param dt the time step
   * @param sync_Uout should Uout be resync via MPI at the end of the step
   */
  template < typename ArrayIn_t,
             typename ArrayOut_t >
  void update_once( ArrayIn_t& Uin, ArrayOut_t& Uout, real_t dt, bool sync_Uout )
  {
    int ndim = this->ndim;

    const Policy& policy = this->policy; 
    ForeachCell& foreach_cell = this->foreach_cell;

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    // Initializing output array 
    // TODO : remove this and copy Uin->Uout in timeloop or field creation logic
    foreach_cell.foreach_cell( "FiniteVolume_RK2::init",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell) 
    {
      ConsState uC = policy.getConsState(Uin, iCell);
      policy.setConsState(Uout, iCell, uC);
    });

    // Setting the ghosts to 0 to accumulate fluxes
    foreach_cell.foreach_ghost_cell( "FiniteVolume_RK2::resetting_ghosts",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell) 
    {
      ConsState empty_state{};
      policy.setConsState(Uout, iCell, empty_state);
    });

    auto fm_prim = PrimState::getFieldManager().get_id2index();
    PatchArray::Ref Qpatch_ = foreach_cell.reserve_patch_tmp("Qpatch", 2, 2, (ndim == 3)?2:0, fm_prim, State_traits<PrimState>::nvars);

    foreach_cell.foreach_patch( "FiniteVolume_RK2::update", 
      PATCH_LAMBDA( const ForeachCell::Patch& patch )
    {
      PatchArray Qpatch = patch.allocate_tmp(Qpatch_);

      patch.foreach_cell( Qpatch, 
        CELL_LAMBDA( const CellIndex& iCell_Qpatch )
      {
        CellIndex iCell_Uin = Uin.getShape().convert_index_ghost(iCell_Qpatch);
        int level_diff = iCell_Uin.level_diff();
        ConsState u = {};
        if (iCell_Uin.is_boundary())
          u = policy.getBoundaryValue(Uin, iCell_Uin, cellmetadata);
        else if (level_diff < 0) {
          int subcell_count = 
          foreach_sibling(ndim, iCell_Uin, Uin.getShape(),
            [&](const CellIndex& iCell_neigh) {
              ConsState uloc = policy.getConsState(Uin, iCell_neigh);
              u += uloc;
            });
          u /= subcell_count;
        }
        else
          u = policy.getConsState(Uin, iCell_Uin);
        
        const PrimState q = policy.consToPrim( u );
        policy.setPrimState( Qpatch, iCell_Qpatch, q );
      });

      patch.foreach_cell( Uout.getShape(), 
        CELL_LAMBDA( const CellIndex& iCell_Uout )
      {
        // Return Slope at position iCell
        auto get_slope = [&](const CellIndex &iCell_Uin, const CellIndex &iCell_Qpatch, ComponentIndex3D dir) 
        {        
          const PrimState qC = policy.getPrimState(Qpatch, iCell_Qpatch );
          offset_t off_m{}; off_m[dir] = -1;
          const PrimState qL = policy.getPrimState(Qpatch, iCell_Qpatch + off_m); 
          offset_t off_p{}; off_p[dir] =  1;
          const PrimState qR = policy.getPrimState(Qpatch, iCell_Qpatch + off_p); 
        
          //!\ Neighbor cells in Qpatch are averaged cells -> size iCell_L != size iCell_Qpatch_L
          CellIndex iCell_L = iCell_Uin.getNeighbor_ghost(off_m, Uout.getShape());
          CellIndex iCell_R = iCell_Uin.getNeighbor_ghost(off_p, Uout.getShape());   

          // Getting the length right and left
          // Smaller -> use averaged same-size cell -> 1*dx
          // Bigger -> same-size cell in Qpatch has vame value as actual bigger cell -> dx/2 + dx             
          constexpr real_t sizes[] = {1.0, 1.0, 1.5}; 
          const real_t dL = sizes[iCell_L.level_diff()+1];
          const real_t dR = sizes[iCell_R.level_diff()+1];  

          // Computing minmod slope for the direction
          PrimState slope = policy.compute_slope( qL, qC, qR, dL, dR);
          return slope;
        }; // get_slope


        auto process_dir = [&](const CellIndex &iCell_Uin, const CellIndex &iCell_Qpatch, ComponentIndex3D dir) {
          // Getting centered value and slope
          PrimState qC0 = policy.getPrimState( Qpatch, iCell_Qpatch );
          PrimState slope_C = get_slope(iCell_Uin, iCell_Qpatch, dir);
          real_t size_C = cellmetadata.getCellSize(iCell_Uin)[dir];

          real_t dim_fac = (ndim == 2 ? 0.5 : 0.25);
  
          // Compute left side flux
          ConsState fluxL {};
          {
            offset_t off_m{}; 
            off_m[dir] = -1;
            const CellIndex iCell_Uin_m = iCell_Uin.getNeighbor_ghost(off_m, Uin.getShape());
            if( iCell_Uin_m.is_boundary() )
            {
              fluxL = policy.getBoundaryFlux(Uin, iCell_Uin_m, cellmetadata);
            }
            else
            {  
              int Ldiff = iCell_Uin_m.level_diff();
              if (Ldiff >= 0) 
              {
                CellIndex iCell_Qpatch_m = iCell_Qpatch + off_m;
                PrimState qL0 = policy.getPrimState( Qpatch, iCell_Qpatch_m );
                PrimState slope_L = get_slope(iCell_Uin_m, iCell_Qpatch_m, dir);
                real_t size_L = cellmetadata.getCellSize(iCell_Uin_m)[dir];

                // Reconstructing
                PrimState qL = qL0 + 0.5 * slope_L;
                PrimState qC = qC0 - 0.5 * slope_C;

                // Solving
                fluxL = policy.riemann_solver(qL, qC, dir);
                
                // Adding flux to the neighbor if it is bigger
                if (Ldiff == 1) 
                {
                  ConsState du_n = fluxL * - dim_fac * dt / size_L;
                  policy.atomic_addConsState(Uout, iCell_Uin_m, du_n);
                }
              } // If smaller we skip
            }
          }

          // Compute right side flux
          ConsState fluxR {};
          {      
            offset_t off_p{}; 
            off_p[dir] = 1;
            const CellIndex iCell_Uin_p = iCell_Uin.getNeighbor_ghost(off_p, Uin.getShape());
            if( iCell_Uin_p.is_boundary() )
            {
              fluxR = policy.getBoundaryFlux(Uin, iCell_Uin_p, cellmetadata);
            }
            else
            {
              int Rdiff = iCell_Uin_p.level_diff();
              if (Rdiff >= 0) 
              {
                CellIndex iCell_Qpatch_p = iCell_Qpatch + off_p;
                PrimState qR0 = policy.getPrimState( Qpatch, iCell_Qpatch_p );
                PrimState slope_R = get_slope(iCell_Uin_p, iCell_Qpatch_p, dir);
                real_t size_R = cellmetadata.getCellSize(iCell_Uin_p)[dir];

                // Reconstructing
                PrimState qC = qC0 + 0.5 * slope_C;
                PrimState qR = qR0 - 0.5 * slope_R;

                // Solving
                fluxR = policy.riemann_solver(qC, qR, dir);

                // Adding flux to the neighbor if it is bigger
                if (Rdiff == 1)
                {
                  ConsState du_n = fluxR * dim_fac * dt / size_R;
                  policy.atomic_addConsState(Uout, iCell_Uin_p, du_n);
                }          
              }
            }
          } 

          ConsState du = (fluxL-fluxR) * dt / size_C;
          return du;
        };


        CellIndex iCell_Qpatch = Qpatch.getShape().convert_index(iCell_Uout);

        ConsState du{};
        du += process_dir(iCell_Uout, iCell_Qpatch, IX);
        du += process_dir(iCell_Uout, iCell_Qpatch, IY);
        if (ndim == 3)
          du += process_dir(iCell_Uout, iCell_Qpatch, IZ);

        policy.atomic_addConsState(Uout, iCell_Uout, du);
      });
    });
      

    // Reducing the ghosts to accumulate the flux in the data arrays 
    int ghost_count = 2;
    GhostCommunicator_partial_blocks ghost_comm ( 
      foreach_cell.get_amr_mesh().getMesh(),
      Uout.getShape(),
      ghost_count );
    ghost_comm.reduce_ghosts( Uout );

    if (sync_Uout)
      ghost_comm.exchange_ghosts( Uout );
  }

private:
  ForeachCell& foreach_cell;
  
  Timers& timers;  
  Policy policy;

  int ndim;
  real_t smallr, smallp;
};

} // namespace dyablo

