#pragma once

#include "hyperbolic/HyperbolicUpdate_base.h"

#include <type_traits>

#include "hyperbolic/policy/HyperbolicPolicy_base.h"
#include "mpi/GhostCommunicator_partial_blocks.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "states/State_Ops.h"

namespace dyablo {
namespace{
using CellIndex     = ForeachCell::CellIndex;
using FieldAccessor = UserData::FieldAccessor;
using offset_t      = typename CellIndex::offset_t;
using PatchArray = ForeachCell::CellArray_patch;

}// namespace
}// namespace dyablo

namespace dyablo {

namespace impl{
namespace {
template <typename T, typename = int, typename=int>
struct HasDensityAndPressure : std::false_type { };

template <typename T>
struct HasDensityAndPressure <T, decltype((void) T::rho, 0), decltype((void) T::p, 0)> : std::true_type { };
}
}

/**
 * @brief Euler update algorithm
 * 
 * @tparam State the type of state to treat
 */
template<typename Policy>
class Hyperbolic_euler : public HyperbolicUpdate {
  static_assert( is_HyperbolicPolicy_v<Policy>,
  "Policy must be wrapped in HyperbolicPolicy_base");

public:
  using PrimState = typename Policy::PrimState;
  using ConsState = typename Policy::ConsState;

public:
  Hyperbolic_euler(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    timers(timers),
    policy_params(Policy::getParams(configMap)),
    ndim(configMap.getValue<int>("mesh", "ndim", 3)),
    smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
    smallp( configMap.getValue<real_t>("hydro","smallp", 1e-10) ),
    slope_enabled( configMap.getValue<bool>("hydro","slope_enabled", true) )
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
    bool slope_enabled = this->slope_enabled;

    const Policy policy( this->policy_params, scalar_data ); 
    Timers& timers = this->timers; 
    ForeachCell& foreach_cell = this->foreach_cell;

    FieldAccessor Uin = policy.getUin(U);
    FieldAccessor Uout = policy.getUout(U);
    
    timers.get("HyperbolicUpdate_euler").start();

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    ForeachCell::SearchMode_neighbor search_neighbor( this->foreach_cell.get_amr_mesh().getLightOctree(), ForeachCell::SearchMode_neighbor::ORIGIN );
    ForeachCell::SearchMode_local search_local( ForeachCell::SearchMode_local::ASSERT );

    // Initializing output array 
    // TODO : remove this and copy Uin->Uout in timeloop or field creation logic
    foreach_cell.foreach_cell( "Hyperbolic_euler::init",
      Uout.getShape(),
      KOKKOS_LAMBDA(const CellIndex &iCell) 
    {
      ConsState uC = policy.getConsState(Uin, iCell);
      policy.setConsState(Uout, iCell, uC);
    });

    // Setting the ghosts to 0 to accumulate fluxes
    foreach_cell.foreach_ghost_cell( "Hyperbolic_euler::resetting_ghosts",
      Uout.getShape(),
      KOKKOS_LAMBDA(const CellIndex &iCell) 
    {
      ConsState empty_state{};
      policy.setConsState(Uout, iCell, empty_state);
    });

    int nb_ghosts = slope_enabled ? 2 : 1;
    PatchArray::Ref Qpatch_ = foreach_cell.reserve_patch_tmp("Qpatch", nb_ghosts, nb_ghosts, (ndim == 3)?nb_ghosts:0, State_traits<PrimState>::nvars);

    foreach_cell.foreach_patch( "Hyperbolic_euler::update", 
      PATCH_LAMBDA( const ForeachCell::Patch& patch )
    {
      PatchArray Qpatch = patch.allocate_tmp(Qpatch_);

      patch.foreach_cell( Qpatch, 
        CELL_LAMBDA( const CellIndex& iCell_Qpatch )
      {
        ForeachCell::SearchMode_neighbor search_neighbor_origin( cellmetadata.getLightOctree(), ForeachCell::SearchMode_neighbor::ORIGIN );
        auto shape = Uin.getShape();
        CellIndex::Status iCell_Uin_status = shape.convert_index_status(iCell_Qpatch, search_neighbor_origin);
        int level_diff = CellIndex::level_diff(iCell_Uin_status);
        ConsState u = {};
        if (CellIndex::is_boundary(iCell_Uin_status))
        {
          CellIndex iCell_Uin = shape.convert_index<CellIndex::BOUNDARY>( iCell_Qpatch, search_neighbor_origin );
          u = policy.getBoundaryValue(Uin, iCell_Uin, cellmetadata);
        }
        else if (level_diff < 0) { 
          CellIndex iCell_Uin = shape.convert_index<CellIndex::SMALLER>( iCell_Qpatch, search_neighbor_origin );
          int subcell_count = 
          foreach_sibling_gathered(ndim, iCell_Uin,
            [&](const CellIndex& iCell_neigh) {
              ConsState uloc = policy.getConsState(Uin, iCell_neigh);
              u += uloc;
            });
          u /= subcell_count;
        }
        else
        {
          using StatusFilter_t = CellIndex::StatusFilter<CellIndex::LOCAL_TO_BLOCK, CellIndex::SAME_SIZE, CellIndex::BIGGER>;
          CellIndex iCell_Uin = shape.convert_index<StatusFilter_t>( iCell_Qpatch, search_neighbor_origin, iCell_Uin_status );
          u = policy.getConsState(Uin, iCell_Uin);
        }
        
        const PrimState q = policy.consToPrim( u );
        policy.setPrimState( Qpatch, iCell_Qpatch, q );
      });

      patch.foreach_cell( Uout.getShape(), 
        KOKKOS_LAMBDA( const CellIndex& iCell_Uout )
      {
        ForeachCell::SearchMode_neighbor search_neighbor( cellmetadata.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );
        ForeachCell::SearchMode_local search_local( ForeachCell::SearchMode_local::INVALID );

        // Return Slope at position iCell
        auto get_slope = [&](int level_diff_L, int level_diff_R, const CellIndex &iCell_Qpatch, ComponentIndex3D dir) 
        {
          if(!slope_enabled)
            return PrimState{};

          const PrimState qC = policy.getPrimState(Qpatch, iCell_Qpatch );
          const PrimState qL = policy.getPrimState(Qpatch, iCell_Qpatch.getNeighbor( -(dir==IX), -(dir==IY), -(dir==IZ), search_local, CellIndex::LOCAL_TO_BLOCK )); 
          const PrimState qR = policy.getPrimState(Qpatch, iCell_Qpatch.getNeighbor(  (dir==IX),  (dir==IY),  (dir==IZ), search_local, CellIndex::LOCAL_TO_BLOCK ));
        
          // Getting the length right and left
          // Smaller -> use averaged same-size cell -> 1*dx
          // Bigger -> same-size cell in Qpatch has vame value as actual bigger cell -> dx/2 + dx             
          const real_t dL = level_diff_L > 0 ? 1.5 : 1;
          const real_t dR = level_diff_R > 0 ? 1.5 : 1;

          // Computing minmod slope for the direction
          PrimState slope = policy.compute_slope( qL, qC, qR, dL, dR);
          return slope;
        }; // get_slope

        CellIndex iCell_Qpatch = Qpatch.getShape().convert_index(iCell_Uout, search_local, CellIndex::Status::LOCAL_TO_BLOCK);
        PrimState qC0 = policy.getPrimState( Qpatch, iCell_Qpatch );
        auto size_C0 = cellmetadata.getCellSize(iCell_Uout);

        auto process_dir = [&](const CellIndex &iCell_Uin, const CellIndex &iCell_Qpatch, ComponentIndex3D dir) {
          // Getting centered value and slope 
          auto iCell_Uin_m_status = iCell_Uin.getNeighborStatus(-(dir==IX), -(dir==IY), -(dir==IZ), search_neighbor);
          auto iCell_Uin_p_status = iCell_Uin.getNeighborStatus( (dir==IX),  (dir==IY),  (dir==IZ), search_neighbor);   
          int Ldiff = CellIndex::level_diff(iCell_Uin_m_status);
          int Rdiff = CellIndex::level_diff(iCell_Uin_p_status);

          PrimState slope_C = get_slope(Ldiff, Rdiff, iCell_Qpatch, dir);
          real_t size_C = size_C0[dir];

          real_t dim_fac = (ndim == 2 ? 0.5 : 0.25);
  
          // Compute left side flux
          ConsState fluxL {};
          {
            PrimState qC = qC0 - 0.5 * slope_C;

            if( CellIndex::is_boundary(iCell_Uin_m_status) )
            {
              const CellIndex iCell_Uin_m = iCell_Uin.getNeighbor<CellIndex::BOUNDARY>(-(dir==IX), -(dir==IY), -(dir==IZ), search_neighbor);
              fluxL = policy.getBoundaryFlux(Uin, iCell_Uin_m, qC, cellmetadata);
            }
            else
            {  
              if (Ldiff >= 0) 
              {
                CellIndex iCell_Qpatch_m = iCell_Qpatch.getNeighbor<CellIndex::LOCAL_TO_BLOCK>( -(dir==IX), -(dir==IY), -(dir==IZ), search_local ); 
                PrimState qL0 = policy.getPrimState( Qpatch, iCell_Qpatch_m );

                //R neighbor is center cell, smaller if iCell_Uin_m was bigger
                int level_diff_m_R = -Ldiff;
                int level_diff_m_L; 
                if( CellIndex::is_local(iCell_Uin_m_status) )
                { //L neighbor can be non-local, compute level_diff with 2*offset
                  auto iCell_Uin_mm_status = iCell_Uin.getNeighborStatus(-2*(dir==IX), -2*(dir==IY), -2*(dir==IZ), search_neighbor);
                  level_diff_m_L = CellIndex::level_diff(iCell_Uin_mm_status);                
                }
                else
                { //iCell_Uin_m is in a different block
                  // We assume (bx>=2), LL neighbor is in same block as L
                  level_diff_m_L = 0;
                }                
                PrimState slope_L = get_slope(level_diff_m_L, level_diff_m_R, iCell_Qpatch_m, dir);

                // Reconstructing
                PrimState qL = qL0 + 0.5 * slope_L;

                // Solving
                fluxL = policy.riemann_solver(qL, qC, dir);
                
                // Adding flux to the neighbor if it is bigger
                if (Ldiff == 1) 
                {
                  real_t size_L = 2 * size_C;
                  ConsState du_n = fluxL * (- dim_fac * dt / size_L);
                  const CellIndex iCell_Uin_m = iCell_Uin.getNeighbor<CellIndex::BIGGER>(-(dir==IX), -(dir==IY), -(dir==IZ), search_neighbor);
                  policy.atomic_addConsState(Uout, iCell_Uin_m, du_n);
                }
              } // If smaller we skip
            }
          }

          // Compute right side flux
          ConsState fluxR {};
          {      
            PrimState qC = qC0 + 0.5 * slope_C;

            if( CellIndex::is_boundary(iCell_Uin_p_status) )
            {
              const CellIndex iCell_Uin_p = iCell_Uin.getNeighbor<CellIndex::BOUNDARY>( (dir==IX),  (dir==IY),  (dir==IZ), search_neighbor);
              fluxR = policy.getBoundaryFlux(Uin, iCell_Uin_p, qC, cellmetadata);
            }
            else
            {
              if (Rdiff >= 0) 
              {
                CellIndex iCell_Qpatch_p = iCell_Qpatch.getNeighbor<CellIndex::LOCAL_TO_BLOCK>(  (dir==IX),  (dir==IY),  (dir==IZ), search_local ); 
                PrimState qR0 = policy.getPrimState( Qpatch, iCell_Qpatch_p );
                
                //L neighbor is center cell, smaller if iCell_Uin_p was bigger
                int level_diff_p_L = -Rdiff;
                int level_diff_p_R;
                if( CellIndex::is_local(iCell_Uin_p_status) )
                { //L neighbor can be non-local, compute level_diff with 2*offset
                  auto iCell_Uin_pp_status = iCell_Uin.getNeighborStatus(2*(dir==IX), 2*(dir==IY), 2*(dir==IZ), search_neighbor);
                  level_diff_p_R = CellIndex::level_diff(iCell_Uin_pp_status);                
                }
                else
                { //iCell_Uin_m is in a different block
                  // We assume (bx>=2), LL neighbor is in same block as L
                  level_diff_p_R = 0;
                }  
                PrimState slope_R = get_slope(level_diff_p_L, level_diff_p_R, iCell_Qpatch_p, dir);

                // Reconstructing
                PrimState qR = qR0 - 0.5 * slope_R;

                // Solving
                fluxR = policy.riemann_solver(qC, qR, dir);

                // Adding flux to the neighbor if it is bigger
                if (Rdiff == 1)
                {
                  real_t size_R = 2 * size_C;
                  ConsState du_n = fluxR * (dim_fac * dt / size_R);
                  CellIndex iCell_Uin_p = iCell_Uin.getNeighbor<CellIndex::BIGGER>( (dir==IX),  (dir==IY),  (dir==IZ), search_neighbor);
                  policy.atomic_addConsState(Uout, iCell_Uin_p, du_n);
                }
              }
            }
          } 

          ConsState du = (fluxL-fluxR) * (dt / size_C);
          return du;
        };


        ConsState du{};
        du += process_dir(iCell_Uout, iCell_Qpatch, IX);
        du += process_dir(iCell_Uout, iCell_Qpatch, IY);
        if (ndim == 3)
           du += process_dir(iCell_Uout, iCell_Qpatch, IZ);
        policy.atomic_addConsState(Uout, iCell_Uout, du);
      });
    });
      

    // Reducing the ghosts to accumulate the flux in the data arrays 
    int ghost_count = 1;
    GhostCommunicator_partial_blocks ghost_comm ( 
      foreach_cell.get_amr_mesh(),
      Uout.getShape(),
      ghost_count );
    ghost_comm.reduce_ghosts( Uout );
    
    if constexpr ( Policy::has_postProcess() )
    {
      foreach_cell.foreach_cell( "HyperbolicUpdate::post-process", Uout.getShape(),
        KOKKOS_LAMBDA(  const ForeachCell::CellIndex& iCell)
      {
        ConsState u = policy.getConsState(Uout, iCell);
        ConsState u_pp = policy.postProcess( u );
        policy.setConsState( Uout, iCell, u_pp );
      });
    }
    
    policy.printWarnings();

    timers.get("HyperbolicUpdate_euler").stop();
  }

private:
  ForeachCell& foreach_cell;
  
  Timers& timers;  
  typename Policy::Params policy_params;

  int ndim;
  real_t smallr, smallp;
  bool slope_enabled;
};

} // namespace dyablo

