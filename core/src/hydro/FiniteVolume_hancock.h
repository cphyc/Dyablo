#pragma once

#include "HydroUpdate_base.h"
#include "HydroUpdate_utils.h"
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

template<typename Policy>
class FiniteVolume_hancock : public HydroUpdate {
  static_assert( is_FiniteVolumePolicy_v<Policy>,
  "Policy must be wrapped in FiniteVolumePolicy_base");

public:
  using PrimState = typename Policy::PrimState;
  using ConsState = typename Policy::ConsState;

public:
  FiniteVolume_hancock(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    timers(timers),
    policy(configMap),
    ndim(configMap.getValue<int>("mesh", "ndim", 3)),
    gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
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
    real_t gamma0 = this->gamma0;
    int ndim = this->ndim;

    const Policy& policy = this->policy; 
    Timers& timers = this->timers; 
    ForeachCell& foreach_cell = this->foreach_cell;

    FieldAccessor Uin = policy.getUin(U);
    FieldAccessor Uout = policy.getUout(U);
    
    timers.get("FiniteVolume_hancock").start();

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    // Initializing output array 
    // TODO : remove this and copy Uin->Uout in timeloop or field creation logic
    foreach_cell.foreach_cell( "FiniteVolume_euler::init",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell) 
    {
      ConsState uC = policy.getConsState(Uin, iCell);
      policy.setConsState(Uout, iCell, uC);
    });

    // Setting the ghosts to 0 to accumulate fluxes
    foreach_cell.foreach_ghost_cell( "FiniteVolume_euler::resetting_ghosts",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell) 
    {
      ConsState empty_state{};
      policy.setConsState(Uout, iCell, empty_state);
    });

    // Create abstract temporary ghosted arrays for patches 
    using PatchArray = ForeachCell::CellArray_patch;
    FieldManager fm_prim = Policy::PrimState::getFieldManager();
    PatchArray::Ref HalfStep_ = foreach_cell.reserve_patch_tmp("HalfStep", 1, 1, (ndim==3)?1:0, fm_prim.get_id2index(), fm_prim.nbfields());
    PatchArray::Ref SlopesX_ = foreach_cell.reserve_patch_tmp("SlopesX", 1, 1, (ndim==3)?1:0, fm_prim.get_id2index(), fm_prim.nbfields());
    PatchArray::Ref SlopesY_ = foreach_cell.reserve_patch_tmp("SlopesY", 1, 1, (ndim==3)?1:0, fm_prim.get_id2index(), fm_prim.nbfields());
    PatchArray::Ref SlopesZ_;
    if( ndim == 3 )
      SlopesZ_ = foreach_cell.reserve_patch_tmp("SlopesZ", 1, 1, 1, fm_prim.get_id2index(), fm_prim.nbfields());

    // Iterate over cells
    foreach_cell.foreach_patch( "FiniteVolume_euler::update",
      PATCH_LAMBDA(const ForeachCell::Patch& patch)
    {
      PatchArray SlopesX = patch.allocate_tmp(SlopesX_);
      PatchArray SlopesY = patch.allocate_tmp(SlopesY_);
      PatchArray SlopesZ;
      if( ndim == 3 )
        SlopesZ = patch.allocate_tmp(SlopesZ_);
      PatchArray HalfStep = patch.allocate_tmp(HalfStep_);

      patch.foreach_cell( HalfStep.getShape(),
        CELL_LAMBDA(const CellIndex& iCell_tmp)
      {
        // Return Slope at position iCell
        auto compute_slope = [&](const CellIndex &iCell, ComponentIndex3D dir) 
        {       
          auto get_neighbor_prim_value = [&]( const CellIndex& iCell_n, const CellIndex::offset_t& off )
          {
            ConsState u {};
            // Getting left value
            int level_diff = iCell_n.level_diff();
            if (iCell_n.is_boundary())
              u = policy.getBoundaryValue(Uin, iCell_n, cellmetadata);
            else if (level_diff < 0) {
              int subcell_count = 
              foreach_smaller_neighbor(ndim, iCell_n, off, Uin.getShape(),
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
        };

        auto compute_half_step = [&]  ( PrimState q,
                              PrimState sx,
                              PrimState sy,
                              PrimState sz,
                              real_t dtdx, real_t dtdy, real_t dtdz )
        {
          // retrieve variations = dx * slopes
          sx*=0.5;
          sy*=0.5;
          sz*=0.5;

          PrimState half_step{};
          if( ndim == 3 )
          {
            half_step.rho = q.rho + (-q.u * sx.rho - sx.u * q.rho) * dtdx 
                                  + (-q.v * sy.rho - sy.v * q.rho) * dtdy 
                                  + (-q.w * sz.rho - sz.w * q.rho) * dtdz;
            half_step.u   = q.u + (-q.u * sx.u - sx.p / q.rho) * dtdx 
                                + (-q.v * sy.u) * dtdy 
                                + (-q.w * sz.u) * dtdz;
            half_step.v   = q.v + (-q.u * sx.v) * dtdx 
                                + (-q.v * sy.v - sy.p / q.rho) * dtdy 
                                + (-q.w * sz.v) * dtdz;
            half_step.w   = q.w + (-q.u * sx.w) * dtdx  
                                + (-q.v * sy.w) * dtdy 
                                + (-q.w * sz.w - sz.p / q.rho) * dtdz;
            half_step.p   = q.p + (-q.u * sx.p - sx.u * gamma0 * q.p) * dtdx 
                                + (-q.v * sy.p - sy.v * gamma0 * q.p) * dtdy 
                                + (-q.w * sz.p - sz.w * gamma0 * q.p) * dtdz;
          }
          else
          {
            half_step.rho = q.rho + (-q.u * sx.rho - sx.u * q.rho) * dtdx 
                                  + (-q.v * sy.rho - sy.v * q.rho) * dtdy;
            half_step.u   = q.u + (-q.u * sx.u - sx.p / q.rho) * dtdx 
                                + (-q.v * sy.u) * dtdy;
            half_step.v   = q.v + (-q.u * sx.v) * dtdx 
                                + (-q.v * sy.v - sy.p / q.rho) * dtdy;
            half_step.p   = q.p + (-q.u * sx.p - sx.u * gamma0 * q.p) * dtdx 
                                + (-q.v * sy.p - sy.v * gamma0 * q.p) * dtdy;
          }
          return half_step;
        };

        CellIndex iCell_Uin = Uin.getShape().convert_index_ghost(iCell_tmp);

        if( iCell_Uin.is_valid() && iCell_Uin.level_diff() >= 0 )
        { //Compute slopes only inside of domain and skip smaller neighbors
          ConsState u = policy.getConsState( Uin, iCell_Uin );
          PrimState q = policy.consToPrim(u);

          auto size = cellmetadata.getCellSize(iCell_Uin);

          PrimState sx = compute_slope(iCell_Uin, IX);
          PrimState sy = compute_slope(iCell_Uin, IY);
          PrimState sz {};
          if(ndim == 3)
            sz = compute_slope(iCell_Uin, IZ);

          PrimState q_half = compute_half_step( q, 
                                        sx, sy, sz, 
                                        dt/size[IX], dt/size[IY], dt/size[IZ]);

          policy.setPrimState( SlopesX, iCell_tmp, sx );
          policy.setPrimState( SlopesY, iCell_tmp, sy );
          if(ndim == 3)
            policy.setPrimState( SlopesZ, iCell_tmp, sz );
          policy.setPrimState( HalfStep, iCell_tmp, q_half );
        }
      });

      patch.foreach_cell( Uout.getShape(),
        CELL_LAMBDA(const CellIndex& iCell)
      {
        auto process_dir = [&](const CellIndex &iCell_U, ComponentIndex3D dir) {
          auto get_slope = [&](const CellIndex &iCell_tmp, ComponentIndex3D dir)
          {
            if( dir==IX )
              return policy.getPrimState( SlopesX, iCell_tmp );
            else if( dir==IY )
              return policy.getPrimState( SlopesY, iCell_tmp );
            else
              return policy.getPrimState( SlopesZ, iCell_tmp );
          };       
          
          // Getting centered value and slope
          CellIndex iCell_tmp = HalfStep.getShape().convert_index( iCell_U );
          PrimState slope_C = get_slope(iCell_tmp, dir);       
          PrimState qC_half = policy.getPrimState( HalfStep, iCell_tmp );
          auto size_C = cellmetadata.getCellSize(iCell_U);

          real_t dim_fac = (ndim == 2 ? 0.5 : 0.25);

          // Compute left side flux
          ConsState fluxL {};
          {
            offset_t off_m{}; 
            off_m[dir] = -1;
            const CellIndex iCell_m_U = iCell_U.getNeighbor_ghost(off_m, Uin.getShape());
            if( iCell_m_U.is_boundary() )
            {
              fluxL = policy.getBoundaryFlux(Uin, iCell_m_U, cellmetadata);
            }
            else
            {  
              int Ldiff = iCell_m_U.level_diff();
              if (Ldiff >= 0) 
              {       
                
                const CellIndex iCell_m_tmp = iCell_tmp.getNeighbor( off_m );
                PrimState slope_L = get_slope(iCell_m_tmp, dir);
                auto size_L = cellmetadata.getCellSize(iCell_m_U);

                PrimState qL_half = policy.getPrimState( HalfStep, iCell_m_tmp );

                // Reconstructing
                PrimState qL = qL_half + 0.5 * slope_L;
                PrimState qC = qC_half - 0.5 * slope_C;

                // Solving
                fluxL = policy.riemann_solver(qL, qC, dir);
                
                // Adding flux to the neighbor if it is bigger
                if (Ldiff == 1) 
                {
                  ConsState du_n = fluxL * - dim_fac * dt / size_L[dir];
                  policy.atomic_addConsState(Uout, iCell_m_U, du_n);
                }
              } // If smaller we skip
            }
          }

          // Compute right side flux
          ConsState fluxR {};
          {      
            offset_t off_p{}; 
            off_p[dir] = 1;
            const CellIndex iCell_p_U = iCell_U.getNeighbor_ghost(off_p, Uin.getShape());
            if( iCell_p_U.is_boundary() )
            {
              fluxR = policy.getBoundaryFlux(Uin, iCell_p_U, cellmetadata);
            }
            else
            {
              int Rdiff = iCell_p_U.level_diff();
              if (Rdiff >= 0) 
              {
                const CellIndex iCell_p_tmp = iCell_tmp.getNeighbor( off_p );
                PrimState slope_R = get_slope(iCell_p_tmp, dir);
                auto size_R = cellmetadata.getCellSize(iCell_p_U);

                PrimState qR_half = policy.getPrimState( HalfStep, iCell_p_tmp );

                // Reconstructing
                PrimState qC = qC_half + 0.5 * slope_C;
                PrimState qR = qR_half - 0.5 * slope_R;

                // Solving
                fluxR = policy.riemann_solver(qC, qR, dir);

                // Adding flux to the neighbor if it is bigger
                if (Rdiff == 1)
                {
                  ConsState du_n = fluxR * dim_fac * dt / size_R[dir];
                  policy.atomic_addConsState(Uout, iCell_p_U, du_n);
                }          
              }
            }
          } 

          ConsState du = (fluxL-fluxR) * dt / size_C[dir];
          return du;
        };

        ConsState du{};
        du += process_dir(iCell, IX);
        du += process_dir(iCell, IY);
        if (ndim == 3)
          du += process_dir(iCell, IZ);
        policy.atomic_addConsState(Uout, iCell, du);
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

    timers.get("FiniteVolume_hancock").stop();
  }

private:
  ForeachCell& foreach_cell;
  
  Timers& timers;  
  Policy policy;

  int ndim;
  real_t gamma0, smallr, smallp;
};

} // namespace dyablo

