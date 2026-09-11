#include "hyperbolic/HyperbolicUpdate_base.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "utils/config/ConfigMap.h"

#include "mpi/GhostCommunicator.h"

#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"

#ifdef __CUDA_ARCH__
#include "math_constants.h"
#endif

class Timers;
class ConfigMap;

namespace dyablo {


namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using CellIndex = ForeachCell::CellIndex;
using FieldArray = UserData::FieldAccessor;
 
}
/**
 * @brief Solves the equations with a Hancock timestepping on 
 * small blocks.
 * 
 * This solver should only be used for comparison with cell-based AMR
 * as it is the only one allowing for bx=by=bz=1.
*/
template <typename Policy>
class HyperbolicUpdate_hancock_oneneighbor : public HyperbolicUpdate{
public: 
  using PrimState = typename Policy::PrimState;
  using ConsState = typename Policy::ConsState;

  HyperbolicUpdate_hancock_oneneighbor(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : foreach_cell(foreach_cell),
      policy_params(Policy::getParams(configMap)),
      timers(timers)
  {}

  void update( UserData& U, ScalarSimulationData& scalar_data) 
  {
    real_t dt = scalar_data.get<real_t>("dt");
    int ndim = foreach_cell.getDim();
 
    using GhostedArray = ForeachCell::CellArray_global_ghosted;

    Policy policy( this->policy_params, scalar_data );
    ForeachCell& foreach_cell = this->foreach_cell;
    
    int nb_ghosts = 1;
    GhostCommunicator ghost_comm(foreach_cell.get_amr_mesh(), U.getShape(), nb_ghosts );

    timers.get("HyperbolicUpdate_hancock_oneneighbor").start();

    UserData::FieldAccessor Uin = policy.getUin(U);
    UserData::FieldAccessor Uout = policy.getUout(U);
    uint32_t nbFields_prim = State_traits<PrimState>::nvars;
    
    GhostedArray Q = foreach_cell.allocate_ghosted_array( "Q", nbFields_prim );

    // Fill Q with primitive variables
    foreach_cell.foreach_cell("HyperbolicUpdate_hancock_oneneighbor::convertToPrimitives", Q, 
      KOKKOS_LAMBDA(const CellIndex& iCell_Q)
    { 
      ConsState uLoc = policy.getConsState( Uin, iCell_Q );
      PrimState qLoc = policy.consToPrim( uLoc );
      policy.setPrimState( Q, iCell_Q, qLoc );
    });
    // Primitive variables of ghost cells are needed to compute slopes
    ghost_comm.exchange_ghosts(Q);

    // Create arrays to store slopes
    GhostedArray Slopes_x = foreach_cell.allocate_ghosted_array( "Slopes_x", nbFields_prim );
    GhostedArray Slopes_y = foreach_cell.allocate_ghosted_array( "Slopes_Y", nbFields_prim );
    GhostedArray Slopes_z;
    if(ndim == 3)
      Slopes_z = foreach_cell.allocate_ghosted_array( "Slopes_z", nbFields_prim );

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    // Fill slope arrays
    foreach_cell.foreach_cell("HyperbolicUpdate_hancock_oneneighbor::compute_slopes", Q, 
      KOKKOS_LAMBDA(const CellIndex& iCell_Q)
    { 
      const auto& lmesh = cellmetadata.getLightOctree();
      ForeachCell::SearchMode_neighbor search_neighbor( lmesh, ForeachCell::SearchMode_neighbor::CLOSEST );

      PrimState qC = policy.getPrimState( Q, iCell_Q );
      auto compute_slope = [&](ComponentIndex3D dir)
      {
        auto get_neighbor_val = [&](int side)
        {
          CellIndex::offset_t offset{};
          offset[dir] = side;
          CellIndex iCell_n0 = iCell_Q.getNeighbor( offset, search_neighbor);
          int level_diff = iCell_n0.level_diff();

          PrimState q;
          if( iCell_n0.is_boundary() )
          {
            ConsState u = policy.getBoundaryValue(Uin, iCell_n0, cellmetadata);
            q = policy.consToPrim( u );
          }
          else if ( level_diff >= 0 )
            q = policy.getPrimState( Q, iCell_n0 );
          else //if (level_diff < 0)
          {
            int subcell_count = 
            foreach_smaller_neighbor_scattered(ndim, iCell_n0, offset, lmesh,
              [&](const CellIndex& iCell_n) {
                PrimState qloc = policy.getPrimState(Q, iCell_n);
                q += qloc;
              });
            q /= subcell_count;
          }

          real_t dx;
          if( level_diff == 0 )
            dx = 1;
          else if (level_diff > 0)
            dx = 2;
          else
            dx = 0.5;

          return std::make_pair(q, dx);
        };

        const auto [qL, dL] = get_neighbor_val(-1);
        const auto [qR, dR] = get_neighbor_val(+1);
        PrimState slope = policy.compute_slope( qL, qC, qR, dL, dR);
        return slope;
      };

      PrimState slopeX = compute_slope( IX );
      policy.setPrimState( Slopes_x, iCell_Q, slopeX );
      PrimState slopeY = compute_slope( IY );
      policy.setPrimState( Slopes_y, iCell_Q, slopeY );
      if(ndim == 3)
      {
        PrimState slopeZ = compute_slope( IZ );
        policy.setPrimState( Slopes_z, iCell_Q, slopeZ );
      }
    });
    // Slopes of ghost cells are needed to compute flux
    ghost_comm.exchange_ghosts(Slopes_x);
    ghost_comm.exchange_ghosts(Slopes_y);
    if(ndim == 3)
      ghost_comm.exchange_ghosts(Slopes_z);

    // Compute flux and update Uout
    foreach_cell.foreach_cell("HyperbolicUpdate_hancock_oneneighbor::flux_and_update", Q, 
      KOKKOS_LAMBDA(const CellIndex& iCell)
    { 
      ForeachCell::SearchMode_neighbor search_neighbor( cellmetadata.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

      ForeachCell::CellMetaData::pos_t cell_size = cellmetadata.getCellSize(iCell);
      ForeachCell::CellMetaData::pos_t pos_c = cellmetadata.getCellCenter(iCell);

      // Return the flux entering the cell at dir,sign interface
      auto flux = [&]( ComponentIndex3D dir, int sign )
      {
        CellIndex::offset_t offset{};
        offset[dir] = sign;
        CellIndex iCell_n0 = iCell.getNeighbor( offset, search_neighbor );

        // Reconstruct state at position off{xyz} in iCell (normalized positions in [-1,1], (0,0,0) is center of cell )
        using real_offset = Kokkos::Array<real_t, 3>;
        auto reconstruct_state = [&]( const CellIndex& iCell, const real_offset& off )
        {
          PrimState q  = policy.getPrimState( Q, iCell );
          PrimState sx = policy.getPrimState( Slopes_x, iCell );
          PrimState sy = policy.getPrimState( Slopes_y, iCell );
          PrimState sz;
          if(ndim == 3)
            sz = policy.getPrimState( Slopes_z, iCell );
          auto cell_size = cellmetadata.getCellSize(iCell);
          const real_t dtdx = dt/cell_size[IX];
          const real_t dtdy = dt/cell_size[IY];
          const real_t dtdz = dt/cell_size[IZ];
          PrimState q_half = policy.compute_half_step(q, sx, sy, sz, dtdx, dtdy, dtdz );
          return q_half + 0.5 * (off[IX] * sx + off[IY] * sy + off[IZ] * sz);
        };

        if( iCell_n0.is_boundary() )
        {
          real_offset offset_c{
            (real_t)offset[IX],
            (real_t)offset[IY],
            (real_t)offset[IZ]
          };
          PrimState qin_reconstructed = reconstruct_state( iCell, offset_c );
          ConsState res = policy.getBoundaryFlux( Uin, iCell_n0, qin_reconstructed, cellmetadata );
          return res;
        }
        else if( iCell_n0.level_diff() >= 0 )
        { // Only one neighbor
          real_offset offset_c{
            (real_t)offset[IX],
            (real_t)offset[IY],
            (real_t)offset[IZ]
          };
          PrimState qin_reconstructed = reconstruct_state( iCell, offset_c );
          
          real_offset offset_n{
            (real_t)-offset[IX],
            (real_t)-offset[IY],
            (real_t)-offset[IZ]
          };
          // Correct reconstruction position if neighbor cell is 
          // bigger to center of smaller cell's interface
          if( iCell_n0.level_diff() > 0 )
          {
            ForeachCell::CellMetaData::pos_t pos_n = cellmetadata.getCellCenter(iCell_n0);  
            for( int facedir = 0; facedir < ndim; facedir++ )
              if( facedir != dir ) // Foreach direction inside face (orthogonal to dir)
              {
                offset_n[facedir] += (pos_c[facedir] > pos_n[facedir])? 0.5 : -0.5;
              }        
          }
          PrimState qout_reconstructed = reconstruct_state( iCell_n0, offset_n );
          PrimState& qL = (sign == -1) ? qout_reconstructed : qin_reconstructed;
          PrimState& qR = (sign == -1) ? qin_reconstructed  : qout_reconstructed;
          ConsState flux = policy.riemann_solver( qL, qR, dir );
          return flux;
        }
        else //if( iCell_n0.level_diff() < 0 )
        { // Multiple smaller neighbors

          PrimState qC  = policy.getPrimState( Q, iCell );
          PrimState sx = policy.getPrimState( Slopes_x, iCell );
          PrimState sy = policy.getPrimState( Slopes_y, iCell );
          PrimState sz;
          if(ndim == 3)
            sz = policy.getPrimState( Slopes_z, iCell );
          auto cell_size = cellmetadata.getCellSize(iCell);
          const real_t dtdx = dt/cell_size[IX];
          const real_t dtdy = dt/cell_size[IY];
          const real_t dtdz = dt/cell_size[IZ];
          PrimState qC_half = policy.compute_half_step(qC, sx, sy, sz, dtdx, dtdy, dtdz );

          // Accumulate fluxes from neighbors of initial cell
          ConsState flux;
          int di_count = (offset[IX]==0)?2:1;
          int dj_count = (offset[IY]==0)?2:1;
          int dk_count = (ndim==3 && offset[IZ]==0)?2:1;
          for( int32_t dk=0; dk<dk_count; dk++ )
          for( int32_t dj=0; dj<dj_count; dj++ )
          for( int32_t di=0; di<di_count; di++ )
          {            
            CellIndex iCell_n = iCell_n0.getNeighbor({di,dj,dk}, search_neighbor);

            // Reconstruct in state at center of neighbor cell's interface
            real_offset offset_c{
              (offset[IX] != 0) ? (real_t)offset[IX] : di-0.5, 
              (offset[IY] != 0) ? (real_t)offset[IY] : dj-0.5, 
              (ndim==2 || offset[IZ] != 0) ? (real_t)offset[IZ] : dk-0.5
            };
            // Equivalent to :
            // `PrimState qin_reconstructed = reconstruct_state( iCell, offset_c);`
            PrimState qin_reconstructed = qC_half + 0.5 * (offset_c[IX] * sx + offset_c[IY] * sy + offset_c[IZ] * sz);

            // Reconstruct out state at center of neighbor cell's interface
            real_offset offset_n{
              (real_t)-offset[IX],
              (real_t)-offset[IY],
              (real_t)-offset[IZ]
            };
            PrimState qout_reconstructed = reconstruct_state( iCell_n, offset_n );

            PrimState& qL = (sign == -1) ? qout_reconstructed : qin_reconstructed;
            PrimState& qR = (sign == -1) ? qin_reconstructed  : qout_reconstructed;
            ConsState flux_contrib = policy.riemann_solver( qL, qR, dir );
            real_t dim_fac = (ndim == 2 ? 0.5 : 0.25);
            flux += dim_fac * flux_contrib;
          }
          return flux;
        }
      };

      ConsState uC = policy.getConsState( Uin, iCell );
      uC += (flux( IX, -1 ) - flux( IX, +1 )) * dt / cell_size[IX];
      uC += (flux( IY, -1 ) - flux( IY, +1 )) * dt / cell_size[IY];
      if (ndim == 3)
        uC += (flux( IZ, -1 ) - flux( IZ, +1 )) * dt / cell_size[IZ];
      policy.setConsState(Uout, iCell, uC);
    });

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

    timers.get("HyperbolicUpdate_hancock_oneneighbor").stop();
  }
  private:
    ForeachCell& foreach_cell;
    typename Policy::Params policy_params;     
    Timers& timers;
};

class HyperbolicPolicy_Hydro_Hancock_impl : public HyperbolicPolicy_Hydro_impl_default
{
public : 
  using HyperbolicPolicy_Hydro_impl_default::HyperbolicPolicy_Hydro_impl_default;

  KOKKOS_INLINE_FUNCTION
  PrimState compute_half_step(PrimState q,
                              PrimState sx, PrimState sy, PrimState sz,
                              real_t dtdx, real_t dtdy, real_t dtdz ) const
  {
    int ndim = this->ndim;
    real_t gamma0 = this->gamma0;

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
  }
};

class HyperbolicPolicy_Hydro_Hancock : public HyperbolicPolicy_base< HyperbolicPolicy_Hydro_Hancock_impl >
{
public:
  using HyperbolicPolicy_base::HyperbolicPolicy_base;

  KOKKOS_INLINE_FUNCTION
  PrimState compute_half_step(PrimState q,
    PrimState sx, PrimState sy, PrimState sz,
    real_t dtdx, real_t dtdy, real_t dtdz ) const
  {
    return impl.compute_half_step(q, sx, sy, sz, dtdx, dtdy, dtdz );
  }
};

class HydroUpdate_hancock_oneneighbor 
  : public HyperbolicUpdate_hancock_oneneighbor<HyperbolicPolicy_Hydro_Hancock>
{
public:
  using HyperbolicUpdate_hancock_oneneighbor<HyperbolicPolicy_Hydro_Hancock>::HyperbolicUpdate_hancock_oneneighbor;
};

} //namespace dyablo 
  
FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_hancock_oneneighbor, 
                  "HydroUpdate_hancock_oneneighbor")