#include "MapUserData_base.h"

#include "amr/CellIndexRemapper.h"
#include "user_data/FieldAccessor.h"

namespace dyablo {

/**
 * Implementation of MapUserData using linear interpolation
 * For fine -> coarse we simply average the values in the small cells
 * For coarse -> fine we use a linear interpolation
 **/
class MapUserData_linear : public MapUserData_base{
public: 
  MapUserData_linear(
    ConfigMap& configMap,
    ForeachCell& foreach_cell,
    Timers& timers )
  : MapUserData_base( configMap, foreach_cell, timers)
  {}
  
  ~MapUserData_linear(){}

  void save_old_mesh(UserData& user_data) override
  {
    MapUserData_base::save_old_mesh(user_data);
    this->cellmetadata_old = std::make_unique<ForeachCell::CellMetaData>(foreach_cell.getCellMetaData());
  }

  void remap_aux( const UserData::FieldAccessor& Uin, const UserData::FieldAccessor& Uout, const CellIndexRemapper& remapper ) override
  {
    using CellIndex = ForeachCell::CellIndex;
    using pos_t = ForeachCell::CellMetaData::pos_t;
    int ndim = foreach_cell.getDim();
    int nbfields = Uin.nbFields();

    ForeachCell::CellMetaData &cellmetadata_in = *(this->cellmetadata_old);
    auto cellmetadata_out = foreach_cell.getCellMetaData();
    
    auto remap = [&](){
      // Detect if a coarsened octant needs ghost values
      int ghost_coarsen_count = 0;
      
      foreach_cell.reduce_cell( "MapUserData_mean::remap", Uout.getShape(),
        KOKKOS_LAMBDA( const CellIndex& iCell_Uout, int& ghost_coarsen_count )
      {
        ForeachCell::SearchMode_neighbor search_neighbor_in( cellmetadata_in.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

        CellIndex iCell_Uin = remapper.get_old_cell( iCell_Uout );

        auto ldiff = iCell_Uin.level_diff();
        if( ldiff >= 0 ) // Same size or coarse -> fine
        {
          // Copy the original cell in the new one
          for(int ivar=0; ivar<nbfields; ivar++)
            Uout.at_ivar( iCell_Uout, ivar ) = Uin.at_ivar( iCell_Uin, ivar );

          if (ldiff > 0) { // coarse -> fine
            const pos_t dh = cellmetadata_in.getCellSize(iCell_Uin); // Original cell size

            // 1. Calculating position difference between new and old cells
            const pos_t pos_old = cellmetadata_in.getCellCenter(iCell_Uin);
            const pos_t pos_new = cellmetadata_out.getCellCenter(iCell_Uout);
            const pos_t dpos {pos_new[IX]-pos_old[IX], 
                              pos_new[IY]-pos_old[IY], 
                              pos_new[IZ]-pos_old[IZ]};

            // 2. Retrieving stencil, and computing level differences
            const CellIndex iCell_mx = iCell_Uin.getNeighbor({-1, 0, 0}, search_neighbor_in); 
            const CellIndex iCell_px = iCell_Uin.getNeighbor({ 1, 0, 0}, search_neighbor_in);
            const CellIndex iCell_my = iCell_Uin.getNeighbor({ 0,-1, 0}, search_neighbor_in);
            const CellIndex iCell_py = iCell_Uin.getNeighbor({ 0, 1, 0}, search_neighbor_in);
            const int ldiffLx = iCell_mx.level_diff();
            const int ldiffRx = iCell_px.level_diff();
            const int ldiffLy = iCell_my.level_diff();
            const int ldiffRy = iCell_py.level_diff();

            CellIndex iCell_mz{}, iCell_pz{};
            int ldiffLz = 0, ldiffRz = 0;
            if (ndim == 3) {
              iCell_mz = iCell_Uin.getNeighbor({ 0, 0,-1}, search_neighbor_in);
              iCell_pz = iCell_Uin.getNeighbor({ 0, 0, 1}, search_neighbor_in);
              ldiffLz = iCell_mz.level_diff();
              ldiffRz = iCell_pz.level_diff();
            }


            // 3. Calculating gradients
            auto get_gradient = [&](const CellIndex &iCellL, const CellIndex &iCellC, const CellIndex &iCellR, int ldiffL, int ldiffR, real_t dh, int ivar) {
              const bool bcL = iCellL.is_boundary();
              const bool bcR = iCellR.is_boundary();

              const real_t dS[3] {0.75, 1.0, 1.5}; 
              
              // If any side is a boundary we return the other gradient
              real_t grad;
              if (bcL)
                grad = (Uin.at_ivar(iCellR, ivar) - Uin.at_ivar(iCellC, ivar)) / (dh * dS[ldiffR+1]);
              else if (bcR)
                grad = (Uin.at_ivar(iCellC, ivar) - Uin.at_ivar(iCellL, ivar)) / (dh * dS[ldiffL+1]);
              // We apply minmod  
              else { 
                const real_t gL = (Uin.at_ivar(iCellR, ivar) - Uin.at_ivar(iCellC, ivar)) / (dh * dS[ldiffR+1]);
                const real_t gR = (Uin.at_ivar(iCellC, ivar) - Uin.at_ivar(iCellL, ivar)) / (dh * dS[ldiffL+1]);
              
                if (gL*gR < 0.0)
                  grad = 0.0;
                else if (Kokkos::abs(gL) < Kokkos::abs(gR))
                  grad = gL;
                else
                  grad = gR;
              }

              return grad;
            };

            for (int ivar=0; ivar<nbfields; ivar++) {
              const real_t gx = get_gradient(iCell_mx, iCell_Uin, iCell_px, ldiffLx, ldiffRx, dh[IX], ivar);
              const real_t gy = get_gradient(iCell_my, iCell_Uin, iCell_py, ldiffLy, ldiffRy, dh[IY], ivar);
              
              Uout.at_ivar(iCell_Uout, ivar) += dpos[IX]*gx + dpos[IY]*gy;

              if (ndim == 3) {
                const real_t gz = get_gradient(iCell_mz, iCell_Uin, iCell_pz, ldiffLz, ldiffRz, dh[IZ], ivar);
                Uout.at_ivar(iCell_Uout, ivar) += dpos[IZ]*gz;
              }
            }
          }
        }
        else // fine -> coarse
        {
          for(int ivar=0; ivar<nbfields; ivar++)
            Uout.at_ivar( iCell_Uout, ivar ) = 0;

          int nsubcells = (ndim-1) * 2 * 2;
          for(int32_t dz=0; dz<(ndim-1); dz++)
            for(int32_t dy=0; dy<2; dy++)
              for(int32_t dx=0; dx<2; dx++)
              {
                CellIndex iCell_Uin_n = iCell_Uin.getNeighbor({dx,dy,dz}, search_neighbor_in);
                ghost_coarsen_count += iCell_Uin_n.iOct().isGhost ? 1 : 0;
                for(int ivar=0; ivar<nbfields; ivar++)
                  Uout.at_ivar( iCell_Uout, ivar ) += Uin.at_ivar( iCell_Uin_n, ivar ) / nsubcells;
              }
        }
      }, ghost_coarsen_count);
      if( ghost_coarsen_count > 0 )
        std::cout << "Warning : detected ghost subcells in coarsened octant during remap - this is so unlikely that it was never tested" << std::endl;

      return ghost_coarsen_count;
    };

    remap();
  }

protected:
  std::unique_ptr<ForeachCell::CellMetaData> cellmetadata_old;
};

} // namespace dyablo;

FACTORY_REGISTER( dyablo::MapUserDataFactory , dyablo::MapUserData_linear, "MapUserData_linear")
