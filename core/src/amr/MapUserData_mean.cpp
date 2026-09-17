#include "MapUserData_base.h"

#include "amr/CellIndexRemapper.h"
#include "user_data/UserData.h"
#include "user_data/FieldAccessor.h"
#include "foreach_cell/ForeachCell_utils.h"

namespace dyablo {

/**
 * Implementation of MapUserData using mean of smaller cells when coarseneing
 **/
class MapUserData_mean : public MapUserData_base{
public: 
  MapUserData_mean(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : MapUserData_base( configMap, foreach_cell, timers)
  {}
  
  ~MapUserData_mean(){}

  void save_old_mesh(UserData& user_data) override
  {
    MapUserData_base::save_old_mesh(user_data);
    this->cellmetadata_old = std::make_unique<ForeachCell::CellMetaData>(foreach_cell.getCellMetaData());
  }

  void remap_aux( const UserData::FieldAccessor& Uin, const UserData::FieldAccessor& Uout, const CellIndexRemapper& remapper ) override
  {
    using CellIndex = ForeachCell::CellIndex;
    int nbfields = Uin.nbFields();
    int ndim = foreach_cell.getDim(); 

    ForeachCell::CellMetaData &cellmetadata_in = *(this->cellmetadata_old);
      
    foreach_cell.foreach_cell( "MapUserData_mean::remap", Uout.getShape(),
      KOKKOS_LAMBDA( const CellIndex& iCell_Uout )
    {
      const auto& lmesh = cellmetadata_in.getLightOctree();
      ForeachCell::SearchMode_neighbor search_neighbor_in( cellmetadata_in.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

      CellIndex iCell_Uin = remapper.get_old_cell( iCell_Uout );

      if( iCell_Uin.level_diff() >= 0 )
      {
        for(int ivar=0; ivar<nbfields; ivar++)
          Uout.at_ivar( iCell_Uout, ivar ) = Uin.at_ivar( iCell_Uin, ivar );
      }
      else
      {
        for(int ivar=0; ivar<nbfields; ivar++)
          Uout.at_ivar( iCell_Uout, ivar ) = 0;

        int nsubcells = (ndim-1) * 2 * 2;
        foreach_sibling_scattered( ndim, iCell_Uin, lmesh,
            [&](const CellIndex& iCell_Uin_n)
        {
          for(int ivar=0; ivar<nbfields; ivar++)
            Uout.at_ivar( iCell_Uout, ivar ) += Uin.at_ivar( iCell_Uin_n, ivar ) / nsubcells;
        });
      }
    });
  }
protected:
  std::unique_ptr<ForeachCell::CellMetaData> cellmetadata_old;
};

} // namespace dyablo;

FACTORY_REGISTER( dyablo::MapUserDataFactory , dyablo::MapUserData_mean, "MapUserData_mean")
