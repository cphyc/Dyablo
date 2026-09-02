#include "MapUserData_base.h"

#include "amr/CellIndexRemapper.h"
#include "user_data/UserData.h"
#include "user_data/FieldAccessor.h"

namespace dyablo {

void MapUserData_base::save_old_mesh( UserData& U )
{
  this->lmesh_old = this->foreach_cell.get_amr_mesh().getLightOctree();
  this->ghost_comm_full = std::make_unique<GhostCommunicator_full_blocks>( this->foreach_cell.get_amr_mesh(), U.getShape(), -1 );
}

void MapUserData_base::remap( UserData& user_data )
{ 
  CellIndexRemapper remapper( this->lmesh_old, this->foreach_cell );

  UserData::FieldAccessor Uin = user_data.backup_and_realloc();
  UserData::FieldAccessor Uout;
  {
    std::vector<UserData::FieldAccessor::FieldInfo> all_fields;
    int i=0;
    for( const std::string& field : user_data.getEnabledFields() )
      all_fields.push_back({field, i++});
    Uout = user_data.getAccessor( all_fields );
  }

  using OctantIndex = LightOctree::OctantIndex;
  int ndim = foreach_cell.getDim();   

  {
    LightOctree& lmesh_old = this->lmesh_old;
    const LightOctree& lmesh_new = this->foreach_cell.get_amr_mesh().getLightOctree();

    uint32_t nbOcts_new = lmesh_new.getNumOctants();

    // // (version where ghost coarsened are not at end of interval)
    // // Get a list of all coarsened ghost octants
    // Kokkos::View< uint32_t* > coarsened_ghosts("coarsened_ghosts", nbGhosts_old);
    // uint32_t coarsened_ghosts_count = 0;
    // Kokkos::parallel_scan( "Remap::list_ghost_coarsened", nbOcts_new,
    //   KOKKOS_LAMBDA( uint32_t iOct_new, uint32_t& count, bool final )
    // {
    //   OctantIndex iOct_old = remapper.get_old_octant({iOct_new, false});
    //   int level_old = lmesh_old.getLevel(iOct_old);
    //   int level_new = lmesh_new.getLevel({iOct_new, false});
    //   // Skip if not coarsened
    //   if( level_new < level_old )
    //   {
    //     // If last sibling is not ghost, everything is local
    //     int8_t dz_max = (ndim == 2)? 0:1;
    //     auto ns = lmesh_old.findNeighbors(iOct_old, {1,1,dz_max});
    //     DYABLO_ASSERT_KOKKOS_DEBUG( ns.size() == 1, "Sibling not found" );
    //     auto iOct_old_last = ns[0];
    //     if( iOct_old_last.isGhost ) 
    //     {
    //       for( int8_t dz=0; dz<=dz_max; dz++ )
    //       for( int8_t dy=0; dy<=1; dy++ )
    //       for( int8_t dx=0; dx<=1; dx++ )
    //       {
    //         auto ns = lmesh_old.findNeighbors(iOct_old, {dx,dy,dz});
    //         DYABLO_ASSERT_KOKKOS_DEBUG( ns.size() == 1, "Sibling not found" );
    //         auto iOct_old_n = ns[0];
    //         if( iOct_old_n.isGhost ) 
    //         {
    //           count++;
    //           if(final)
    //             coarsened_ghosts( count ) = iOct_old_n.iOct;
    //         }
    //       }
    //     }
    //   }
    // }, coarsened_ghosts_count);

    Kokkos::View< uint32_t* > coarsened_ghosts("coarsened_ghosts", 7);
    uint32_t coarsened_ghosts_count = 0;
    Kokkos::parallel_reduce( "Remap::list_ghost_coarsened", 1,
      KOKKOS_LAMBDA( uint32_t dummy, uint32_t& count)
    {
      // Coarsened ghosts can only happen for last octant of process
      uint32_t iOct_new = nbOcts_new - 1;
      OctantIndex iOct_old = remapper.get_old_octant({iOct_new, false});
      int level_old = lmesh_old.getLevel(iOct_old);
      int level_new = lmesh_new.getLevel({iOct_new, false});
      // Skip if not coarsened
      if( level_new < level_old )
      {
        // If last sibling is not ghost, everything is local
        int32_t dz_max = (ndim == 2)? 0:1;
        for( int32_t dz=0; dz<=dz_max; dz++ )
        for( int32_t dy=0; dy<=1; dy++ )
        for( int32_t dx=0; dx<=1; dx++ )
        {
          DYABLO_ASSERT_KOKKOS_DEBUG( !lmesh_old.isBoundary(iOct_old, {dx,dy,dz}), "Sibling should be inside domain" )

          OctantIndex iOct_old_n = lmesh_old.findNeighbor(iOct_old, {dx,dy,dz});
          if( iOct_old_n.isGhost ) 
          {
            coarsened_ghosts( count ) = iOct_old_n.iOct;
            count++;
          }
        }
      }
    }, coarsened_ghosts_count);

    Kokkos::resize( coarsened_ghosts, coarsened_ghosts_count );

    GhostCommunicator_full_blocks::OctSubset subset(*ghost_comm_full, coarsened_ghosts);
    this->ghost_comm_full->exchange_ghosts_subset(Uin, subset);
  }

  remap_aux( Uin, Uout, remapper );

  // Deallocate fields_old before reallocating empty fields
  Uin = UserData::FieldAccessor();

  user_data.extend_fields();
}



} //namespace dyablo 