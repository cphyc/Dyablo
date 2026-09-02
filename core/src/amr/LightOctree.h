#pragma once

#include "LightOctree_forward.h"

//#include "amr/LightOctree_hashmap.h"
#include "LightOctree_hashmap_precompute.h"

namespace dyablo{

//using LightOctree_hashmap_Impl = LightOctree_hashmap;
using LightOctree_hashmap_Impl = LightOctree_hashmap_precompute;

class LightOctree : public LightOctree_hashmap_Impl
{
public:
  using  LightOctree_hashmap_Impl::LightOctree_hashmap_Impl;
};

}

namespace dyablo::LightOctree_tools{

/**
 * Iterate over neighbor octants, call f for each
 * @param lmesh LightOctree used for fetching the neighbors
 * @param iOct_origin Origin octant
 * @param iOct_neighbor Neighbor octant found with lmesh.getNeighbor(iOct_origin, offset)
 * @param offset is the offset that was applied to get iOct_neighbor
 * @param f function alled for each neighbor octant 
 * @returns number of neighbors processed
 * Note : Make sure that iOct_neighbor matches origin and offset, and doesn't fall in a boundary 
 **/
template< typename F >
KOKKOS_INLINE_FUNCTION
int foreach_neighbor_octant( const LightOctree& lmesh, 
                             const LightOctree::OctantIndex& iOct_origin,
                             const LightOctree::OctantIndex& iOct_neighbor,
                             const LightOctree::offset_t& offset,
                             const F& f )
{
  DYABLO_ASSERT_KOKKOS_DEBUG( !lmesh.isBoundary(iOct_origin, offset) 
                            && iOct_neighbor.iOct == lmesh.findNeighbor(iOct_origin, offset).iOct 
                            , "foreach_neighbor_octant : iOct_neighbor must be lmesh.getNeighbor(iOct_origin, offset)");

  if( lmesh.getLevel( iOct_neighbor ) <= lmesh.getLevel( iOct_origin ) )
  {
    f( iOct_neighbor );
    return 1;
  }
  else //( lmesh.getLevel( iOct_neighbor ) > lmesh.getLevel( iOct_origin ) )
  {
    int ndim = lmesh.getNdim();  
    int sz_max = (ndim==2) ? 0 : (offset[IZ]==0); // No offset in z in 2D
    int sy_max = (offset[IY]==0);
    int sx_max = (offset[IX]==0); // Constrained to plane adjacent to neighbor if offset in this direction
    
    int count = 0;
    for( int sz=0; sz<=sz_max; sz++ )
    for( int sy=0; sy<=sy_max; sy++ )
    for( int sx=0; sx<=sx_max; sx++ )
    {
      count++;
      constexpr bool ghosts_enabled = true;
      LightOctree::OctantIndex iOct_smaller_n = lmesh.findNeighbor<ghosts_enabled>( iOct_neighbor, {sx, sy, sz} );
      DYABLO_ASSERT_KOKKOS_DEBUG( lmesh.getLevel(iOct_smaller_n) == lmesh.getLevel(iOct_neighbor), "All neighbors should be at same level" );
      f( iOct_smaller_n );
    }
      return count;
  }
}

};
