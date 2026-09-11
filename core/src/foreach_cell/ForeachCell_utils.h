#pragma once

#include "ForeachCell.h"

namespace dyablo {

namespace{

using CellIndex = ForeachCell::CellIndex;
using CellArray_global_ghosted = ForeachCell::CellArray_global_ghosted;

}

/**
 * @brief Iterate over smaller neighbor cells
 * for example, neighbors in 3D are the 4 cells that are in contact with the original cell
 * @tparam enable_different_block when off all neighbors are supposed to live in the same octant,
 *         this is true when block size is pair (obviously not working in cell-based)
 * @param ndim 2D or 3D
 * @param iCell_n smaller neigbors origin (neighbor with the smallest morton index)
 * @param offset is the offset that was applied to get iCell_n
 * @param lmesh LightOctree used to find neighbor octants when enable_different_block = on
 * @param apply_neighbor is a (const CellIndex&) -> void functor that performs an operation with 
 *                       each sibling
 * @returns number of neighbor cells
 **/
template< bool enable_different_block, typename Func >
KOKKOS_INLINE_FUNCTION
int foreach_smaller_neighbor( int ndim, const CellIndex& iCell_n, const CellIndex::offset_t& offset, const LightOctree& lmesh, const Func& apply_neighbor )
{
  DYABLO_ASSERT_KOKKOS_DEBUG( iCell_n.level_diff() == -1, "iCell must be smaller neighbor for foreach_smaller_neighbor" );
  DYABLO_ASSERT_KOKKOS_DEBUG( enable_different_block || ( iCell_n.bx()%2 == 0 && iCell_n.by()%2 == 0 && (ndim==2 || iCell_n.bz()%2 == 0) ),
    "enable_different_block must be activated for cell-based or odd block size" );

  int di_count = (offset[IX]==0)?2:1;
  int dj_count = (offset[IY]==0)?2:1;
  int dk_count = (ndim==3 && offset[IZ]==0)?2:1;
  for( int32_t dk=0; dk<dk_count; dk++ )
  for( int32_t dj=0; dj<dj_count; dj++ )
  for( int32_t di=0; di<di_count; di++ )
  {
      CellIndex iCell_ghost;
      if constexpr ( enable_different_block )
      {
        // Looking for same-size (local or remote)
        // neighbor can't be smaller (Assert if it is)
        ForeachCell::SearchMode_neighbor search_mode( lmesh, ForeachCell::SearchMode_neighbor::ASSERT );
        iCell_ghost = iCell_n.getNeighbor<CellIndex::LOCAL_TO_BLOCK, CellIndex::SAME_SIZE>( di, dj, dk, search_mode );
      }        
      else
      {
        // Looking for local cell (Assert if it's not)
        ForeachCell::SearchMode_local search_mode( ForeachCell::SearchMode_local::ASSERT );
        iCell_ghost = iCell_n.getNeighbor<CellIndex::LOCAL_TO_BLOCK>( di, dj, dk, search_mode );
      }
      apply_neighbor(iCell_ghost);
  }
  return di_count*dj_count*dk_count;
}

template< typename Func>
KOKKOS_INLINE_FUNCTION
int foreach_smaller_neighbor_gathered( int ndim, const CellIndex& iCell_n, const CellIndex::offset_t& offset, const Func& apply_neighbor )
{
  LightOctree* lmesh = nullptr;
  return foreach_smaller_neighbor<false>(ndim, iCell_n, offset, *lmesh, apply_neighbor);
}

template< typename Func>
KOKKOS_INLINE_FUNCTION
int foreach_smaller_neighbor_scattered( int ndim, const CellIndex& iCell_n, const CellIndex::offset_t& offset, const LightOctree& lmesh, const Func& apply_neighbor )
{
  return foreach_smaller_neighbor<true>(ndim, iCell_n, offset, lmesh, apply_neighbor);
}

/**
 * Iterate over sibling cells
 * @tparam enable_different_block when off all siblings are supposed to live in the same octant,
 *         this is true when block size is pair (obviously not working in cell-based)
 * @param ndim 2D or 3D
 * @param iCell_n first cell from the bigger supercell (sibling with the smallest morton index)
 * @param lmesh LightOctree used to find neighbor octants when enable_different_block = on
 * @param apply_neighbor is a (const CellIndex&) -> void functor that performs an operation with 
 *                       each sibling
 * @returns number of sibling cells
 * NOTE : for example in 3D, sibings are the 8 cells that form a bigger supercell
 **/
template< bool enable_different_block, typename Func >
KOKKOS_INLINE_FUNCTION
int foreach_sibling( int ndim, const CellIndex& iCell_n, const LightOctree& lmesh, const Func& apply_sibling )
{
  // enable_different_block must be activated for cell-based or odd block size
  DYABLO_ASSERT_KOKKOS_DEBUG( enable_different_block || ( iCell_n.bx()%2 == 0 && iCell_n.by()%2 == 0 && (ndim==2 || iCell_n.bz()%2 == 0) ),
    "enable_different_block must be activated for cell-based or odd block size" );
  int dk_count = ndim==3?2:1;
  for( int32_t dk=0; dk<dk_count; dk++ )
  for( int32_t dj=0; dj<2; dj++ )
  for( int32_t di=0; di<2; di++ )
  {
      CellIndex iCell_ghost;
      if constexpr ( enable_different_block )
      {
        // Looking for same-size (local or remote)
        // neighbor can't be smaller (Assert if it is)
        ForeachCell::SearchMode_neighbor search_mode( lmesh, ForeachCell::SearchMode_neighbor::ASSERT );
        iCell_ghost = iCell_n.getNeighbor<CellIndex::LOCAL_TO_BLOCK, CellIndex::SAME_SIZE>( di, dj, dk, search_mode );
      }        
      else
      {
        // Looking for local cell (Assert if it's not)
        ForeachCell::SearchMode_local search_mode( ForeachCell::SearchMode_local::ASSERT );
        iCell_ghost = iCell_n.getNeighbor<CellIndex::LOCAL_TO_BLOCK>( di, dj, dk, search_mode );
      }
      apply_sibling(iCell_ghost);
  }
  return 2*2*dk_count;
}

/// foreach_sibling with enable_different_block = false, lmesh is not needed
template< typename Func >
KOKKOS_INLINE_FUNCTION
int foreach_sibling_gathered( int ndim, const CellIndex& iCell_n, const Func& apply_sibling )
{
  LightOctree* lmesh = nullptr;
  return foreach_sibling<false>(ndim, iCell_n, *lmesh, apply_sibling);
}

/// foreach_sibling with enable_different_block = true
template< typename Func >
KOKKOS_INLINE_FUNCTION
int foreach_sibling_scattered( int ndim, const CellIndex& iCell_n, const LightOctree& lmesh, const Func& apply_sibling )
{
  return foreach_sibling<true>(ndim, iCell_n, lmesh, apply_sibling);
}

} // namespace dyablo