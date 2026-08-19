#pragma once

#include "ForeachCell.h"

namespace dyablo {

namespace{

using CellIndex = ForeachCell::CellIndex;
using CellArray_global_ghosted = ForeachCell::CellArray_global_ghosted;

}

/**
 * Iterate over neighbors when current cell is a smaller neighbor
 * @tparam ndim 2D/3D
 * @param offset is the offset that was applied to get current CellIndex
 * @param apply_neighbor is a (const CellIndex&) -> void functor that performs an operation with 
 *                       each neighbor smaller cell
 * @param search_mode how to search siblings of origin neighbor cell
 * @returns number of sibling cells
 * NOTE : level_diff() must be -1
 * NOTE : for example, neighbors in 3D are the 4 cells that are in contact with the original cell
 **/
template< typename SearchMode_t, typename Func >
KOKKOS_INLINE_FUNCTION
int foreach_smaller_neighbor( int ndim, const CellIndex& iCell, const CellIndex::offset_t& offset, const SearchMode_t& search_mode, const Func& apply_neighbor )
{
  DYABLO_ASSERT_KOKKOS_DEBUG( iCell.level_diff() == -1, "iCell must be smaller neighbor for foreach_smaller_neighbor" );
  [[maybe_unused]]constexpr bool enable_different_block = std::is_same_v<SearchMode_t, ForeachCell::SearchMode_neighbor> ;
  DYABLO_ASSERT_KOKKOS_DEBUG( enable_different_block || ( iCell.bx()%2 == 0 && iCell.by()%2 == 0 && (ndim==2 || iCell.bz()%2 == 0) ),
    "enable_different_block must be activated for cell-based or odd block size" );

  int di_count = (offset[IX]==0)?2:1;
  int dj_count = (offset[IY]==0)?2:1;
  int dk_count = (ndim==3 && offset[IZ]==0)?2:1;
  for( int32_t dk=0; dk<dk_count; dk++ )
  for( int32_t dj=0; dj<dj_count; dj++ )
  for( int32_t di=0; di<di_count; di++ )
  {
      CellIndex iCell_ghost = iCell.getNeighbor( di, dj, dk, search_mode );
      apply_neighbor(iCell_ghost);
  }
  return di_count*dj_count*dk_count;
}

template< int ndim, typename SearchMode_t, typename Func >
KOKKOS_INLINE_FUNCTION
int foreach_smaller_neighbor( const CellIndex& iCell, const CellIndex::offset_t& offset, const SearchMode_t& search_mode, const Func& apply_neighbor )
{
  return foreach_smaller_neighbor( ndim, iCell, offset, search_mode, apply_neighbor );
}

/**
 * Iterate over sibling cells
 * @param apply_neighbor is a (const CellIndex&) -> void functor that performs an operation with 
 *                       each sibling cell
 * @param iCell first cell from the bigger supercell (sibling with the smallest morton index)
 * @param apply_neighbor is a (const CellIndex&) -> void functor that performs an operation with 
 *                       each sibling
 * @returns number of sibling cells
 * NOTE : for example in 3D, sibings are the 8 cells that form a bigger supercell
 **/
template< typename SearchMode_t, typename Func >
KOKKOS_INLINE_FUNCTION
int foreach_sibling( int ndim, const CellIndex& iCell, const SearchMode_t& search_mode, const Func& apply_sibling )
{
  // enable_different_block must be activated for cell-based or odd block size
  [[maybe_unused]] constexpr bool enable_different_block = std::is_same_v<SearchMode_t, ForeachCell::SearchMode_neighbor> ;
  DYABLO_ASSERT_KOKKOS_DEBUG( enable_different_block || ( iCell.bx()%2 == 0 && iCell.by()%2 == 0 && (ndim==2 || iCell.bz()%2 == 0) ),
    "enable_different_block must be activated for cell-based or odd block size" );
  int dk_count = ndim==3?2:1;
  for( int32_t dk=0; dk<dk_count; dk++ )
  for( int32_t dj=0; dj<2; dj++ )
  for( int32_t di=0; di<2; di++ )
  {
      CellIndex iCell_ghost;
      if constexpr ( enable_different_block )
        iCell_ghost = iCell.getNeighbor( di, dj, dk, search_mode );
      else
        iCell_ghost = iCell.getNeighbor<CellIndex::LOCAL_TO_BLOCK>( di, dj, dk, search_mode );
      apply_sibling(iCell_ghost);
  }
  return 2*2*dk_count;
}

template< int ndim, typename SearchMode_t, typename Func >
KOKKOS_INLINE_FUNCTION
int foreach_sibling( const CellIndex& iCell, const SearchMode_t& search_mode, const Func& apply_sibling )
{
  return foreach_sibling(ndim, iCell, search_mode, apply_sibling);
}

} // namespace dyablo