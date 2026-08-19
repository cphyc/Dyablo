#pragma once

#include "ForeachCell.h"
#include "foreach_cell/Binned_iOcts.hpp"

namespace dyablo {

template<bool _leaves_local, bool _leaves_ghost, bool _intermediates_local, bool _intermediates_ghost>
class IterationSpace_subset_impl
{
private:

  ForeachCell::IterationSpace_fullArray_impl<_leaves_local, _leaves_ghost, _intermediates_local, _intermediates_ghost> full_array;

  Kokkos::View<uint32_t*> iOcts_locals;
  Kokkos::View<uint32_t*> iOcts_ghosts;
  Kokkos::View<uint32_t*> iOcts_intermediates;
  Kokkos::View<uint32_t*> iOcts_intermediate_ghosts;

public:
  IterationSpace_subset_impl(  const ForeachCell::CellArray_shape& iter_space_level,
                              Kokkos::View<uint32_t*>& iOcts_locals,
                              Kokkos::View<uint32_t*>& iOcts_ghosts,
                              Kokkos::View<uint32_t*>& iOcts_intermediates,
                              Kokkos::View<uint32_t*>& iOcts_intermediate_ghosts)
  : full_array( iter_space_level ),
    iOcts_locals(iOcts_locals),
    iOcts_ghosts(iOcts_ghosts),
    iOcts_intermediates(iOcts_intermediates),
    iOcts_intermediate_ghosts(iOcts_intermediate_ghosts)
  {}

  KOKKOS_INLINE_FUNCTION
  uint32_t bx() const         { return full_array.bx(); }
  KOKKOS_INLINE_FUNCTION
  uint32_t by() const         { return full_array.by(); }
  KOKKOS_INLINE_FUNCTION
  uint32_t bz() const         { return full_array.bz(); }

  KOKKOS_INLINE_FUNCTION
  uint32_t iOct_count() const { return full_array.iOct_count(); }

  KOKKOS_INLINE_FUNCTION
  ForeachCell::CellIndex getCellIndex(uint32_t iOct_in, uint32_t iCell_ijk, uint32_t i, uint32_t j, uint32_t k) const
  {
    ForeachCell::CellIndex iCell = full_array.getCellIndex(iOct_in, iCell_ijk, i, j, k);
    LightOctree::OctantIndex iOct_raw = iCell.iOct();

    uint32_t iOct_filtered = 0;
    if(      !iOct_raw.isGhost && !iOct_raw.isIntermediate )
        iOct_filtered = iOcts_locals(iOct_raw.iOct);
    else if(  iOct_raw.isGhost && !iOct_raw.isIntermediate )
        iOct_filtered = iOcts_ghosts(iOct_raw.iOct);
    else if( !iOct_raw.isGhost &&  iOct_raw.isIntermediate )
        iOct_filtered = iOcts_intermediates(iOct_raw.iOct);
    else//if( iOct_raw.isGhost &&  iOct_raw.isIntermediate )
        iOct_filtered = iOcts_intermediate_ghosts(iOct_raw.iOct);
  
        
    LightOctree::OctantIndex iOct_filtered_struct = iOct_raw;
    iOct_filtered_struct.iOct = iOct_filtered;

    iCell.set_iOct( iOct_filtered_struct );

    return iCell;
  }
};

/***
 * Class to generate Iterationspace instances for per-level operations
 * 
 * This precomputes IterationSpaces for every level to quickly get 
 * per-level IterationSpaces for multiple levels.
 ***/
class IterationSpace_levels
{
public:
  IterationSpace_levels(const LightOctree& lmesh, int level_max)
    : binned_iOcts_levels(lmesh, level_max)
  {/*Empty*/}

  /***
   * Generate an IterationSpace for foreach_cell that filters only cells from a specific AMR level
   * 
   * @tparam leaves_local, leaves_ghost, intermediates_local select what kind of cells are enabled (same as IterationSpace_fullArray_impl)
   * @param level IterationSpace will select only cells at this level
   * @param shape shape of the (full) array from which cells are filtered
   */
  template<bool leaves_local, bool leaves_ghost, bool intermediates_local, bool intermediates_ghost>
  IterationSpace_subset_impl<leaves_local, leaves_ghost, intermediates_local, intermediates_ghost> getIterationSpace(int level, const ForeachCell::CellArray_shape& shape )
  {
      uint32_t bx = shape.bx;
      uint32_t by = shape.by;
      uint32_t bz = shape.bz;
      uint32_t nbFields = shape.nbFields;

      Kokkos::View<uint32_t*> iOcts_locals, iOcts_ghosts, iOcts_intermediates, iOcts_intermediate_ghosts;
      if( leaves_local ) iOcts_locals = binned_iOcts_levels.get_iOcts_leaves(level);
      if( leaves_ghost ) iOcts_ghosts = binned_iOcts_levels.get_iOcts_ghost_leaves(level);
      if( intermediates_local ) iOcts_intermediates = binned_iOcts_levels.get_iOcts_intermediates(level);
      if( intermediates_ghost ) iOcts_intermediate_ghosts = binned_iOcts_levels.get_iOcts_ghost_intermediates(level);
      
      ForeachCell::CellArray_shape iter_space{
          .bx=bx, .by=by, .bz=bz, 
          .nbFields = nbFields,
          .nbOcts = (uint32_t)iOcts_locals.size(),
          .nbGhosts = (uint32_t)iOcts_ghosts.size(),
          .nbIntermediateOcts = (uint32_t)iOcts_intermediates.size(),
          .nbIntermediateGhosts = (uint32_t)iOcts_intermediate_ghosts.size(),
      };

      return IterationSpace_subset_impl<leaves_local, leaves_ghost, intermediates_local, intermediates_ghost>( iter_space, iOcts_locals, iOcts_ghosts, iOcts_intermediates, iOcts_intermediate_ghosts );
  }

private:
  Binned_iOcts_levels binned_iOcts_levels;

};

} //namespace dyablo