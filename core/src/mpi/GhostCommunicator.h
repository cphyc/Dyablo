#pragma once

#include "amr/AMRmesh.h"

#include "GhostCommunicator_partial_blocks.h"
#include "GhostCommunicator_full_blocks.h"

namespace dyablo {

template< typename Impl >
/***
 * Interface to implement for a GhostCommunicator
 ***/
class GhostCommunicator_impl : protected Impl
{
public:
  using CellArray_shape = AMRBlockForeachCell_CellArray_impl::CellArray_shape;

  /**
   * @param mesh AMR mesh to determline neighborhood
   * @param shape shape of the blocks in the arrays
   * @param ghost_count number for ghosts needed for stencil operations (this is a minimum, actual comms can recieve more ghosts)
   * @param mpi_comm you know what it is
   **/
  GhostCommunicator_impl( const AMRmesh& mesh, 
                          const ForeachCell::CellArray_global_ghosted::Shape_t& shape, 
                          int ghost_count, 
                          const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
  : Impl(mesh.getMesh(), shape, ghost_count, mpi_comm)
  {}

  static std::string name()
  {
    return Impl::name();
  }

  /// Number of ghost blocks (possibly partial blocks) for this mesh 
  uint32_t getNumGhosts() const
  {
    return Impl::getNumGhosts();
  }
  uint32_t getNumIntermediateGhosts() const
  {
    return Impl::getNumIntermediateGhosts();
  }

  /***
   * Send ghosts cells for the selected Fields in the accessor
   * Ghosts from other fields in UserData WILL NOT be modified
   * Cells at a distance greater than ghost_count from the local domain have undefined value
   * (they may be exchanged or not depending on the backend)
   ***/
  void exchange_ghosts( const UserData::FieldAccessor& U ) const
  {
    Impl::exchange_ghosts(U);
  }
  void exchange_ghosts_at_level( const UserData::FieldAccessor& U, const uint8_t level ) const
  {
    Impl::exchange_ghosts_at_level(U, level);
  }
  void exchange_intermediate_ghosts( const UserData::FieldAccessor& U ) const
  {
    Impl::exchange_intermediate_ghosts(U);
  }
  void exchange_intermediate_ghosts_at_level( const UserData::FieldAccessor& U, const uint8_t level ) const
  {
    Impl::exchange_intermediate_ghosts_at_level(U, level);
  }

  /***
   * Send ghosts cells for all fields in the CellArray
   * Cells at a distance greater than ghost_count from the local domain have undefined value
   * (they may be exchanged or not depending on the backend)
   ***/
  void exchange_ghosts( const ForeachCell::CellArray_global_ghosted& U ) const
  {
    Impl::exchange_ghosts(U);
  }
  void exchange_ghosts_at_level( const ForeachCell::CellArray_global_ghosted& U, const uint8_t level ) const
  {
    Impl::exchange_ghosts_at_level(U, level);
  }
  void exchange_intermediate_ghosts( const ForeachCell::CellArray_global_ghosted& U ) const
  {
    Impl::exchange_intermediate_ghosts(U);
  }
  void exchange_intermediate_ghosts_at_level( const ForeachCell::CellArray_global_ghosted& U, const uint8_t level ) const
  {
    Impl::exchange_intermediate_ghosts_at_level(U, level);
  }


  /***
   * Reduce ghosts cells for the selected Fields in the accessor
   * BE SURE TO SET ALL YOUR GHOSTS TO ZERO TO AVOID ISSUES
   * 
   * Other fields from UserData WILL NOT be modified
   * Ghost Cells un neighboring blocks at a distance greater than ghost_count from 
   * the local domain may or may not be exchanged depending on the backend, be sure to set them to zero
   ***/
  void reduce_ghosts( UserData::FieldAccessor& U ) const
  {
    Impl::reduce_ghosts(U);
  }
  void reduce_ghosts_at_level( UserData::FieldAccessor& U, const uint8_t level ) const
  {
    Impl::reduce_ghosts_at_level(U, level);
  }
  void reduce_intermediate_ghosts( UserData::FieldAccessor& U ) const
  {
    Impl::reduce_intermediate_ghosts(U);
  }
  void reduce_intermediate_ghosts_at_level( UserData::FieldAccessor& U, const uint8_t level) const
  {
    Impl::reduce_intermediate_ghosts_at_level(U, level);
  }

  void reduce_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
  {
    Impl::reduce_ghosts(U);
  }
  void reduce_ghosts_at_level( ForeachCell::CellArray_global_ghosted& U, const uint8_t level ) const
  {
    Impl::reduce_ghosts_at_level(U, level);
  }
  void reduce_intermediate_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
  {
    Impl::reduce_intermediate_ghosts(U);
  }
  void reduce_intermediate_ghosts_at_level( ForeachCell::CellArray_global_ghosted& U, const uint8_t level) const
  {
    Impl::reduce_intermediate_ghosts_at_level(U, level);
  }

  void init_intermediates( 
    const AMRmesh& mesh, 
    const ForeachCell::CellArray_global_ghosted::Shape_t& shape, 
    const uint32_t ghost_count, 
    const MpiComm& mpi_comm )
  {
    Impl::init_intermediates(mesh.getMesh(), shape, ghost_count, mpi_comm);
  }

  void sort_ghosts_by_levels(const LightOctree& lmesh, const uint8_t level_max)
  {
    Impl::sort_ghosts_by_levels(lmesh, level_max);
  }
  void sort_intermediate_ghosts_by_levels(const LightOctree& lmesh, const uint8_t level_max)
  {
    Impl::sort_intermediate_ghosts_by_levels(lmesh, level_max);
  }

};

// GhostCommunicator partial block implementation is only compatible with AMRmesh_hashmap_new, use full block otherwise
using GhostCommunicator = GhostCommunicator_impl< std::conditional_t< std::is_same_v<AMRmesh::Impl_t, AMRmesh_hashmap_new>, 
                                                                      GhostCommunicator_partial_blocks, 
                                                                      GhostCommunicator_full_blocks > >;

}// namespace dyablo
