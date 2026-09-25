#pragma once

#include "amr/AMRmesh.h"

#include "GhostCommunicator_partial_blocks.h"

namespace dyablo {

template< typename Impl >
/***
 * Interface to implement for a GhostCommunicator
 ***/
class GhostCommunicator_impl : public Impl
{
public:
  /**
   * @param mesh AMR mesh to determline neighborhood
   * @param shape shape of the blocks in the arrays
   * @param ghost_count number for ghosts needed for stencil operations (this is a minimum, actual comms can recieve more ghosts)
   * @param mpi_comm you know what it is
   **/
  GhostCommunicator_impl( const AMRmesh& mesh, 
                          const ForeachCell::CellArray_global_ghosted::Shape_t& shape, 
                          int ghost_count,
                          bool intermediates = false, 
                          const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
  : Impl(mesh, shape, ghost_count, intermediates, mpi_comm)
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

  bool has_intermediates() const
  {
    return Impl::has_intermediates();
  }

  using OctSubset = typename Impl::OctSubset;

  /***
   * Send ghosts cells for the selected Fields in the accessor
   * Ghosts from other fields in UserData WILL NOT be modified
   * Cells at a distance greater than ghost_count from the local domain have undefined value
   * (they may be exchanged or not depending on the backend)
   ***/
  template<typename T = real_t>
  void exchange_ghosts( const UserData::FieldAccessor_t<T>& U ) const
  {
    Impl::exchange_ghosts(U);
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

  template<typename T = real_t>
  void exchange_ghosts_subset( const UserData::FieldAccessor_t<T>& U, const OctSubset& subset ) const
  {
    Impl::exchange_ghosts_subset(U, subset);
  }


  /***
   * Reduce ghosts cells for the selected Fields in the accessor
   * BE SURE TO SET ALL YOUR GHOSTS TO ZERO TO AVOID ISSUES
   * 
   * Other fields from UserData WILL NOT be modified
   * Ghost Cells un neighboring blocks at a distance greater than ghost_count from 
   * the local domain may or may not be exchanged depending on the backend, be sure to set them to zero
   ***/
  template<typename T = real_t>
  void reduce_ghosts( UserData::FieldAccessor_t<T>& U ) const
  {
    Impl::reduce_ghosts(U);
  }

  void reduce_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
  {
    Impl::reduce_ghosts(U);
  }

};

using GhostCommunicator = GhostCommunicator_partial_blocks;

}// namespace dyablo
