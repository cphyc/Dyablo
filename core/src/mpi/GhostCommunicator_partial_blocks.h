#pragma once

#include "user_data/UserData.h"

namespace dyablo {

class GhostCommunicator_partial_blocks;
struct GhostCommunicator_partial_blocks_Pdata;
struct GhostCommunicator_partial_blocks_OctSubset_Pdata;

struct GhostCommunicator_partial_blocks_OctSubset
{
  GhostCommunicator_partial_blocks_OctSubset(const GhostCommunicator_partial_blocks_OctSubset& o);

  GhostCommunicator_partial_blocks_OctSubset(const GhostCommunicator_partial_blocks& comm_full, Kokkos::View<uint32_t*> subset_iOcts_recv );
  GhostCommunicator_partial_blocks_OctSubset(const GhostCommunicator_partial_blocks& comm_full, Kokkos::View<uint32_t*> subset_iOcts_send, Kokkos::View<uint32_t*> subset_iOcts_recv );
  ~GhostCommunicator_partial_blocks_OctSubset();
  using Pdata = GhostCommunicator_partial_blocks_OctSubset_Pdata;
  std::unique_ptr<Pdata> pdata;
};

/**
 * Ghost communicator for partial blocks (ghosts) 
 * then serialize/deserialize in Kokkos kernels and use CUDA-aware MPI 
 **/
 class GhostCommunicator_partial_blocks
{
public:
    GhostCommunicator_partial_blocks() = delete;
    GhostCommunicator_partial_blocks( GhostCommunicator_partial_blocks&& ) = default;
    GhostCommunicator_partial_blocks( const GhostCommunicator_partial_blocks& );
    GhostCommunicator_partial_blocks( const AMRmesh& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape, uint32_t ghost_count, bool intermediates=false, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() );
    ~GhostCommunicator_partial_blocks();

    static std::string name()
    {
      return "GhostCommunicator_partial_blocks";
    }

    bool has_intermediates() const;

    Kokkos::View<uint32_t*> iOcts_send() const;

    /// @copydoc GhostCommunicator_base::getNumGhosts
    uint32_t getNumGhosts() const;

    template<typename T = real_t>
    void exchange_ghosts( const UserData::FieldAccessor_t<T>& U) const;
    template<typename T = real_t>
    void exchange_ghosts( const UserData::FieldAccessor_fulltree_t<T>& U) const;
    void exchange_ghosts( const ForeachCell::CellArray_global_ghosted& U ) const;

    using OctSubset = GhostCommunicator_partial_blocks_OctSubset;

    template<typename T = real_t>
    void exchange_ghosts_subset( const UserData::FieldAccessor_t<T>& U, const OctSubset& subset ) const;
    template<typename T = real_t>
    void exchange_ghosts_subset( const UserData::FieldAccessor_fulltree_t<T>& U, const OctSubset& subset ) const;

    template<typename T = real_t>
    void reduce_ghosts( UserData::FieldAccessor_t<T>& U ) const;
    template<typename T = real_t>
    void reduce_ghosts( UserData::FieldAccessor_fulltree_t<T>& U ) const;
    void reduce_ghosts( ForeachCell::CellArray_global_ghosted& U ) const; 

    template<typename T = real_t>
    void reduce_ghosts_subset( UserData::FieldAccessor_t<T>& U, const OctSubset& subset ) const;
    template<typename T = real_t>
    void reduce_ghosts_subset( UserData::FieldAccessor_fulltree_t<T>& U, const OctSubset& subset ) const;

//private:
    using Pdata = GhostCommunicator_partial_blocks_Pdata;
    std::unique_ptr<Pdata> pdata;  
    GhostCommunicator_partial_blocks( const Pdata& );
};

} // namespace dyablo