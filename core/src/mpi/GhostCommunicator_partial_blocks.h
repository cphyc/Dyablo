#pragma once

#include "user_data/UserData.h"
#include "user_data/FieldAccessor.h"

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

    // Arrays convertible to an accessor with a dynamic number of fields (FieldAccessor_dynamic_convertible)
    // are converted to FieldAccessor_dynamic_t and communicated by the non-template overloads

    template< FieldAccessor_dynamic_convertible Array_t >
    void exchange_ghosts( const Array_t& U ) const
    {
      exchange_ghosts( FieldAccessor_dynamic_t<Array_t>( U ) );
    }
    void exchange_ghosts( const UserData::FieldAccessor_impl<DYNAMIC_FIELD_COUNT>& U ) const;
    void exchange_ghosts( const UserData::FieldAccessor_fulltree_impl<DYNAMIC_FIELD_COUNT>& U ) const;

    using OctSubset = GhostCommunicator_partial_blocks_OctSubset;

    template< FieldAccessor_dynamic_convertible Array_t >
    void exchange_ghosts_subset( const Array_t& U, const OctSubset& subset ) const
    {
      exchange_ghosts_subset( FieldAccessor_dynamic_t<Array_t>( U ), subset );
    }
    void exchange_ghosts_subset( const UserData::FieldAccessor_impl<DYNAMIC_FIELD_COUNT>& U, const OctSubset& subset ) const;
    void exchange_ghosts_subset( const UserData::FieldAccessor_fulltree_impl<DYNAMIC_FIELD_COUNT>& U, const OctSubset& subset ) const;

    template< FieldAccessor_dynamic_convertible Array_t >
    void reduce_ghosts( Array_t& U ) const
    {
      FieldAccessor_dynamic_t<Array_t> U_dynamic( U );
      reduce_ghosts( U_dynamic );
    }
    void reduce_ghosts( UserData::FieldAccessor_impl<DYNAMIC_FIELD_COUNT>& U ) const;
    void reduce_ghosts( UserData::FieldAccessor_fulltree_impl<DYNAMIC_FIELD_COUNT>& U ) const;

    template< FieldAccessor_dynamic_convertible Array_t >
    void reduce_ghosts_subset( Array_t& U, const OctSubset& subset ) const
    {
      FieldAccessor_dynamic_t<Array_t> U_dynamic( U );
      reduce_ghosts_subset( U_dynamic, subset );
    }
    void reduce_ghosts_subset( UserData::FieldAccessor_impl<DYNAMIC_FIELD_COUNT>& U, const OctSubset& subset ) const;
    void reduce_ghosts_subset( UserData::FieldAccessor_fulltree_impl<DYNAMIC_FIELD_COUNT>& U, const OctSubset& subset ) const;

//private:
    using Pdata = GhostCommunicator_partial_blocks_Pdata;
    std::unique_ptr<Pdata> pdata;  
    GhostCommunicator_partial_blocks( const Pdata& );
};

} // namespace dyablo