#pragma once

#include "user_data/UserData.h"
#include "user_data/FieldAccessor.h"
#include "mpi/ViewCommunicator.h"

namespace dyablo {

class GhostCommunicator_full_blocks : protected ViewCommunicator
{
private:
    bool intermediates;
public:
    template< typename AMRmesh_t >
    GhostCommunicator_full_blocks( const AMRmesh_t& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& /*shape*/,  int /*ghost_count*/, bool intermediates=false, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
    : ViewCommunicator( ViewCommunicator::from_mesh(amr_mesh, intermediates, mpi_comm) ),
      intermediates(intermediates)
    {}

    static std::string name()
    {
      return "GhostCommunicator_full_blocks";
    }

    bool has_intermediates() const
    {
      return intermediates;
    }

    Kokkos::View<uint32_t*> iOcts_send() const;
     
    /// @copydoc GhostCommunicator_base::getNumGhosts
    uint32_t getNumGhosts() const
    {
      return ViewCommunicator::getNumGhosts();
    }

    // Arrays convertible to an accessor with a dynamic number of fields (FieldAccessor_dynamic_convertible)
    // are converted to FieldAccessor_dynamic_t and communicated by the non-template overloads

    template< FieldAccessor_dynamic_convertible Array_t >
    void exchange_ghosts( const Array_t& U ) const
    {
      exchange_ghosts( FieldAccessor_dynamic_t<Array_t>( U ) );
    }
    void exchange_ghosts( const UserData::FieldAccessor_impl<DYNAMIC_FIELD_COUNT>& U ) const;
    void exchange_ghosts( const UserData::FieldAccessor_fulltree_impl<DYNAMIC_FIELD_COUNT>& U ) const;

    struct OctSubset
    {
      /***
       * Initialize with only iOct of ghost (recieved) octants
       * @param subset_iOcts_recv are iOcts of ghosts to filter and recieve
       * List of octants to send will be computed internally
       **/
      OctSubset(const GhostCommunicator_full_blocks& comm_full, Kokkos::View<uint32_t*> subset_iOcts_recv );
      /***
       * Initialize with list of ghost octants (recieved), but also a list of octants to send
       * This is for optimization purpose : computing the list of octants to send (requires an 
       * MPI communication) can then be skipped. The list must match what would be computed internally 
       * by the other constructor
       * @param subset_iOcts_recv are iOcts of ghosts to filter and recieve
       * @param subset_iOcts_send are indexes of filtered iOcts in comm_full send list 
       * NOTE : iOcts comm_full.to_send(subset_iOcts_send(i)) will be sent, not subset_iOcts_send(i)
       */
      OctSubset(const GhostCommunicator_full_blocks& comm_full, Kokkos::View<uint32_t*> subset_iOcts_send, Kokkos::View<uint32_t*> subset_iOcts_recv );

      uint32_t nbGhosts() const
      {
        return partial_comm->getNumGhosts();
      }
    
      std::shared_ptr<ViewCommunicator> partial_comm;
      Kokkos::View<uint32_t*> subset_iOcts;
    };

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

    // -- Special optimization paths --
    // Avoid conversion to dynamic FieldAccessor 
    /// Communicates all fields at once (not converted to an accessor : that would communicate fields one by one)
    void exchange_ghosts( ForeachCell::CellArray_global_ghosted& U ) const;
    void reduce_ghosts( ForeachCell::CellArray_global_ghosted& U ) const;
  };

} // namespace dyablo