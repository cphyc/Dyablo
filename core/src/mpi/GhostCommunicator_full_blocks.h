#pragma once

#include "user_data/UserData.h"
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

    template<typename T = real_t>
    void exchange_ghosts( const UserData::FieldAccessor_t<T>& U ) const;
    template<typename T = real_t>
    void exchange_ghosts( const UserData::FieldAccessor_fulltree_t<T>& U ) const;

    void exchange_ghosts( ForeachCell::CellArray_global_ghosted& U ) const;

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
};

} // namespace dyablo