#pragma once

#include <vector>
#include "Kokkos_Core.hpp"
#include "foreach_cell/ForeachCell.h"

namespace dyablo {

/**
 * Ghost communicator for partial blocks (ghosts) 
 * then serialize/deserialize in Kokkos kernels and use CUDA-aware MPI 
 **/
class GhostCommunicator_partial_blocks
{
public: 
    using CellArray_shape = AMRBlockForeachCell_CellArray_impl::CellArray_shape;

    GhostCommunicator_partial_blocks( const AMRmesh_hashmap_new& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape,  int ghost_count, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() );

    void init( const AMRmesh_hashmap_new& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape, uint32_t ghost_count, const MpiComm& mpi_comm );
    
    void init_intermediates( 
      const AMRmesh_hashmap_new& amr_mesh, 
      const ForeachCell::CellArray_global_ghosted::Shape_t& shape, 
      const uint32_t ghost_count,  
      const MpiComm& mpi_comm );
    static std::string name()
    {
      return "GhostCommunicator_partial_blocks";
    }

    /// @copydoc GhostCommunicator_base::getNumGhosts
    uint32_t getNumGhosts() const
    {
      return this->m_local_ghost_octants;
    }
    uint32_t getNumIntermediateGhosts() const
    {
      return this->m_local_intermediate_ghost_octants;
    }

    /**
     * TODO : doc
     **/
    template< typename CellArray_t >
    void exchanging_ghosts( 
      const CellArray_t& U, 
      const int num_vars,
      const std::vector<int>& send_sizes,
      const std::vector<int>& recv_sizes,
      const Kokkos::View< uint32_t* >& send_iOct, 
      const Kokkos::View< uint32_t* >& send_iCell, 
      const Kokkos::View< uint32_t* >& recv_iOct, 
      const Kokkos::View< uint32_t* >& recv_iCell, 
      const bool isIntermediate) const
    {
      using CellIndex = ForeachCell::CellIndex;

      uint32_t total_send_size = send_iOct.size(), total_recv_size = recv_iOct.size(); // send/recv buffer size (number of cells)    
      uint32_t bx=U.getShape().bx, by=U.getShape().by, bz=U.getShape().bz ; // Block size

      if( num_vars*total_send_size == 0 )
        return;

      Kokkos::View< real_t*, Kokkos::LayoutLeft > send_buffer("exchange_ghosts::send_buffer", num_vars*total_send_size );

      Kokkos::parallel_for("exchange_ghosts::pack", total_send_size*num_vars,
        KOKKOS_LAMBDA( uint32_t ipack )
      {
        uint32_t ighost = ipack/num_vars;
        uint32_t ivar = ipack%num_vars;

        uint32_t iOct = send_iOct(ighost);
        uint32_t iCell = send_iCell(ighost);
        uint32_t i = iCell%bx; 
        uint32_t j = (iCell/bx)%by; 
        uint32_t k = (iCell/bx)/by;

        CellIndex cell_index { {iOct, false, isIntermediate}, i, j, k, bx, by, bz };
        send_buffer( ipack ) = U.at_ivar( cell_index, ivar );
      });


      Kokkos::View< real_t* > recv_buffer("exchange_ghosts::recv_buffer", num_vars*total_recv_size ); 
      #ifdef MPI_IS_CUDA_AWARE 
        Kokkos::fence();
        mpi_comm.MPI_Alltoallv( send_buffer.data(), send_sizes.data(), recv_buffer.data(), recv_sizes.data() );
        Kokkos::fence();
      #else
        {
          auto send_buffer_host = Kokkos::create_mirror_view(send_buffer);
          auto recv_buffer_host = Kokkos::create_mirror_view(recv_buffer);

          Kokkos::deep_copy(send_buffer_host, send_buffer);
          mpi_comm.MPI_Alltoallv( send_buffer_host.data(), send_sizes.data(), recv_buffer_host.data(), recv_sizes.data() );
          Kokkos::deep_copy(recv_buffer, recv_buffer_host);
        }  
      #endif

      Kokkos::parallel_for("exchange_ghosts::unpack", total_recv_size*num_vars,
        KOKKOS_LAMBDA( uint32_t ipack )
      {
        uint32_t ighost = ipack/num_vars;
        uint32_t ivar = ipack%num_vars;

        uint32_t iOct = recv_iOct(ighost);
        uint32_t iCell = recv_iCell(ighost);
        uint32_t i = iCell%bx; 
        uint32_t j = (iCell/bx)%by; 
        uint32_t k = (iCell/bx)/by;

        CellIndex cell_index { {iOct, true, isIntermediate}, i, j, k, bx, by, bz };
        U.at_ivar( cell_index, ivar ) = recv_buffer( ipack );
      });
    }

    template< typename CellArray_t >
    void exchange_ghosts( const CellArray_t& U) const
    {
      uint32_t num_vars = U.nbFields(); // number of vars for each cell

      // Number of values to send are cell_count * num_vars 
      std::vector<int> send_sizes = this->m_send_cell_count;
      for( auto& v : send_sizes )
        v*=num_vars;
      std::vector<int> recv_sizes = this->m_recv_cell_count;  
      for( auto& v : recv_sizes )
        v*=num_vars;

      const Kokkos::View< uint32_t* >& send_iOct = this->m_send_iOct;
      const Kokkos::View< uint32_t* >& send_iCell = this->m_send_iCell;
      const Kokkos::View< uint32_t* >& recv_iOct = this->m_recv_iOct;
      const Kokkos::View< uint32_t* >& recv_iCell = this->m_recv_iCell;
      exchanging_ghosts( U, num_vars, send_sizes, recv_sizes, send_iOct, send_iCell, recv_iOct, recv_iCell, false );
    }

    template< typename CellArray_t >
    void exchange_ghosts_at_level( const CellArray_t& U, const uint8_t level) const
    {
      uint32_t num_vars = U.nbFields(); // number of vars for each cell

      // Number of values to send are cell_count * num_vars 
      std::vector<int> send_sizes = this->m_send_cell_count_per_level[level];
      for( auto& v : send_sizes )
        v*=num_vars;
      std::vector<int> recv_sizes = this->m_recv_cell_count_per_level[level];  
      for( auto& v : recv_sizes )
        v*=num_vars;

      const std::vector<uint32_t>& send_total_ghost_cells_per_level = this->m_send_total_ghost_cells_per_level;
      const std::vector<uint32_t>& recv_total_ghost_cells_per_level = this->m_recv_total_ghost_cells_per_level;
      const auto send_pair_range = std::make_pair(send_total_ghost_cells_per_level[level], send_total_ghost_cells_per_level[level+1]);
      const auto recv_pair_range = std::make_pair(recv_total_ghost_cells_per_level[level], recv_total_ghost_cells_per_level[level+1]);
      
      const Kokkos::View< uint32_t* >& send_iOct = Kokkos::subview(this->m_send_iOct_per_level, send_pair_range);
      const Kokkos::View< uint32_t* >& send_iCell = Kokkos::subview(this->m_send_iCell_per_level, send_pair_range);;
      const Kokkos::View< uint32_t* >& recv_iOct = Kokkos::subview(this->m_recv_iOct_per_level, recv_pair_range);
      const Kokkos::View< uint32_t* >& recv_iCell = Kokkos::subview(this->m_recv_iCell_per_level, recv_pair_range);
      exchanging_ghosts( U, num_vars, send_sizes, recv_sizes, send_iOct, send_iCell, recv_iOct, recv_iCell, false );
    }

    template< typename CellArray_t >
    void exchange_intermediate_ghosts( const CellArray_t& U) const
    {
      uint32_t num_vars = U.nbFields(); // number of vars for each cell

      // Number of values to send are cell_count * num_vars 
      std::vector<int> send_sizes = this->m_send_intermediate_cell_count;
      for( auto& v : send_sizes )
        v*=num_vars;
      std::vector<int> recv_sizes = this->m_recv_intermediate_cell_count;  
      for( auto& v : recv_sizes )
        v*=num_vars;

      const Kokkos::View< uint32_t* >& send_iOct = this->m_send_intermediate_iOct;
      const Kokkos::View< uint32_t* >& send_iCell = this->m_send_intermediate_iCell;
      const Kokkos::View< uint32_t* >& recv_iOct = this->m_recv_intermediate_iOct;
      const Kokkos::View< uint32_t* >& recv_iCell = this->m_recv_intermediate_iCell;
      exchanging_ghosts( U, num_vars, send_sizes, recv_sizes, send_iOct, send_iCell, recv_iOct, recv_iCell, true );
    }

    template< typename CellArray_t >
    void exchange_intermediate_ghosts_at_level( const CellArray_t& U, const uint8_t level) const
    {
      uint32_t num_vars = U.nbFields(); // number of vars for each cell

      // Number of values to send are cell_count * num_vars 
      std::vector<int> send_sizes = this->m_send_intermediate_cell_count_per_level[level];
      for( auto& v : send_sizes )
        v*=num_vars;
      std::vector<int> recv_sizes = this->m_recv_intermediate_cell_count_per_level[level];  
      for( auto& v : recv_sizes )
        v*=num_vars;

      const std::vector<uint32_t>& send_total_ghost_cells_per_level = this->m_send_total_intermediate_ghost_cells_per_level;
      const std::vector<uint32_t>& recv_total_ghost_cells_per_level = this->m_recv_total_intermediate_ghost_cells_per_level;
      const auto send_pair_range = std::make_pair(send_total_ghost_cells_per_level[level], send_total_ghost_cells_per_level[level+1]);
      const auto recv_pair_range = std::make_pair(recv_total_ghost_cells_per_level[level], recv_total_ghost_cells_per_level[level+1]);
      
      const Kokkos::View< uint32_t* >& send_iOct = Kokkos::subview(this->m_send_intermediate_iOct_per_level, send_pair_range);
      const Kokkos::View< uint32_t* >& send_iCell = Kokkos::subview(this->m_send_intermediate_iCell_per_level, send_pair_range);;
      const Kokkos::View< uint32_t* >& recv_iOct = Kokkos::subview(this->m_recv_intermediate_iOct_per_level, recv_pair_range);
      const Kokkos::View< uint32_t* >& recv_iCell = Kokkos::subview(this->m_recv_intermediate_iCell_per_level, recv_pair_range);
      exchanging_ghosts( U, num_vars, send_sizes, recv_sizes, send_iOct, send_iCell, recv_iOct, recv_iCell, true );
    }

    /**
     * TODO : doc
     **/
    template< typename CellArray_t >
    void reducing_ghosts( 
      CellArray_t& U, 
      const int num_vars,
      const std::vector<int>& send_sizes,
      const std::vector<int>& recv_sizes,
      const Kokkos::View< uint32_t* >& send_iOct, 
      const Kokkos::View< uint32_t* >& send_iCell, 
      const Kokkos::View< uint32_t* >& recv_iOct, 
      const Kokkos::View< uint32_t* >& recv_iCell, 
      const bool isIntermediate) const
    {
      using CellIndex = ForeachCell::CellIndex;

      uint32_t total_send_size = send_iOct.size(), total_recv_size = recv_iOct.size(); // send/recv buffer size (number of cells)    
      uint32_t bx=U.getShape().bx, by=U.getShape().by, bz=U.getShape().bz ; // Block size
      
      if( num_vars*total_send_size == 0 )
        return;

      Kokkos::View< real_t*, Kokkos::LayoutLeft > send_buffer("reduce_ghosts::send_buffer", num_vars*total_send_size );
    
      Kokkos::parallel_for("reduce_ghosts::pack", total_send_size*num_vars,
        KOKKOS_LAMBDA( uint32_t ipack )
      {
        uint32_t ighost = ipack/num_vars;
        uint32_t ivar = ipack%num_vars;

        uint32_t iOct = send_iOct(ighost);
        uint32_t iCell = send_iCell(ighost);
        uint32_t i = iCell%bx; 
        uint32_t j = (iCell/bx)%by; 
        uint32_t k = (iCell/bx)/by;

        CellIndex cell_index { {iOct, true, isIntermediate}, i, j, k, bx, by, bz };
        send_buffer( ipack ) = U.at_ivar( cell_index, ivar );
      });
      
      Kokkos::View< real_t* > recv_buffer("exchange_ghosts::recv_buffer", num_vars*total_recv_size ); 
      #ifdef MPI_IS_CUDA_AWARE 
        Kokkos::fence();
        mpi_comm.MPI_Alltoallv( send_buffer.data(), send_sizes.data(), recv_buffer.data(), recv_sizes.data() );
        Kokkos::fence();
      #else
        {
          auto send_buffer_host = Kokkos::create_mirror_view(send_buffer);
          auto recv_buffer_host = Kokkos::create_mirror_view(recv_buffer);

          Kokkos::deep_copy(send_buffer_host, send_buffer);
          mpi_comm.MPI_Alltoallv( send_buffer_host.data(), send_sizes.data(), recv_buffer_host.data(), recv_sizes.data() );
          Kokkos::deep_copy(recv_buffer, recv_buffer_host);
        }  
      #endif

      Kokkos::parallel_for("reduce_ghosts::unpack", total_recv_size*num_vars,
        KOKKOS_LAMBDA( uint32_t ipack )
      {
        uint32_t ighost = ipack/num_vars;
        uint32_t ivar = ipack%num_vars;

        uint32_t iOct = recv_iOct(ighost);
        uint32_t iCell = recv_iCell(ighost);
        uint32_t i = iCell%bx; 
        uint32_t j = (iCell/bx)%by; 
        uint32_t k = (iCell/bx)/by;

        CellIndex cell_index { {iOct, false, isIntermediate}, i, j, k, bx, by, bz };
        Kokkos::atomic_add( &U.at_ivar( cell_index, ivar ), recv_buffer( ipack ) );
      });
    }

    template< typename CellArray_t >
    void reduce_ghosts( CellArray_t& U ) const
    {
      uint32_t num_vars = U.nbFields(); // number of vars for each cell
      // Note : sends and recvs counts are swapped for reduce
      std::vector<int> send_sizes = this->m_recv_cell_count;
      for( auto& v : send_sizes )
        v*=num_vars;
      std::vector<int> recv_sizes = this->m_send_cell_count;  
      for( auto& v : recv_sizes )
        v*=num_vars;

      const Kokkos::View< uint32_t* >& send_iOct = this->m_recv_iOct;
      const Kokkos::View< uint32_t* >& send_iCell = this->m_recv_iCell;
      const Kokkos::View< uint32_t* >& recv_iOct = this->m_send_iOct;
      const Kokkos::View< uint32_t* >& recv_iCell = this->m_send_iCell;
      reducing_ghosts( U, num_vars, send_sizes, recv_sizes, send_iOct, send_iCell, recv_iOct, recv_iCell, false );
    }

    template< typename CellArray_t >
    void reduce_ghosts_at_level( CellArray_t& U, const uint8_t level) const
    {
      uint32_t num_vars = U.nbFields(); // number of vars for each cell
      // Note : sends and recvs counts are swapped for reduce
      std::vector<int> send_sizes = this->m_recv_cell_count_per_level[level];
      for( auto& v : send_sizes )
        v*=num_vars;
      std::vector<int> recv_sizes = this->m_send_cell_count_per_level[level];  
      for( auto& v : recv_sizes )
        v*=num_vars;

      const std::vector<uint32_t>& send_total_ghost_cells_per_level = this->m_send_total_ghost_cells_per_level;
      const std::vector<uint32_t>& recv_total_ghost_cells_per_level = this->m_recv_total_ghost_cells_per_level;
      const auto send_pair_range = std::make_pair(send_total_ghost_cells_per_level[level], send_total_ghost_cells_per_level[level+1]);
      const auto recv_pair_range = std::make_pair(recv_total_ghost_cells_per_level[level], recv_total_ghost_cells_per_level[level+1]);

      const Kokkos::View< uint32_t* >& send_iOct = Kokkos::subview(this->m_recv_iOct_per_level, recv_pair_range);
      const Kokkos::View< uint32_t* >& send_iCell = Kokkos::subview(this->m_recv_iCell_per_level, recv_pair_range);;
      const Kokkos::View< uint32_t* >& recv_iOct = Kokkos::subview(this->m_send_iOct_per_level, send_pair_range);
      const Kokkos::View< uint32_t* >& recv_iCell = Kokkos::subview(this->m_send_iCell_per_level, send_pair_range);
      reducing_ghosts( U, num_vars, send_sizes, recv_sizes, send_iOct, send_iCell, recv_iOct, recv_iCell, false );
    }

    template< typename CellArray_t >
    void reduce_intermediate_ghosts( CellArray_t& U) const
    {
      uint32_t num_vars = U.nbFields(); // number of vars for each cell

      // Note : sends and recvs counts are swapped for reduce

      // Number of values to send are cell_count * num_vars 
      std::vector<int> send_sizes = this->m_recv_intermediate_cell_count;
      for( auto& v : send_sizes )
        v*=num_vars;
      std::vector<int> recv_sizes = this->m_send_intermediate_cell_count;  
      for( auto& v : recv_sizes )
        v*=num_vars;

      const Kokkos::View< uint32_t* >& send_iOct = this->m_recv_intermediate_iOct;
      const Kokkos::View< uint32_t* >& send_iCell = this->m_recv_intermediate_iCell;
      const Kokkos::View< uint32_t* >& recv_iOct = this->m_send_intermediate_iOct;
      const Kokkos::View< uint32_t* >& recv_iCell = this->m_send_intermediate_iCell;
      reducing_ghosts( U, num_vars, send_sizes, recv_sizes, send_iOct, send_iCell, recv_iOct, recv_iCell, true );
    }
    template< typename CellArray_t >
    void reduce_intermediate_ghosts_at_level( CellArray_t& U, const uint8_t level) const
    {
      uint32_t num_vars = U.nbFields(); // number of vars for each cell
      // Note : sends and recvs counts are swapped for reduce
      std::vector<int> send_sizes = this->m_recv_intermediate_cell_count_per_level[level];
      for( auto& v : send_sizes )
        v*=num_vars;
      std::vector<int> recv_sizes = this->m_send_intermediate_cell_count_per_level[level];  
      for( auto& v : recv_sizes )
        v*=num_vars;

      const std::vector<uint32_t> send_total_ghost_cells_per_level = this->m_send_total_intermediate_ghost_cells_per_level;
      const std::vector<uint32_t> recv_total_ghost_cells_per_level = this->m_recv_total_intermediate_ghost_cells_per_level;
      const auto send_pair_range = std::make_pair(send_total_ghost_cells_per_level[level], send_total_ghost_cells_per_level[level+1]);
      const auto recv_pair_range = std::make_pair(recv_total_ghost_cells_per_level[level], recv_total_ghost_cells_per_level[level+1]);

      const Kokkos::View< uint32_t* >& send_iOct = Kokkos::subview(this->m_recv_intermediate_iOct_per_level, recv_pair_range);
      const Kokkos::View< uint32_t* >& send_iCell = Kokkos::subview(this->m_recv_intermediate_iCell_per_level, recv_pair_range);;
      const Kokkos::View< uint32_t* >& recv_iOct = Kokkos::subview(this->m_send_intermediate_iOct_per_level, send_pair_range);
      const Kokkos::View< uint32_t* >& recv_iCell = Kokkos::subview(this->m_send_intermediate_iCell_per_level, send_pair_range);
      reducing_ghosts( U, num_vars, send_sizes, recv_sizes, send_iOct, send_iCell, recv_iOct, recv_iCell, true );
    }

    void sort_ghosts_by_levels(const LightOctree& lmesh, const uint8_t level_max)
    {
      const uint32_t mpi_size = mpi_comm.MPI_Comm_size();
      const uint32_t mpi_rank = mpi_comm.MPI_Comm_rank();
      const std::vector<int>& send_cell_count = this->m_send_cell_count;
      const std::vector<int>& recv_cell_count = this->m_recv_cell_count;
      const Kokkos::View< uint32_t* >& send_iCell = this->m_send_iCell;
      const Kokkos::View< uint32_t* >& send_iOct = this->m_send_iOct;
      const Kokkos::View< uint32_t* >& recv_iCell = this->m_recv_iCell;
      const Kokkos::View< uint32_t* >& recv_iOct = this->m_recv_iOct;
      const uint32_t total_send_size = send_iOct.size(), total_recv_size = recv_iOct.size(); // send/recv buffer size (number of cells)    

      // Cumulative count
      std::vector<uint32_t> send_cell_offsets(mpi_size + 1, 0);
      std::vector<uint32_t> recv_cell_offsets(mpi_size + 1, 0);
      std::partial_sum(send_cell_count.begin(), send_cell_count.end(), send_cell_offsets.begin() + 1);
      std::partial_sum(recv_cell_count.begin(), recv_cell_count.end(), recv_cell_offsets.begin() + 1);
      std::vector< std::vector<int> > send_cell_count_per_level(level_max + 1, std::vector<int>(mpi_size, 0));
      std::vector< std::vector<int> > recv_cell_count_per_level(level_max + 1, std::vector<int>(mpi_size, 0));
      // Count how many cells we send per level and rank
      Kokkos::View<uint32_t**> send_cell_count_per_level_view_device("send_cell_count_per_level_view_device", level_max+1, mpi_size); 
      Kokkos::deep_copy(send_cell_count_per_level_view_device, 0);
      for (uint32_t irank = 0; irank < mpi_size; irank++){
        if (irank != mpi_rank){
          const uint32_t rank_offset = send_cell_offsets[irank];
          Kokkos::parallel_for("GhostCommunicator_partial_blocks::sort_ghosts_by_levels", send_cell_count[irank],
            KOKKOS_LAMBDA( uint32_t iGhost )
          {
            iGhost += rank_offset;
            const uint32_t iOct = send_iOct(iGhost);
            const uint8_t level = lmesh.getLevel({iOct, false});
            Kokkos::atomic_fetch_add(&send_cell_count_per_level_view_device(level, irank), 1);
          });
        }
      }
      const auto send_cell_count_per_level_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_cell_count_per_level_view_device);
      for (uint8_t ilevel = 0; ilevel <= level_max; ilevel++)
      for (uint32_t irank = 0; irank < mpi_size; irank++)
        send_cell_count_per_level[ilevel][irank] = send_cell_count_per_level_host(ilevel, irank);
      
      // Count how many cells we receive per level and rank
      std::vector<uint32_t> send_total_ghost_cells_per_level(level_max + 1, 0);
      std::vector<uint32_t> recv_total_ghost_cells_per_level(level_max + 1, 0);
      for (uint8_t ilevel=0; ilevel<=level_max; ilevel++)
      {
        std::vector<int> send_sizes_tmp( mpi_size );
        std::vector<int> recv_sizes_tmp( mpi_size );
        send_sizes_tmp = send_cell_count_per_level[ilevel];
        mpi_comm.MPI_Alltoall( send_sizes_tmp.data(), 1, recv_sizes_tmp.data(), 1);
        recv_cell_count_per_level[ilevel] = recv_sizes_tmp;
        send_total_ghost_cells_per_level[ilevel] = std::reduce(send_cell_count_per_level[ilevel].begin(), send_cell_count_per_level[ilevel].end());
        recv_total_ghost_cells_per_level[ilevel] = std::reduce(recv_cell_count_per_level[ilevel].begin(), recv_cell_count_per_level[ilevel].end());
      }

      std::vector<uint32_t> send_total_ghost_cells_offset_per_level(send_total_ghost_cells_per_level.size() + 1, 0);
      std::vector<uint32_t> recv_total_ghost_cells_offset_per_level(recv_total_ghost_cells_per_level.size() + 1, 0);
      // Total number of cells per level (accumulated)
      std::partial_sum(send_total_ghost_cells_per_level.begin(), send_total_ghost_cells_per_level.end(), send_total_ghost_cells_offset_per_level.begin() + 1);
      std::partial_sum(recv_total_ghost_cells_per_level.begin(), recv_total_ghost_cells_per_level.end(), recv_total_ghost_cells_offset_per_level.begin() + 1);

      // Sort the ghosts by level
      Kokkos::View< uint32_t* > send_iOct_per_level("send_iOct_per_level", total_send_size );
      Kokkos::View< uint32_t* > send_iCell_per_level("send_iCell_per_level", total_send_size );
      Kokkos::View< uint32_t* > recv_iOct_per_level("recv_iOct_per_level", total_recv_size );
      Kokkos::View< uint32_t* > recv_iCell_per_level("recv_iCell_per_level", total_recv_size );

      for (uint8_t ilevel = 0; ilevel <= level_max; ilevel++){
        const uint32_t level_offset_send = send_total_ghost_cells_offset_per_level[ilevel];
        const uint32_t level_offset_recv = recv_total_ghost_cells_offset_per_level[ilevel];
        uint32_t rank_offset_send(0), rank_offset_recv(0);
        for (uint32_t irank = 0; irank < mpi_size; irank++){
          if (irank != mpi_rank){
            uint32_t offset_send(0), offset_recv(0);
            const uint32_t send_cell_offsets_rank = send_cell_offsets[irank];
            const uint32_t recv_cell_offsets_rank = recv_cell_offsets[irank];
            Kokkos::parallel_scan("GhostCommunicator_partial_blocks::sort_send_ghosts_by_levels", send_cell_count[irank], //send_cell_count_per_level[ilevel][irank],
              KOKKOS_LAMBDA( uint32_t iGhost, uint32_t& index, bool final )
            {
              iGhost += send_cell_offsets_rank;
              const uint32_t iOct = send_iOct(iGhost);
              const uint8_t level = lmesh.getLevel({iOct, false});
              if (level == ilevel) {
                const uint32_t iCell = send_iCell(iGhost);
                if ( final ) {
                  send_iOct_per_level(index + level_offset_send + rank_offset_send) = iOct;
                  send_iCell_per_level(index + level_offset_send + rank_offset_send) = iCell;
                }
                index++;
              }
            }, offset_send);
            Kokkos::parallel_scan("GhostCommunicator_partial_blocks::sort_recv_ghosts_by_levels", recv_cell_count[irank], //send_cell_count_per_level[ilevel][irank],
              KOKKOS_LAMBDA( uint32_t iGhost, uint32_t& index, bool final )
            {
              iGhost += recv_cell_offsets_rank;
              const uint32_t iOct = recv_iOct(iGhost);
              const uint8_t level = lmesh.getLevel({iOct, true});
              if (level == ilevel) {
                const uint32_t iCell = recv_iCell(iGhost);
                if ( final ) {
                  recv_iOct_per_level(index + level_offset_recv + rank_offset_recv) = iOct;
                  recv_iCell_per_level(index + level_offset_recv + rank_offset_recv) = iCell;
                }
                index++;
              }
            }, offset_recv);
            DYABLO_ASSERT_KOKKOS_DEBUG(send_cell_count_per_level[ilevel][irank] == static_cast<int>(offset_send), "Wrong count of send cells per level");
            DYABLO_ASSERT_KOKKOS_DEBUG(recv_cell_count_per_level[ilevel][irank] == static_cast<int>(offset_recv), "Wrong count of recv cells per level");
            rank_offset_send += offset_send;
            rank_offset_recv += offset_recv;
          }
        }
      }
      this->m_send_total_ghost_cells_per_level = send_total_ghost_cells_offset_per_level;
      this->m_recv_total_ghost_cells_per_level = recv_total_ghost_cells_offset_per_level;
      this->m_send_cell_count_per_level = send_cell_count_per_level;
      this->m_recv_cell_count_per_level = recv_cell_count_per_level;
      this->m_send_iOct_per_level = send_iOct_per_level;
      this->m_send_iCell_per_level = send_iCell_per_level;
      this->m_recv_iOct_per_level = recv_iOct_per_level;
      this->m_recv_iCell_per_level = recv_iCell_per_level;      
    }

    void sort_intermediate_ghosts_by_levels(const LightOctree& lmesh, const uint8_t level_max)
    {
      const uint32_t mpi_size = mpi_comm.MPI_Comm_size();
      const uint32_t mpi_rank = mpi_comm.MPI_Comm_rank();
      const std::vector<int>& send_cell_count = this->m_send_intermediate_cell_count;
      const std::vector<int>& recv_cell_count = this->m_recv_intermediate_cell_count;
      const Kokkos::View< uint32_t* >& send_iCell = this->m_send_intermediate_iCell;
      const Kokkos::View< uint32_t* >& send_iOct = this->m_send_intermediate_iOct;
      const Kokkos::View< uint32_t* >& recv_iCell = this->m_recv_intermediate_iCell;
      const Kokkos::View< uint32_t* >& recv_iOct = this->m_recv_intermediate_iOct;
      const uint32_t total_send_size = send_iOct.size(), total_recv_size = recv_iOct.size(); // send/recv buffer size (number of cells)    

      // Cumulative count
      std::vector<uint32_t> send_cell_offsets(mpi_size + 1, 0);
      std::vector<uint32_t> recv_cell_offsets(mpi_size + 1, 0);
      std::partial_sum(send_cell_count.begin(), send_cell_count.end(), send_cell_offsets.begin() + 1);
      std::partial_sum(recv_cell_count.begin(), recv_cell_count.end(), recv_cell_offsets.begin() + 1);
      std::vector< std::vector<int> > send_cell_count_per_level(level_max + 1, std::vector<int>(mpi_size, 0));
      std::vector< std::vector<int> > recv_cell_count_per_level(level_max + 1, std::vector<int>(mpi_size, 0));
      // Count how many cells we send per level and rank
      Kokkos::View<int**> send_cell_count_per_level_view_device("send_cell_count_per_level_view_device", level_max+1, mpi_size); 
      Kokkos::deep_copy(send_cell_count_per_level_view_device, 0);
      for (uint32_t irank = 0; irank < mpi_size; irank++){
        if (irank != mpi_rank){
          const uint32_t rank_offset = send_cell_offsets[irank];
          Kokkos::parallel_for("GhostCommunicator_partial_blocks::sort_ghosts_by_levels", send_cell_count[irank],
            KOKKOS_LAMBDA( uint32_t iGhost )
          {
            iGhost += rank_offset;
            const uint32_t iOct = send_iOct(iGhost);
            const uint8_t level = lmesh.getLevel({iOct, false, true});
            Kokkos::atomic_fetch_add(&send_cell_count_per_level_view_device(level, irank), 1);
          });
        }
      }
      const auto send_cell_count_per_level_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_cell_count_per_level_view_device);
      for (uint8_t ilevel = 0; ilevel <= level_max; ilevel++)
      for (uint32_t irank = 0; irank < mpi_size; irank++)
        send_cell_count_per_level[ilevel][irank] = send_cell_count_per_level_host(ilevel, irank);
      
      // Count how many cells we receive per level and rank
      std::vector<uint32_t> send_total_ghost_cells_per_level(level_max + 1, 0);
      std::vector<uint32_t> recv_total_ghost_cells_per_level(level_max + 1, 0);
      for (uint8_t ilevel=0; ilevel<=level_max; ilevel++)
      {
        std::vector<int> send_sizes_tmp( mpi_size );
        std::vector<int> recv_sizes_tmp( mpi_size );
        send_sizes_tmp = send_cell_count_per_level[ilevel];
        mpi_comm.MPI_Alltoall( send_sizes_tmp.data(), 1, recv_sizes_tmp.data(), 1);
        recv_cell_count_per_level[ilevel] = recv_sizes_tmp;
        send_total_ghost_cells_per_level[ilevel] = std::reduce(send_cell_count_per_level[ilevel].begin(), send_cell_count_per_level[ilevel].end());
        recv_total_ghost_cells_per_level[ilevel] = std::reduce(recv_cell_count_per_level[ilevel].begin(), recv_cell_count_per_level[ilevel].end());
      }

      std::vector<uint32_t> send_total_ghost_cells_offset_per_level(send_total_ghost_cells_per_level.size() + 1, 0);
      std::vector<uint32_t> recv_total_ghost_cells_offset_per_level(recv_total_ghost_cells_per_level.size() + 1, 0);
      // Total number of cells per level (accumulated)
      std::partial_sum(send_total_ghost_cells_per_level.begin(), send_total_ghost_cells_per_level.end(), send_total_ghost_cells_offset_per_level.begin() + 1);
      std::partial_sum(recv_total_ghost_cells_per_level.begin(), recv_total_ghost_cells_per_level.end(), recv_total_ghost_cells_offset_per_level.begin() + 1);

      // Sort the ghosts by level
      Kokkos::View< uint32_t* > send_iOct_per_level("send_iOct_per_level", total_send_size );
      Kokkos::View< uint32_t* > send_iCell_per_level("send_iCell_per_level", total_send_size );
      Kokkos::View< uint32_t* > recv_iOct_per_level("recv_iOct_per_level", total_recv_size );
      Kokkos::View< uint32_t* > recv_iCell_per_level("recv_iCell_per_level", total_recv_size );

      for (uint8_t ilevel = 0; ilevel <= level_max; ilevel++){
        const uint32_t level_offset_send = send_total_ghost_cells_offset_per_level[ilevel];
        const uint32_t level_offset_recv = recv_total_ghost_cells_offset_per_level[ilevel];
        uint32_t rank_offset_send(0), rank_offset_recv(0);
        for (uint32_t irank = 0; irank < mpi_size; irank++){
          if (irank != mpi_rank){
            uint32_t offset_send(0), offset_recv(0);
            const uint32_t send_cell_offsets_rank = send_cell_offsets[irank];
            const uint32_t recv_cell_offsets_rank = recv_cell_offsets[irank];
            Kokkos::parallel_scan("GhostCommunicator_partial_blocks::sort_send_ghosts_by_levels", send_cell_count[irank], //send_cell_count_per_level[ilevel][irank],
              KOKKOS_LAMBDA( uint32_t iGhost, uint32_t& index, bool final )
            {
              iGhost += send_cell_offsets_rank;
              const uint32_t iOct = send_iOct(iGhost);
              const uint8_t level = lmesh.getLevel({iOct, false, true});
              if (level == ilevel) {
                const uint32_t iCell = send_iCell(iGhost);
                if ( final ) {
                  send_iOct_per_level(index + level_offset_send + rank_offset_send) = iOct;
                  send_iCell_per_level(index + level_offset_send + rank_offset_send) = iCell;
                }
                index++;
              }
            }, offset_send);
            Kokkos::parallel_scan("GhostCommunicator_partial_blocks::sort_recv_ghosts_by_levels", recv_cell_count[irank], //send_cell_count_per_level[ilevel][irank],
              KOKKOS_LAMBDA( uint32_t iGhost, uint32_t& index, bool final )
            {
              iGhost += recv_cell_offsets_rank;
              const uint32_t iOct = recv_iOct(iGhost);
              const uint8_t level = lmesh.getLevel({iOct, true, true});
              if (level == ilevel) {
                const uint32_t iCell = recv_iCell(iGhost);
                if ( final ) {
                  recv_iOct_per_level(index + level_offset_recv + rank_offset_recv) = iOct;
                  recv_iCell_per_level(index + level_offset_recv + rank_offset_recv) = iCell;
                }
                index++;
              }
            }, offset_recv);
            DYABLO_ASSERT_KOKKOS_DEBUG(send_cell_count_per_level[ilevel][irank] == static_cast<int>(offset_send), "Wrong intermediate count of send cells per level");
            DYABLO_ASSERT_KOKKOS_DEBUG(recv_cell_count_per_level[ilevel][irank] == static_cast<int>(offset_recv), "Wrong intermediate count of recv cells per level");
            rank_offset_send += offset_send;
            rank_offset_recv += offset_recv;
          }
        }
      }
      this->m_send_total_intermediate_ghost_cells_per_level = send_total_ghost_cells_offset_per_level;
      this->m_recv_total_intermediate_ghost_cells_per_level = recv_total_ghost_cells_offset_per_level;
      this->m_send_intermediate_cell_count_per_level = send_cell_count_per_level;
      this->m_recv_intermediate_cell_count_per_level = recv_cell_count_per_level;
      this->m_send_intermediate_iOct_per_level = send_iOct_per_level;
      this->m_send_intermediate_iCell_per_level = send_iCell_per_level;
      this->m_recv_intermediate_iOct_per_level = recv_iOct_per_level;
      this->m_recv_intermediate_iCell_per_level = recv_iCell_per_level;      
    }

private:
    uint32_t m_local_ghost_octants; // Number of octants to allocate for ghosts
    std::vector<int> m_send_cell_count; // number of cells to send to each process
    std::vector<int> m_recv_cell_count; // number of cells to recv from each process
    Kokkos::View< uint32_t* > m_send_iOct; // send_iOct(ighost) iOct of cell to pack to position ighost in send buffer
    Kokkos::View< uint32_t* > m_send_iCell; // send_iCell(ighost) iCell of cell to pack to position ighost in send buffer
    Kokkos::View< uint32_t* > m_recv_iOct; // recv_iOct(ighost) iOct of cell to unpack from position ighost in recv buffer
    Kokkos::View< uint32_t* > m_recv_iCell; // recv_iCell(ighost) iCell of cell to unpack from position ighost in recv buffer
    // Intermediate
    uint32_t m_local_intermediate_ghost_octants; // Number of octants to allocate for ghosts
    std::vector<int> m_send_intermediate_cell_count; // number of cells to send to each process
    std::vector<int> m_recv_intermediate_cell_count; // number of cells to recv from each process
    Kokkos::View< uint32_t* > m_send_intermediate_iOct; // send_iOct(ighost) iOct of cell to pack to position ighost in send buffer
    Kokkos::View< uint32_t* > m_send_intermediate_iCell; // send_iCell(ighost) iCell of cell to pack to position ighost in send buffer
    Kokkos::View< uint32_t* > m_recv_intermediate_iOct; // recv_iOct(ighost) iOct of cell to unpack from position ighost in recv buffer
    Kokkos::View< uint32_t* > m_recv_intermediate_iCell; // recv_iCell(ighost) iCell of cell to unpack from position ighost in recv buffer

    // Per level
    std::vector<uint32_t> m_send_total_ghost_cells_per_level; // Number of ghosts cells per level (accumulated)
    std::vector<uint32_t> m_recv_total_ghost_cells_per_level; // Number of ghosts cells per level (accumulated)
    std::vector<std::vector<int>> m_send_cell_count_per_level; // number of cells to send to each process per level
    std::vector<std::vector<int>> m_recv_cell_count_per_level; // number of cells to recv from each process
    Kokkos::View< uint32_t* > m_send_iOct_per_level;
    Kokkos::View< uint32_t* > m_send_iCell_per_level;
    Kokkos::View< uint32_t* > m_recv_iOct_per_level;
    Kokkos::View< uint32_t* > m_recv_iCell_per_level;
    // Intermediate per level
    std::vector<uint32_t> m_send_total_intermediate_ghost_cells_per_level; // Number of octants to allocate for ghosts
    std::vector<uint32_t> m_recv_total_intermediate_ghost_cells_per_level; // Number of octants to allocate for ghosts
    std::vector<std::vector<int>> m_send_intermediate_cell_count_per_level; // number of cells to send to each process
    std::vector<std::vector<int>> m_recv_intermediate_cell_count_per_level; // number of cells to recv from each process
    Kokkos::View< uint32_t* > m_send_intermediate_iOct_per_level;
    Kokkos::View< uint32_t* > m_send_intermediate_iCell_per_level;
    Kokkos::View< uint32_t* > m_recv_intermediate_iOct_per_level;
    Kokkos::View< uint32_t* > m_recv_intermediate_iCell_per_level;
    
    MpiComm mpi_comm;    
};

} // namespace dyablo