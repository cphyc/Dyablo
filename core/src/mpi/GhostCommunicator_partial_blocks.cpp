#include "GhostCommunicator_partial_blocks.h"
#include "DyabloSession.hpp"
#include "utils/mpi/MpiBufferPool.h"
#include "user_data/FieldAccessor.h"

namespace dyablo {

struct GhostCommunicator_partial_blocks_Pdata
{
  bool intermediates;
  MpiComm mpi_comm;
  uint32_t bx, by, bz, ghost_count;

  std::vector<int>        ghostmap_send_sizes;
  Kokkos::View<int*>      ghostmap_send_masks;
  Kokkos::View<uint32_t*> ghostmap_send_iOcts;

  std::vector<int>        ghostmap_recv_sizes;
  Kokkos::View<int*>      ghostmap_recv_masks;

  uint32_t m_local_ghost_octants; // Number of octants to allocate for ghosts
  std::vector<int> m_send_cell_count; // number of cells to send to each process
  std::vector<int> m_recv_cell_count; // number of cells to recv from each process
  Kokkos::View< uint32_t* > m_send_iOct; // send_iOct(ighost) iOct of cell to pack to position ighost in send buffer
  Kokkos::View< uint32_t* > m_send_iCell; // send_iCell(ighost) iCell of cell to pack to position ighost in send buffer
  Kokkos::View< uint32_t* > m_recv_iOct; // recv_iOct(ighost) iOct of cell to unpack from position ighost in recv buffer
  Kokkos::View< uint32_t* > m_recv_iCell; // recv_iCell(ighost) iCell of cell to unpack from position ighost in recv buffer
};

struct GhostCommunicator_partial_blocks_OctSubset_Pdata
{
  GhostCommunicator_partial_blocks comm_subset;
};

namespace{

struct Precomputed_Mask_Cells
{
  using key_t = std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>;
  using map_t = std::map< key_t, Precomputed_Mask_Cells>;

  Kokkos::View<uint32_t*> facemask_count; // Number of cells for each mask
  Kokkos::View<uint32_t**, Kokkos::LayoutRight> facemask_iCells; // Cell indexes for each mask
};

Precomputed_Mask_Cells::map_t& facemask_cache()
{
  static Precomputed_Mask_Cells::map_t cache = []()
  {
    DyabloSession::get_DyabloSession().call_before_finalize([&]()
    {
      cache.clear();
    });

    return Precomputed_Mask_Cells::map_t();
  }();

  return cache;
}

const Precomputed_Mask_Cells& precompute_facemask_cells( uint32_t bx, uint32_t by, uint32_t bz, uint32_t ghost_count )
{
  using key_t = std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>;

  Precomputed_Mask_Cells::map_t& precomputed_map = facemask_cache();
  key_t key{bx,by,bz,ghost_count};
  
  if( precomputed_map.count(key) == 0 )  
  {
    using GhostMap_t = AMRmesh::GhostMap_t;
    using CellMask = AMRmesh::GhostMap_t::CellMask;
    using Face = AMRmesh::GhostMap_t::Face;

    int max_icells = bx*by*bz;
    Kokkos::View<uint32_t*> facemask_count("facemask_count", GhostMap_t::MASK_COUNT);// number of cells to add for facemask
    Kokkos::View<uint32_t**, Kokkos::LayoutRight> facemask_iCells("facemask_iCells", GhostMap_t::MASK_COUNT, max_icells);// cells to add for facemask

    auto facemask_count_host = Kokkos::create_mirror_view(facemask_count);

    // Check if face is included in mask
    auto has_face = [](CellMask mask, Face face){ return mask & (1 << face); };
    // Add cells of block to facemask_* delimited by region (xmin:xmax, ...)
    auto add_cells = [&](CellMask mask, uint32_t xmin, uint32_t xmax, uint32_t ymin, uint32_t ymax, uint32_t zmin, uint32_t zmax)
    {
      DYABLO_ASSERT_HOST_DEBUG( xmin <= xmax && xmax <= bx, "add_cells, xmin, xmax error, should have 0 <= xmin (" << xmin << ") <= xmax (" << xmax << ") <= bx (" << bx << ")" );
      DYABLO_ASSERT_HOST_DEBUG( ymin <= ymax && ymax <= by, "add_cells, ymin, ymax error, should have 0 <= ymin (" << ymin << ") <= ymax (" << ymax << ") <= by (" << by << ")" );
      DYABLO_ASSERT_HOST_DEBUG( zmin <= zmax && zmax <= bz, "add_cells, zmin, zmax error, should have 0 <= zmin (" << zmin << ") <= zmax (" << zmax << ") <= bz (" << bz << ")" );
      uint32_t dx = (xmax-xmin);
      uint32_t dy = (ymax-ymin);
      uint32_t dz = (zmax-zmin);

      uint32_t i0 = facemask_count_host(mask);
      facemask_count_host(mask) += dx*dy*dz;

      Kokkos::parallel_for( "add_cells", dx*dy*dz,
        KOKKOS_LAMBDA( uint32_t i )
      {
        uint32_t x = i%dx;
        uint32_t y = (i/dx)%dy;
        uint32_t z = (i/dx)/dy;
        x += xmin;
        y += ymin;
        z += zmin;

        uint32_t iCell = x + bx*y + bx*by*z;
        DYABLO_ASSERT_KOKKOS_DEBUG( i0 + i < facemask_iCells.extent(1), "precompute_facemask_cells : too many cells");
        facemask_iCells(mask, i0 + i) = iCell;
      });
    };

    DYABLO_ASSERT_HOST_RELEASE( ghost_count <= bx || ghost_count <= by, "GhostCommunicator_partial_blocks::init : ghost_count ("<<ghost_count<<") not compatible with block size (" << bx << "," << by << "," << bz << ")"  );

    int faces_mask_count = (1 << Face::FACE_COUNT);
    for( CellMask mask = 1; mask < faces_mask_count; mask++  )
    {

      // Uncomment to add full block each time for debug
      // add_cells( mask, 0, bx, 0, by, 0, bz );
      // continue;

      uint32_t xmin = 0, xmax = bx; // avoid duplicates : all cells with x<xmin are already added
      if( has_face(mask, Face::XL) )
      {
        uint32_t xl_xmax = ghost_count;
        add_cells( mask, 0, xl_xmax, 0, by, 0, bz );
        xmin = xl_xmax;
      }
      if( has_face(mask, Face::XR) )
      {
        uint32_t xr_xmin = std::max( bx-ghost_count, xmin ); // avoid adding cells already added when ghost_count < 2*bx
        add_cells( mask, xr_xmin, bx, 0, by, 0, bz );
        xmax = xr_xmin;
      }
      
      uint32_t ymin = 0, ymax = by;
      if( has_face(mask, Face::YL) )
      {
        uint32_t yl_ymax = ghost_count;
        add_cells( mask, xmin, xmax, 0, yl_ymax, 0, bz );
        ymin = yl_ymax;
      }
      if( has_face(mask, Face::YR) )
      {
        uint32_t yr_ymin = std::max( by-ghost_count, ymin );
        add_cells( mask, xmin, xmax, yr_ymin, by, 0, bz );
        ymax = yr_ymin;
      }

      uint32_t zmin = 0;
      if( has_face(mask, Face::ZL) )
      {
        uint32_t zl_zmax = std::min( ghost_count, bz );
        add_cells( mask, xmin, xmax, ymin, ymax, 0, zl_zmax );
        zmin = zl_zmax;
      }
      if( has_face(mask, Face::ZR) )
      {
        uint32_t zr_zmin = std::max( (int)bz-(int)ghost_count, (int)zmin);
        add_cells( mask, xmin, xmax, ymin, ymax, zr_zmin, bz );
      }
    }

    // Last mask is full block
    add_cells( GhostMap_t::MASK_FULL_BLOCK, 0, bx, 0, by, 0, bz );

    Kokkos::deep_copy( facemask_count, facemask_count_host );

    precomputed_map[key].facemask_count = facemask_count;
    precomputed_map[key].facemask_iCells = facemask_iCells;
  }
  
  return precomputed_map.at( key );
}


struct CellList
{
  std::vector<int> sizes;
  Kokkos::View< uint32_t* > iOcts;
  Kokkos::View< uint32_t* > iCells;
};

template< typename T >
CellList list_cells(
  const Precomputed_Mask_Cells& precomputed_cells,
  const std::vector<int>& ghostmap_sizes, 
  const T& ghostmap_iOcts, 
  Kokkos::View<int*> ghostmap_masks)
{
  using GhostMap_t = AMRmesh::GhostMap_t;

  int mpi_size = ghostmap_sizes.size();
  uint32_t nbOcts = ghostmap_masks.size();
  const Kokkos::View<uint32_t*>& facemask_count = precomputed_cells.facemask_count; // Number of cells for each mask
  const Kokkos::View<uint32_t**, Kokkos::LayoutRight> facemask_iCells = precomputed_cells.facemask_iCells; // Cell indexes for each mask

  std::vector<int> sizes( mpi_size );
  // Count cells for each process
  Kokkos::View<uint32_t*> offset_iOct("offset_iOct", nbOcts);
  uint32_t first_rank_iOct = 0;
  uint32_t offset = 0;
  for( int rank=0; rank<mpi_size; rank++ )
  {
    Kokkos::parallel_scan("GhostCommunicator_partial_blocks::count_cells", ghostmap_sizes[rank],
      KOKKOS_LAMBDA( uint32_t iOct, int& count, bool final )
    {
      GhostMap_t::CellMask facemask = ghostmap_masks(first_rank_iOct+iOct);
      if( final )
        offset_iOct( first_rank_iOct+iOct ) = count + offset;
      count += facemask_count(facemask);
    }, sizes[rank]);
    first_rank_iOct += ghostmap_sizes[rank];
    offset+=sizes[rank];
  }

  // Allocate cell containers
  uint32_t total_cell_count = std::accumulate( sizes.begin(), sizes.end(), 0 );
  Kokkos::View<uint32_t*> iOcts = Kokkos::View<uint32_t*>( "iOcts", total_cell_count );
  Kokkos::View<uint32_t*> iCells = Kokkos::View<uint32_t*>( "iCells", total_cell_count ); 

  // Fill cell containers
  
  Kokkos::parallel_for("GhostCommunicator_partial_blocks::list_cells", 
    Kokkos::TeamPolicy<>(nbOcts, Kokkos::AUTO),
    KOKKOS_LAMBDA( const Kokkos::TeamPolicy<>::member_type& team )
  {
    uint32_t iOct = team.league_rank();
    GhostMap_t::CellMask facemask = ghostmap_masks(iOct);
    uint32_t iCell_begin = offset_iOct(iOct);
    uint32_t current_send_iOct = ghostmap_iOcts(iOct);

    Kokkos::parallel_for( Kokkos::TeamVectorRange(team, facemask_count(facemask)),
      [&]( uint32_t i )
    {
      DYABLO_ASSERT_KOKKOS_DEBUG( (iCell_begin + i) < total_cell_count, "GhostCommunicator_partial_blocks::list_cells out of bounds " );
      iOcts( iCell_begin + i ) = current_send_iOct;
      iCells( iCell_begin + i ) = facemask_iCells( facemask, i);
    });
  });
  
  return {sizes, iOcts, iCells};
}

GhostCommunicator_partial_blocks_Pdata init(const AMRmesh::GhostMap_t& gm, const ForeachCell::CellArray_global_ghosted::Shape_t& shape, uint32_t ghost_count, bool intermediates, const MpiComm& mpi_comm )
{
  using GhostMap_t = AMRmesh::GhostMap_t;

  int mpi_size = mpi_comm.MPI_Comm_size();

  DYABLO_ASSERT_HOST_RELEASE( ghost_count <= shape.bx || ghost_count <= shape.by, "GhostCommunicator_partial_blocks::init : ghost_count ("<<ghost_count<<") not compatible with block size (" << shape.bx << "," << shape.by << "," << shape.bz << ")"  );

  // Exchange ghostmap sizes 
  auto& to_send = intermediates ? gm.to_send_intermediates : gm.to_send_leaves;
  std::vector<int> ghostmap_send_sizes( mpi_size );
  {
    auto ghostmap_send_sizes_host = Kokkos::create_mirror_view(to_send.send_sizes);
    Kokkos::deep_copy( ghostmap_send_sizes_host, to_send.send_sizes );
    for( int i=0; i<mpi_size; i++ )
      ghostmap_send_sizes[i] = ghostmap_send_sizes_host(i);
  }
  std::vector<int> ghostmap_recv_sizes( mpi_size );
  mpi_comm.MPI_Alltoall(  ghostmap_send_sizes.data(), 1, 
                          ghostmap_recv_sizes.data(), 1);
  uint32_t total_ghostmap_recv_count = std::accumulate( ghostmap_recv_sizes.begin(), 
                                                        ghostmap_recv_sizes.end(), 0 );

  using CellMask = GhostMap_t::CellMask;
  // Send mask to recieving ranks
  const Kokkos::View<CellMask*>& ghostmap_send_masks = to_send.send_cell_masks;
  Kokkos::View<CellMask*>  ghostmap_recv_masks("ghostmap_recv_masks", total_ghostmap_recv_count);

  #ifdef MPI_IS_CUDA_AWARE 
  {
    mpi_comm.MPI_Alltoallv( ghostmap_send_masks.data(), ghostmap_send_sizes.data(),
                            ghostmap_recv_masks.data(), ghostmap_recv_sizes.data());
  }
  #else
  {
    auto ghostmap_send_masks_host = Kokkos::create_mirror_view(ghostmap_send_masks);
    auto ghostmap_recv_masks_host = Kokkos::create_mirror_view(ghostmap_recv_masks);
    Kokkos::deep_copy(ghostmap_send_masks_host, ghostmap_send_masks);
    mpi_comm.MPI_Alltoallv( ghostmap_send_masks_host.data(), ghostmap_send_sizes.data(), 
                            ghostmap_recv_masks_host.data(), ghostmap_recv_sizes.data() );
    Kokkos::deep_copy(ghostmap_recv_masks, ghostmap_recv_masks_host);
  }
  #endif

  // Precompute cells for each mask type
  Precomputed_Mask_Cells precomputed_cells = precompute_facemask_cells( shape.bx, shape.by, shape.bz, ghost_count );
  
  // Compute list of cells to send
  CellList send_cells = list_cells( precomputed_cells, ghostmap_send_sizes, to_send.send_iOcts, to_send.send_cell_masks);

  // Compute list of cells to recieve
  CellList recv_cells = list_cells( precomputed_cells, ghostmap_recv_sizes, KOKKOS_LAMBDA(uint32_t iOct){return iOct;}, ghostmap_recv_masks);

  GhostCommunicator_partial_blocks_Pdata pdata{intermediates, mpi_comm};
  pdata.bx = shape.bx;
  pdata.by = shape.by;
  pdata.bz = shape.bz;
  pdata.ghost_count = ghost_count;

  pdata.ghostmap_send_sizes = ghostmap_send_sizes;
  pdata.ghostmap_send_masks = ghostmap_send_masks;
  pdata.ghostmap_send_iOcts = to_send.send_iOcts;

  pdata.ghostmap_recv_sizes = ghostmap_recv_sizes;
  pdata.ghostmap_recv_masks = ghostmap_recv_masks;

  pdata.m_send_cell_count = send_cells.sizes;
  pdata.m_recv_cell_count = recv_cells.sizes;
  pdata.m_send_iOct = send_cells.iOcts;
  pdata.m_send_iCell = send_cells.iCells; 
  pdata.m_recv_iOct = recv_cells.iOcts;
  pdata.m_recv_iCell = recv_cells.iCells;
  pdata.m_local_ghost_octants = total_ghostmap_recv_count;

  return pdata;
}

template< typename CellArray_t >
void exchange_ghosts_aux( const GhostCommunicator_partial_blocks::Pdata& pdata, const CellArray_t& U)
{
  using CellIndex = ForeachCell::CellIndex;
  using value_t = std::decay_t<decltype( U.at_ivar( std::declval<CellIndex>(), 0 ) )>; // Field type (double or float)

  MpiBufferPool& mpi_pool = DyabloSession::get_MpiBufferPool();

  uint32_t num_vars = U.nbFields(); // number of vars for each cell

  bool intermediates = pdata.intermediates;
  // Number of values to send are cell_count * num_vars 
  std::vector<int> send_sizes = pdata.m_send_cell_count;
  for( auto& v : send_sizes )
    v*=num_vars;
  std::vector<int> recv_sizes = pdata.m_recv_cell_count;  
  for( auto& v : recv_sizes )
    v*=num_vars;

  const Kokkos::View< uint32_t* >& send_iOct = pdata.m_send_iOct;
  const Kokkos::View< uint32_t* >& send_iCell = pdata.m_send_iCell;
  const Kokkos::View< uint32_t* >& recv_iOct = pdata.m_recv_iOct;
  const Kokkos::View< uint32_t* >& recv_iCell = pdata.m_recv_iCell;
  uint32_t total_send_size = send_iOct.size(), total_recv_size = recv_iOct.size(); // send/recv buffer size (number of cells)    
  uint32_t bx=U.getShape().bx, by=U.getShape().by, bz=U.getShape().bz ; // Block size

  Kokkos::View< value_t* > send_buffer = mpi_pool.MPI_Alloc_view<Kokkos::View< value_t* >>( "exchange_ghosts::send_buffer", num_vars*total_send_size );

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

    CellIndex cell_index { {iOct, false, intermediates}, i, j, k, bx, by, bz };
    send_buffer( ipack ) = U.at_ivar( cell_index, ivar );
  });

  Kokkos::View< value_t* > recv_buffer = mpi_pool.MPI_Alloc_view<Kokkos::View< value_t* >>( "exchange_ghosts::recv_buffer", num_vars*total_recv_size );
#ifdef MPI_IS_CUDA_AWARE 
  Kokkos::fence();
  pdata.mpi_comm.MPI_Alltoallv( send_buffer.data(), send_sizes.data(), recv_buffer.data(), recv_sizes.data() );
  Kokkos::fence();
#else
  {
    auto send_buffer_host = Kokkos::create_mirror_view(send_buffer);
    auto recv_buffer_host = Kokkos::create_mirror_view(recv_buffer);

    Kokkos::deep_copy(send_buffer_host, send_buffer);
    pdata.mpi_comm.MPI_Alltoallv( send_buffer_host.data(), send_sizes.data(), recv_buffer_host.data(), recv_sizes.data() );
    Kokkos::deep_copy(recv_buffer, recv_buffer_host);
  }  
#endif

  mpi_pool.MPI_Free_view( send_buffer );

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

    CellIndex cell_index { {iOct, true, intermediates}, i, j, k, bx, by, bz };
    U.at_ivar( cell_index, ivar ) = recv_buffer( ipack );
  });

  mpi_pool.MPI_Free_view( recv_buffer );
}

template< typename CellArray_t >
void reduce_ghosts_aux( const GhostCommunicator_partial_blocks::Pdata& pdata, CellArray_t& U)
{
  using CellIndex = ForeachCell::CellIndex;
  using value_t = std::decay_t<decltype( U.at_ivar( std::declval<CellIndex>(), 0 ) )>; // Field type (double or float)

  bool intermediates = pdata.intermediates;
  uint32_t num_vars = U.nbFields(); // number of vars for each cell

  // Note : sends and recvs counts are swapped for reduce

  // Number of values to send are cell_count * num_vars 
  std::vector<int> send_sizes = pdata.m_recv_cell_count;
  for( auto& v : send_sizes )
    v*=num_vars;
  std::vector<int> recv_sizes = pdata.m_send_cell_count;  
  for( auto& v : recv_sizes )
    v*=num_vars;

  const Kokkos::View< uint32_t* >& send_iOct = pdata.m_recv_iOct;
  const Kokkos::View< uint32_t* >& send_iCell = pdata.m_recv_iCell;
  const Kokkos::View< uint32_t* >& recv_iOct = pdata.m_send_iOct;
  const Kokkos::View< uint32_t* >& recv_iCell = pdata.m_send_iCell;
  uint32_t total_send_size = send_iOct.size(), total_recv_size = recv_iOct.size(); // send/recv buffer size (number of cells)    
  uint32_t bx=U.getShape().bx, by=U.getShape().by, bz=U.getShape().bz ; // Block size

  Kokkos::View< value_t*, Kokkos::LayoutLeft > send_buffer("reduce_ghosts::send_buffer", num_vars*total_send_size );

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

    CellIndex cell_index { {iOct, true, intermediates}, i, j, k, bx, by, bz };
    send_buffer( ipack ) = U.at_ivar( cell_index, ivar );
  });
  
  Kokkos::View< value_t* > recv_buffer("exchange_ghosts::recv_buffer", num_vars*total_recv_size ); 
#ifdef MPI_IS_CUDA_AWARE 
  Kokkos::fence();
  pdata.mpi_comm.MPI_Alltoallv( send_buffer.data(), send_sizes.data(), recv_buffer.data(), recv_sizes.data() );
  Kokkos::fence();
#else
  {
    auto send_buffer_host = Kokkos::create_mirror_view(send_buffer);
    auto recv_buffer_host = Kokkos::create_mirror_view(recv_buffer);

    Kokkos::deep_copy(send_buffer_host, send_buffer);
    pdata.mpi_comm.MPI_Alltoallv( send_buffer_host.data(), send_sizes.data(), recv_buffer_host.data(), recv_sizes.data() );
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

    CellIndex cell_index { {iOct, false, intermediates}, i, j, k, bx, by, bz };
    Kokkos::atomic_add( &U.at_ivar( cell_index, ivar ), recv_buffer( ipack ) );
  });
}


struct Subset_GhostMap
{
  std::vector<int>        ghostmap_send_sizes; 
  std::vector<int>        ghostmap_recv_sizes;    
  Kokkos::View<uint32_t*> ghostmap_send_iGhosts;
  Kokkos::View<uint32_t*> ghostmap_recv_iGhosts; 
};


/**
 * Computes sizes to send/recv form/to other processes knowing the sizes of the full communicator and the indices 
 * of the subset octants in the original octant list
 * @param sizes_full sizes to send/recv in full comm
 * @param subset_iOcts indices of subset octs in full comm oct list
 **/
std::vector<int> compute_subset_sizes(
  const std::vector<int>& sizes_full,
  const Kokkos::View<uint32_t*>& subset_iOcts)
{
  int mpi_size = sizes_full.size();

  Kokkos::View<uint32_t*> rank_begin_full("rank_begin_full", mpi_size);
  auto rank_begin_full_host = Kokkos::create_mirror_view( rank_begin_full );
  uint32_t niOct_full = 0;
  for( int i=0; i<mpi_size; i++ )
  {
    rank_begin_full_host(i) = niOct_full;
    niOct_full+=sizes_full[i];
  }
  Kokkos::deep_copy( rank_begin_full, rank_begin_full_host );

  Kokkos::View<uint32_t*> rank_begin_subset("rank_begin_subset", mpi_size);
  uint32_t niOct_subset = subset_iOcts.size();
  Kokkos::parallel_for( "OctSubset::count_sizes", niOct_subset,
    KOKKOS_LAMBDA( uint32_t i )
  {   
    // Find start of rank in subset
    uint32_t next = i < niOct_subset-1 ? subset_iOcts(i+1) : niOct_full;
    DYABLO_ASSERT_KOKKOS_DEBUG( subset_iOcts(i) < next, "OctSubset_init_with_send : subset_iOcts is not increasing" );
    for( int rank=0; rank<mpi_size; rank++ )
    {
      if(subset_iOcts(i) < rank_begin_full(rank) && rank_begin_full(rank) <= next )
        rank_begin_subset(rank) = i+1;
    }
  });

  auto rank_begin_subset_host = Kokkos::create_mirror_view( rank_begin_subset );
  Kokkos::deep_copy( rank_begin_subset_host, rank_begin_subset );

  std::vector<int> subset_sizes( mpi_size );
  for( int i=0; i<mpi_size; i++ )
  {
    uint32_t next = i < mpi_size-1 ? rank_begin_subset_host(i+1) : niOct_subset;
    subset_sizes[i] = next - rank_begin_subset_host(i);
  }

  return subset_sizes;
}

Subset_GhostMap compute_subset_ghostmap_with_send( 
  const MpiComm& mpi_comm,
  const std::vector<int>& send_sizes_full,
  const Kokkos::View<uint32_t*>& send_iOcts_full,
  const std::vector<int>& recv_sizes_full,
  const Kokkos::View<uint32_t*>& subset_iOcts_send,
  const Kokkos::View<uint32_t*>& subset_iOcts_recv )
{
  auto send_sizes = compute_subset_sizes(send_sizes_full, subset_iOcts_send);
  auto recv_sizes = compute_subset_sizes(recv_sizes_full, subset_iOcts_recv);

  return {
    send_sizes,
    recv_sizes,    
    subset_iOcts_send,
    subset_iOcts_recv
  };
}

Subset_GhostMap compute_subset_ghostmap( 
  const MpiComm& mpi_comm,
  const std::vector<int>& send_sizes_full,
  const Kokkos::View<uint32_t*>& send_iOcts_full,
  const std::vector<int>& recv_sizes_full,
  const Kokkos::View<uint32_t*>& subset_iOcts)
{
  int mpi_size = mpi_comm.MPI_Comm_size();  

  Kokkos::View<int*> recv_sizes_full_device("recv_sizes_full_device", mpi_size);
  {
    std::vector<int> recv_sizes_full_copy = recv_sizes_full;
    Kokkos::View<int*>::host_mirror_type recv_sizes_full_host( recv_sizes_full_copy.data(), mpi_size );
    Kokkos::deep_copy(recv_sizes_full_device, recv_sizes_full_host);
  }
   
  Kokkos::View<uint32_t*> recv_iGhosts = subset_iOcts;

  std::vector<int> recv_sizes(mpi_size); 
  std::vector<int> send_sizes(mpi_size); 
  Kokkos::View<uint32_t*> send_iGhosts;
  {
    // Compute ghost index local to ranks 
    // 0s are every first ghost from each rank
    Kokkos::View<uint32_t*> recv_iOcts_ranklocal("recv_iOcts_ranklocal", recv_iGhosts.size());
    Kokkos::View<int*> recv_sizes_device("recv_sizes_device", mpi_size);
    Kokkos::parallel_for( "OctSubset::convert_to_ranklocal", recv_iGhosts.size(),
      KOKKOS_LAMBDA( uint32_t i )
    { 
      uint32_t iGhost_local = recv_iGhosts(i);

      // Find origin rank of iGhost_local
      uint32_t iGhost_rank_begin = 0;
      int rank = 0;
      for( rank=0; rank < mpi_size; rank++ )
      {
        uint32_t iGhost_rank_end = iGhost_rank_begin + recv_sizes_full_device(rank);
        if( iGhost_local < iGhost_rank_end )
          break;

        iGhost_rank_begin = iGhost_rank_end;
      }
      DYABLO_ASSERT_KOKKOS_DEBUG( 0<=rank && rank<mpi_size, "OctSubset error - could not find subset octant rank." );
      DYABLO_ASSERT_KOKKOS_DEBUG( iGhost_rank_begin <= iGhost_local && iGhost_local < iGhost_rank_begin + recv_sizes_full_device(rank), "OctSubset error - could not find subset octant rank" );
      
      Kokkos::atomic_inc(&recv_sizes_device(rank));
      recv_iOcts_ranklocal(i) = iGhost_local - iGhost_rank_begin;
    });

    // Exchange sizes
    {
      Kokkos::View<int*>::host_mirror_type recv_sizes_host( recv_sizes.data(), mpi_size );
      Kokkos::deep_copy( recv_sizes_host, recv_sizes_device );

      mpi_comm.MPI_Alltoall( recv_sizes.data(), 1, send_sizes.data(), 1 );
    }


    // Exchange ranklocal iOcts
    int total_send_count = std::accumulate( send_sizes.begin(), send_sizes.end(), 0 );
    Kokkos::View<uint32_t*> send_iOcts_ranklocal("send_iOcts_ranklocal", total_send_count);
    #ifdef MPI_IS_CUDA_AWARE
    {
      mpi_comm.MPI_Alltoallv(  recv_iOcts_ranklocal.data(), recv_sizes.data(),
                               send_iOcts_ranklocal.data(), send_sizes.data() );
    }
    #else
    {
      auto recv_iOcts_ranklocal_host = Kokkos::create_mirror_view( recv_iOcts_ranklocal );
      auto send_iOcts_ranklocal_host = Kokkos::create_mirror_view( send_iOcts_ranklocal );
      Kokkos::deep_copy( recv_iOcts_ranklocal_host, recv_iOcts_ranklocal );
      mpi_comm.MPI_Alltoallv(  recv_iOcts_ranklocal_host.data(), recv_sizes.data(),
                               send_iOcts_ranklocal_host.data(), send_sizes.data() );
      Kokkos::deep_copy( send_iOcts_ranklocal, send_iOcts_ranklocal_host );
    }
    #endif
    
    send_iGhosts = Kokkos::View<uint32_t*>( "send_iGhosts", total_send_count );
    int i_rank_begin = 0;
    int iGhost_rank_begin = 0;
    for( int rank=0; rank<mpi_size; rank++ )
    {
      Kokkos::parallel_for("OctSubset::unpack_requested_octs", send_sizes[rank],
        KOKKOS_LAMBDA( int i_rank )
      {
        int i = i_rank + i_rank_begin;
        // Position in full send_buffer
        uint32_t iGhost = send_iOcts_ranklocal(i) + iGhost_rank_begin;
        send_iGhosts(i) = iGhost;
      });
      i_rank_begin += send_sizes[rank];
      iGhost_rank_begin += send_sizes_full[rank];
    }    
  } 

  return {
    send_sizes,
    recv_sizes,    
    send_iGhosts,
    recv_iGhosts
  };
}

template< typename View_t > 
View_t filter_view( 
    const View_t& array_full,
    const Kokkos::View<uint32_t*>& ghostmap_iGhosts )
{
  int nbOcts = ghostmap_iGhosts.size();
  View_t array_filtered( array_full.label(), nbOcts );
  Kokkos::parallel_for( "Filter ghostmap masks", nbOcts,
    KOKKOS_LAMBDA( uint32_t i )
  {
    array_filtered(i) = array_full( ghostmap_iGhosts(i) );
  });

  return array_filtered;
};

GhostCommunicator_partial_blocks_OctSubset_Pdata init_subset_aux( const GhostCommunicator_partial_blocks& comm_full, Subset_GhostMap& subset_map )
{
  const MpiComm& mpi_comm = comm_full.pdata->mpi_comm;

  const std::vector<int>& ghostmap_send_sizes = subset_map.ghostmap_send_sizes;
  const std::vector<int>& ghostmap_recv_sizes = subset_map.ghostmap_recv_sizes;
  const Kokkos::View<uint32_t*>& ghostmap_send_iGhosts = subset_map.ghostmap_send_iGhosts;
  const Kokkos::View<uint32_t*>& ghostmap_recv_iGhosts = subset_map.ghostmap_recv_iGhosts;

  Kokkos::View<int*> ghostmap_recv_masks = filter_view( comm_full.pdata->ghostmap_recv_masks, ghostmap_recv_iGhosts );
  Kokkos::View<int*> ghostmap_send_masks = filter_view( comm_full.pdata->ghostmap_send_masks, ghostmap_send_iGhosts );
  Kokkos::View<uint32_t*> ghostmap_send_iOcts = filter_view( comm_full.pdata->ghostmap_send_iOcts, ghostmap_send_iGhosts );

  bool intermediates = comm_full.pdata->intermediates;
  uint32_t bx = comm_full.pdata->bx;
  uint32_t by = comm_full.pdata->by;
  uint32_t bz = comm_full.pdata->bz;
  uint32_t ghost_count = comm_full.pdata->ghost_count;
  Precomputed_Mask_Cells precomputed_cells = precompute_facemask_cells( bx, by, bz, ghost_count );
  // Compute list of cells to send
  CellList send_cells = list_cells( precomputed_cells, ghostmap_send_sizes, ghostmap_send_iOcts, ghostmap_send_masks);
  // Compute list of cells to recieve
  CellList recv_cells = list_cells( precomputed_cells, ghostmap_recv_sizes, ghostmap_recv_iGhosts, ghostmap_recv_masks);

  GhostCommunicator_partial_blocks_Pdata pdata{intermediates, mpi_comm};
  pdata.bx = bx;
  pdata.by = by;
  pdata.bz = bz;
  pdata.ghost_count = ghost_count;
  pdata.m_local_ghost_octants = comm_full.pdata->m_local_ghost_octants; // Number of octants to ALLOCATE for ghosts, not number exchanged

  pdata.ghostmap_send_sizes = ghostmap_send_sizes;
  pdata.ghostmap_send_masks = ghostmap_send_masks;
  pdata.ghostmap_send_iOcts = ghostmap_send_iOcts;

  pdata.ghostmap_recv_sizes = ghostmap_recv_sizes;
  pdata.ghostmap_recv_masks = ghostmap_recv_masks;

  pdata.m_send_cell_count = send_cells.sizes;
  pdata.m_recv_cell_count = recv_cells.sizes;
  pdata.m_send_iOct = send_cells.iOcts;
  pdata.m_send_iCell = send_cells.iCells; 
  pdata.m_recv_iOct = recv_cells.iOcts;
  pdata.m_recv_iCell = recv_cells.iCells;

  return GhostCommunicator_partial_blocks_OctSubset_Pdata{ 
    GhostCommunicator_partial_blocks(pdata),
  };  
}

GhostCommunicator_partial_blocks_OctSubset_Pdata init_subset( const GhostCommunicator_partial_blocks& comm_full, Kokkos::View<uint32_t*> subset_iOcts )
{
  const MpiComm& mpi_comm = comm_full.pdata->mpi_comm;

  Subset_GhostMap subset_map = compute_subset_ghostmap(
    mpi_comm,
    comm_full.pdata->ghostmap_send_sizes,
    comm_full.pdata->ghostmap_send_iOcts,
    comm_full.pdata->ghostmap_recv_sizes,
    subset_iOcts);

  return init_subset_aux( comm_full, subset_map );
}

GhostCommunicator_partial_blocks_OctSubset_Pdata init_subset_with_send( const GhostCommunicator_partial_blocks& comm_full, Kokkos::View<uint32_t*> subset_iOcts_send, Kokkos::View<uint32_t*> subset_iOcts_recv )
{
  const MpiComm& mpi_comm = comm_full.pdata->mpi_comm;

  Subset_GhostMap subset_map = compute_subset_ghostmap_with_send(
    mpi_comm,
    comm_full.pdata->ghostmap_send_sizes,
    comm_full.pdata->ghostmap_send_iOcts,
    comm_full.pdata->ghostmap_recv_sizes,
    subset_iOcts_send,
    subset_iOcts_recv);

  return init_subset_aux( comm_full, subset_map );
}

} // namespace

GhostCommunicator_partial_blocks::GhostCommunicator_partial_blocks( const AMRmesh& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape, uint32_t ghost_count, bool intermediates, const MpiComm& mpi_comm )
  : GhostCommunicator_partial_blocks(init( amr_mesh.getGhostMap(), shape, ghost_count, intermediates, mpi_comm) )
{}

GhostCommunicator_partial_blocks::GhostCommunicator_partial_blocks( const GhostCommunicator_partial_blocks& o )
: GhostCommunicator_partial_blocks( *(o.pdata) )
{}

GhostCommunicator_partial_blocks::GhostCommunicator_partial_blocks( const GhostCommunicator_partial_blocks::Pdata& o )
: pdata( std::make_unique<Pdata>( o ) )
{}

GhostCommunicator_partial_blocks::~GhostCommunicator_partial_blocks()
{}

uint32_t GhostCommunicator_partial_blocks::getNumGhosts() const
{
  return pdata->m_local_ghost_octants;
}

bool GhostCommunicator_partial_blocks::has_intermediates() const
{
  return pdata->intermediates;
}

Kokkos::View<uint32_t*> GhostCommunicator_partial_blocks::iOcts_send() const
{
  return pdata->ghostmap_send_iOcts;
}

template<typename T>
void GhostCommunicator_partial_blocks::exchange_ghosts( const UserData::FieldAccessor_t<T>& U) const
{
  DYABLO_ASSERT_HOST_RELEASE( !this->has_intermediates(), "Trying to exchange intermediates with FieldAccessor without intermediates" );
  exchange_ghosts_aux(*pdata, U);
}

template<typename T>
void GhostCommunicator_partial_blocks::exchange_ghosts( const UserData::FieldAccessor_fulltree_t<T>& U) const
{
  exchange_ghosts_aux(*pdata, U);
}

void GhostCommunicator_partial_blocks::exchange_ghosts( const ForeachCell::CellArray_global_ghosted& U) const
{
  exchange_ghosts_aux(*pdata, U);
}

template<typename T>
void GhostCommunicator_partial_blocks::reduce_ghosts( UserData::FieldAccessor_t<T>& U) const
{
  DYABLO_ASSERT_HOST_RELEASE( !this->has_intermediates(), "Trying to exchange intermediates with FieldAccessor without intermediates" );
  reduce_ghosts_aux(*pdata, U);
}

template<typename T>
void GhostCommunicator_partial_blocks::reduce_ghosts( UserData::FieldAccessor_fulltree_t<T>& U) const
{
  reduce_ghosts_aux(*pdata, U);
}

void GhostCommunicator_partial_blocks::reduce_ghosts( ForeachCell::CellArray_global_ghosted& U) const
{
  reduce_ghosts_aux(*pdata,U);
}

GhostCommunicator_partial_blocks_OctSubset::GhostCommunicator_partial_blocks_OctSubset( const GhostCommunicator_partial_blocks_OctSubset& o )
: pdata( std::make_unique<Pdata>(*o.pdata) )
{}


GhostCommunicator_partial_blocks_OctSubset::GhostCommunicator_partial_blocks_OctSubset( const GhostCommunicator_partial_blocks& comm_full, Kokkos::View<uint32_t*> subset_iOcts_recv )
: pdata( std::make_unique<Pdata>( Pdata{ init_subset(comm_full, subset_iOcts_recv) } ) )
{}

GhostCommunicator_partial_blocks_OctSubset::GhostCommunicator_partial_blocks_OctSubset( const GhostCommunicator_partial_blocks& comm_full, Kokkos::View<uint32_t*> subset_iOcts_send, Kokkos::View<uint32_t*> subset_iOcts_recv)
: pdata( std::make_unique<Pdata>( Pdata{ init_subset_with_send(comm_full, subset_iOcts_send, subset_iOcts_recv) } ) )
{}

GhostCommunicator_partial_blocks_OctSubset::~GhostCommunicator_partial_blocks_OctSubset()
{}

template<typename T>
void GhostCommunicator_partial_blocks::exchange_ghosts_subset( const UserData::FieldAccessor_t<T>& U, const OctSubset& subset ) const
{
  DYABLO_ASSERT_HOST_RELEASE( !this->has_intermediates(), "Trying to exchange intermediates with FieldAccessor without intermediates" );
  subset.pdata->comm_subset.exchange_ghosts(U);
}

template<typename T>
void GhostCommunicator_partial_blocks::exchange_ghosts_subset( const UserData::FieldAccessor_fulltree_t<T>& U, const OctSubset& subset ) const
{
  subset.pdata->comm_subset.exchange_ghosts(U);
}

template<typename T>
void GhostCommunicator_partial_blocks::reduce_ghosts_subset( UserData::FieldAccessor_t<T>& U, const OctSubset& subset ) const
{
  DYABLO_ASSERT_HOST_RELEASE( !this->has_intermediates(), "Trying to exchange intermediates with FieldAccessor without intermediates" );
  subset.pdata->comm_subset.reduce_ghosts(U);
}

template<typename T>
void GhostCommunicator_partial_blocks::reduce_ghosts_subset( UserData::FieldAccessor_fulltree_t<T>& U, const OctSubset& subset ) const
{
  subset.pdata->comm_subset.reduce_ghosts(U);
}

// real_t is double or float : instantiating both covers it without duplicates
#define DYABLO_INSTANTIATE_FIELD_TYPE(T) \
template void GhostCommunicator_partial_blocks::exchange_ghosts<T>(const UserData::FieldAccessor_t<T>&) const; \
template void GhostCommunicator_partial_blocks::exchange_ghosts<T>(const UserData::FieldAccessor_fulltree_t<T>&) const; \
template void GhostCommunicator_partial_blocks::reduce_ghosts<T>(UserData::FieldAccessor_t<T>&) const; \
template void GhostCommunicator_partial_blocks::reduce_ghosts<T>(UserData::FieldAccessor_fulltree_t<T>&) const; \
template void GhostCommunicator_partial_blocks::exchange_ghosts_subset<T>(const UserData::FieldAccessor_t<T>&, const OctSubset&) const; \
template void GhostCommunicator_partial_blocks::exchange_ghosts_subset<T>(const UserData::FieldAccessor_fulltree_t<T>&, const OctSubset&) const; \
template void GhostCommunicator_partial_blocks::reduce_ghosts_subset<T>(UserData::FieldAccessor_t<T>&, const OctSubset&) const; \
template void GhostCommunicator_partial_blocks::reduce_ghosts_subset<T>(UserData::FieldAccessor_fulltree_t<T>&, const OctSubset&) const;
DYABLO_INSTANTIATE_FIELD_TYPE(double)
DYABLO_INSTANTIATE_FIELD_TYPE(float)
#undef DYABLO_INSTANTIATE_FIELD_TYPE

} // namespace dyablo