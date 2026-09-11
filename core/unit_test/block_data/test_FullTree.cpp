/**
 * \file test_CommGhosts.cpp
 * \author A. Durocher
 * Test intermediate cells computations for full tree
 * 
 * Leaves are filled with cell center positions, intermediate cells are filled with average values from children, 
 * Intermediate cells should have cell center as values.
 */
#include "gtest/gtest.h"

#include "amr/AMRmesh.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

#include "user_data/UserData.h"
#include "user_data/FieldAccessor.h"
#include "utils/config/ConfigMap.h"
#include "mpi/GhostCommunicator_Subset_levels.hpp"
#include "foreach_cell/IterationSpace_levels.hpp"

namespace dyablo {
namespace {

using OctantIndex = LightOctree::OctantIndex;
using CellIndex = ForeachCell::CellIndex;
using CellArray_shape = ForeachCell::CellArray_shape;


} // namespace
} // namespace dyablo


/***
 * Fill cells with cell centers, average to parent cells until level_min and check if value is still cell center. 
 * Leaves are filled with cell center position.
 * Level by level from fine to corase : 
 *  - getChildren mode :
 *    - iterate over parents, use getChildren + foreach_sibling to get mean value
 *    - MPI ghosts are exchanged
 *  - getParent mode :
 *    - iterate over children and accumulate in parent with getparent
 *    - MPI reduction is performed to update parent values
 * All Cells including intermediates should have their cell center matching cell position
 ***/
void test_FullTree_average_children(bool getParent_mode)
{
  bool getChildren_mode = !getParent_mode;

  using namespace dyablo;

  std::cout << "// =========================================\n";
  std::cout << "// Testing FullTree Average ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  int ndim = 3;
  int level_min = 3;
  int level_max = 7;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    amr_mesh = std::make_shared<AMRmesh>(ndim, std::array<bool,3>{false,false,false}, level_min, level_max);
    //amr_mesh->setBalanceCodimension(ndim);
    //uint32_t idx = 0;
    //amr_mesh->setBalance(idx,true);
    // mr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    //amr_mesh->setPeriodic(4);
    //amr_mesh->setPeriodic(5);

    int local_rank = amr_mesh->getMpiComm().MPI_Comm_rank();

    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1, 1);      
    amr_mesh->adapt();
    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1, 1);      
    amr_mesh->adapt();
    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1, 1);      
    amr_mesh->adapt();
    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1, 1);      
    amr_mesh->adapt();

    amr_mesh->loadBalance(0);
  }

  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = 8;
  ConfigMap configMap ("");
  configMap.getValue<uint32_t>("amr", "bx", bx);
  configMap.getValue<uint32_t>("amr", "by", by);
  configMap.getValue<uint32_t>("amr", "bz", bz);
  
  ForeachCell foreach_cell(*amr_mesh, configMap);  
  UserData U ( configMap, foreach_cell );

  U.new_fields({"px", "dummy", "py", "pz"});

  std::cout << "Initialize User Data..." << std::endl;

  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  using pos_t = ForeachCell::CellMetaData::pos_t;

  { // Initialize U
    enum VarIndex_test{Px,Py,Pz,Dummy};

    UserData::FieldAccessor Uin = U.getAccessor( {{"px", Px}, {"py", Py}, {"pz", Pz}, {"dummy", Dummy}} );
    foreach_cell.foreach_cell( "Init_U", U.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      auto c = cells.getCellCenter( iCell );
      Uin.at(iCell, Px) = c[IX];
      Uin.at(iCell, Py) = c[IY];
      Uin.at(iCell, Pz) = c[IZ];
      Uin.at(iCell, Dummy) = 99;
    });
  }

  U.new_intermediate_fields( {"px","py","pz"} );

  enum VarIndex_test{Px,Py,Pz};
  UserData::FieldAccessor_fulltree Ua = U.getAccessor_fulltree( {{"px", Px}, {"py", Py}, {"pz", Pz}} );

  const LightOctree& lmesh = amr_mesh->getLightOctree();
  GhostCommunicator_full_blocks ghost_communicator_leaves( *amr_mesh, Ua.getShape(), -1, false );
  GhostCommunicator_full_blocks ghost_communicator_intermediates( *amr_mesh, Ua.getShape(), -1, true );

  constexpr bool with_locals = true;
  constexpr bool without_locals = false;
  constexpr bool with_ghosts = true;
  constexpr bool without_ghosts = false;
  constexpr bool with_intermediates = true;
  constexpr bool without_intermediates = false;
  Subset_levels level_subsets( lmesh, amr_mesh->get_level_max() );
  IterationSpace_levels iterationspace_levels( lmesh, amr_mesh->get_level_max() );

  if( getChildren_mode )
  {
    for( int level = level_max-1; level >= level_min; level-- )
    {
      // Exhange children of current level to have all needed siblings

      //ghost_communicator_leaves.exchange_ghosts( Ua );
      auto subset_level_leaves = level_subsets.getGhostCommunicatorSubset_level(level+1, ghost_communicator_leaves);
      ghost_communicator_leaves.exchange_ghosts_subset( Ua, subset_level_leaves );

      //ghost_communicator_intermediates.exchange_ghosts( Ua );
      auto subset_level_intermediates = level_subsets.getGhostCommunicatorSubset_level(level+1, ghost_communicator_intermediates);
      ghost_communicator_intermediates.exchange_ghosts_subset( Ua, subset_level_intermediates );
    
      auto iter_space_level_intermediates = iterationspace_levels.getIterationSpace<without_locals, without_ghosts, with_intermediates, without_ghosts>(level, Ua.getShape());
      foreach_cell.foreach_cell("average_parent_cell", iter_space_level_intermediates,
        KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
      {
        ForeachCell::SearchMode_intermediates search_neighbor_intermediate(lmesh, ForeachCell::SearchMode_intermediates::BiggerNeighborMode::ASSERT);

        ForeachCell::CellIndex iCell_c0 = iCell.getChildren(lmesh);

        pos_t p{};
        int ns = foreach_sibling_scattered( ndim, iCell_c0, lmesh,
          [&]( const ForeachCell::CellIndex& iCell_c )
        {
          real_t px = Ua.at(iCell_c, Px);
          real_t py = Ua.at(iCell_c, Py);
          real_t pz = Ua.at(iCell_c, Pz);

          p[IX] += px;
          p[IY] += py;
          p[IZ] += pz;
        });

        Ua.at( iCell, Px ) = p[IX]/ns;
        Ua.at( iCell, Py ) = p[IY]/ns;
        Ua.at( iCell, Pz ) = p[IZ]/ns;
      });
    }
  }
  else //if( getParent_mode )
  {
    for( int level = level_max; level > level_min; level-- )
    {
      // Reset intermediate ghosts
      ForeachCell::IterationSpace_fullArray_impl<without_locals, without_ghosts, without_intermediates, with_ghosts> iter_space_inter_ghosts(Ua.getShape());
      foreach_cell.foreach_cell( "zero ghosts", iter_space_inter_ghosts, 
        KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
      {
        Ua.at( iCell, Px ) = 0;
        Ua.at( iCell, Py ) = 0;
        Ua.at( iCell, Pz ) = 0;
      }); 

      auto iter_space_level = iterationspace_levels.getIterationSpace<with_locals, without_ghosts, with_intermediates, without_ghosts>(level, Ua.getShape());
      foreach_cell.foreach_cell("average_parent_cell", iter_space_level,
        KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
      {
        ForeachCell::SearchMode_intermediates search_neighbor_intermediate(lmesh, ForeachCell::SearchMode_intermediates::BiggerNeighborMode::ASSERT);

        ForeachCell::CellIndex iCell_p = iCell.getParent(lmesh);

        int ns = 8; //number of suboctants

        real_t px = Ua.at(iCell, Px);
        real_t py = Ua.at(iCell, Py);
        real_t pz = Ua.at(iCell, Pz);

        Kokkos::atomic_add( &Ua.at( iCell_p, Px ), px/ns );
        Kokkos::atomic_add( &Ua.at( iCell_p, Py ), py/ns );
        Kokkos::atomic_add( &Ua.at( iCell_p, Pz ), pz/ns );
      }); 

      // Reduce ghosts for parent level to have up to date parents
      
      //ghost_communicator_intermediates.reduce_ghosts( Ua );
      auto subset_level_intermediates = level_subsets.getGhostCommunicatorSubset_level(level-1, ghost_communicator_intermediates);
      ghost_communicator_intermediates.reduce_ghosts_subset( Ua, subset_level_intermediates );
    }
  }
  
  ForeachCell::IterationSpace_fullArray_impl<with_locals, without_ghosts, with_intermediates, without_ghosts> iter_space_with_intermediates(Ua.getShape());
  int error_count = 0;
  int total_count = 0;
  const uint32_t total_cells = bx * by * bz * (amr_mesh->getNumOctants() + amr_mesh->getNumIntermediates());
  foreach_cell.reduce_cell( "test_values", iter_space_with_intermediates,
    KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell, int& error_count, int& total_count )
  {
    total_count++;
    auto test_equal = [&](real_t a, real_t b)
    {
      //EXPECT_DOUBLE_EQ(a, b);
      if( a != b )
      {
        error_count++;
        printf("%f != %f\n", a, b);
      }
    };

    auto pos = cells.getCellCenter(iCell);
    test_equal( pos[IX], Ua.at(iCell, Px) );
    test_equal( pos[IY], Ua.at(iCell, Py) );
    test_equal( pos[IZ], Ua.at(iCell, Pz) );    
  }, error_count, total_count);

  EXPECT_EQ(0, error_count);
  EXPECT_EQ(total_cells, total_count);
}

TEST(dyablo, test_FullTree_average_getChildren)
{
  using namespace dyablo;
  bool getChildren_mode = false;
  test_FullTree_average_children(getChildren_mode);
}

TEST(dyablo, test_FullTree_average_getParent)
{
  using namespace dyablo;
  bool getParent_mode = true;
  test_FullTree_average_children(getParent_mode);
}


/***
 * Fill cells with cell centers, compute average with neighbors and check if value is still cell center. 
 * - All cells (includeing intermediates) are filled with cell center position
 * - MPI Communication is performed
 * - add mean of neighbors to each intermediate
 * - Verify intermediates have 2*position
 ***/
template< bool diagonal >
void test_FullTree_average_stencil()
{
  using namespace dyablo;

  std::cout << "// =========================================\n";
  std::cout << "// Testing FullTree Average ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  int ndim = 3;
  int level_min = 3;
  int level_max = 7;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    amr_mesh = std::make_shared<AMRmesh>(ndim, std::array<bool,3>{false,false,false}, level_min, level_max);
    //amr_mesh->setBalanceCodimension(ndim);
    //uint32_t idx = 0;
    //amr_mesh->setBalance(idx,true);
    // mr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    //amr_mesh->setPeriodic(4);
    //amr_mesh->setPeriodic(5);

    int local_rank = amr_mesh->getMpiComm().MPI_Comm_rank();

    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1, 1);      
    amr_mesh->adapt();
    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1, 1);      
    amr_mesh->adapt();
    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1, 1);      
    amr_mesh->adapt();
    if( local_rank == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1, 1);      
    amr_mesh->adapt();

    amr_mesh->loadBalance(0);
  }

  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = 8;
  ConfigMap configMap ("");
  configMap.getValue<uint32_t>("amr", "bx", bx);
  configMap.getValue<uint32_t>("amr", "by", by);
  configMap.getValue<uint32_t>("amr", "bz", bz);
  
  ForeachCell foreach_cell(*amr_mesh, configMap);  
  UserData U ( configMap, foreach_cell );

  U.new_fields({"px", "dummy", "py", "pz"});

  std::cout << "Initialize User Data..." << std::endl;

  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  using pos_t = ForeachCell::CellMetaData::pos_t;

  U.new_intermediate_fields( {"px","py","pz"} );

  constexpr bool with_locals = true;
  constexpr bool without_locals = false;
  constexpr bool without_ghosts = false;
  constexpr bool with_intermediates = true;

  enum VarIndex_test{Px,Py,Pz};
  UserData::FieldAccessor_fulltree Uin = U.getAccessor_fulltree( {{"px", Px}, {"py", Py}, {"pz", Pz}} );

  ForeachCell::IterationSpace_fullArray_impl<with_locals, without_ghosts, with_intermediates, without_ghosts> iter_space_with_intermediates(Uin.getShape());

  { // Initialize U
    foreach_cell.foreach_cell( "Init_U", iter_space_with_intermediates,
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      auto c = cells.getCellCenter( iCell );
      Uin.at(iCell, Px) = c[IX];
      Uin.at(iCell, Py) = c[IY];
      Uin.at(iCell, Pz) = c[IZ];
    });
  }

  const LightOctree& lmesh = amr_mesh->getLightOctree();  
  GhostCommunicator_partial_blocks ghost_communicator_leaves( *amr_mesh, Uin.getShape(), 1, false );
  GhostCommunicator_partial_blocks ghost_communicator_intermediates( *amr_mesh, Uin.getShape(), 1, true );

  ghost_communicator_leaves.exchange_ghosts( Uin );
  ghost_communicator_intermediates.exchange_ghosts( Uin );

  U.new_fields({"px_new", "py_new", "pz_new"});
  U.new_intermediate_fields( {"px_new","py_new","pz_new"} );
  UserData::FieldAccessor_fulltree Uout = U.getAccessor_fulltree( {{"px_new", Px}, {"py_new", Py}, {"pz_new", Pz}} );
  ForeachCell::IterationSpace_fullArray_impl<without_locals, without_ghosts, with_intermediates, without_ghosts> iter_space_only_intermediates(Uout.getShape());

  foreach_cell.foreach_cell("stencil_mean", iter_space_with_intermediates,
    KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
  {
    // iCell is intermediate, we should always find a same size neighbor
    ForeachCell::SearchMode_intermediates search_neighbor_intermediate(lmesh, ForeachCell::SearchMode_intermediates::BiggerNeighborMode::GETLEAF);

    pos_t p{};
    
    auto add_neighbor = [&]( const CellIndex::offset_t& off )
    {
      ForeachCell::CellIndex iCell_n = iCell.getNeighbor( off, search_neighbor_intermediate );

      DYABLO_ASSERT_KOKKOS_RELEASE( iCell_n.level_diff() >= 0, "getNeighbor with intermediates shouldn't return small cell" );

      real_t px,py,pz;
      if( iCell_n.is_boundary() )
      {
        auto pos = cells.getCellCenter( iCell );
        auto size = cells.getCellSize( iCell );

        px = pos[IX] + off[IX] * size[IX];
        py = pos[IY] + off[IY] * size[IY];
        pz = pos[IZ] + off[IZ] * size[IZ];
      }
      else if (iCell_n.level_diff() == 1) // Bigger
      {
        using pos_t = Kokkos::Array<real_t,3>;

        DYABLO_ASSERT_KOKKOS_RELEASE( !iCell.iOct().isIntermediate, "Intermediates should always have same size neighbor" );

        auto pos_c = cells.getCellCenter(iCell);
        auto size_c = cells.getCellSize(iCell);
        
        pos_t expected_pos{
          pos_c[IX] += off[IX] * size_c[IX],
          pos_c[IY] += off[IY] * size_c[IY],
          pos_c[IZ] += off[IZ] * size_c[IZ],
        };
        
        //auto pos_n = cells.getCellCenter(iCell_n);
        pos_t pos_n {
          Uin.at(iCell_n, Px),
          Uin.at(iCell_n, Py),
          Uin.at(iCell_n, Pz),
        };
        
        // Select the same-size virtual neighbor among bigger cell's sectors
        pos_t offset_quadrant;
        offset_quadrant[IX] = pos_n[IX] > expected_pos[IX] ? -1. : +1.;
        offset_quadrant[IY] = pos_n[IY] > expected_pos[IY] ? -1. : +1.;
        offset_quadrant[IZ] = pos_n[IZ] > expected_pos[IZ] ? -1. : +1.;

        px = pos_n[IX] + offset_quadrant[IX] * size_c[IX] / 2;            
        py = pos_n[IY] + offset_quadrant[IY] * size_c[IY] / 2;            
        pz = pos_n[IZ] + offset_quadrant[IZ] * size_c[IZ] / 2; 
      }
      else // Same size
      {
        px = Uin.at(iCell_n, Px);
        py = Uin.at(iCell_n, Py);
        pz = Uin.at(iCell_n, Pz);
      }

      p[IX] += px;
      p[IY] += py;
      p[IZ] += pz;
    };

    int nneighbor;
    if( diagonal )
    {
      for( int32_t dx=-1; dx<=1; dx++ )
      for( int32_t dy=-1; dy<=1; dy++ )
      for( int32_t dz=-1; dz<=1; dz++ )
      {
        add_neighbor(CellIndex::offset_t{dx, dy, dz});
      }
      nneighbor = 27;
    }
    else
    {
      add_neighbor(CellIndex::offset_t{-1, 0, 0});
      add_neighbor(CellIndex::offset_t{ 0,-1, 0});
      add_neighbor(CellIndex::offset_t{ 0, 0,-1});
      add_neighbor(CellIndex::offset_t{ 1, 0, 0});
      add_neighbor(CellIndex::offset_t{ 0, 1, 0});
      add_neighbor(CellIndex::offset_t{ 0, 0, 1});
      nneighbor = 6;
    }   

    Uout.at( iCell, Px ) = p[IX]/nneighbor;
    Uout.at( iCell, Py ) = p[IY]/nneighbor;
    Uout.at( iCell, Pz ) = p[IZ]/nneighbor;
  }); 
  
  int error_count = 0;
  int total_count = 0;
  const uint32_t total_cells = bx * by * bz * (amr_mesh->getNumIntermediates() + amr_mesh->getNumOctants());
  foreach_cell.reduce_cell( "test_values", iter_space_with_intermediates,
    KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell, int& error_count, int& total_count )
  {
    total_count++;
    auto test_equal = [&](real_t a, real_t b)
    {
      //EXPECT_DOUBLE_EQ(a, b);
      if( a != b )
      {
        error_count++;
        printf("%f != %f\n", a, b);
      }
    };

    auto pos = cells.getCellCenter(iCell);

    test_equal( pos[IX], Uout.at(iCell, Px) );
    test_equal( pos[IY], Uout.at(iCell, Py) );
    test_equal( pos[IZ], Uout.at(iCell, Pz) ); 

  }, error_count, total_count);

  EXPECT_EQ(0, error_count);
  EXPECT_EQ(total_cells, total_count);
}

TEST(dyablo, test_FullTree_average_stencil_cross)
{
  using namespace dyablo;
  test_FullTree_average_stencil<false>();
}

TEST(dyablo, test_FullTree_average_stencil_corners)
{
  using namespace dyablo;
  test_FullTree_average_stencil<true>();
}

