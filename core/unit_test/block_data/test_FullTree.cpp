/**
 * \file test_CommGhosts.cpp
 * \author A. Durocher
 * Test intermediate cells computations for full tree
 * 
 * Leaves are filled with cell center positions, intermediate cells are filled with average values from children, 
 * Intermediate cells should have cell center as values.
 */
#include "gtest/gtest.h"

#include "mpi/ViewCommunicator.h"

#include "legacy/utils_block.h"
#include "amr/AMRmesh.h"
#include "utils/io/AMRMesh_output_vtk.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

#include "mpi/GhostCommunicator.h"
#include "UserData.h"
#include "utils/config/ConfigMap.h"


void test_FullTree()
{
  using namespace dyablo;

  std::cout << "// =========================================\n";
  std::cout << "// Testing FullTree Average ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  int ndim = 3;
  int level_min = 3;
  int level_max = 7;
  const int min_multigrid_level = 1;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, level_min, level_max);
    //amr_mesh->setBalanceCodimension(ndim);
    //uint32_t idx = 0;
    //amr_mesh->setBalance(idx,true);
    // mr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    //amr_mesh->setPeriodic(4);
    //amr_mesh->setPeriodic(5);

    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
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

  enum VarIndex_test{Px,Py,Pz,Dummy};
  UserData::FieldAccessor Ua = U.getAccessor( {{"px", Px}, {"py", Py}, {"pz", Pz}} );
  UserData::FieldAccessor Udummy = U.getAccessor( {{"dummy", Dummy}} );

  GhostCommunicator ghost_communicator( *amr_mesh, U.getShape(), 2 );
  ghost_communicator.exchange_ghosts( Ua );

  U.new_intermediate_fields( {"px","py","pz"} );
  UserData::FieldAccessor Uintermediate = U.getAccessor_intermediate( {{"px", Px}, {"py", Py}, {"pz", Pz}} );

  for( int level = level_max; level >= min_multigrid_level; level-- )
  {
    foreach_cell.foreach_intermediate_cell( "average_parent_cell", Uintermediate.getShape(),
    KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
    {
      if( cells.getLevel(iCell) == level )
      {
        ForeachCell::CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());

        //pos_t parent_pos = cells.getCellCenter( iCell );
        //pos_t child_pos =  cells.getCellCenter( iCell_c0 );

        pos_t p{};
        int ns = foreach_sibling( ndim, iCell_c0, Uintermediate.getShape(),
          [&]( const ForeachCell::CellIndex& iCell_c )
        {
          real_t px, py, pz;

          if( iCell_c.iOct.isIntermediate )
          {
            px = Uintermediate.at(iCell_c, Px);
            py = Uintermediate.at(iCell_c, Py);
            pz = Uintermediate.at(iCell_c, Pz);
          }
          else
          {
            px = Ua.at(iCell_c, Px);
            py = Ua.at(iCell_c, Py);
            pz = Ua.at(iCell_c, Pz);
          }

          p[IX] += px;
          p[IY] += py;
          p[IZ] += pz;
        });

        Uintermediate.at( iCell, Px ) = p[IX]/ns;
        Uintermediate.at( iCell, Py ) = p[IY]/ns;
        Uintermediate.at( iCell, Pz ) = p[IZ]/ns;
      }
    }); 
    
  }

  int error_count = 0;
  foreach_cell.reduce_intermediate_cell( "test_values", Uintermediate.getShape(),
    KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell, int& error_count )
  {
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
    test_equal( pos[IX], Uintermediate.at(iCell, Px) );
    test_equal( pos[IY], Uintermediate.at(iCell, Py) );
    test_equal( pos[IZ], Uintermediate.at(iCell, Pz) );    
  }, error_count);

  EXPECT_EQ(0, error_count);
}

TEST(dyablo, test_FullTree)
{
  using namespace dyablo;
  test_FullTree();
}