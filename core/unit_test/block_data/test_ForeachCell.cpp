/**
 * \file test_LoadBalance.cpp
 * \author A. Durocher
 * Tests loadbalance MPI communication
 */

#include "gtest/gtest.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

namespace dyablo
{

// =======================================================================
// =======================================================================
void run_test()
{
  std::cout << "// =========================================\n";
  std::cout << "// Testing ForeachCell ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    int ndim = 3;
    amr_mesh = std::make_shared<AMRmesh>(ndim, std::array<bool,3>{false,false,false}, 3, 5);
    //uint32_t idx = 0;
    //amr_mesh->setBalance(idx,true);
    // mr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    //amr_mesh->setPeriodic(4);
    //amr_mesh->setPeriodic(5);


    if( amr_mesh->getMpiComm().MPI_Comm_rank() == 0 )
    {
      EXPECT_GT( amr_mesh->getNumOctants() , 121 ) << "Internal test error : not enough octants in MPI rank 0";

      // Refine initial 47 (final smaller 47..54)
      amr_mesh->setMarker(47,1);
      // Refine around initial 78
      amr_mesh->setMarker(65 ,1);
      amr_mesh->setMarker(72 ,1);
      amr_mesh->setMarker(73 ,1);
      amr_mesh->setMarker(67 ,1);
      amr_mesh->setMarker(74 ,1);
      amr_mesh->setMarker(75 ,1);
      amr_mesh->setMarker(76 ,1);
      amr_mesh->setMarker(77 ,1);
      amr_mesh->setMarker(71 ,1);
      amr_mesh->setMarker(69 ,1);
      //78
      amr_mesh->setMarker(81 ,1);
      amr_mesh->setMarker(88 ,1);
      amr_mesh->setMarker(89 ,1);
      amr_mesh->setMarker(79 ,1);
      amr_mesh->setMarker(85 ,1);
      amr_mesh->setMarker(92 ,1);
      amr_mesh->setMarker(93 ,1);
      amr_mesh->setMarker(97 ,1);
      amr_mesh->setMarker(104 ,1);
      amr_mesh->setMarker(105 ,1);
      amr_mesh->setMarker(99 ,1);
      amr_mesh->setMarker(106 ,1);
      amr_mesh->setMarker(107 ,1);
      amr_mesh->setMarker(113 ,1);
      amr_mesh->setMarker(120 ,1);
      amr_mesh->setMarker(121 ,1);
    }

    amr_mesh->adapt();
    amr_mesh->loadBalance();
  }

  // Content of .ini file used ton configure configmap and HydroParams
  std::string configmap_str = "";
  ConfigMap configMap(configmap_str);

  // Fill default values
  constexpr int ndim = 3;
  configMap.getValue<int>("run", "ndim", ndim);
  uint32_t bx = configMap.getValue<uint32_t>("amr", "bx", 8);
  uint32_t by = configMap.getValue<uint32_t>("amr", "by", 1);
  uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", (ndim==2)?1:4);
  configMap.getValue<real_t>("mesh", "xmin", -2);
  configMap.getValue<real_t>("mesh", "ymin", 0);
  configMap.getValue<real_t>("mesh", "zmin", 1);
  configMap.getValue<real_t>("mesh", "xmax", 2);
  configMap.getValue<real_t>("mesh", "ymax", 2);
  configMap.getValue<real_t>("mesh", "zmax", 4);

  // Create ForeachCell
  std::cout << "Initialize ForeachCell..." << std::endl;
  ForeachCell foreach_cell( *amr_mesh, configMap );
  ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

  std::cout << "Allocate U..." << std::endl;
  
  enum VarIndex_U { IU, IV, IW, COUNT };
  ForeachCell::CellArray_global_ghosted U = foreach_cell.allocate_ghosted_array("U", VarIndex_U::COUNT);
  std::cout << "Fill U..." << std::endl;
  {
    foreach_cell.foreach_cell( "Fill_U", U, 
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      auto pos = cells.getCellCenter(iCell);
      U.at( iCell, IU ) = pos[IX];
      U.at( iCell, IV ) = pos[IY];
      U.at( iCell, IW ) = pos[IZ];
    });
  }

  std::cout << "Apply stencil..." << std::endl;
  ForeachCell::CellArray_global_ghosted U2 = foreach_cell.allocate_ghosted_array("U2", VarIndex_U::COUNT);
  foreach_cell.foreach_cell( "Stencil_U2", U, 
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
  {
    const auto& lmesh = cells.getLightOctree();
    ForeachCell::SearchMode_neighbor search_neighbor( lmesh, ForeachCell::SearchMode_neighbor::CLOSEST );

    auto append_offset = [&]( ForeachCell::CellIndex::offset_t offset )
    {
      ForeachCell::CellIndex iCell_n = iCell.getNeighbor( offset, search_neighbor );

      // Res will be center of same-size virtual neighbor
      Kokkos::Array<real_t, 3> res = {};
      if( iCell_n.is_boundary() )
      {
        // Compute neighbor position from local cell position
        auto pos = cells.getCellCenter(iCell);
        auto size = cells.getCellSize(iCell);
        res[IX] = pos[IX] + size[IX]*offset[IX];
        res[IY] = pos[IY] + size[IY]*offset[IY];
        res[IZ] = pos[IZ] + size[IZ]*offset[IZ];
      }
      else
      {
        if( iCell_n.level_diff() == 0 )
        {
          // Just get same-size neighbor position
          auto pos = cells.getCellCenter(iCell_n);
          res = pos;
        }
        else if( iCell_n.level_diff() == -1 )
        {
          // Mean of positions, shifted to center of bigger cell
          int n_smaller_neighbors = 
            foreach_smaller_neighbor_scattered( ndim, iCell_n, offset, lmesh,
            [&]( const ForeachCell::CellIndex& iCell_sn )
          {
            auto pos = cells.getCellCenter(iCell_sn);
            auto size = cells.getCellSize(iCell_sn);
            res[IX] += pos[IX] + offset[IX]*size[IX]*0.5;
            res[IY] += pos[IY] + offset[IY]*size[IY]*0.5;
            res[IZ] += pos[IZ] + offset[IZ]*size[IZ]*0.5;
          });

          res[IX] /= n_smaller_neighbors;
          res[IY] /= n_smaller_neighbors;
          res[IZ] /= n_smaller_neighbors;
        }
        else if( iCell_n.level_diff() == 1 )
        {
          auto pos_c = cells.getCellCenter(iCell);
          auto pos_n = cells.getCellCenter(iCell_n);
          auto size_n = cells.getCellSize(iCell_n);
          // Select the same-size virtual neighbor among biger cell's sectors
          Kokkos::Array<real_t,3> offset_quadrant;
          offset_quadrant[IX] = pos_n[IX] > pos_c[IX] ? -1. : +1.;
          offset_quadrant[IY] = pos_n[IY] > pos_c[IY] ? -1. : +1.;
          if( ndim == 3 )
            offset_quadrant[IZ] = pos_n[IZ] > pos_c[IZ] ? -1. : +1.;

          res[IX] = pos_n[IX] + offset_quadrant[IX] * size_n[IX] / 4;            
          res[IY] = pos_n[IY] + offset_quadrant[IY] * size_n[IY] / 4;            
          res[IZ] = pos_n[IZ] + offset_quadrant[IZ] * size_n[IZ] / 4;            
        }
      }        

      int nneighbors = (ndim==2)?4:6;
      U2.at( iCell, IU ) += res[IX]/nneighbors;
      U2.at( iCell, IV ) += res[IY]/nneighbors;
      U2.at( iCell, IW ) += res[IZ]/nneighbors;
    };

    append_offset({-1,+0,+0});
    append_offset({+1,+0,+0});
    append_offset({+0,-1,+0});
    append_offset({+0,+1,+0});
    if( ndim == 3 )
    {
      append_offset({+0,+0,-1});
      append_offset({+0,+0,+1});
    }
  });

  std::cout << "Check U2..." << std::endl;
  auto Udevice = U.subview_U();
  auto U2device = U.subview_U();
  auto Uhost = Kokkos::create_mirror_view(Udevice);
  auto U2host = Kokkos::create_mirror_view(U2device);
  Kokkos::deep_copy( Uhost, Udevice );
  Kokkos::deep_copy( U2host, U2device );
  
  for( uint32_t iOct=0; iOct < amr_mesh->getNumOctants(); iOct++ )
    for( uint32_t i=0; i<bx*by*bz; i++ )
    {
      EXPECT_DOUBLE_EQ( Uhost(i, iOct, 0 ), U2host(i, iOct, 0 ) );
      EXPECT_DOUBLE_EQ( Uhost(i, iOct, 1 ), U2host(i, iOct, 1 ) );
      EXPECT_DOUBLE_EQ( Uhost(i, iOct, 2 ), U2host(i, iOct, 2 ) );
    }

} // run_test



} // namespace dyablo

TEST(Test_ForeachCell, foreach_cell_mean_position_matches_center)
{
  dyablo::run_test();
}