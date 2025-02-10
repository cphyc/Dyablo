/**
 * \author M.-A. Breton, based on A. Durocher's implementation for Conjugate Gradient
 * Tests Multigrid gravity solver
 *
 * This file tests the self-gravity solver.
 * An AMR mesh is created and filled with a Herquist density profile
 * The self-gravity solver is called to compute the gravity potential associated
 * to the density profile. Then the resulting gravity potential is compared* to
 * the analytical Hernquist gravity potential.
 *
 * This test generates paraview output files.
 * In test_GravitySolver_iter0000001.xmf
 *  - iphy : the computed gravity potential
 *  - igx : numerical gravity potential shifted to fit* the analytical potential
 *  - igy : the analytical Hernquist gravity potential ("periodic" with 2 repetitions)
 *  - igz : |igx-igy|/|igy| ()
 *
 * Note : numerical values are only compared to analytical values at the center
 * of the domain. This is because the analytical solution is not accurate on
 * borders because of boundary conditions.
 */

#include "gtest/gtest.h"
#include "amr/AMRmesh.h"
#include "amr/CellIndexRemapper.h"
#include "amr/MapUserData.h"
#include "mpi/GhostCommunicator.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "gravity/GravitySolver_cg.h"
#include "io/IOManager.h"
#include <mpi.h>

namespace dyablo {
  namespace constants {
    constexpr double Pi = 3.14;
  }

  //https://arxiv.org/pdf/1712.07070.pdf , Appendix A
  struct Hernquist {
    constexpr static real_t G = 1 / (4 * dyablo::constants::Pi);
    constexpr static real_t rho0 = 1;
    constexpr static real_t r0 = 0.01;
    constexpr static real_t L = 1;

    constexpr static real_t M = 2 * dyablo::constants::Pi * r0 * r0 * r0 * rho0;

    KOKKOS_INLINE_FUNCTION
      static real_t rho(real_t r2)
    {
      real_t r = std::sqrt(r2) * L;
      return rho0 / (r / r0 * std::pow(1 + r / r0, 3));
    }

    KOKKOS_INLINE_FUNCTION
      static real_t phi(real_t r2)
    {
      real_t r = std::sqrt(r2) * L;
      return -G * M / (r + r0);
    }

    KOKKOS_INLINE_FUNCTION
      static real_t phi_periodic(real_t x, real_t y, real_t z)
    {
      constexpr int dmax = 2;
      real_t res = 0;
      for (int ix = -dmax; ix <= +dmax; ix++)
        for (int iy = -dmax; iy <= +dmax; iy++)
          for (int iz = -dmax; iz <= +dmax; iz++)
          {
            real_t r2 = (x + ix) * (x + ix) + (y + iy) * (y + iy) + (z + iz) * (z + iz);
            res += phi(r2);
          }
      return res;
    }

    // density trigger for refinement
    static real_t rhomax(int level)
    {
      int GAMER_level = level + 2 - 6; // for 4³ blocks - level_min = 6
      return 1E-3 * std::pow(4, GAMER_level);
    }
  };

  real_t MPI_Allreduce_scalar(real_t local_v)
  {
    real_t res;
    MPI_Allreduce(&local_v, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return res;
  }

  std::shared_ptr<AMRmesh> mesh_amrgrid_semiperiodic_sphere()
  {
    int level_min = 4;
    int level_max = level_min + 6; // + 6;
    uint32_t bx = 4, by = 4, bz = 4;


    std::cout << "// =========================================\n";
    std::cout << "// Testing GravitySolver_multigrid...\n";
    std::cout << "// Grid : amr - blocks " << bx << " -  levels " << level_min << " -> " << level_max << " \n";
    std::cout << "// Boundary conditions : (periodic, periodic, periodic) \n";
    std::cout << "// =========================================\n";

    std::cout << "Create mesh..." << std::endl;
    std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
    {
      int ndim = 3;

      amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool, 3>{true, true, true}, level_min, level_max);

      for (int level = level_min + 1; level < level_max; level++)
      {
        for (uint32_t iOct = 0; iOct < amr_mesh->getNumOctants(); iOct++)
        {
          auto oct_pos = amr_mesh->getCoordinates(iOct);
          real_t oct_size = amr_mesh->getSize(iOct)[0];

          for (uint32_t c = 0; c < bx * by * bz; c++)
          {
            uint32_t cz = c / (bx * by);
            uint32_t cy = (c - cz * bx * by) / bx;
            uint32_t cx = c - cz * bx * by - cy * bx;

            real_t x = oct_pos[IX] + (cx + 0.5) * oct_size / bx - 0.5;
            real_t y = oct_pos[IY] + (cy + 0.5) * oct_size / by - 0.5;
            real_t z = oct_pos[IZ] + (cz + 0.5) * oct_size / bz - 0.5;

            real_t r2 = x * x + y * y + z * z;

            if (Hernquist::rho(r2) > Hernquist::rhomax(level))
            {
              amr_mesh->setMarker(iOct, 1);
              break;
            }
          }
        }
        amr_mesh->adapt();
      }

      uint8_t levels = 6;
      amr_mesh->loadBalance(levels);
    }

    std::cout << "End Mesh creation." << std::endl;

    return amr_mesh;
  }



  /// Tests convergence with 
  void test_GravitySolver(std::shared_ptr<AMRmesh> amr_mesh)
  {
    // Content of .ini file used ton configure configmap and HydroParams
    std::string configmap_str =
      "[run]\n"
      "solver_name=Hydro_Muscl_Block_3D \n"
      "[output]\n"
      "outputPrefix=test_GravitySolver_multigrid5\n"
      "write_variables=rho,gphi,gx,gy,gz,res,solution,rhs\n"
      "[amr]\n"
      "use_block_data=yes\n"
      "bx=4\n"
      "by=4\n"
      "bz=4\n"
      "[mesh]\n"
      "ndim=3\n"
      "boundary_type_xmin=periodic\n"
      "boundary_type_xmax=periodic\n"
      "boundary_type_ymin=periodic\n"
      "boundary_type_ymax=periodic\n"
      "boundary_type_zmin=periodic\n"
      "boundary_type_zmax=periodic\n"
      "[gravity]\n"
      "gravity_type=field\n"
      "G=1\n"
      "\n";
    ConfigMap configMap(configmap_str);
    ForeachCell foreach_cell(*amr_mesh, configMap);

    Kokkos::Array<BoundaryConditionType, 3> boundarycondition = {
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING)
    };

    std::cout << "Initialize User Data..." << std::endl;

    enum VarIndex_gravity {
      Irho,
      Igx,
      Igy,
      Igz,
      Iphi,
      Isolution,
      Irhs,
      Iresidual,
    };

    UserData U_(configMap, foreach_cell);
    U_.new_fields({ "rho", "gx", "gy", "gz", "gphi", "solution", "rhs",  "res" });
    U_.new_intermediate_fields( {"rho","gphi", "solution", "rhs", "res"} );


    UserData::FieldAccessor U = U_.getAccessor({
      {"rho", Irho},
      {"gphi", Iphi},
      {"gx", Igx},
      {"gy", Igy},
      {"gz", Igz},
      {"solution", Isolution},
      {"rhs", Irhs},
      { "res", Iresidual },
      });
    
    UserData::FieldAccessor Uintermediate = U_.getAccessor_intermediate( {
      {"rho", Irho},
      {"gphi", Iphi},
      {"solution", Isolution},
      {"rhs", Irhs},
      { "res", Iresidual },
      });


    {
      // Initialize U
      auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_cell("Init", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        auto pos = cells.getCellCenter(iCell);
        real_t x = pos[IX] - 0.5;
        real_t y = pos[IY] - 0.5;
        real_t z = pos[IZ] - 0.5;

        real_t r2 = x * x + y * y + z * z;

        U.at(iCell, Irho) = Hernquist::rho(r2);
      });

      GhostCommunicator ghost_comm(*amr_mesh, U.getShape(), 1);
      ghost_comm.exchange_ghosts(U);
    }

    Timers timers;
    ScalarSimulationData scalar_data;
    std::unique_ptr<IOManager> iomanager = IOManagerFactory::make_instance(
      "IOManager_hdf5",
      configMap,
      foreach_cell,
      timers
    );
    int iter = 0;
    int time = 0;

    auto cells = foreach_cell.getCellMetaData();
    real_t ndim = configMap.getValue<real_t>("mesh", "ndim", 0);
    real_t xmin = configMap.getValue<real_t>("mesh", "xmin", 0);
    real_t ymin = configMap.getValue<real_t>("mesh", "ymin", 0);
    real_t zmin = configMap.getValue<real_t>("mesh", "zmin", 0);
    real_t xmax = configMap.getValue<real_t>("mesh", "xmax", 0);
    real_t ymax = configMap.getValue<real_t>("mesh", "ymax", 0);
    real_t zmax = configMap.getValue<real_t>("mesh", "zmax", 0);

    // Min/max AMR level, and intermediate per level
    const LightOctree& lmesh = U.getShape().lmesh;
    const uint32_t numOctants = lmesh.getNumOctants();
    uint32_t max_level_in_amr = 0;
    Kokkos::parallel_for( "Get max level in AMR", 
      Kokkos::RangePolicy<>(0,numOctants), 
      [=, &max_level_in_amr]( const uint32_t iOct )
    {
      Kokkos::atomic_fetch_max( &max_level_in_amr, lmesh.getLevel({iOct, false}) );
    });

    uint32_t min_level_in_amr = 100;
    Kokkos::parallel_for( "Get min level in AMR", 
      Kokkos::RangePolicy<>(0,numOctants), 
      [=, &min_level_in_amr]( const uint32_t iOct )
    {
      Kokkos::atomic_fetch_min( &min_level_in_amr, lmesh.getLevel({iOct, false}) );
    });

    const uint32_t min_level_multigrid = 0;
    const uint32_t max_level = lmesh.get_level_max();
    const uint32_t level_coarse = lmesh.get_level_min();
    printf("min_level AMR %d, max_level AMR = %d\n", min_level_in_amr, max_level_in_amr);
    printf("coarse_level ICs %d, max_level ICs = %d\n", level_coarse, max_level);
    printf("numOcts %d, numGhosts = %d, numTotal = %u\n", lmesh.getNumOctants(), lmesh.getNumGhosts(), lmesh.getNumOctants()+lmesh.getNumGhosts());

    const uint32_t nlevel = max_level_in_amr - min_level_multigrid + 1;
    Kokkos::View<int*> octs_per_level("octs_per_level", nlevel);
    Kokkos::View<int*> octs_intermediate_per_level("octs_intermediate_per_level", nlevel);
    // Count number of octs in AMR per level
    Kokkos::parallel_for( "Count number of octs per level", 
    Kokkos::RangePolicy<>(0,numOctants), 
      KOKKOS_LAMBDA( const uint32_t iOct )
      {
        const uint32_t level = lmesh.getLevel({iOct, false});
        Kokkos::atomic_fetch_add( &octs_per_level(level-min_level_multigrid), 1 );
      }
    );
    // Count number of intermediate octs per level      
    Kokkos::parallel_for( "Count number of intermediate octs per level", 
    Kokkos::RangePolicy<>(0,numOctants), 
      KOKKOS_LAMBDA( const uint32_t ioct_local )
      {
        const LightOctree_base::OctantIndex iOct = {ioct_local, false};
        uint32_t level = lmesh.getLevel(iOct);
        auto logical_coords = lmesh.getStorage().get_logical_coords(iOct);
        while (logical_coords[IX] % 2 == 0 && logical_coords[IY] % 2 == 0 && logical_coords[IZ] % 2 == 0 && level > min_level_multigrid) {
          level--;
          logical_coords[IX] /= 2;
          logical_coords[IY] /= 2;
          logical_coords[IZ] /= 2;
          Kokkos::atomic_fetch_add( &octs_intermediate_per_level(level-min_level_multigrid), 1 );
        } 
      }
    );

    for(uint32_t ilevel = min_level_multigrid; ilevel <= max_level_in_amr; ilevel++) 
      printf("Finished Level %d, octs %u intermediate %u\n", ilevel, octs_per_level(ilevel-min_level_multigrid), octs_intermediate_per_level(ilevel-min_level_multigrid));



    // MULTIGRID 

    auto average_4bigger_neighbors = [&](const ForeachCell::CellIndex iCell, ForeachCell::CellIndex::offset_t offset) {
      real_t result = 0;
      uint32_t counter = 0;
      const int8_t offset_x = (!abs(offset[0])); 
      const int8_t offset_y = (!abs(offset[1])); 
      const int8_t offset_z = (!abs(offset[2]));
      const Kokkos::Array<real_t, 4> tmp = {9./16, 3./16, 3./16, 1./16};
      for (int8_t i = 0; i <= offset_x; i++)
      for (int8_t j = 0; j <= offset_y; j++)
      for (int8_t k = 0; k <= offset_z; k++)
      {
        const int8_t shift_x = offset[IX] + i * (2*(iCell.i % 2) - 1);
        const int8_t shift_y = offset[IY] + j * (2*(iCell.j % 2) - 1);
        const int8_t shift_z = offset[IZ] + k * (2*(iCell.k % 2) - 1);
        ForeachCell::CellIndex::offset_t shift = {shift_x, shift_y, shift_z};
        ForeachCell::CellIndex iCell_coarse = iCell.getNeighbor_ghost(shift, U.getShape());
        if (iCell_coarse.status == ForeachCell::CellIndex::BIGGER ) result += tmp[counter]*U.at(iCell_coarse, Iphi);
        else {
          const ForeachCell::CellIndex iCell_parent = iCell_coarse.getParent(Uintermediate.getShape());
          result += tmp[counter]*Uintermediate.at(iCell_parent, Iphi);
        }
        counter++;
      }
      return result; 
    };

    auto average_8bigger_neighbors = [&](const ForeachCell::CellIndex iCell, ForeachCell::CellIndex::offset_t offset) {
      real_t result = 0;
      uint32_t counter = 0;
      const int8_t offset_x = (!abs(offset[0])); 
      const int8_t offset_y = (!abs(offset[1])); 
      const int8_t offset_z = (!abs(offset[2]));
      Kokkos::Array<real_t, 4> tmp = {9./32, 3./32, 3./32, 1./32};
      for (int8_t i = 0; i <= offset_x; i++)
      for (int8_t j = 0; j <= offset_y; j++)
      for (int8_t k = 0; k <= offset_z; k++){
        for (int8_t l = 0; l <= 1; l++)
        {
          const int8_t shift_x = l*offset[IX] + 2. * i * (2*(iCell.i % 2) - 1);
          const int8_t shift_y = l*offset[IY] + 2. * j * (2*(iCell.j % 2) - 1);
          const int8_t shift_z = l*offset[IZ] + 2. * k * (2*(iCell.k % 2) - 1);
          const ForeachCell::CellIndex::offset_t shift = {shift_x, shift_y, shift_z};
          const ForeachCell::CellIndex iCell_coarse = iCell.getNeighbor_ghost(shift, U.getShape());
          if (iCell_coarse.status == ForeachCell::CellIndex::BIGGER ) {
            result += tmp[counter]*U.at(iCell_coarse, Iphi);
            auto lc = lmesh.get_logical_coords(iCell_coarse.iOct);
            printf(" %d Leaf-------- %d %d %d ijk %d %d %d\n", counter, lc[IX], lc[IY], lc[IZ], iCell_coarse.i, iCell_coarse.j, iCell_coarse.k);
            printf("counter %d tmp %f U.at %.5e\n", counter, tmp[counter], U.at(iCell_coarse, Iphi));
          }
          else {
            const ForeachCell::CellIndex iCell_parent = iCell_coarse.getParent(Uintermediate.getShape());
            auto lc = lmesh.get_logical_coords(iCell_parent.iOct);
            printf(" %d Intermediate %d %d %d ijk %d %d %d\n", counter, lc[IX], lc[IY], lc[IZ], iCell_parent.i, iCell_parent.j, iCell_parent.k);
            result += tmp[counter]*Uintermediate.at(iCell_parent, Iphi);
            printf("counter %d tmp %f U.at %.5e\n", counter, tmp[counter], Uintermediate.at(iCell_parent, Iphi));
          }
        }
        counter++;
      }
      return result; 
    };

    auto residual_norm_sqr = [&](const uint32_t level){
      real_t residual_sqr_leaves = 0;
      real_t residual_sqr_intermediate = 0;
      foreach_cell.reduce_cell("Compute residual norm", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell, real_t & update_residual_sqr)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          real_t residual_tmp = U.at(iCell, Iresidual);
          update_residual_sqr += residual_tmp * residual_tmp;
        }
      }, Kokkos::Sum<real_t>(residual_sqr_leaves));
      foreach_cell.reduce_intermediate_cell("Compute residual norm", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell, real_t & update_residual_sqr)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          real_t residual_tmp = Uintermediate.at(iCell, Iresidual);
          update_residual_sqr += residual_tmp * residual_tmp;
        }
      }, Kokkos::Sum<real_t>(residual_sqr_intermediate));
      // printf("Level %d Residual**2, leaves %.3e intermediate %.3e\n", level, residual_sqr_leaves, residual_sqr_intermediate);
      return residual_sqr_leaves + residual_sqr_intermediate;
    };

    auto initialise_lhs = [&](const uint32_t level){
      foreach_cell.foreach_cell("Write potential", U.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        uint32_t current_level = cells.getLevel(iCell);
        if (level == current_level){
          const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
          U.at(iCell, Isolution) = -U.at(iCell, Irhs) / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ]));
        }
      });
      foreach_cell.foreach_intermediate_cell("Write potential", Uintermediate.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        uint32_t current_level = cells.getLevel(iCell);
        if (level == current_level){
          const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
          Uintermediate.at(iCell, Isolution) = -Uintermediate.at(iCell, Irhs) / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ])) ;
        }
      });
    };

    // Coarse level and coarser
    auto residual_intermediate_uniform = [&](const uint32_t level) {
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_intermediate_cell("Residual intermediate", Uintermediate.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
          Kokkos::Array<real_t, 3> contrib_L, contrib_R;
          real_t neighbors = 0;
          for ( ComponentIndex3D dir : {IX,IY,IZ} )
          {
            ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
            ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
            ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
            DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
            ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
            DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
            if ( iCell_L.status == ForeachCell::CellIndex::BIGGER ) {
              iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
              contrib_L[dir] = U.at(iCell_L, Isolution);
            } else contrib_L[dir] = Uintermediate.at(iCell_L, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L[dir] = 0;
            if ( iCell_R.status == ForeachCell::CellIndex::BIGGER ) {
              iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
              contrib_R[dir] = U.at(iCell_R, Isolution);
            } else contrib_R[dir] = Uintermediate.at(iCell_R, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R[dir] = 0;
            neighbors += (contrib_L[dir] + contrib_R[dir]) / (size[dir] * size[dir]);
          }
          const real_t laplacian_solution = neighbors - Uintermediate.at(iCell, Isolution) * ( 2./(size[IX]*size[IX]) + 2./(size[IY]*size[IY]) + 2./(size[IZ]*size[IZ]) );
          Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
        }
      });
    };
    auto residual_uniform = [&](const uint32_t level) {
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_cell("Residual", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
          Kokkos::Array<real_t, 3> contrib_L, contrib_R;
          real_t neighbors = 0;
          for ( ComponentIndex3D dir : {IX,IY,IZ} )
          {
            ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
            ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
            ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
            DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() <= 0, "Leaf cell at coarse level cannot have coarser leaf neighbor" );
            ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
            DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() <= 0, "Leaf cell at coarse level cannot have coarser leaf neighbor" );
            if ( iCell_L.status == ForeachCell::CellIndex::SMALLER ) {
              iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
              contrib_L[dir] = Uintermediate.at(iCell_L, Isolution);
            } else contrib_L[dir] = U.at(iCell_L, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L[dir] = 0;
            if ( iCell_R.status == ForeachCell::CellIndex::SMALLER ) {
              iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
              contrib_R[dir] = Uintermediate.at(iCell_R, Isolution);
            } else contrib_R[dir] = U.at(iCell_R, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R[dir] = 0;
            neighbors += (contrib_L[dir] + contrib_R[dir]) / (size[dir] * size[dir]);
          }
          const real_t laplacian_solution = neighbors - U.at(iCell, Isolution) * ( 2./(size[IX]*size[IX]) + 2./(size[IY]*size[IY]) + 2./(size[IZ]*size[IZ]) );
          U.at(iCell, Iresidual) = U.at(iCell, Irhs) - laplacian_solution;
        }
      });
      foreach_cell.foreach_intermediate_cell("Residual intermediate", Uintermediate.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
          Kokkos::Array<real_t, 3> contrib_L, contrib_R;
          real_t neighbors = 0;
          for ( ComponentIndex3D dir : {IX,IY,IZ} )
          {
            ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
            ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
            ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
            DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
            ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
            DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
            if ( iCell_L.status == ForeachCell::CellIndex::BIGGER ) {
              iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
              contrib_L[dir] = U.at(iCell_L, Isolution);
            } else contrib_L[dir] = Uintermediate.at(iCell_L, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L[dir] = 0;
            if ( iCell_R.status == ForeachCell::CellIndex::BIGGER ) {
              iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
              contrib_R[dir] = U.at(iCell_R, Isolution);
            } else contrib_R[dir] = Uintermediate.at(iCell_R, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R[dir] = 0;
            neighbors += (contrib_L[dir] + contrib_R[dir]) / (size[dir] * size[dir]);
          }
          const real_t laplacian_solution = neighbors - Uintermediate.at(iCell, Isolution) * ( 2./(size[IX]*size[IX]) + 2./(size[IY]*size[IY]) + 2./(size[IZ]*size[IZ]) );
          Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
        }
      });
    };

    auto residual_leaves_amr_correction = [&](const uint32_t level) {
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_cell("Residual leaves", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
          Kokkos::Array<real_t, 3> contrib_L, contrib_R;
          real_t laplacian_solution = 0;
          //real_t neighbors = 0;
          const real_t central_solution = U.at(iCell, Isolution);
          for ( ComponentIndex3D dir : {IX,IY,IZ} )
          {
            ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
            ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
            ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
            ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
            real_t f_L(1), f_R(1), f_C(2);
            if ( iCell_L.status == ForeachCell::CellIndex::SMALLER ) { 
              iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
              contrib_L[dir] = Uintermediate.at(iCell_L, Isolution);
            } else if (iCell_L.status == ForeachCell::CellIndex::BIGGER) {
                contrib_L[dir] = 0; // Zero BC for correction terms
                f_L = 8./3.;
                f_R = 4./3.;
                f_C = 4.;
            } else contrib_L[dir] = U.at(iCell_L, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L[dir] = 0;
            if ( iCell_R.status == ForeachCell::CellIndex::SMALLER ) { 
              iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
              contrib_R[dir] = Uintermediate.at(iCell_R, Isolution);
            } else if (iCell_R.status == ForeachCell::CellIndex::BIGGER) {
                contrib_R[dir] = 0; // Zero BC for correction terms
                f_R = 8./3.;
                f_L = 4./3.;
                f_C = 4.;
            } else contrib_R[dir] = U.at(iCell_R, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R[dir] = 0;
            laplacian_solution += (f_L*contrib_L[dir] + f_R*contrib_R[dir] - f_C*central_solution) / (size[dir] * size[dir]);

          }
          U.at(iCell, Iresidual) = U.at(iCell, Irhs) - laplacian_solution;
        }
      });
    };

    auto residual_intermediate_amr_correction = [&](const uint32_t level) {
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_intermediate_cell("Residual intermediate", Uintermediate.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
          Kokkos::Array<real_t, 3> contrib_L, contrib_R;
          real_t laplacian_solution = 0;
          const real_t central_solution = Uintermediate.at(iCell, Isolution);
          for ( ComponentIndex3D dir : {IX,IY,IZ} )
          {
            real_t f_L(1), f_R(1), f_C(2);
            ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
            ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
            ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
            DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
            ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
            DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
            if ( iCell_L.status == ForeachCell::CellIndex::BIGGER ) {
              contrib_L[dir] = 0; // Zero BC for correction terms
              f_L = 8./3.;
              f_R = 4./3.;
              f_C = 4.;
            } else contrib_L[dir] = Uintermediate.at(iCell_L, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L[dir] = 0;
            if ( iCell_R.status == ForeachCell::CellIndex::BIGGER ) {
              contrib_R[dir] = 0; // Zero BC for correction terms
              f_R = 8./3.;
              f_L = 4./3.;
              f_C = 4.;
            } else contrib_R[dir] = Uintermediate.at(iCell_R, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R[dir] = 0;
            laplacian_solution += (f_L*contrib_L[dir] + f_R*contrib_R[dir] - f_C*central_solution) / (size[dir] * size[dir]);
          }
          Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
        }
      });
    };

    auto residual_amr_finest = [&](const uint32_t level) {
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_cell("Residual leaves", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
          Kokkos::Array<real_t, 3> contrib_L, contrib_R;
          real_t laplacian_solution = 0;
          const real_t central_solution = U.at(iCell, Isolution);
          for ( ComponentIndex3D dir : {IX,IY,IZ} )
          {
            ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
            ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
            ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
            ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
            real_t f_L(1), f_R(1), f_C(2);
            if ( iCell_L.status == ForeachCell::CellIndex::SMALLER ) { 
              iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
              contrib_L[dir] = Uintermediate.at(iCell_L, Isolution);
            } else if (iCell_L.status == ForeachCell::CellIndex::BIGGER) {
                //contrib_L[dir] = U.at(iCell_L, Iphi); // TODO: find better interpolation from coarser level?
                contrib_L[dir] = average_4bigger_neighbors(iCell, off_L);
                f_L = 8./15.;
                f_R = 4./5.;
                f_C = 4./3.;
                /* contrib_L[dir] = average_8bigger_neighbors(iCell, off_L);
                f_L = 8./3.;
                f_R = 4./3.;
                f_C = 4.; */
            } else contrib_L[dir] = U.at(iCell_L, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L[dir] = 0;
            if ( iCell_R.status == ForeachCell::CellIndex::SMALLER ) { 
              iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
              contrib_R[dir] = Uintermediate.at(iCell_R, Isolution);
            } else if (iCell_R.status == ForeachCell::CellIndex::BIGGER) {
                //contrib_R[dir] = U.at(iCell_R, Iphi); // TODO: find better interpolation from coarser level?
                contrib_R[dir] = average_4bigger_neighbors(iCell, off_R);
                f_R = 8./15.;
                f_L = 4./5.;
                f_C = 4./3.;
                /* contrib_R[dir] = average_8bigger_neighbors(iCell, off_R);
                f_R = 8./3.;
                f_L = 4./3.;
                f_C = 4.; */
            } else contrib_R[dir] = U.at(iCell_R, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R[dir] = 0;
            laplacian_solution += (f_L*contrib_L[dir] + f_R*contrib_R[dir] - f_C*central_solution) / (size[dir] * size[dir]);
          }
          U.at(iCell, Iresidual) = U.at(iCell, Irhs) - laplacian_solution;
        }
      });
      foreach_cell.foreach_intermediate_cell("Residual intermediate", Uintermediate.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
          Kokkos::Array<real_t, 3> contrib_L, contrib_R;
          real_t neighbors = 0;
          for ( ComponentIndex3D dir : {IX,IY,IZ} )
          {
            ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
            ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
            ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
            DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
            ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
            DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
            if ( iCell_L.status == ForeachCell::CellIndex::BIGGER ) {
              iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
              contrib_L[dir] = U.at(iCell_L, Isolution);
            } else contrib_L[dir] = Uintermediate.at(iCell_L, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L[dir] = 0;
            if ( iCell_R.status == ForeachCell::CellIndex::BIGGER ) {
              iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
              contrib_R[dir] = U.at(iCell_R, Isolution);
            } else contrib_R[dir] = Uintermediate.at(iCell_R, Isolution);
            if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R[dir] = 0;
            neighbors += (contrib_L[dir] + contrib_R[dir]) / (size[dir] * size[dir]);
          }
          const real_t laplacian_solution = neighbors - Uintermediate.at(iCell, Isolution) * ( 2./(size[IX]*size[IX]) + 2./(size[IY]*size[IY]) + 2./(size[IZ]*size[IZ]) );
          Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
        }
      });
    };

    auto restriction = [&](const uint32_t level) {
      foreach_cell.foreach_intermediate_cell( "Restrict on intermediate", Uintermediate.getShape(),
      KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
      {
        const uint32_t current_level = cells.getLevel(iCell);
        if( current_level == level )
        {
          ForeachCell::CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
          real_t residual = 0;
          int ns = foreach_sibling( ndim, iCell_c0, Uintermediate.getShape(),
            [&]( const ForeachCell::CellIndex& iCell_c )
          {
            if ( iCell_c.iOct.isIntermediate )  residual += Uintermediate.at(iCell_c, Iresidual);
            else  residual += U.at(iCell_c, Iresidual);
            
          });
          Uintermediate.at( iCell, Irhs ) = residual/ns;
        }
      }); 
    };

    auto prolongation0_inject = [&](const uint32_t level) {
      foreach_cell.foreach_intermediate_cell( "Prolongation", Uintermediate.getShape(),
      KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
      {
        const uint32_t current_level = cells.getLevel(iCell);
        if( current_level == level - 1 )
        {
          ForeachCell::CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
          const real_t solution = Uintermediate.at( iCell, Isolution );
          //const real_t rhs = Uintermediate.at( iCell, Irhs );
          foreach_sibling( ndim, iCell_c0, Uintermediate.getShape(),
            [&]( const ForeachCell::CellIndex& iCell_c )
          {
            if( iCell_c.iOct.isIntermediate )
            {
              Uintermediate.at(iCell_c, Isolution) = solution;
              //Uintermediate.at(iCell_c, Irhs) = rhs;
            }
            else
            {
              U.at(iCell_c, Isolution) = solution;
              //U.at(iCell_c, Irhs) = rhs;
            }
          });
        }
      }); 
    };

    auto prolongation0 = [&](const uint32_t level) {
      foreach_cell.foreach_intermediate_cell( "Prolongation", Uintermediate.getShape(),
      KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
      {
        const uint32_t current_level = cells.getLevel(iCell);
        if( current_level == level - 1 )
        {
          const real_t correction = Uintermediate.at( iCell, Isolution );
          const ForeachCell::CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
          foreach_sibling( ndim, iCell_c0, Uintermediate.getShape(),
            [&]( const ForeachCell::CellIndex& iCell_c )
          {
            if( iCell_c.iOct.isIntermediate ) Uintermediate.at(iCell_c, Isolution) += correction;
            else  U.at(iCell_c, Isolution) += correction;
          });
        }
      }); 
    };

    auto get_neighbor_value = [&](const ForeachCell::CellIndex iCell, const ForeachCell::CellIndex::offset_t offset){
      ForeachCell::CellIndex iCell_n = iCell.getNeighbor_ghost_intermediate(offset, Uintermediate.getShape());
      if (iCell_n.level_diff() != 0){
        iCell_n = iCell.getNeighbor_ghost(offset, U.getShape());
        return U.at( iCell_n, Isolution );
      } else return Uintermediate.at( iCell_n, Isolution );
    };

    auto prolongation = [&](const uint32_t level) {
      const real_t f0 = 27.0 / 64;
      const real_t f1 = 9.0 / 64;
      const real_t f2 = 3.0 / 64;
      const real_t f3 = 1.0 / 64;
      foreach_cell.foreach_intermediate_cell( "Prolongation", Uintermediate.getShape(),
      KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
      {
        const uint32_t current_level = cells.getLevel(iCell);
        if( current_level == level - 1 )
        {
          // Get coarse cell values to interpolate from
          const real_t tmp111 = Uintermediate.at( iCell, Isolution );
          const real_t tmp0 = f0 * tmp111;

          const real_t tmp000 = get_neighbor_value(iCell, {-1, -1, -1});
          const real_t tmp001 = get_neighbor_value(iCell, {-1, -1, 0});
          const real_t tmp002 = get_neighbor_value(iCell, {-1, -1, 1});
          const real_t tmp010 = get_neighbor_value(iCell, {-1, 0, -1});
          const real_t tmp011 = get_neighbor_value(iCell, {-1, 0, 0});
          const real_t tmp012 = get_neighbor_value(iCell, {-1, 0, 1});
          const real_t tmp020 = get_neighbor_value(iCell, {-1, 1, -1});
          const real_t tmp021 = get_neighbor_value(iCell, {-1, 1, 0});
          const real_t tmp022 = get_neighbor_value(iCell, {-1, 1, 1});
          const real_t tmp100 = get_neighbor_value(iCell, {0, -1, -1});
          const real_t tmp101 = get_neighbor_value(iCell, {0, -1, 0});
          const real_t tmp102 = get_neighbor_value(iCell, {0, -1, 1});
          const real_t tmp110 = get_neighbor_value(iCell, {0, 0, -1});
          const real_t tmp112 = get_neighbor_value(iCell, {0, 0, 1});
          const real_t tmp120 = get_neighbor_value(iCell, {0, 1, -1});
          const real_t tmp121 = get_neighbor_value(iCell, {0, 1, 0});
          const real_t tmp122 = get_neighbor_value(iCell, {0, 1, 1});
          const real_t tmp200 = get_neighbor_value(iCell, {1, -1, -1});
          const real_t tmp201 = get_neighbor_value(iCell, {1, -1, 0});
          const real_t tmp202 = get_neighbor_value(iCell, {1, -1, 1});
          const real_t tmp210 = get_neighbor_value(iCell, {1, 0, -1});
          const real_t tmp211 = get_neighbor_value(iCell, {1, 0, 0});
          const real_t tmp212 = get_neighbor_value(iCell, {1, 0, 1});
          const real_t tmp220 = get_neighbor_value(iCell, {1, 1, -1});
          const real_t tmp221 = get_neighbor_value(iCell, {1, 1, 0});
          const real_t tmp222 = get_neighbor_value(iCell, {1, 1, 1});
          // Interpolate to children
          const ForeachCell::CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
          // foreach_sibling
          // CellIndex get_sibling_cell TODO: check if sibling cells are necessary leaf or intermediate, similar to iCell_c0
          if( iCell_c0.iOct.isIntermediate ){
            CellIndex iCell000 = iCell_c0.getNeighbor_ghost_intermediate({0, 0, 0}, Uintermediate.getShape());
            Uintermediate.at(iCell000, Isolution) += tmp0
                + f1 * (tmp011 + tmp101 + tmp110)
                + f2 * (tmp001 + tmp010 + tmp100)
                + f3 * tmp000;
            CellIndex iCell001 = iCell_c0.getNeighbor_ghost_intermediate({0, 0, 1}, Uintermediate.getShape());
            Uintermediate.at(iCell001, Isolution) += tmp0
                + f1 * (tmp011 + tmp101 + tmp112)
                + f2 * (tmp001 + tmp012 + tmp102)
                + f3 * tmp002;
            CellIndex iCell010 = iCell_c0.getNeighbor_ghost_intermediate({0, 1, 0}, Uintermediate.getShape());
            Uintermediate.at(iCell010, Isolution) += tmp0
                + f1 * (tmp011 + tmp121 + tmp110)
                + f2 * (tmp021 + tmp010 + tmp120)
                + f3 * tmp020;
            CellIndex iCell011 = iCell_c0.getNeighbor_ghost_intermediate({0, 1, 1}, Uintermediate.getShape());
            Uintermediate.at(iCell011, Isolution) += tmp0
                    + f1 * (tmp011 + tmp121 + tmp112)
                    + f2 * (tmp021 + tmp012 + tmp122)
                    + f3 * tmp022;
            CellIndex iCell100 = iCell_c0.getNeighbor_ghost_intermediate({1, 0, 0}, Uintermediate.getShape());
            Uintermediate.at(iCell100, Isolution) += tmp0
                    + f1 * (tmp211 + tmp101 + tmp110)
                    + f2 * (tmp201 + tmp210 + tmp100)
                    + f3 * tmp200;
            CellIndex iCell101 = iCell_c0.getNeighbor_ghost_intermediate({1, 0, 1}, Uintermediate.getShape());
            Uintermediate.at(iCell101, Isolution) += tmp0
                    + f1 * (tmp211 + tmp101 + tmp112)
                    + f2 * (tmp201 + tmp212 + tmp102)
                    + f3 * tmp202;
            CellIndex iCell110 = iCell_c0.getNeighbor_ghost_intermediate({1, 1, 0}, Uintermediate.getShape());
            Uintermediate.at(iCell110, Isolution) += tmp0
                    + f1 * (tmp211 + tmp121 + tmp110)
                    + f2 * (tmp221 + tmp210 + tmp120)
                    + f3 * tmp220;
            CellIndex iCell111 = iCell_c0.getNeighbor_ghost_intermediate({1, 1, 1}, Uintermediate.getShape());
            Uintermediate.at(iCell111, Isolution) += tmp0
                    + f1 * (tmp211 + tmp121 + tmp112)
                    + f2 * (tmp221 + tmp212 + tmp122)
                    + f3 * tmp222;
          }
          else{
            CellIndex iCell000 = iCell_c0.getNeighbor_ghost({0, 0, 0}, U.getShape());
            U.at(iCell000, Isolution) += tmp0
                + f1 * (tmp011 + tmp101 + tmp110)
                + f2 * (tmp001 + tmp010 + tmp100)
                + f3 * tmp000;
            CellIndex iCell001 = iCell_c0.getNeighbor_ghost({0, 0, 1}, U.getShape());
            U.at(iCell001, Isolution) += tmp0
                + f1 * (tmp011 + tmp101 + tmp112)
                + f2 * (tmp001 + tmp012 + tmp102)
                + f3 * tmp002;
            CellIndex iCell010 = iCell_c0.getNeighbor_ghost({0, 1, 0}, U.getShape());
            U.at(iCell010, Isolution) += tmp0
                + f1 * (tmp011 + tmp121 + tmp110)
                + f2 * (tmp021 + tmp010 + tmp120)
                + f3 * tmp020;
            CellIndex iCell011 = iCell_c0.getNeighbor_ghost({0, 1, 1}, U.getShape());
            U.at(iCell011, Isolution) += tmp0
                    + f1 * (tmp011 + tmp121 + tmp112)
                    + f2 * (tmp021 + tmp012 + tmp122)
                    + f3 * tmp022;
            CellIndex iCell100 = iCell_c0.getNeighbor_ghost({1, 0, 0}, U.getShape());
            U.at(iCell100, Isolution) += tmp0
                    + f1 * (tmp211 + tmp101 + tmp110)
                    + f2 * (tmp201 + tmp210 + tmp100)
                    + f3 * tmp200;
            CellIndex iCell101 = iCell_c0.getNeighbor_ghost({1, 0, 1}, U.getShape());
            U.at(iCell101, Isolution) += tmp0
                    + f1 * (tmp211 + tmp101 + tmp112)
                    + f2 * (tmp201 + tmp212 + tmp102)
                    + f3 * tmp202;
            CellIndex iCell110 = iCell_c0.getNeighbor_ghost({1, 1, 0}, U.getShape());
            U.at(iCell110, Isolution) += tmp0
                    + f1 * (tmp211 + tmp121 + tmp110)
                    + f2 * (tmp221 + tmp210 + tmp120)
                    + f3 * tmp220;
            CellIndex iCell111 = iCell_c0.getNeighbor_ghost({1, 1, 1}, U.getShape());
            U.at(iCell111, Isolution) += tmp0
                    + f1 * (tmp211 + tmp121 + tmp112)
                    + f2 * (tmp221 + tmp212 + tmp122)
                    + f3 * tmp222;
          }
        }
      }); 
    };

    auto prolongation_on_intermediate = [&](const uint32_t level) {
      const real_t f0 = 27.0 / 64;
      const real_t f1 = 9.0 / 64;
      const real_t f2 = 3.0 / 64;
      const real_t f3 = 1.0 / 64;
      foreach_cell.foreach_intermediate_cell( "Prolongation", Uintermediate.getShape(),
      KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
      {
        const uint32_t current_level = cells.getLevel(iCell);
        if( current_level == level - 1 )
        {
          // Get coarse cell values to interpolate from
          const real_t tmp111 = Uintermediate.at( iCell, Isolution );
          const real_t tmp0 = f0 * tmp111;

          const real_t tmp000 = get_neighbor_value(iCell, {-1, -1, -1});
          const real_t tmp001 = get_neighbor_value(iCell, {-1, -1, 0});
          const real_t tmp002 = get_neighbor_value(iCell, {-1, -1, 1});
          const real_t tmp010 = get_neighbor_value(iCell, {-1, 0, -1});
          const real_t tmp011 = get_neighbor_value(iCell, {-1, 0, 0});
          const real_t tmp012 = get_neighbor_value(iCell, {-1, 0, 1});
          const real_t tmp020 = get_neighbor_value(iCell, {-1, 1, -1});
          const real_t tmp021 = get_neighbor_value(iCell, {-1, 1, 0});
          const real_t tmp022 = get_neighbor_value(iCell, {-1, 1, 1});
          const real_t tmp100 = get_neighbor_value(iCell, {0, -1, -1});
          const real_t tmp101 = get_neighbor_value(iCell, {0, -1, 0});
          const real_t tmp102 = get_neighbor_value(iCell, {0, -1, 1});
          const real_t tmp110 = get_neighbor_value(iCell, {0, 0, -1});
          const real_t tmp112 = get_neighbor_value(iCell, {0, 0, 1});
          const real_t tmp120 = get_neighbor_value(iCell, {0, 1, -1});
          const real_t tmp121 = get_neighbor_value(iCell, {0, 1, 0});
          const real_t tmp122 = get_neighbor_value(iCell, {0, 1, 1});
          const real_t tmp200 = get_neighbor_value(iCell, {1, -1, -1});
          const real_t tmp201 = get_neighbor_value(iCell, {1, -1, 0});
          const real_t tmp202 = get_neighbor_value(iCell, {1, -1, 1});
          const real_t tmp210 = get_neighbor_value(iCell, {1, 0, -1});
          const real_t tmp211 = get_neighbor_value(iCell, {1, 0, 0});
          const real_t tmp212 = get_neighbor_value(iCell, {1, 0, 1});
          const real_t tmp220 = get_neighbor_value(iCell, {1, 1, -1});
          const real_t tmp221 = get_neighbor_value(iCell, {1, 1, 0});
          const real_t tmp222 = get_neighbor_value(iCell, {1, 1, 1});
          // Interpolate to children
          const ForeachCell::CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
          // foreach_sibling
          // CellIndex get_sibling_cell TODO: check if sibling cells are necessary leaf or intermediate, similar to iCell_c0
          if( iCell_c0.iOct.isIntermediate ){
            CellIndex iCell000 = iCell_c0.getNeighbor_ghost_intermediate({0, 0, 0}, Uintermediate.getShape());
            Uintermediate.at(iCell000, Isolution) += tmp0
                + f1 * (tmp011 + tmp101 + tmp110)
                + f2 * (tmp001 + tmp010 + tmp100)
                + f3 * tmp000;
            CellIndex iCell001 = iCell_c0.getNeighbor_ghost_intermediate({0, 0, 1}, Uintermediate.getShape());
            Uintermediate.at(iCell001, Isolution) += tmp0
                + f1 * (tmp011 + tmp101 + tmp112)
                + f2 * (tmp001 + tmp012 + tmp102)
                + f3 * tmp002;
            CellIndex iCell010 = iCell_c0.getNeighbor_ghost_intermediate({0, 1, 0}, Uintermediate.getShape());
            Uintermediate.at(iCell010, Isolution) += tmp0
                + f1 * (tmp011 + tmp121 + tmp110)
                + f2 * (tmp021 + tmp010 + tmp120)
                + f3 * tmp020;
            CellIndex iCell011 = iCell_c0.getNeighbor_ghost_intermediate({0, 1, 1}, Uintermediate.getShape());
            Uintermediate.at(iCell011, Isolution) += tmp0
                    + f1 * (tmp011 + tmp121 + tmp112)
                    + f2 * (tmp021 + tmp012 + tmp122)
                    + f3 * tmp022;
            CellIndex iCell100 = iCell_c0.getNeighbor_ghost_intermediate({1, 0, 0}, Uintermediate.getShape());
            Uintermediate.at(iCell100, Isolution) += tmp0
                    + f1 * (tmp211 + tmp101 + tmp110)
                    + f2 * (tmp201 + tmp210 + tmp100)
                    + f3 * tmp200;
            CellIndex iCell101 = iCell_c0.getNeighbor_ghost_intermediate({1, 0, 1}, Uintermediate.getShape());
            Uintermediate.at(iCell101, Isolution) += tmp0
                    + f1 * (tmp211 + tmp101 + tmp112)
                    + f2 * (tmp201 + tmp212 + tmp102)
                    + f3 * tmp202;
            CellIndex iCell110 = iCell_c0.getNeighbor_ghost_intermediate({1, 1, 0}, Uintermediate.getShape());
            Uintermediate.at(iCell110, Isolution) += tmp0
                    + f1 * (tmp211 + tmp121 + tmp110)
                    + f2 * (tmp221 + tmp210 + tmp120)
                    + f3 * tmp220;
            CellIndex iCell111 = iCell_c0.getNeighbor_ghost_intermediate({1, 1, 1}, Uintermediate.getShape());
            Uintermediate.at(iCell111, Isolution) += tmp0
                    + f1 * (tmp211 + tmp121 + tmp112)
                    + f2 * (tmp221 + tmp212 + tmp122)
                    + f3 * tmp222;
          }
        }
      }); 
    };

    auto zero_solution = [&](const uint32_t level){
      foreach_cell.foreach_cell("Write potential", U.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        uint32_t current_level = cells.getLevel(iCell);
        if (level == current_level){
          U.at(iCell, Isolution) = 0;
        }
      });
      foreach_cell.foreach_intermediate_cell("Write potential", Uintermediate.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        uint32_t current_level = cells.getLevel(iCell);
        if (level == current_level){
          Uintermediate.at(iCell, Isolution) = 0;
        }
      });
    };

    auto zero_solution_residual_rhs = [&](const uint32_t level){
      foreach_cell.foreach_cell("Write potential", U.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        uint32_t current_level = cells.getLevel(iCell);
        if (level == current_level){
          U.at(iCell, Isolution) = 0;
          U.at(iCell, Irhs) = 0;
          U.at(iCell, Iresidual) = 0;
        }
      });
      foreach_cell.foreach_intermediate_cell("Write potential", Uintermediate.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        uint32_t current_level = cells.getLevel(iCell);
        if (level == current_level){
          Uintermediate.at(iCell, Isolution) = 0;
          Uintermediate.at(iCell, Irhs) = 0;
          Uintermediate.at(iCell, Iresidual) = 0;
        }
      });
    };

    auto solution_to_potential = [&](const uint32_t level){
      foreach_cell.foreach_cell("Write potential", U.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        uint32_t current_level = cells.getLevel(iCell);
        if (level == current_level){
          U.at(iCell, Iphi) = U.at(iCell, Isolution);
        }
      });
      foreach_cell.foreach_intermediate_cell("Write potential", Uintermediate.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        uint32_t current_level = cells.getLevel(iCell);
        if (level == current_level){
          Uintermediate.at(iCell, Iphi) = Uintermediate.at(iCell, Isolution);
        }
      });
    };

    auto isRed = [](const ForeachCell::CellIndex iCell) {
      uint32_t idx = iCell.i + iCell.j + iCell.k;
      return (int)(idx % 2 != 0);
    };
    auto isBlack = [](const ForeachCell::CellIndex iCell) {
      uint32_t idx = iCell.i + iCell.j + iCell.k;
      return (int)(idx % 2 == 0);
    };


    // AMR, loop on all leaves
    // GS on correction term. Bigger neighbors set to zero
    auto gauss_seidel_intermediate_amr_correction = [&](auto is_coloured, const uint32_t level) {
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_intermediate_cell("Gauss-Seidel", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          if (is_coloured(iCell)) {
            const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
            Kokkos::Array<real_t, 3> f_L, f_R, f_C;
            real_t neighbors(0), contrib_L(0), contrib_R(0);
            const real_t w_relax(1.);
            for ( const ComponentIndex3D dir : {IX,IY,IZ} )
            {
              f_L[dir] = 1.;
              f_R[dir] = 1.;
              f_C[dir] = 2.;
              ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
              ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
              ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, U.getShape());
              ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, U.getShape());
              DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate L cell cannot have smaller intermediate neighbor" );
              DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate R cell cannot have smaller intermediate neighbor" );
              if (iCell_L.status == ForeachCell::CellIndex::BIGGER) {
                contrib_L = 0; // Zero BC for correction terms
                f_L[dir] = 8./3.;
                f_R[dir] = 4./3.;
                f_C[dir] = 4.;
              } else contrib_L = U.at(iCell_L, Isolution);
              if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;
              if (iCell_R.status == ForeachCell::CellIndex::BIGGER) {
                contrib_R = 0; // Zero BC for correction terms
                f_R[dir] = 8./3.;
                f_L[dir] = 4./3.;
                f_C[dir] = 4.;
              } else contrib_R = U.at(iCell_R, Isolution);
              if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;
              neighbors += (f_L[dir] * contrib_L + f_R[dir] * contrib_R) / (size[dir] * size[dir]);                   
            }
            U.at(iCell, Isolution) += w_relax * (
              (neighbors - U.at(iCell, Irhs)) / ( 
              f_C[IX] / (size[IX]*size[IX]) + f_C[IY] / (size[IY]*size[IY]) + f_C[IZ] / (size[IZ]*size[IZ])
              ) - U.at(iCell, Isolution)
            );
          }
        }
      });
    };
    
    auto gauss_seidel_leaves_amr_correction = [&](auto is_coloured, const uint32_t level) {
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_cell("Gauss-Seidel", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          if (is_coloured(iCell)) {
            const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
            Kokkos::Array<real_t, 3> f_L, f_R, f_C;
            real_t neighbors(0), contrib_L(0), contrib_R(0);
            const real_t w_relax(1.);
            for ( const ComponentIndex3D dir : {IX,IY,IZ} )
            {
              f_L[dir] = 1.;
              f_R[dir] = 1.;
              f_C[dir] = 2.;
              ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
              ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
              ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
              ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
              if ( iCell_L.status == ForeachCell::CellIndex::SMALLER ) {
                iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
                contrib_L = Uintermediate.at(iCell_L, Isolution);
              } else if (iCell_L.status == ForeachCell::CellIndex::BIGGER) {
                contrib_L = 0; // Zero BC for correction terms
                f_L[dir] = 8./3.;
                f_R[dir] = 4./3.;
                f_C[dir] = 4.;
              } else contrib_L = U.at(iCell_L, Isolution);
              if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;
              if ( iCell_R.status == ForeachCell::CellIndex::SMALLER ) {
                iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
                contrib_R = Uintermediate.at(iCell_R, Isolution);
              } else if (iCell_R.status == ForeachCell::CellIndex::BIGGER) {
                contrib_R = 0; // Zero BC for correction terms
                f_R[dir] = 8./3.;
                f_L[dir] = 4./3.;
                f_C[dir] = 4.;
              } else contrib_R = U.at(iCell_R, Isolution);
              if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;
              neighbors += (f_L[dir] * contrib_L + f_R[dir] * contrib_R) / (size[dir] * size[dir]);                   
            }
            U.at(iCell, Isolution) += w_relax * (
              (neighbors - U.at(iCell, Irhs)) / ( 
              f_C[IX] / (size[IX]*size[IX]) + f_C[IY] / (size[IY]*size[IY]) + f_C[IZ] / (size[IZ]*size[IZ])
              ) - U.at(iCell, Isolution)
            );
          }
        }
      });
    };

    // GS on potential. Bigger neighbors are necessarily Iphi

    auto gauss_seidel_leaves_amr_finest = [&](auto is_coloured, const uint32_t level) {
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_cell("Gauss-Seidel", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          if (is_coloured(iCell)) {
            const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
            Kokkos::Array<real_t, 3> f_L, f_R, f_C;
            real_t neighbors(0), contrib_L(0), contrib_R(0);
            const real_t w_relax(1.);
            for ( const ComponentIndex3D dir : {IX,IY,IZ} )
            {
              f_L[dir] = 1.;
              f_R[dir] = 1.;
              f_C[dir] = 2.;
              ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
              ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
              ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
              ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
              if ( iCell_L.status == ForeachCell::CellIndex::SMALLER ) {
                iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
                contrib_L = Uintermediate.at(iCell_L, Isolution);
              } else if (iCell_L.status == ForeachCell::CellIndex::BIGGER) {
                //contrib_L = U.at(iCell_L, Iphi); // TODO: find better interpolation from coarser level?
                contrib_L = average_4bigger_neighbors(iCell, off_L);
                f_L[dir] = 8./15.;
                f_R[dir] = 4./5.;
                f_C[dir] = 4./3.;
                /* contrib_L = average_8bigger_neighbors(iCell, off_L);
                f_L[dir] = 8./3.;
                f_R[dir] = 4./3.;
                f_C[dir] = 4.; */
              } else contrib_L = U.at(iCell_L, Isolution);
              if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;
              if ( iCell_R.status == ForeachCell::CellIndex::SMALLER ) {
                iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
                contrib_R = Uintermediate.at(iCell_R, Isolution);
              } else if (iCell_R.status == ForeachCell::CellIndex::BIGGER) {
                //contrib_R = U.at(iCell_R, Iphi); // TODO: find better interpolation from coarser level?
                contrib_R = average_4bigger_neighbors(iCell, off_R);
                f_R[dir] = 8./15.;
                f_L[dir] = 4./5.;
                f_C[dir] = 4./3.;
                /* contrib_R = average_8bigger_neighbors(iCell, off_R);
                f_R[dir] = 8./3.;
                f_L[dir] = 4./3.;
                f_C[dir] = 4.; */
              } else contrib_R = U.at(iCell_R, Isolution);
              if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;
              neighbors += (f_L[dir] * contrib_L + f_R[dir] * contrib_R) / (size[dir] * size[dir]);             
            }
            U.at(iCell, Isolution) += w_relax * (
              (neighbors - U.at(iCell, Irhs)) / ( 
              f_C[IX] / (size[IX]*size[IX]) + f_C[IY] / (size[IY]*size[IY]) + f_C[IZ] / (size[IZ]*size[IZ])
              ) - U.at(iCell, Isolution)
            );
          }
        }
      });
    };

    // PM-like, loop on all leaves
    auto gauss_seidel_leaves_uniform = [&](auto is_coloured, const uint32_t level) {
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_cell("Gauss-Seidel", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          if (is_coloured(iCell)) {
            const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
            Kokkos::Array<real_t, 3> contrib_L, contrib_R;
            real_t neighbors = 0;
            const real_t w_relax(1.);
            for ( const ComponentIndex3D dir : {IX,IY,IZ} )
            {
              ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
              ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
              ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
              ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
              if ( iCell_L.level_diff() != 0 ) {
                iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
                contrib_L[dir] = Uintermediate.at(iCell_L, Isolution);
              } else contrib_L[dir] = U.at(iCell_L, Isolution);
              if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L[dir] = 0;
              if ( iCell_R.level_diff() != 0 ) {
                iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
                contrib_R[dir] = Uintermediate.at(iCell_R, Isolution);
              } else contrib_R[dir] = U.at(iCell_R, Isolution);
              if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R[dir] = 0;
              neighbors += (contrib_L[dir] + contrib_R[dir]) / (size[dir] * size[dir]);             
            }
            U.at(iCell, Isolution) += w_relax * (
              (neighbors - U.at(iCell, Irhs)) / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ]))
              - U.at(iCell, Isolution)
            );
          }
        }
      });
    };
      
    // PM-like, loop on all intermediate octants
    auto gauss_seidel_intermediate = [&](auto is_coloured, const uint32_t level) {
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_intermediate_cell("Gauss-Seidel", Uintermediate.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          if (is_coloured(iCell)) {
            const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
            Kokkos::Array<real_t, 3> contrib_L, contrib_R;
            real_t neighbors = 0;
            const real_t w_relax(1.);
            for ( ComponentIndex3D dir : {IX,IY,IZ} )
            {
              ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
              ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
              ForeachCell::CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
              ForeachCell::CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
              if ( iCell_L.level_diff() != 0 ) {
                iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
                contrib_L[dir] = U.at(iCell_L, Isolution);
              } else contrib_L[dir] = Uintermediate.at(iCell_L, Isolution);
              if ( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L[dir] = 0;
              if ( iCell_R.level_diff() != 0 ) {
                iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
                contrib_R[dir] = U.at(iCell_R, Isolution);
              } else contrib_R[dir] = Uintermediate.at(iCell_R, Isolution);
              if ( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R[dir] = 0;
              neighbors += (contrib_L[dir] + contrib_R[dir]) / (size[dir] * size[dir]);
            }
            Uintermediate.at(iCell, Isolution) += w_relax * (
              (neighbors - Uintermediate.at(iCell, Irhs)) / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ]))
              - Uintermediate.at(iCell, Isolution)
            );
          }
        }
      });
    };
    

    auto smoothing_intermediate_amr_correction = [&](const uint32_t nIterations, const uint32_t level) {
      for (uint32_t i = 0; i < nIterations; i++) {
        gauss_seidel_intermediate_amr_correction(isRed, level);
        gauss_seidel_intermediate_amr_correction(isBlack, level);
      }
    };
    auto smoothing_uniform = [&](const uint32_t nIterations, const uint32_t level) {
      for (uint32_t i = 0; i < nIterations; i++) {
        gauss_seidel_leaves_uniform(isRed, level);
        gauss_seidel_intermediate(isRed, level);
        gauss_seidel_leaves_uniform(isBlack, level);
        gauss_seidel_intermediate(isBlack, level);
      }
    };
    auto smoothing_all_amr_correction = [&](const uint32_t nIterations, const uint32_t level) {
      for (uint32_t i = 0; i < nIterations; i++) {
        gauss_seidel_leaves_amr_correction(isRed, level);
        gauss_seidel_intermediate_amr_correction(isRed, level);
        gauss_seidel_leaves_amr_correction(isBlack, level);
        gauss_seidel_intermediate_amr_correction(isBlack, level);
      }
    };
    auto smoothing_all_amr_finest = [&](const uint32_t nIterations, const uint32_t level) {
      for (uint32_t i = 0; i < nIterations; i++) {
        gauss_seidel_leaves_amr_finest(isRed, level);
        gauss_seidel_intermediate(isRed, level);
        gauss_seidel_leaves_amr_finest(isBlack, level);
        gauss_seidel_intermediate(isBlack, level);
      }
    };

    // RUN
    const uint32_t Npre = 2; // Number of pre smoothing
    const uint32_t Npost = 2; // Number of post smoothing
    const uint32_t Ncycles = 1; // Number of multigrid cycles
    real_t rho_mean = 0;

    std::function<void (uint32_t)> V_cycle_uniform = [&](const uint32_t level) {
      smoothing_uniform(Npre, level);
      residual_uniform(level);
      restriction(level - 1);
      initialise_lhs(level - 1);
      if (level == min_level_multigrid + 1) {
        smoothing_uniform(Npre, level - 1);
      }
      else V_cycle_uniform(level - 1);
      
      prolongation(level);
      smoothing_uniform(Npost, level);
    };

    std::function<void (uint32_t, uint32_t)> V_cycle_amr = [&](const uint32_t current_level, const uint32_t finest_level) {

      if ( current_level == finest_level ) {
        smoothing_all_amr_finest(Npre, current_level);
        residual_amr_finest(current_level);
      } else if ( current_level == finest_level - 1 ) {
        smoothing_intermediate_amr_correction(Npre, current_level);
        residual_intermediate_amr_correction(current_level);
      } else { // TODO: Subdomain of intermediate cells
        smoothing_intermediate_amr_correction(Npre, current_level);
        residual_intermediate_amr_correction(current_level);
      }

      restriction(current_level - 1);
      initialise_lhs(current_level - 1);

      if (finest_level == level_coarse + 1) {
        smoothing_intermediate_amr_correction(Npre, current_level);
      } else if ( level_coarse == (current_level - 1) ) { // TODO: Subdomain of intermediate cells
        smoothing_intermediate_amr_correction(Npre, current_level);
      } else V_cycle_amr(current_level - 1, finest_level); 
      
      if ( current_level == finest_level ) {
        prolongation(current_level);
        smoothing_all_amr_finest(Npost, current_level);
      } else {
        prolongation_on_intermediate(current_level);  
        smoothing_all_amr_correction(Npost, current_level);
      }    
      
    };

    // Compute rho mean
    foreach_cell.reduce_cell("Compute rho_mean", U.getShape(),
    KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell, real_t & update_rhomean)
    {
      ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t rhoi = U.at(iCell, Irho);
      update_rhomean += rhoi * size[IX] * size[IY] * size[IZ];
    }, Kokkos::Sum<real_t>(rho_mean));
    real_t Vtot = (xmax - xmin) * (ymax - ymin) * (zmax - zmin);
    rho_mean = MPI_Allreduce_scalar(rho_mean) / Vtot;
    printf("rhomean = %.5e\n", rho_mean);
    
    // Initialize RHS and solution on leaves
    foreach_cell.foreach_cell("Init RHS and potential", U.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
    {
      ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      const real_t rho = U.at(iCell, Irho) - rho_mean;
      U.at(iCell, Irhs) = rho;
      U.at(iCell, Isolution) = -rho / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ]));
    });

    // Initialize RHS on intermediate levels. Solution will be interpolated from coarser levels so no need to initialize
    for( uint32_t level = max_level_in_amr + 1; level >= level_coarse; level-- )
    {
      foreach_cell.foreach_intermediate_cell( "average_parent_cell", Uintermediate.getShape(),
      KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if( current_level == level )
        {
          ForeachCell::CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
          real_t rho = 0;
          int ns = foreach_sibling( ndim, iCell_c0, Uintermediate.getShape(),
            [&]( const ForeachCell::CellIndex& iCell_c )
          {
            if ( iCell_c.iOct.isIntermediate ) rho += Uintermediate.at(iCell_c, Irho);
            else rho += U.at(iCell_c, Irho);
          });
          rho /= ns;
          Uintermediate.at( iCell, Irho ) = rho;
          Uintermediate.at( iCell, Irhs ) = rho - rho_mean;
        }
      }); 
    }

    residual_uniform(level_coarse);
    solution_to_potential(level_coarse);
    printf("Level %d Residual norm init  V %f\n", level_coarse, std::sqrt(residual_norm_sqr(level_coarse)));

    scalar_data.set<int>("iter", iter++);
    scalar_data.set<real_t>("time", time++);
    iomanager->save_snapshot(U_, scalar_data);
    
    // Multigrid
    printf("Coarse Multigrid\n");
    for(uint32_t i = 0; i < 3; i++)
    { 
      V_cycle_uniform(level_coarse);
      residual_uniform(level_coarse);
      solution_to_potential(level_coarse);
      printf("Level %d Residual norm after V %.5e\n", level_coarse, std::sqrt(residual_norm_sqr(level_coarse)));
      scalar_data.set<int>("iter", iter++);
      scalar_data.set<real_t>("time", time++);
      iomanager->save_snapshot(U_, scalar_data);
    }

    printf("AMR Multigrid\n");
    for (uint32_t ilevel = level_coarse+1; ilevel <= max_level_in_amr; ilevel++) {
      zero_solution(ilevel);
      prolongation(ilevel);
      zero_solution_residual_rhs(ilevel-1);
      solution_to_potential(ilevel);
      residual_amr_finest(ilevel);
      scalar_data.set<int>("iter", iter++);
      scalar_data.set<real_t>("time", time++);
      iomanager->save_snapshot(U_, scalar_data);

      for(uint32_t i = 0; i < 30; i++)
      { 
        V_cycle_amr(ilevel, ilevel);
        residual_amr_finest(ilevel);
        solution_to_potential(ilevel);
        printf("Level %d Residual norm after V %.5e\n", ilevel, std::sqrt(residual_norm_sqr(ilevel)));
        scalar_data.set<int>("iter", iter++);
        scalar_data.set<real_t>("time", time++);
        iomanager->save_snapshot(U_, scalar_data);
      }
      /* for (uint32_t i = 0; i < 30; i++) {
        smoothing_amr_finest(Npre, ilevel);
        residual_amr_finest(ilevel);
        solution_to_potential(ilevel);
        printf("Residual norm after V %.5e\n", std::sqrt(residual_norm_sqr(ilevel)));
        scalar_data.set<int>("iter", iter++);
        scalar_data.set<real_t>("time", time++);
        iomanager->save_snapshot(U_, scalar_data);
      } */
    }

    // Test against analytical prediction
    real_t rcore = 2*Hernquist::r0;
    {
      real_t Phi_ana_mean = 0;
      real_t Phi_num_mean = 0;
      real_t Vcore = 0;
      auto cells = foreach_cell.getCellMetaData();
      foreach_cell.reduce_cell( "Init", U.getShape(),
        KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& Phi_ana_mean, real_t& Phi_num_mean, real_t& Vcore )
      {
        auto pos = cells.getCellCenter(iCell);
        auto size = cells.getCellSize(iCell);

        real_t x = pos[IX]-0.5;
        real_t y = pos[IY]-0.5;
        real_t z = pos[IZ]-0.5;

        real_t r2 = x*x + y*y + z*z;
        real_t Vcell = size[IX]*size[IY]*size[IZ];
        if( r2 < rcore*rcore )
        {
          Phi_ana_mean += Hernquist::phi_periodic(x,y,z)*Vcell;
          Phi_num_mean += U.at(iCell, Iphi)*Vcell;
          Vcore += Vcell;
        } 
      }, Phi_ana_mean, Phi_num_mean, Vcore);

      Phi_ana_mean /= Vcore;
      Phi_num_mean /= Vcore;

      int err_count=0;
      foreach_cell.reduce_cell( "Init", U.getShape(),
        KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, int err_count )
      {
        auto pos = cells.getCellCenter(iCell);
        real_t x = pos[IX]-0.5;
        real_t y = pos[IY]-0.5;
        real_t z = pos[IZ]-0.5;

        real_t r2 = x*x + y*y + z*z;

        real_t Phi_ana = Hernquist::phi_periodic(x,y,z);
        real_t Phi_num = U.at( iCell, Iphi ) + (Phi_ana_mean-Phi_num_mean);
        real_t Phi_err = std::abs(Phi_ana-Phi_num)/std::abs(Phi_ana);

        if( r2 < rcore*rcore )
          if( abs(Phi_ana - Phi_num) > 1E-2 )
            err_count ++;
        
        U.at(iCell, Igx) = Phi_ana;
        U.at(iCell, Igy) = Phi_num;
        U.at(iCell, Igz) = Phi_err;
      }, err_count);
      EXPECT_EQ(err_count, 0);
    }

  }

} // namespace dyablo

TEST(Test_GravitySolver_multigrid, mesh_amrgrid_semiperiodic_sphere)
{
  dyablo::test_GravitySolver(dyablo::mesh_amrgrid_semiperiodic_sphere());
}


