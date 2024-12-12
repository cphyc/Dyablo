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
    int level_max = level_min + 6;
    uint32_t bx = 4, by = 4, bz = 4;


    std::cout << "// =========================================\n";
    std::cout << "// Testing GravitySolver_gc...\n";
    std::cout << "// Grid : amr - blocks " << bx << " -  levels " << level_min << " -> " << level_max << " \n";
    std::cout << "// Boundary conditions : (absorbing, absorbing, periodic) \n";
    std::cout << "// =========================================\n";

    std::cout << "Create mesh..." << std::endl;
    std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
    {
      int ndim = 3;

      amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool, 3>{false, false, true}, level_min, level_max);

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
      "outputPrefix=test_GravitySolver\n"
      "write_variables=rho,gphi,gx,gy,gz,res\n"
      "[amr]\n"
      "use_block_data=yes\n"
      "bx=4\n"
      "by=4\n"
      "bz=4\n"
      "[mesh]\n"
      "ndim=3\n"
      "boundary_type_xmin=absorbing\n"
      "boundary_type_xmax=absorbing\n"
      "boundary_type_ymin=absorbing\n"
      "boundary_type_ymax=absorbing\n"
      "boundary_type_zmin=periodic\n"
      "boundary_type_zmax=periodic\n"
      "[gravity]\n"
      "gravity_type=field\n"
      "G=1\n"
      "\n";
    ConfigMap configMap(configmap_str);
    ForeachCell foreach_cell(*amr_mesh, configMap);


    std::cout << "Initialize User Data..." << std::endl;

    enum VarIndex_gravity {
      Irho,
      Igx,
      Igy,
      Igz,
      Iphi,
      Iresidual,
      Ineighbor
    };

    UserData U_(configMap, foreach_cell);
    U_.new_fields({ "rho", "gx", "gy", "gz", "gphi", "res", "neighbor_contrib" });

    UserData::FieldAccessor U = U_.getAccessor({
      {"rho", Irho},
      {"gphi", Iphi},
      { "res", Iresidual },
      { "neighbor_contrib", Ineighbor }
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



    const uint32_t Npre = 2;
    const uint32_t Npost = 1;
    // Compute rho_mean
    real_t rho_mean = 0;
    auto cells = foreach_cell.getCellMetaData();
    real_t xmin = configMap.getValue<real_t>("mesh", "xmin", 0);
    real_t ymin = configMap.getValue<real_t>("mesh", "ymin", 0);
    real_t zmin = configMap.getValue<real_t>("mesh", "zmin", 0);
    real_t xmax = configMap.getValue<real_t>("mesh", "xmax", 0);
    real_t ymax = configMap.getValue<real_t>("mesh", "ymax", 0);
    real_t zmax = configMap.getValue<real_t>("mesh", "zmax", 0);
    foreach_cell.reduce_cell("Compute rho_mean", U.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell, real_t & update_rhomean)
    {
      ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t rhoi = U.at(iCell, Irho);
      update_rhomean += rhoi * size[IX] * size[IY] * size[IZ];
    }, Kokkos::Sum<real_t>(rho_mean));
    real_t Vtot = (xmax - xmin) * (ymax - ymin) * (zmax - zmin);
    rho_mean = MPI_Allreduce_scalar(rho_mean) / Vtot;
    // Initialize potential
    {
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_cell("Init Potential", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
        const real_t rho = (U.at(iCell, Irho) / rho_mean - 1) * size[IX] * size[IY] * size[IZ];
        U.at(iCell, Iphi) = -rho / (2. / (size[IX] * size[IX]) + 2. / (size[IY] * size[IY]) + 2. / (size[IZ] * size[IZ]));
      });
    }


    const LightOctree& lmesh = U.getShape().lmesh;
    lmesh.buildFullTree();
    /* const uint32_t numOctants = lmesh.getNumOctants();
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
    printf("min_level %d, max_level = %d\n", min_level_in_amr, max_level_in_amr);


    const uint32_t min_level_multigrid = 1;
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
        auto logical_coords = lmesh.get_logical_coord(iOct);
        while (logical_coords[IX] % 2 == 0 && logical_coords[IY] % 2 == 0 && logical_coords[IZ] % 2 == 0 && level > min_level_multigrid) {
          level--;
          logical_coords = lmesh.get_parent_logical_coord(logical_coords);
          Kokkos::atomic_fetch_add( &octs_intermediate_per_level(level-min_level_multigrid), 1 );
        } 
      }
    );
    for(uint32_t ilevel = min_level_multigrid; ilevel <= max_level_in_amr; ilevel++) printf("Finished Level %d, octs %u intermediate %u\n", ilevel, octs_per_level(ilevel-min_level_multigrid), octs_intermediate_per_level(ilevel-min_level_multigrid));


    uint64_t numIntermediate = 0;
    for (uint32_t ilevel = 0; ilevel < nlevel; ilevel++) numIntermediate += octs_intermediate_per_level(ilevel);
    printf("numIntermediate %lu\n", numIntermediate); */
    // Create intermediate hashmap

    // Initialize cells from intermediate octs
    
    /* build_fulltree(U);

    auto isRed = [](const ForeachCell::CellIndex iCell) {
      uint32_t idx = iCell.i + iCell.j + iCell.k;
      return (int)(idx % 2 != 0);
      };
    auto isBlack = [](const ForeachCell::CellIndex iCell) {
      uint32_t idx = iCell.i + iCell.j + iCell.k;
      return (int)(idx % 2 == 0);
      };


    // Loop on all leaves, accounts for contribution from smaller neighbors and contributes to bigger neighbors
    auto gauss_seidel_leaves = [&](auto is_coloured, const uint32_t level) {
      const LightOctree& lmesh = U.getShape().lmesh;
      const auto cells = foreach_leaf_cell.getCellMetaData();
      foreach_cell.foreach_leaf_cell("Gauss-Seidel", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          if (is_coloured(iCell)) {
            uint8_t neighbor_contribution = [&](ForeachCell::CellIndex iCell, const ComponentIndex3D dir) {
              const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
              ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
              ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
              ForeachCell::CellIndex iCell_L = iCell.getNeighbor_leaf_ghost(off_L, U);
              ForeachCell::CellIndex iCell_R = iCell.getNeighbor_leaf_ghost(off_R, U);
              real_t size_sqr;
              if (iCell_L.level_diff() == 1) {
                size_sqr = 0.75 * size[dir] * size[dir];
                // Contrib_L alreay injected from the 4 smaller neighbors
                real_t contrib_R = get_value_same_level(U, iCell_R, Iphi, off_R);
                U.at(iCell, Ineighbor) += 8. * contrib_R / (7. * size[dir] * size[dir]);
              }
              else if (iCell_L.level_diff() == -1) {
                size_sqr = 1.5 * size[dir] * size[dir];
                atomic_write_in_bigger_neighbor(U, iCell_L);
                real_t contrib_L = get_value_from_bigger_neighbor(U, iCell_L, Iphi, off_L);
                real_t contrib_R = get_value_same_level(U, iCell_R, Iphi, off_R);
                U.at(iCell, Ineighbor) += 4. * contrib_R / (5. * size[dir] * size[dir]);
              }
              else if (iCell_R.level_diff() == 1) {
                size_sqr = 0.75 * size[dir] * size[dir];
                // Contrib_R alreay injected from the 4 smaller neighbors
                real_t contrib_L = get_value_same_level(U, iCell_L, Iphi, off_L);
                U.at(iCell, Ineighbor) += 8 * contrib_L / (7. * size[dir] * size[dir]);
              }
              else if (iCell_R.level_diff() == -1) {
                size_sqr = 1.5 * size[dir] * size[dir];
                atomic_write_in_bigger_neighbor(U, iCell_R);
                real_t contrib_L = get_value_from_bigger_neighbor(U, iCell_L, Iphi, off_L);
                real_t contrib_L = get_value_same_level(U, iCell_L, Iphi, off_L);
                U.at(iCell, Ineighbor) += 4. * contrib_L / (5. * size[dir] * size[dir]);
              }
              else { // Both neighbors at same level
                real_t contrib_L = get_value_same_level(U, iCell_L, Iphi, off_L);
                real_t contrib_R = get_value_same_level(U, iCell_R, Iphi, off_R);
                U.at(iCell, Ineighbor) += (contrib_L + contrib_R) / (size[dir] * size[dir]);
              }
              return size_sqr;
              }
            const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
            const real_t rho = (U.at(iCell, Irho) / rho_mean - 1) * size[IX] * size[IY] * size[IZ];
            const uint8_t size_sqr_x = neighbor_contribution(iCell, IX);
            const uint8_t size_sqr_y = neighbor_contribution(iCell, IY);
            const uint8_t size_sqr_z = neighbor_contribution(iCell, IZ);
            U.at(iCell, Iphi) = (U.at(iCell, Ineighbor) - rho) / (2. / size_sqr_x + 2. / size_sqr_y + 2. / size_sqr_z);
            U.at(iCell, Iresidual) = rho - U.at(iCell, Iphi);
          }
        }
      });
      };
    // PM-like, loop on all octants on the full tree
    auto gauss_seidel_uniform = [&](auto is_coloured, const uint32_t level) {
      const LightOctree& lmesh = U.getShape().lmesh;
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_cell("Gauss-Seidel", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          if (is_coloured(iCell)) {
            uint8_t neighbor_contribution = [&](ForeachCell::CellIndex iCell, const ComponentIndex3D dir) {
              const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
              const real_t size_sqr = size * size;
              ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
              ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
              ForeachCell::CellIndex iCell_L = iCell.getNeighbor_fulltree_ghost(off_L, U);
              ForeachCell::CellIndex iCell_R = iCell.getNeighbor_fulltree_ghost(off_R, U);
              real_t contrib_L = get_value_same_level(U, iCell_L, Iphi, off_L);
              real_t contrib_R = get_value_same_level(U, iCell_R, Iphi, off_R);
              U.at(iCell, Ineighbor) += (contrib_L + contrib_R) / (size[dir] * size[dir]);
              return size_sqr;
              }
            const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
            const real_t rho = (U.at(iCell, Irho) / rho_mean - 1) * size[IX] * size[IY] * size[IZ];
            const uint8_t size_sqr_x = neighbor_contribution(iCell, IX);
            const uint8_t size_sqr_y = neighbor_contribution(iCell, IY);
            const uint8_t size_sqr_z = neighbor_contribution(iCell, IZ);
            U.at(iCell, Iphi) = (U.at(iCell, Ineighbor) - rho) / (2. / size_sqr_x + 2. / size_sqr_y + 2. / size_sqr_z);
            U.at(iCell, Iresidual) = rho - U.at(iCell, Iphi);
          }
        }
      });
      };
    // PM-like, loop on all intermediate octants
    auto gauss_seidel_intermediate = [&](auto is_coloured, const uint32_t level) {
      const LightOctree& lmesh = U.getShape().lmesh;
      const auto cells = foreach_cell.getCellMetaData();
      foreach_cell.foreach_intermediate_cell("Gauss-Seidel", U.getShape(),
        KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell)
      {
        const uint32_t current_level = lmesh.getLevel(iCell.iOct);
        if (current_level == level) {
          if (is_coloured(iCell)) {
            uint8_t neighbor_contribution = [&](ForeachCell::CellIndex iCell, const ComponentIndex3D dir) {
              const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
              const real_t size_sqr = size * size;
              ForeachCell::CellIndex::offset_t off_L = {}; off_L[dir] = -1;
              ForeachCell::CellIndex::offset_t off_R = {}; off_R[dir] = +1;
              ForeachCell::CellIndex iCell_L = iCell.getNeighbor_fulltree_ghost(off_L, U);
              ForeachCell::CellIndex iCell_R = iCell.getNeighbor_fulltree_ghost(off_R, U);
              real_t contrib_L = get_value_same_level(U, iCell_L, Iphi, off_L);
              real_t contrib_R = get_value_same_level(U, iCell_R, Iphi, off_R);
              U.at(iCell, Ineighbor) += (contrib_L + contrib_R) / (size[dir] * size[dir]);
              return size_sqr;
              }
            const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
            const real_t rho = (U.at(iCell, Irho) / rho_mean - 1) * size[IX] * size[IY] * size[IZ];
            const uint8_t size_sqr_x = neighbor_contribution(iCell, IX);
            const uint8_t size_sqr_y = neighbor_contribution(iCell, IY);
            const uint8_t size_sqr_z = neighbor_contribution(iCell, IZ);
            U.at(iCell, Iphi) = (U.at(iCell, Ineighbor) - rho) / (2. / size_sqr_x + 2. / size_sqr_y + 2. / size_sqr_z);
            U.at(iCell, Iresidual) = rho - U.at(iCell, Iphi);
          }
        }
      });
      };

    template<typename Function>
    auto smoothing = [&](const uint32_t nIterations, const uint32_t level, Function gaussSeidel) {
      for (uint32_t i = 0; i < nIterations; i++) {
        gaussSeidel(isRed, level);
        gaussSeidel(isBlack, level);
      }
      };

    auto V_cycle_uniform = [&](const uint32_t level) {
      smoothing(Npre, level, gauss_seidel_uniform);
      restriction_uniform;
      if (level == level_min_multigrid) {
        smoothing(Npre, level - 1, gauss_seidel_uniform);
      }
      else {
        V_cycle_uniform(level - 1);
      }
      prolongation_uniform;
      smoothing(Npost, level, gauss_seidel_uniform);
      }

      auto V_cycle_amr = [&](const uint32_t level) {
      smoothing(Npre, level, gauss_seidel_leaves);
      smoothing(Npre, level, gauss_seidel_intermediate);
      restriction_amr;
      if (level == level_min + 1) {
        V_cycle_uniform(level + 1);
      }
      else {
        V_cycle_amr(level - 1);
      }
      prolongation_amr;
      smoothing(Npost, level, gauss_seidel_uniform);
      }

    const uint32_t max_level_in_mesh = get_max_level_available(*amr_mesh);

    if (max_level_in_mesh == level_coarse)
      V_cycle_uniform(level_coarse);
    else
      V_cycle_amr(max_level_in_mesh); */

    real_t residual_norm;
    foreach_cell.reduce_cell("Compute residual norm", U.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex & iCell, real_t & update_residual_sqr)
    {
      real_t residual = U.at(iCell, Iresidual);
      update_residual_sqr += residual * residual;
    }, Kokkos::Sum<real_t>(residual_norm));
    residual_norm = std::sqrt(residual_norm);




    scalar_data.set<int>("iter", iter++);
    scalar_data.set<real_t>("time", time++);
    iomanager->save_snapshot(U_, scalar_data);

  }

} // namespace dyablo

TEST(Test_GravitySolver_multigrid, mesh_amrgrid_semiperiodic_sphere)
{
  dyablo::test_GravitySolver(dyablo::mesh_amrgrid_semiperiodic_sphere());
}


