#include "GravitySolver_multigrid.h"

#include "morton_utils.h"
#include "utils/monitoring/Timers.h"
#include "mpi/GhostCommunicator.h"
#include "foreach_cell/ForeachCell_utils.h"
#include <mpi.h>

namespace dyablo { 

using GlobalArray = typename ForeachCell::CellArray_global;
using GhostedArray = typename ForeachCell::CellArray_global_ghosted;
using CellIndex = typename ForeachCell::CellIndex;

enum VarIndex_MG
{
  Irho, Igx, Igy, Igz, Iphi,
  Isolution, Irhs, Iresidual, Imask
};

/**
 * @brief Structure storing all the necessary information for the 
 * resolution of the conjugate gradient
 **/
struct GravitySolver_multigrid::Data{
  ForeachCell& foreach_cell;
  
  Timers& timers;  

  int ndim;
  real_t xmin, ymin, zmin;
  real_t xmax, ymax, zmax;

  Kokkos::Array<BoundaryConditionType, 3> boundarycondition;

  bool cosmo_run;
  real_t four_Pi_G;
  real_t MG_eps;
  uint32_t first_mpi_multigrid_level;
  level_t level_coarse;
  uint32_t Npre, Npost, Ncycles;

  Kokkos::View<uint32_t*> octs_per_level;
  Kokkos::View<uint32_t*> octs_intermediate_per_level;
  Kokkos::View<uint32_t*> ghosts_per_level;
  Kokkos::View<uint32_t*> ghosts_intermediate_per_level;
  Kokkos::View<uint32_t*> octs_per_level_count;
  Kokkos::View<uint32_t*> octs_intermediate_per_level_count;
  Kokkos::View<uint32_t*> ghosts_per_level_count;
  Kokkos::View<uint32_t*> ghosts_intermediate_per_level_count;

  UserData_fields::FieldAccessor U;
  UserData_fields::FieldAccessor Uintermediate;

};

GravitySolver_multigrid::GravitySolver_multigrid(
  ConfigMap& configMap,
  ForeachCell& foreach_cell,
  Timers& timers )
 : pdata(new Data
    {
      foreach_cell,
      timers,
      configMap.getValue<int>("mesh", "ndim", 3),
      configMap.getValue<real_t>("mesh", "xmin", 0.0),
      configMap.getValue<real_t>("mesh", "ymin", 0.0),
      configMap.getValue<real_t>("mesh", "zmin", 0.0),
      configMap.getValue<real_t>("mesh", "xmax", 1.0),
      configMap.getValue<real_t>("mesh", "ymax", 1.0),
      configMap.getValue<real_t>("mesh", "zmax", 1.0),
      {
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING)
      },
      configMap.getValue<bool>("cosmology", "active", false),
      -1.0, // gravity_constant 4*Pi*G, defined later
      configMap.getValue<real_t>("gravity", "MG_eps", 1E-2),
      configMap.getValue<uint32_t>("gravity", "first_mpi_multigrid_level", 2),
      4,    // level coarse
      configMap.getValue<uint32_t>("gravity", "Npre", 2),
      configMap.getValue<uint32_t>("gravity", "Npost", 1),
      configMap.getValue<uint32_t>("gravity", "Ncycles", 2),
    })
{
  int ndim = configMap.getValue<int>("mesh", "ndim", 3);
  if(!pdata->cosmo_run)
    pdata->four_Pi_G = configMap.getValue<real_t>("gravity", "4_Pi_G", 1.0);
  
  DYABLO_ASSERT_HOST_RELEASE( ndim == 3, "GravitySolver_mg can only run in 3D" )
  
  [[maybe_unused]] GravityType gtype = configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_FIELD);
  DYABLO_ASSERT_HOST_RELEASE( gtype == GRAVITY_FIELD, "GravitySolver_mg must have gravity_type=field" );
}

GravitySolver_multigrid::~GravitySolver_multigrid()
{}

//namespace{

/**
 * @brief Computes a three-point finite-difference gradient contribution.
 *
 * @tparam Array_t Type of the data arrays.
 *
 * @param[in] U               Data array containing the primary field.
 * @param[in] Uintermediate   Intermediate data array used for multigrid processing.
 * @param[in] iCell           Index of the cell where the gradient is evaluated.
 * @param[in] iter_space      Shape of the grid or iteration space.
 * @param[in] side            Indicates stencil orientation or side (e.g., +1 or -1).
 * @param[out] w              Output weight used in the gradient computation.
 * @param[out] contrib        Output contribution to the multigrid equation.
 * @param[in] dir             Direction along which the gradient is computed (x, y, or z).
 * @param[in] boundarycondition Array indicating the boundary condition types in each direction.
 */
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::get_weight_and_contrib_gradient_3pt(Array_t& U, Array_t& Uintermediate, const CellIndex& iCell, const Shape_t& iter_space, const int side, real_t& w, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  constexpr real_t half = 0.5;
  constexpr bool search_intermediate = true;
  CellIndex::offset_t offset = {}; offset[dir] = side;
  CellIndex iCell_X = iCell.getNeighbor_ghost(offset, iter_space);
  if (CellIndex::BIGGER == iCell_X.status) 
  {
    contrib = average_8bigger_neighbors(U, Uintermediate, iCell, iter_space, offset);
    w = half;
  } 
  else if (CellIndex::SMALLER == iCell_X.status) 
  {
    iCell_X = iCell.getNeighbor_ghost<search_intermediate>(offset, iter_space);
    contrib = Uintermediate.at(iCell_X, Iphi);
  } 
  else contrib = U.at(iCell_X, Iphi);
  if( BC_ABSORBING == boundarycondition[dir] && iCell_X.is_boundary() ) contrib = 0;
};

/**
 * @brief Computes the three-point finite-difference gradient over the entire domain.
 *
 * @tparam Array_t Type of the data arrays.
 *
 * @param[in]  U               Data array.
 * @param[out] Uintermediate   Intermediate data array.
 */
template< typename Array_t >
void GravitySolver_multigrid::gradient_3pt(Array_t& U, Array_t& Uintermediate)
{
  constexpr real_t zero(0), one(1);
  const auto iter_space = Uintermediate.getShape();
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  constexpr Kokkos::Array< VarIndex, 3 > IG = {Igx, Igy, Igz};

  foreach_cell.foreach_cell("Gravity_mg::construct_force_field", iter_space, 
    KOKKOS_LAMBDA(const CellIndex& iCell)
  { 
    const level_t level = cells.getLevel(iCell);
    const uint32_t nocts1d = 1U << level;
    const Kokkos::Array<real_t, 3> size = {
          1./(nocts1d * iter_space.bx), 
          1./(nocts1d * iter_space.by), 
          1./(nocts1d * iter_space.bz)
    };
    const real_t phi_C = U.at(iCell, Iphi);

    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t phi_L(zero), phi_R(zero), a(one), b(one);
      get_weight_and_contrib_gradient_3pt(U, Uintermediate, iCell, iter_space, -1, a, phi_L, dir, boundarycondition);
      get_weight_and_contrib_gradient_3pt(U, Uintermediate, iCell, iter_space, +1, b, phi_R, dir, boundarycondition);

      const real_t f_L = b/(a*a + a*b);
      const real_t f_R = -a/(a*b + b*b);
      const real_t f_C = (a-b)/(a*b);

      U.at(iCell, IG[dir]) = (f_L * phi_L + f_R * phi_R + f_C * phi_C)/size[dir];
    }
  });
}

/**
 * @brief Computes the five-point finite-difference gradient over the entire domain.
 *
 * @tparam Array_t Type of the data arrays.
 *
 * @param[in]  U               Data array.
 * @param[out] Uintermediate   Intermediate data array.
 */
// TODO: To write 5-pt gradient!
template< typename Array_t >
void GravitySolver_multigrid::gradient_5pt(Array_t& U, Array_t& Uintermediate)
{
  DYABLO_ASSERT_HOST_RELEASE(false, "gradient_5pt not implemented yet");
}


/**
 * @brief Interpolates the contribution at a fine-coarse grid boundary using coarser neighbors.
 * 
 * At fine-coarse resolution boundaries, this function reconstructs the value at a finer cell 
 * by averaging over eight surrounding coarser-level neighbors. 
 * 
 * @tparam Array_t Type of the data arrays.
 * 
 * @param[in] U               Data array
 * @param[in] Uintermediate   Intermediate data array
 * @param[in] iCell           Index of the fine-level cell at which to reconstruct the value.
 * @param[in] iter_space      Grid shape used to compute neighbor positions and bounds.
 * @param[in] offset          Directional offset that determines which neighboring cells to consider.
 * 
 * @return The interpolated value at the fine-coarse boundary cell.
 */
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
real_t GravitySolver_multigrid::average_8bigger_neighbors(const Array_t& U, const Array_t& Uintermediate, const CellIndex& iCell, const Shape_t& iter_space, CellIndex::offset_t offset) 
{
  real_t result = 0;
  uint32_t counter = 0;
  const int8_t offset_x = (!abs(offset[0])); 
  const int8_t offset_y = (!abs(offset[1])); 
  const int8_t offset_z = (!abs(offset[2]));
  constexpr Kokkos::Array<real_t, 4> tmp = {9./32, 3./32, 3./32, 1./32};
  for (int8_t i = 0; i <= offset_x; i++)
  for (int8_t j = 0; j <= offset_y; j++)
  for (int8_t k = 0; k <= offset_z; k++)
  {
    for (int8_t l = 0; l <= 1; l++)
    {
      const int8_t shift_x = 2 * ( l*offset[IX] + i * (2*(iCell.i % 2) - 1) );
      const int8_t shift_y = 2 * ( l*offset[IY] + j * (2*(iCell.j % 2) - 1) );
      const int8_t shift_z = 2 * ( l*offset[IZ] + k * (2*(iCell.k % 2) - 1) );
      const CellIndex::offset_t shift = {shift_x, shift_y, shift_z};
      const CellIndex iCell_coarse = iCell.getNeighbor_ghost(shift, iter_space);
      if (CellIndex::BIGGER == iCell_coarse.status) 
        result += tmp[counter]*U.at(iCell_coarse, Iphi);
      else 
      {
        const CellIndex iCell_parent = iCell_coarse.getParent(iter_space);
        result += tmp[counter]*Uintermediate.at(iCell_parent, Iphi);
      }
    }
    counter++;
  }
  return result; 
}

/**
 * @brief Computes the L2 norm of the residual at a given grid level.
 * 
 * This function calculates the square root of the sum of squared residuals 
 * from both the leaf cells and the intermediate (coarse) cells at a specified multigrid level.
 * It performs parallel reduction over all relevant cells and uses an MPI reduction 
 * to combine values across all processes.
 * 
 * @param[in] level The multigrid level at which the residual norm is computed.
 * 
 * @return The L2 norm of the residual (i.e., sqrt(sum of squares)) at the given level.
 */
real_t GravitySolver_multigrid::residual_norm(const level_t level)
{
  const auto& U = pdata->U;
  const auto& Uintermediate = pdata->Uintermediate;
  const auto iter_space = Uintermediate.getShape();
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  real_t residual_sqr_leaves = 0;
  real_t residual_sqr_intermediate = 0;
  const auto octs = get_subview_octs(level);
  foreach_cell.reduce_cell_in_octants("Compute residual norm", iter_space, octs,
    KOKKOS_LAMBDA(const CellIndex & iCell, real_t & update_residual_sqr)
  {
    const real_t residual_tmp = U.at(iCell, Iresidual);
    update_residual_sqr += residual_tmp * residual_tmp;
  }, Kokkos::Sum<real_t>(residual_sqr_leaves));
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.reduce_intermediate_cell_in_octants("Compute residual norm", iter_space, octs_intermediate,
    KOKKOS_LAMBDA(const CellIndex & iCell, real_t & update_residual_sqr)
  {
    const real_t residual_tmp = Uintermediate.at(iCell, Iresidual);
    update_residual_sqr += residual_tmp * residual_tmp;
  }, Kokkos::Sum<real_t>(residual_sqr_intermediate));

  real_t residual_sqr = residual_sqr_leaves + residual_sqr_intermediate;
  //printf("residual_sqr_leafs = %e residual_sqr_intermediate = %e\n", residual_sqr_leaves, residual_sqr_intermediate);
  residual_sqr = MPI_Allreduce_scalar(residual_sqr);
  return Kokkos::sqrt(residual_sqr);
}

/**
 * @brief Computes the L2 norm of the truncation at a given grid level.
 * 
 * This function calculates the square root of the sum of squared truncation (residual slot) 
 * from both the intermediate (coarser) cells at a specified multigrid level.
 * It performs parallel reduction over all relevant cells and uses an MPI reduction 
 * to combine values across all processes.
 * 
 * @param[in] level The multigrid level at which the truncation norm is computed.
 * 
 * @return The L2 norm of the truncation (i.e., sqrt(sum of squares)) at the given level.
 */
real_t GravitySolver_multigrid::truncation_norm(const level_t level)
{
  const auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  real_t truncation_sqr = 0;
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.reduce_intermediate_cell_in_octants("Compute residual norm", Uintermediate.getShape(), octs_intermediate,
    KOKKOS_LAMBDA(const CellIndex & iCell, real_t & update_residual_sqr)
  {
    const real_t truncation_tmp = Uintermediate.at(iCell, Irhs) - Uintermediate.at(iCell, Iresidual);
    update_residual_sqr += truncation_tmp * truncation_tmp;
  }, Kokkos::Sum<real_t>(truncation_sqr));
  truncation_sqr = MPI_Allreduce_scalar(truncation_sqr);
  return Kokkos::sqrt(truncation_sqr);
}

/**
 * @brief Initializes the left-hand side (solution) from the right-hand side for a Poisson problem.
 * 
 * This function sets the initial guess for the solution of a Poisson equation
 * by applying a simple scaling factor to the right-hand side values.
 * It supports selective initialization over different target cell types 
 * (leaves, intermediates, ghosts) depending on the compile-time `target`.
 * 
 * The scaling factor is derived from a finite difference approximation of the Laplacian operator.
 * 
 * @tparam target The subset of grid cells (e.g., leaves, intermediates, ghosts) to be initialized.
 * @param level The grid level at which the initialization is applied.
 */
template <Target target>
void GravitySolver_multigrid::initialise_lhs_from_rhs(const level_t level)
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  constexpr real_t one = 1;
  constexpr real_t two = 2;
  const real_t factor = - one / (two / (size[IX]*size[IX]) + two / (size[IY]*size[IY]) + two / (size[IZ]*size[IZ]));

  if constexpr (target == Target::LEAVES || target == Target::BOTH || target == Target::ALL_LEAVES || target == Target::ALL) 
  {
    const auto octs = get_subview_octs(level);
    foreach_cell.foreach_cell_in_octants("Initialise potential", iter_space, octs,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      U.at(iCell, Isolution) = U.at(iCell, Irhs) * factor;
    });
  }

  if constexpr (target == Target::INTERMEDIATES || target == Target::BOTH || target == Target::ALL_INTERMEDIATES || target == Target::ALL) 
  {
    const auto octs_intermediate = get_subview_octs_intermediate(level);
    foreach_cell.foreach_intermediate_cell_in_octants("Initialise potential", iter_space, octs_intermediate,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      Uintermediate.at(iCell, Isolution) = Uintermediate.at(iCell, Irhs) * factor;
    });
  }

  if constexpr (target == Target::GHOST_LEAVES || target == Target::BOTH_GHOSTS || target == Target::ALL_LEAVES || target == Target::ALL) 
  {
    const auto octs_ghost = get_subview_ghosts(level);
    foreach_cell.foreach_ghost_cell_in_octants("Initialise potential", iter_space, octs_ghost,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      U.at(iCell, Isolution) = U.at(iCell, Irhs) * factor;
    });
  }

  if constexpr (target == Target::GHOST_INTERMEDIATES || target == Target::BOTH_GHOSTS || target == Target::ALL_INTERMEDIATES || target == Target::ALL) 
  {
    const auto octs_intermediate_ghost = get_subview_ghosts_intermediate(level);
    foreach_cell.foreach_intermediate_ghost_cell_in_octants("Initialise potential", iter_space, octs_intermediate_ghost,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      Uintermediate.at(iCell, Isolution) = Uintermediate.at(iCell, Irhs) * factor;
    }); 
  }
};

/**
 * @brief Copies multiple field components from source to destination across multigrid levels.
 * 
 * This method transfers data between specified field components (e.g., solution, residual)
 * within a range of multigrid levels. Copy operations are selectively applied to different
 * categories of grid cells (leaves, intermediates, ghosts) based on the `target` template parameter.
 * 
 * @tparam target Specifies which cell types (e.g., leaves, intermediates, ghosts) to process.
 * @tparam T The type used to index field components.
 * @tparam N The number of field components to copy.
 * @param level_min The minimum grid level to include.
 * @param level_max The maximum grid level to include.
 * @param iFields_src Array of field indices from which to copy.
 * @param iFields_dst Array of field indices to which to copy.
 */
template <Target target, typename T, size_t N>
void GravitySolver_multigrid::copy_multigrid_fields(const level_t level_min, const level_t level_max, const Kokkos::Array<T, N>& iFields_src, const Kokkos::Array<T, N>& iFields_dst)
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const auto iter_space = Uintermediate.getShape();
  const ForeachCell& foreach_cell = pdata->foreach_cell;

  if constexpr (target == Target::LEAVES || target == Target::BOTH || target == Target::ALL_LEAVES || target == Target::ALL) 
  {
    const auto octs = get_subview_octs(level_min, level_max);
    foreach_cell.foreach_cell_in_octants("Fill leaves", iter_space, octs,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      for (size_t i = 0; i < N; i++)
        U.at(iCell, iFields_dst[i]) = U.at(iCell, iFields_src[i]);
    });
  }
  
  if constexpr (target == Target::INTERMEDIATES || target == Target::BOTH || target == Target::ALL_INTERMEDIATES || target == Target::ALL) 
  {
    const auto octs_intermediate = get_subview_octs_intermediate(level_min, level_max);
    foreach_cell.foreach_intermediate_cell_in_octants("Fill intermediates", iter_space, octs_intermediate,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      for (size_t i = 0; i < N; i++)
        Uintermediate.at(iCell, iFields_dst[i]) = Uintermediate.at(iCell, iFields_src[i]);
    });
  }

  if constexpr (target == Target::GHOST_LEAVES || target == Target::BOTH_GHOSTS || target == Target::ALL_LEAVES || target == Target::ALL) 
  {
    const auto octs = get_subview_ghosts(level_min, level_max);
    foreach_cell.foreach_ghost_cell_in_octants("Fill ghost leaves", iter_space, octs,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      for (size_t i = 0; i < N; i++)
        U.at(iCell, iFields_dst[i]) = U.at(iCell, iFields_src[i]);
    });
  }

  if constexpr (target == Target::GHOST_INTERMEDIATES || target == Target::BOTH_GHOSTS || target == Target::ALL_INTERMEDIATES || target == Target::ALL) 
  {
    const auto octs_intermediate = get_subview_ghosts_intermediate(level_min, level_max);
    foreach_cell.foreach_intermediate_ghost_cell_in_octants("Fill ghost intermediates", iter_space, octs_intermediate,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      for (size_t i = 0; i < N; i++)
        Uintermediate.at(iCell, iFields_dst[i]) = Uintermediate.at(iCell, iFields_src[i]);
    });
  }

};

/**
 * @brief Propagates and accumulates values up the multigrid tree hierarchy.
 * 
 * Starting from a fine cell, this function recursively walks up the multigrid tree 
 * to its coarser ancestors (up to a specified level), applying a decaying contribution 
 * to corresponding fields in each parent cell. It uses atomic additions to safely accumulate 
 * values in parallel computations.
 * 
 * This function is typically used during restriction operations.
 * 
 * @tparam N The number of fields to process.
 * 
 * @param U[in,out] Field accessor used to read and update field values.
 * @param iCell[in,out] The current cell index (will be updated as it walks up the tree).
 * @param iter_space[in] The iteration space that defines the shape of the grid.
 * @param level_stop[in] The target (coarsest) level to stop accumulation.
 * @param level_start[in] The starting (finest) level to begin accumulation.
 * @param factor_init[in] Initial contribution values for each field.
 * @param factor_decay[in] The factor by which contributions decay at each level.
 * @param iFields[in] Array of field indices corresponding to each value in `factor_init`.
 */
template <size_t N>
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::accumulate_up_tree(
    const UserData_fields::FieldAccessor& U,
    CellIndex& iCell,
    const Shape_t& iter_space,
    const level_t level_stop,
    const level_t level_start,
    const Kokkos::Array<real_t, N>& factor_init,
    const real_t factor_decay,
    const Kokkos::Array<int, N>& iFields)
{
  CellIndex iCell_p;
  Kokkos::Array<real_t, N> factor = factor_init;
  for (level_t level_diff = 0; level_diff < level_start - level_stop; level_diff++)
  {
    iCell_p = iCell.getParent(iter_space);
    for (size_t i = 0; i < N; i++)
    {
      Kokkos::atomic_add(&U.at(iCell_p, iFields[i]), factor[i]);
      factor[i] *= factor_decay;
    }
    iCell = std::move(iCell_p);
  }
}

/**
 * @brief Performs multigrid restriction from finer to coarser grid levels.
 * 
 * This method transfers data from finer (child) cells to coarser (parent) cells across
 * the specified multigrid levels using the restriction operator. It accumulates contributions 
 * from child cells into their corresponding parent cells. Only non-ghost cells are considered
 * for restriction.
 * 
 * The restriction operation supports applying to leaf, intermediate, or both types of cells, 
 * determined by the `target` template parameter.
 * 
 * @tparam target Specifies the target cell types to apply the restriction to 
 *         (LEAVES, INTERMEDIATES, or BOTH).
 * @tparam Integer_t The type used for indexing field components.
 * @tparam N The number of field components being restricted.
 * 
 * @param level_stop The coarsest grid level (restriction target).
 * @param level_start The finest grid level (restriction source).
 * @param iFields_children Array of field indices to restrict from (fine level).
 * @param iFields_parents Array of field indices to restrict to (coarse level).
 */
template< Target target, typename Integer_t, size_t N >
void GravitySolver_multigrid::restriction_from_children(const level_t level_stop, const level_t level_start, const Kokkos::Array<Integer_t, N>& iFields_children, const Kokkos::Array<Integer_t, N>& iFields_parents) 
{
  if constexpr (target > Target::BOTH)
    static_assert(target <= Target::BOTH, "Restriction only supports LEAVES, INTERMEDIATES or BOTH targets");

  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const auto iter_space = Uintermediate.getShape();
  ForeachCell& foreach_cell = pdata->foreach_cell;
  constexpr real_t inv_ns = 1./8; 

  if constexpr (target == Target::LEAVES || target == Target::BOTH)
  {
    const auto octs = get_subview_octs(level_start);
    foreach_cell.foreach_cell_in_octants( "Restrict", iter_space, octs,
      KOKKOS_LAMBDA( CellIndex& iCell)
    {
      Kokkos::Array<real_t, N> factor_init;
      for (size_t i = 0; i < N; i++)
        factor_init[i] = U.at(iCell, iFields_children[i]) * inv_ns;

      accumulate_up_tree(Uintermediate, iCell, iter_space, level_stop, level_start, factor_init, inv_ns, iFields_parents);
    }); 
  }
  if constexpr (target == Target::INTERMEDIATES || target == Target::BOTH) 
  {
    const auto octs_intermediate = get_subview_octs_intermediate(level_start);
    foreach_cell.foreach_intermediate_cell_in_octants( "Restrict", iter_space, octs_intermediate,
      KOKKOS_LAMBDA( CellIndex& iCell)
    {
      Kokkos::Array<real_t, N> factor_init;
      for (size_t i = 0; i < N; i++)
        factor_init[i] = Uintermediate.at(iCell, iFields_children[i]) * inv_ns;

      accumulate_up_tree(Uintermediate, iCell, iter_space, level_stop, level_start, factor_init, inv_ns, iFields_parents);
    }); 
  }
}


/**
 * @brief Retrieve the solution value from a neighboring cell.
 * 
 * This function returns the solution value of a neighboring cell 
 * for a given intermediate cell. It first attempts to access the neighbor 
 * using intermediate data. If the neighbor lies on a different level, 
 * it retrieves the solution from the main data array instead.
 * 
 * @tparam Array_t Type of the data arrays.
 * 
 * @param U[in] Main data array containing solution values.
 * @param Uintermediate[in] Intermediate data array containing solution values for finer resolutions.
 * @param iCell[in] Index of the current cell for which the neighbor is being queried.
 * @param iter_space[in] The spatial domain describing the valid index region.
 * @param offset[in] Offset used to determine the relative position of the neighbor.
 * 
 * @return The solution value at the neighboring cell.
 */
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
real_t GravitySolver_multigrid::get_neighbor_value(const Array_t& U, const Array_t& Uintermediate, const CellIndex& iCell, const Shape_t& iter_space, const CellIndex::offset_t offset)
{
  constexpr bool search_intermediate = true;
  CellIndex iCell_n = iCell.getNeighbor_ghost<search_intermediate>(offset, iter_space);
  if (iCell_n.level_diff() == 0)
    return Uintermediate.at( iCell_n, Isolution );

  iCell_n = iCell.getNeighbor_ghost(offset, iter_space);
  return U.at( iCell_n, Isolution );
}

/**
 * @brief Perform first-order prolongation (interpolation) from parent cells.
 * 
 * @tparam target Specifies whether to apply the prolongation to LEAVES, INTERMEDIATES, BOTH, or ALL relevant cell types.
 * 
 * @param level The level from which the child (fine) cells will receive interpolated data from their parent (coarser) cells.
 */
template <Target target>
void GravitySolver_multigrid::prolongation_from_children(const level_t level) 
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const auto iter_space = Uintermediate.getShape();
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  constexpr real_t f0 = 27.0 / 64;
  constexpr real_t f1 = 9.0 / 64;
  constexpr real_t f2 = 3.0 / 64;
  constexpr real_t f3 = 1.0 / 64;

  if constexpr (target == Target::LEAVES || target == Target::BOTH || target == Target::ALL_LEAVES || target == Target::ALL) 
  {
    const auto octs = get_subview_octs(level);
    foreach_cell.foreach_cell_in_octants( "Prolongation", iter_space, octs,
      KOKKOS_LAMBDA( CellIndex& iCell)
    {
      const int8_t shift_x = 2 * (iCell.i % 2) - 1;
      const int8_t shift_y = 2 * (iCell.j % 2) - 1;
      const int8_t shift_z = 2 * (iCell.k % 2) - 1;
      // Get coarse cell values to interpolate from
      CellIndex iCell_p = iCell.getParent(iter_space);
      iCell_p.status = CellIndex::LOCAL_TO_BLOCK;
      const real_t tmp000 = Uintermediate.at( iCell_p, Isolution );
      const real_t tmp001 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {0, 0, shift_z});
      const real_t tmp010 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {0, shift_y, 0});
      const real_t tmp100 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {shift_x, 0, 0});
      const real_t tmp011 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {0, shift_y, shift_z});
      const real_t tmp101 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {shift_x, 0, shift_z});
      const real_t tmp110 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {shift_x, shift_y, 0});
      const real_t tmp111 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {shift_x, shift_y, shift_z});
      // Interpolate
      U.at(iCell, Isolution) += f0*tmp000
          + f1 * (tmp001 + tmp010 + tmp100)
          + f2 * (tmp011 + tmp101 + tmp110)
          + f3 * tmp111;
    }); 
  }

  if constexpr (target == Target::INTERMEDIATES || target == Target::BOTH || target == Target::ALL_INTERMEDIATES || target == Target::ALL) 
  {
    const auto octs_intermediate = get_subview_octs_intermediate(level);
    foreach_cell.foreach_intermediate_cell_in_octants( "Prolongation", iter_space, octs_intermediate,
      KOKKOS_LAMBDA( CellIndex& iCell)
    {
      const int8_t shift_x = 2 * (iCell.i % 2) - 1;
      const int8_t shift_y = 2 * (iCell.j % 2) - 1;
      const int8_t shift_z = 2 * (iCell.k % 2) - 1;
      // Get coarse cell values to interpolate from
      CellIndex iCell_p = iCell.getParent(iter_space);
      iCell_p.status = CellIndex::LOCAL_TO_BLOCK;
      const real_t tmp000 = Uintermediate.at( iCell_p, Isolution );
      const real_t tmp001 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {0, 0, shift_z});
      const real_t tmp010 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {0, shift_y, 0});
      const real_t tmp100 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {shift_x, 0, 0});
      const real_t tmp011 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {0, shift_y, shift_z});
      const real_t tmp101 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {shift_x, 0, shift_z});
      const real_t tmp110 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {shift_x, shift_y, 0});
      const real_t tmp111 = get_neighbor_value(U, Uintermediate, iCell_p, iter_space, {shift_x, shift_y, shift_z});
      // Interpolate
      Uintermediate.at(iCell, Isolution) += f0*tmp000
          + f1 * (tmp001 + tmp010 + tmp100)
          + f2 * (tmp011 + tmp101 + tmp110)
          + f3 * tmp111;
    }); 
  }
}


bool GravitySolver_multigrid::check_neighbors()
{
  const auto iter_space = pdata->Uintermediate.getShape();
  ForeachCell& foreach_cell = pdata->foreach_cell;
  const LightOctree& lmesh = foreach_cell.get_amr_mesh().getLightOctree();
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  foreach_cell.foreach_cell( "Test octs", iter_space,
    KOKKOS_LAMBDA( const CellIndex& iCell )
  {
    for (int8_t ix = -1; ix <= 1; ++ix)
    for (int8_t iy = -1; iy <= 1; ++iy)
    for (int8_t iz = -1; iz <= 1; ++iz)
    {
      CellIndex::offset_t offset = {ix, iy, iz};
      CellIndex iCell_n = iCell.getNeighbor_ghost(offset, iter_space);
      auto level = lmesh.getLevel(iCell_n.iOct);
    }
  });
  return true;
}
/**
 * @brief Verify access to parent cells for all cells at levels > 0.
 * 
 * This method iterates over all regular and intermediate cells in the data 
 * structure and, for cells with level greater than zero, attempts to 
 * retrieve their parent cell index. The purpose is primarily to check that 
 * parent indexing works correctly without errors. No actual data modification 
 * occurs.
 * 
 * @return Always returns true, indicating completion of the check.
 */
bool GravitySolver_multigrid::check_parents()
{
  const auto iter_space = pdata->Uintermediate.getShape();
  ForeachCell& foreach_cell = pdata->foreach_cell;
  const LightOctree& lmesh = foreach_cell.get_amr_mesh().getLightOctree();
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const level_t level_max = lmesh.get_level_max();
  for (level_t ilevel = 1; ilevel <= level_max; ilevel++)
  {
    const auto octs = get_subview_octs(ilevel);
    foreach_cell.foreach_cell_in_octants("Set solution to zero", iter_space, octs,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      const auto lc = lmesh.get_logical_coords(iCell.iOct);
      const auto iCell_p = iCell.getParent(iter_space);
      const auto lcp = lmesh.get_logical_coords(iCell_p.iOct);
      [[maybe_unused]] const bool cond = ( (lc[IX] >> 1) == lcp[IX] && (lc[IY] >> 1) == lcp[IY] && (lc[IZ] >> 1) == lcp[IZ] );
      DYABLO_ASSERT_KOKKOS_DEBUG( cond, "Inconsistency in Child/Parent logical coords" );
    });
    const auto octs_intermediate = get_subview_octs_intermediate(ilevel);
    foreach_cell.foreach_intermediate_cell_in_octants("Set solution to zero", iter_space, octs_intermediate,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      const auto lc = lmesh.get_logical_coords(iCell.iOct);
      const auto iCell_p = iCell.getParent(iter_space);
      const auto lcp = lmesh.get_logical_coords(iCell_p.iOct); 
      [[maybe_unused]] const bool cond = ( (lc[IX] >> 1) == lcp[IX] && (lc[IY] >> 1) == lcp[IY] && (lc[IZ] >> 1) == lcp[IZ] );
      DYABLO_ASSERT_KOKKOS_DEBUG( cond, "Inconsistency in Child/Parent logical coords" );
    });
  }
  return true;
}


/**
 * @brief Compute and initialize the multigrid mask field over specified levels.
 * 
 * This method sets up the mask values used in the multigrid solver for all relevant
 * grid levels between `level_min` and `level_max`. It initializes mask values on 
 * leaf and intermediate cells, including ghost cells, to support multigrid operations.
 * 
 * The mask is initialized differently for MPI and non-MPI levels, accounting for
 * domain decomposition. The method also performs accumulation and reduction steps
 * across the grid hierarchy to propagate mask values upward through the grid levels.
 * Finally, it manages ghost cell communication to synchronize mask values across MPI ranks.
 * 
 * @param U_               The user data object containing solution fields.
 * @param level_min        The minimum grid level on which to compute the mask.
 * @param level_max        The maximum grid level on which to compute the mask.
 * @param ghost_comm_minimal   Ghost communicator for minimal communication between MPI ranks.
 * @param ghost_comm_blockwide Ghost communicator for block-wide communication between MPI ranks.
 */
void GravitySolver_multigrid::compute_mask(UserData& U_, const level_t level_min, const level_t level_max, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide)
{
  const auto& Uintermediate = pdata->Uintermediate;
  const auto iter_space = Uintermediate.getShape();
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const level_t first_mpi_multigrid_level = pdata->first_mpi_multigrid_level;
  const bool has_non_mpi_levels = (level_min < first_mpi_multigrid_level);
  const level_t level_min_mpi = has_non_mpi_levels ? first_mpi_multigrid_level: level_min;
  constexpr real_t inv_ns = 1./8.; 
  constexpr real_t two = 2;

  // Initialise mask
  fill_multigrid_fields<Target::ALL>(level_max, level_max, kokkos_array<int>(Imask), kokkos_array<real_t>(1));
  fill_multigrid_fields<Target::BOTH>(level_min_mpi, level_max - 1, kokkos_array<int>(Imask), kokkos_array<real_t>(-1));
  fill_multigrid_fields<Target::BOTH_GHOSTS>(level_min_mpi, level_max - 1, kokkos_array<int>(Imask), kokkos_array<real_t>(0));

  if (has_non_mpi_levels)
  {
    const MpiComm& mpicomm = foreach_cell.get_amr_mesh().getMpiComm();
    const real_t mpi_size = mpicomm.MPI_Comm_size();
    const real_t mask_initial = -1. / mpi_size; // Used to ensure that when reducing over all MPI, we find -1 by default.
    fill_multigrid_fields<Target::INTERMEDIATES>(first_mpi_multigrid_level - 1, first_mpi_multigrid_level - 1, kokkos_array<int>(Imask), kokkos_array<real_t>(mask_initial));
    fill_multigrid_fields<Target::INTERMEDIATES>(level_min, first_mpi_multigrid_level - 2, kokkos_array<int>(Imask), kokkos_array<real_t>(0));    
  }

  // Compute mask on leaves
  const auto octs = get_subview_octs(level_max);
  foreach_cell.foreach_cell_in_octants( "Mask on domain", iter_space, octs,
    KOKKOS_LAMBDA( CellIndex& iCell)
  {
   accumulate_up_tree(Uintermediate, iCell, iter_space, level_min_mpi - 1, level_max, kokkos_array<real_t>(two * inv_ns), inv_ns, kokkos_array<int>(Imask));
  });
  // Compute mask on intermediate cells
  const auto octs_intermediate = get_subview_octs_intermediate(level_max);
  foreach_cell.foreach_intermediate_cell_in_octants( "Mask on domain", iter_space, octs_intermediate,
    KOKKOS_LAMBDA( CellIndex& iCell)
  {
    accumulate_up_tree(Uintermediate, iCell, iter_space, level_min_mpi - 1, level_max, kokkos_array<real_t>(two * inv_ns), inv_ns, kokkos_array<int>(Imask));
  });
 
  // Proces non-MPI levels if needed
  if (has_non_mpi_levels)
  {
    reduce_nonMPI_levels(first_mpi_multigrid_level - 1, kokkos_array<int>(Imask));
    if (level_min + 1 < first_mpi_multigrid_level)
    {
      const auto octs_intermediate = get_subview_octs_intermediate(first_mpi_multigrid_level - 1);
      foreach_cell.foreach_intermediate_cell_in_octants( "Mask on domain", iter_space, octs_intermediate,
        KOKKOS_LAMBDA( CellIndex& iCell)
      {
        accumulate_up_tree(Uintermediate, iCell, iter_space, level_min, first_mpi_multigrid_level - 1, kokkos_array<real_t>(Uintermediate.at(iCell, Imask) * inv_ns), inv_ns, kokkos_array<int>(Imask));
      });
    }
  }
  // MPI Communications at each level
  for (level_t level = level_max - 1; level >= level_min_mpi; level--) 
  {
    reduce_ghosts_at_level<Target::INTERMEDIATES>(U_, level, {"mask"}, ghost_comm_blockwide);
    exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, level, {"mask"}, ghost_comm_minimal); 
  }
}


/**
 * @brief Fill specified multigrid fields with given values over a range of grid levels.
 * 
 * This templated function sets the values of selected fields on either leaf cells,
 * intermediate cells, or their ghost counterparts, within a specified level range.
 * 
 * The function supports multiple target cell sets defined by the `target` template parameter:
 * - LEAVES: Regular leaf cells
 * - INTERMEDIATES: Intermediate cells
 * - GHOST_LEAVES: Ghost cells associated with leaves
 * - GHOST_INTERMEDIATES: Ghost cells associated with intermediates
 * - BOTH and BOTH_GHOSTS: Combinations of the above
 * - ALL, ALL_LEAVES, ALL_INTERMEDIATES: Convenience aliases to cover multiple categories
 * 
 * For each targeted cell set and level range, the corresponding fields indexed by `iFields`
 * are set to the corresponding values from `values`.
 * 
 * @tparam target       The target cell type(s) to fill (e.g., LEAVES, INTERMEDIATES, BOTH, etc.)
 * @tparam Integer_t    Integer type for field indices
 * @tparam Real_t       Numeric type for field values
 * @tparam N            Number of fields to fill
 * 
 * @param[in] min_level Minimum grid level to fill (inclusive)
 * @param[in] max_level Maximum grid level to fill (inclusive)
 * @param[in] iFields   Array of field indices specifying which fields to fill
 * @param[in] values    Array of values corresponding to each field index
 * 
 * @note This function modifies the internal data arrays `pdata->U` and `pdata->Uintermediate` 
 *       directly by writing the specified values.
 */
template <Target target, typename Integer_t, typename Real_t, size_t N>
void GravitySolver_multigrid::fill_multigrid_fields(const level_t min_level, const level_t max_level, const Kokkos::Array<Integer_t, N>& iFields, const Kokkos::Array<Real_t, N>& values)
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const auto iter_space = Uintermediate.getShape();
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();

  if constexpr (target == Target::LEAVES || target == Target::BOTH || target == Target::ALL_LEAVES || target == Target::ALL) 
  {
    const auto octs = get_subview_octs(min_level, max_level);
    foreach_cell.foreach_cell_in_octants("Fill leaves", iter_space, octs,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      for (size_t i = 0; i < N; i++)
        U.at(iCell, iFields[i]) = values[i];
    });
  }
  
  if constexpr (target == Target::INTERMEDIATES || target == Target::BOTH || target == Target::ALL_INTERMEDIATES || target == Target::ALL) 
  {
    const auto octs_intermediate = get_subview_octs_intermediate(min_level, max_level);
    foreach_cell.foreach_intermediate_cell_in_octants("Fill intermediates", iter_space, octs_intermediate,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      for (size_t i = 0; i < N; i++)
        Uintermediate.at(iCell, iFields[i]) = values[i];
    });
  }

  if constexpr (target == Target::GHOST_LEAVES || target == Target::BOTH_GHOSTS || target == Target::ALL_LEAVES || target == Target::ALL) 
  {
    const auto octs = get_subview_ghosts(min_level, max_level);
    foreach_cell.foreach_ghost_cell_in_octants("Fill ghost leaves", iter_space, octs,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      for (size_t i = 0; i < N; i++)
        U.at(iCell, iFields[i]) = values[i];
    });
  }

  if constexpr (target == Target::GHOST_INTERMEDIATES || target == Target::BOTH_GHOSTS || target == Target::ALL_INTERMEDIATES || target == Target::ALL) 
  {
    const auto octs_intermediate = get_subview_ghosts_intermediate(min_level, max_level);
    foreach_cell.foreach_intermediate_ghost_cell_in_octants("Fill ghost intermediates", iter_space, octs_intermediate,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      for (size_t i = 0; i < N; i++)
        Uintermediate.at(iCell, iFields[i]) = values[i];
    });
  }
}

/**
 * @brief Compute the uniform operator (discrete Laplacian) on solution fields at a given level.
 * 
 * Applies a discrete Laplacian operator to the solution fields on both leaf and intermediate cells
 * at the specified grid level. The Laplacian is computed using uniform grid spacing and considers
 * boundary conditions.
 * 
 * The residual of the Laplacian operation is stored in the corresponding residual fields.
 * 
 * @param[in] level The multigrid level at which to apply the operator.
 * 
 * @note
 * - This function reads from the solution fields `pdata->U` and `pdata->Uintermediate`.
 * - The residual results are written to the fields `Iresidual` in both `pdata->U` (leaves) and
 *   `pdata->Uintermediate` (intermediate cells).
 */
void GravitySolver_multigrid::operator_uniform(const level_t level) 
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  constexpr real_t zero = 0;
  constexpr real_t two = 2;
  const real_t factor = two/(size[IX]*size[IX]) + two/(size[IY]*size[IY]) + two/(size[IZ]*size[IZ]);

  const auto octs = get_subview_octs(level);
  foreach_cell.foreach_cell_in_octants("Operator", iter_space, octs,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    real_t neighbors(zero);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t contrib_L(zero), contrib_R(zero);
      get_contrib_leaves_uniform(U, Uintermediate, iCell, iter_space, -1, contrib_L, dir, boundarycondition);
      get_contrib_leaves_uniform(U, Uintermediate, iCell, iter_space, +1, contrib_R, dir, boundarycondition);
      neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
    }
    const real_t laplacian_solution = neighbors - U.at(iCell, Isolution) * factor;
    U.at(iCell, Iresidual) = laplacian_solution;
  });
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Operator", iter_space, octs_intermediate,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    real_t neighbors(zero);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t contrib_L(zero), contrib_R(zero);
      get_contrib_intermediate_uniform(U, Uintermediate, iCell, iter_space, -1, contrib_L, dir, boundarycondition);
      get_contrib_intermediate_uniform(U, Uintermediate, iCell, iter_space, +1, contrib_R, dir, boundarycondition);
      neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
    }
    const real_t laplacian_solution = neighbors - Uintermediate.at(iCell, Isolution) * factor;
    Uintermediate.at(iCell, Iresidual) = laplacian_solution;
  });
}

/**
 * @brief Compute contribution from a neighbor cell for the intermediate uniform operator.
 * 
 * Retrieves the value of the neighboring cell in a specified direction and side, considering
 * intermediate cells and boundary conditions, to contribute to the discrete Laplacian calculation.
 * 
 * @tparam Array_t The array type used for field access.
 * 
 * @param[in] U The solution data array for leaf cells.
 * @param[in] Uintermediate The solution data array for intermediate cells.
 * @param[in] iCell The current cell index from which to get the neighbor.
 * @param[in] iter_space The iteration space shape defining the grid.
 * @param[in] side The side offset direction (-1 or +1) along the component direction.
 * @param[out] contrib The output contribution value from the neighbor cell.
 * @param[in] dir The direction (component) along which the neighbor is considered (IX, IY, or IZ).
 * @param[in] boundarycondition Array of boundary condition types for each spatial direction.
 */
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::get_contrib_intermediate_uniform(Array_t& U, Array_t& Uintermediate, const CellIndex& iCell, const Shape_t& iter_space, const int side, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  constexpr bool search_intermediate = true;
  CellIndex::offset_t offset = {}; offset[dir] = side;
  CellIndex iCell_X = iCell.getNeighbor_ghost<search_intermediate>(offset, iter_space);
  DYABLO_ASSERT_KOKKOS_DEBUG( iCell_X.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
  if ( CellIndex::BIGGER == iCell_X.status ) 
  {
    iCell_X = iCell.getNeighbor_ghost(offset, iter_space);
    contrib = U.at(iCell_X, Isolution);
  } 
  else contrib = Uintermediate.at(iCell_X, Isolution);
  if ( BC_ABSORBING == boundarycondition[dir] && iCell_X.is_boundary() ) contrib = 0;
}

/**
 * @brief Compute contribution from a neighbor cell for the leaves uniform operator.
 * 
 * Retrieves the value of the neighboring leaf or intermediate cell in a specified direction and side,
 * considering boundary conditions, to contribute to the discrete Laplacian calculation.
 * 
 * @tparam Array_t The array type used for field access.
 * 
 * @param[in] U The solution data array for leaf cells.
 * @param[in] Uintermediate The solution data array for intermediate cells.
 * @param[in] iCell The current cell index from which to get the neighbor.
 * @param[in] iter_space The iteration space shape defining the grid.
 * @param[in] side The side offset direction (-1 or +1) along the component direction.
 * @param[out] contrib The output contribution value from the neighbor cell.
 * @param[in] dir The direction (component) along which the neighbor is considered (IX, IY, or IZ).
 * @param[in] boundarycondition Array of boundary condition types for each spatial direction.
 */
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::get_contrib_leaves_uniform(Array_t& U, Array_t& Uintermediate, const CellIndex& iCell, const Shape_t& iter_space, const int side, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  constexpr bool search_intermediate = true;
  CellIndex::offset_t offset = {}; offset[dir] = side;
  CellIndex iCell_X = iCell.getNeighbor_ghost(offset, iter_space);
  DYABLO_ASSERT_KOKKOS_DEBUG( iCell_X.level_diff() <= 0, "Leaf cell at coarse level cannot have coarser leaf neighbor" );
  if ( CellIndex::SMALLER == iCell_X.status ) 
  {
    iCell_X = iCell.getNeighbor_ghost<search_intermediate>(offset, iter_space);
    contrib = Uintermediate.at(iCell_X, Isolution);
  } 
  else contrib = U.at(iCell_X, Isolution);
  if ( BC_ABSORBING == boundarycondition[dir] && iCell_X.is_boundary() ) contrib = 0;
}

/**
 * @brief Residual on a uniform grid.
 * 
 * Computes the residual of a Poisson equation on a uniform grid.
 * 
 * @tparam target The target grid cells to operate on (LEAVES, INTERMEDIATES, or BOTH).
 * 
 * @param[in] level The grid level.
 * @param[in] U The data array to read solution and RHS values from (leaf cells).
 * @param[in] Uintermediate The data array to read solution and RHS values from (intermediate cells).
 * @param[in] boundarycondition Array of boundary condition types for each spatial direction.
 */
template <Target target> 
void GravitySolver_multigrid::residual_uniform(const level_t level) 
{
  if constexpr (target > Target::BOTH)
    static_assert(target <= Target::BOTH, "Residual uniform can only support LEAVES, INTERMEDIATES or BOTH targets");

  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  constexpr real_t zero = 0;
  constexpr real_t two = 2;
  const real_t factor = two/(size[IX]*size[IX]) + two/(size[IY]*size[IY]) + two/(size[IZ]*size[IZ]);

  if constexpr (target == Target::LEAVES || target == Target::BOTH) 
  {
    const auto octs = get_subview_octs(level);
    foreach_cell.foreach_cell_in_octants("Residual", iter_space,  octs,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      real_t neighbors(zero);
      for ( ComponentIndex3D dir : {IX,IY,IZ} )
      {
        real_t contrib_L(zero), contrib_R(zero);
        get_contrib_leaves_uniform(U, Uintermediate, iCell, iter_space, -1, contrib_L, dir, boundarycondition);
        get_contrib_leaves_uniform(U, Uintermediate, iCell, iter_space, +1, contrib_R, dir, boundarycondition);
        neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
      }
      const real_t laplacian_solution = neighbors - U.at(iCell, Isolution) * factor;
      U.at(iCell, Iresidual) = U.at(iCell, Irhs) - laplacian_solution;
    });
  }
  if constexpr (target == Target::INTERMEDIATES || target == Target::BOTH) 
  {
    const auto octs_intermediate = get_subview_octs_intermediate(level);
    foreach_cell.foreach_intermediate_cell_in_octants("Residual", iter_space, octs_intermediate,
      KOKKOS_LAMBDA(const CellIndex & iCell)
    {
      real_t neighbors(zero);
      for ( ComponentIndex3D dir : {IX,IY,IZ} )
      {
        real_t contrib_L(zero), contrib_R(zero);
        get_contrib_intermediate_uniform(U, Uintermediate, iCell, iter_space, -1, contrib_L, dir, boundarycondition);
        get_contrib_intermediate_uniform(U, Uintermediate, iCell, iter_space, +1, contrib_R, dir, boundarycondition);
        neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
      }
      const real_t laplacian_solution = neighbors - Uintermediate.at(iCell, Isolution) * factor;
      Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
    });
  }
}

/**
 * @brief Compute weight and contribution for intermediate AMR correction.
 * 
 * Calculates the weight and contribution from a neighbor intermediate cell,
 * used for adaptive mesh refinement corrections.
 * 
 * @tparam Array_t Type of the data array.
 * 
 * @param[in] Uintermediate Data array for intermediate cells.
 * @param[in] iCell The current cell index.
 * @param[in] iter_space The iteration space shape.
 * @param[in] side Direction to look for the neighbor (-1 or +1).
 * @param[out] w Computed weight for the neighbor cell contribution.
 * @param[out] contrib Computed contribution from the neighbor cell.
 * @param[in] dir The spatial direction (IX, IY, IZ).
 * @param[in] boundarycondition Boundary conditions for each spatial direction.
 * @param[in] mask Mask value at the current cell.
 */
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::get_weight_and_contrib_intermediate_amr_correction(Array_t& Uintermediate, const CellIndex& iCell, const Shape_t& iter_space, const int side, real_t& w, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition, const real_t mask) 
{
  constexpr real_t half = 0.5;
  constexpr bool search_intermediate = true;
  CellIndex::offset_t offset = {}; offset[dir] = side;
  const CellIndex iCell_X = iCell.getNeighbor_ghost<search_intermediate>(offset, iter_space);
  DYABLO_ASSERT_KOKKOS_DEBUG( iCell_X.level_diff() >= 0, "Intermediate L cell cannot have smaller intermediate neighbor" );
  if (iCell_X.status == CellIndex::BIGGER) w = half;
  else if (Uintermediate.at(iCell_X, Imask) <= 0) 
  {
    const real_t neighbor_mask = Uintermediate.at(iCell_X, Imask);
    w = mask / (mask - neighbor_mask);
  } //else if (Uintermediate.at(iCell_X, Imask) < 1) w = 1;  // First-order reconstruction 
  else contrib = Uintermediate.at(iCell_X, Isolution);
  if (boundarycondition[dir] == BC_ABSORBING && iCell_X.is_boundary()) contrib = 0;
}


/**
 * @brief Apply intermediate AMR correction for Laplacian operator at a given grid level.
 * 
 * Computes the Laplacian operator for intermediate AMR cells.
 * 
 * @param[in] level The grid level on which to apply the correction.
 */
void GravitySolver_multigrid::operator_intermediate_amr_correction(const level_t level) 
{
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  constexpr real_t zero = 0;
  constexpr real_t one = 1;
  constexpr real_t two = 2;

  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Residual", iter_space, octs_intermediate,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const real_t mask = Uintermediate.at(iCell, Imask);
    if (mask <= zero) return;

    real_t laplacian_solution(zero);
    const real_t central_solution = Uintermediate.at(iCell, Isolution);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t a(one), b(one), contrib_L(zero), contrib_R(zero);
      get_weight_and_contrib_intermediate_amr_correction(Uintermediate, iCell, iter_space, -1, a, contrib_L, dir, boundarycondition, mask);
      get_weight_and_contrib_intermediate_amr_correction(Uintermediate, iCell, iter_space, +1, b, contrib_R, dir, boundarycondition, mask);
      const real_t f_L = two / (a * (a + b));
      const real_t f_R = two / (b * (a + b));
      const real_t f_C = two / (a * b);
      laplacian_solution += (f_L*contrib_L + f_R*contrib_R - f_C*central_solution) / (size[dir] * size[dir]);
    }
    Uintermediate.at(iCell, Iresidual) = laplacian_solution;
  });
}

/**
 * @brief Compute residual for intermediate cells with AMR on correction levels, at a given grid level.
 * 
 * Calculates the residual correction for intermediate cells considering adaptive mesh refinement effects.
 * 
 * @param[in] level The grid level on which to compute the residual.
 */
void GravitySolver_multigrid::residual_intermediate_amr_correction(const level_t level) 
{
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  constexpr real_t zero = 0;
  constexpr real_t one = 1;
  constexpr real_t two = 2;

  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Residual", iter_space, octs_intermediate,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const real_t mask = Uintermediate.at(iCell, Imask);
    if (mask <= zero) return;

    real_t laplacian_solution(zero);
    const real_t central_solution = Uintermediate.at(iCell, Isolution);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t a(one), b(one), contrib_L(zero), contrib_R(zero);
      get_weight_and_contrib_intermediate_amr_correction(Uintermediate, iCell, iter_space, -1, a, contrib_L, dir, boundarycondition, mask);
      get_weight_and_contrib_intermediate_amr_correction(Uintermediate, iCell, iter_space, +1, b, contrib_R, dir, boundarycondition, mask);
      const real_t f_L = two / (a * (a + b));
      const real_t f_R = two / (b * (a + b));
      const real_t f_C = two / (a * b);
      laplacian_solution += (f_L*contrib_L + f_R*contrib_R - f_C*central_solution) / (size[dir] * size[dir]);
    }
    Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
  });
}


/**
 * @brief Compute weight and contribution for AMR finest level.
 * 
 * Calculates the weight and solution contribution from neighbor cells
 * on the finest level.
 * 
 * @tparam Array_t The array type for U and Uintermediate.
 * 
 * @param[in,out] U The solution array for leaf cells.
 * @param[in,out] Uintermediate The solution array for intermediate cells.
 * @param[in] iCell The current cell index.
 * @param[in] iter_space The iteration space shape.
 * @param[in] side The direction side (-1 or +1) for the neighbor cell.
 * @param[out] w The weight computed for the neighbor contribution.
 * @param[out] contrib The contribution from the neighbor cell.
 * @param[in] dir The direction index (IX, IY, or IZ).
 * @param[in] boundarycondition Boundary conditions per direction.
 */
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::get_weight_and_contrib_amr_finest(Array_t& U, Array_t& Uintermediate, const CellIndex& iCell, const Shape_t& iter_space, const int side, real_t& w, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  constexpr real_t half = 0.5;
  constexpr bool search_intermediate = true;
  CellIndex::offset_t offset = {}; offset[dir] = side;
  CellIndex iCell_X = iCell.getNeighbor_ghost(offset, iter_space);
  if ( CellIndex::SMALLER == iCell_X.status ) 
  {
    iCell_X = iCell.getNeighbor_ghost<search_intermediate>(offset, iter_space);
    contrib = Uintermediate.at(iCell_X, Isolution);
  } 
  else if (CellIndex::BIGGER == iCell_X.status) 
  {
    contrib = average_8bigger_neighbors(U, Uintermediate, iCell, iter_space, offset);
    w = half;
  } else contrib = U.at(iCell_X, Isolution);
  if ( BC_ABSORBING == boundarycondition[dir] && iCell_X.is_boundary() ) contrib = 0;
}

/**
 * @brief Compute the Laplacian operator on the finest AMR level.
 * 
 * This function computes the Laplacian operator (Poisson equation) using the multigrid method
 * at the finest adaptive mesh refinement (AMR) level. It processes both leaf cells and
 * intermediate cells, applying appropriate neighbor contributions and boundary conditions.
 * 
 * The result is stored in the residual data arrays for later use in the solver.
 * 
 * @param[in] level The AMR level to operate on.
 */
void GravitySolver_multigrid::operator_amr_finest(const level_t level) 
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  constexpr real_t zero = 0;
  constexpr real_t one = 1;
  constexpr real_t two = 2;
  const real_t factor = two/(size[IX]*size[IX]) + two/(size[IY]*size[IY]) + two/(size[IZ]*size[IZ]);

  const auto octs = get_subview_octs(level);
  foreach_cell.foreach_cell_in_octants("Residual", iter_space, octs,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    real_t laplacian_solution(zero);
    const real_t central_solution = U.at(iCell, Isolution);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t a(one), b(one), contrib_L(zero), contrib_R(zero);
      get_weight_and_contrib_amr_finest(U, Uintermediate, iCell, iter_space, -1, a, contrib_L, dir, boundarycondition);
      get_weight_and_contrib_amr_finest(U, Uintermediate, iCell, iter_space, +1, b, contrib_R, dir, boundarycondition);
      const real_t f_L = two / (a * (a + b));
      const real_t f_R = two / (b * (a + b));
      const real_t f_C = two / (a * b);
      laplacian_solution += (f_L*contrib_L + f_R*contrib_R - f_C*central_solution) / (size[dir] * size[dir]);
    }
    U.at(iCell, Iresidual) = laplacian_solution;
  });
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Residual", iter_space, octs_intermediate,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    real_t neighbors(zero);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t contrib_L(zero), contrib_R(zero);
      get_contrib_intermediate_uniform(U, Uintermediate, iCell, iter_space, -1, contrib_L, dir, boundarycondition);
      get_contrib_intermediate_uniform(U, Uintermediate, iCell, iter_space, +1, contrib_R, dir, boundarycondition);
      neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
    }
    const real_t laplacian_solution = neighbors - Uintermediate.at(iCell, Isolution) * factor;
    Uintermediate.at(iCell, Iresidual) = laplacian_solution;
  });
}

/**
 * @brief Compute the residual on the finest AMR level.
 * 
 * Calculates the residual of the Poisson equation on the finest adaptive mesh refinement (AMR) level.
 * 
 * @param[in] level The AMR level to compute the residual for.
 */
void GravitySolver_multigrid::residual_amr_finest(const level_t level) 
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  constexpr real_t zero = 0;
  constexpr real_t one = 1;
  constexpr real_t two = 2;
  const real_t factor = two/(size[IX]*size[IX]) + two/(size[IY]*size[IY]) + two/(size[IZ]*size[IZ]);

  const auto octs = get_subview_octs(level);
  foreach_cell.foreach_cell_in_octants("Residual", iter_space, octs,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    real_t laplacian_solution(zero);
    const real_t central_solution = U.at(iCell, Isolution);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t a(one), b(one), contrib_L(zero), contrib_R(zero);
      get_weight_and_contrib_amr_finest(U, Uintermediate, iCell, iter_space, -1, a, contrib_L, dir, boundarycondition);
      get_weight_and_contrib_amr_finest(U, Uintermediate, iCell, iter_space, +1, b, contrib_R, dir, boundarycondition);
      const real_t f_L = two / (a * (a + b));
      const real_t f_R = two / (b * (a + b));
      const real_t f_C = two / (a * b);
      laplacian_solution += (f_L*contrib_L + f_R*contrib_R - f_C*central_solution) / (size[dir] * size[dir]);
    }
    U.at(iCell, Iresidual) = U.at(iCell, Irhs) - laplacian_solution;
  });
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Residual", iter_space, octs_intermediate,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    real_t neighbors(zero);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t contrib_L(zero), contrib_R(zero);
      get_contrib_intermediate_uniform(U, Uintermediate, iCell, iter_space, -1, contrib_L, dir, boundarycondition);
      get_contrib_intermediate_uniform(U, Uintermediate, iCell, iter_space, +1, contrib_R, dir, boundarycondition);
      neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
    }
    const real_t laplacian_solution = neighbors - Uintermediate.at(iCell, Isolution) * factor;
    Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
  });
}

/**
 * @brief Perform a Gauss-Seidel relaxation sweep on intermediate cells at a given AMR level.
 * 
 * Updates the solution on intermediate cells of the adaptive mesh refinement (AMR) hierarchy
 * by performing a Gauss-Seidel sweep. The update uses neighboring cell contributions weighted
 * appropriately for AMR corrections, considering boundary conditions and mask values.
 * 
 * @tparam Array_t Type of the data array.
 * 
 * @param[in,out] Uintermediate The data array holding intermediate solution, residuals, masks, and RHS.
 * @param[in] iCell The cell index identifying the current cell being updated.
 * @param[in] size The physical cell size in each spatial direction for the current AMR level.
 * @param[in] boundarycondition Array specifying the boundary conditions for each spatial direction.
 */
template< typename Array_t > 
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::gauss_seidel_intermediate_amr_correction(Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  constexpr real_t zero = 0;
  const real_t mask = Uintermediate.at(iCell, Imask);
  if (mask <= zero) return;
  
  constexpr real_t one = 1;
  constexpr real_t two = 2;
  const auto iter_space = Uintermediate.getShape();
  Kokkos::Array<real_t, 3> f_C;
  real_t neighbors(zero);
  constexpr real_t w_relax(1.25);
  for ( const ComponentIndex3D dir : {IX,IY,IZ} )
  {
    real_t a(one), b(one), contrib_L(zero), contrib_R(zero);
    get_weight_and_contrib_intermediate_amr_correction(Uintermediate, iCell, iter_space, -1, a, contrib_L, dir, boundarycondition, mask);
    get_weight_and_contrib_intermediate_amr_correction(Uintermediate, iCell, iter_space, +1, b, contrib_R, dir, boundarycondition, mask);
    const real_t f_L = two / (a * (a + b));
    const real_t f_R = two / (b * (a + b));
    f_C[dir] = two / (a * b);
    neighbors += (f_L * contrib_L + f_R * contrib_R) / (size[dir] * size[dir]);                   
  }

  const real_t sol = Uintermediate.at(iCell, Isolution);
  const real_t rhs = Uintermediate.at(iCell, Irhs);
  const real_t denom =
    f_C[IX] / (size[IX] * size[IX]) +
    f_C[IY] / (size[IY] * size[IY]) +
    f_C[IZ] / (size[IZ] * size[IZ]);

  Uintermediate.at(iCell, Isolution) += w_relax * ( (neighbors - rhs) / denom - sol);
}

/**
 * @brief Perform a Gauss-Seidel relaxation sweep on leaf cells at the finest AMR level.
 * 
 * Updates the solution (e.g., gravitational potential) on leaf cells of the adaptive mesh refinement (AMR)
 * hierarchy at the finest level using a Gauss-Seidel sweep. The update accounts for neighboring cell values,
 * boundary conditions, and relaxation weighting.
 * 
 * @tparam Array_t Type of the array containing solution and auxiliary data.
 * 
 * @param[in,out] U The array holding solution, residuals, right-hand side, and related data on leaf cells.
 * @param[in] Uintermediate The array holding intermediate-level data.
 * @param[in] iCell The cell index identifying the current leaf cell being updated.
 * @param[in] size The physical cell size in each spatial direction at the current AMR level.
 * @param[in] boundarycondition Array specifying the boundary conditions for each spatial direction.
 */
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::gauss_seidel_leaves_amr_finest(Array_t& U, Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  const auto iter_space = Uintermediate.getShape();
  Kokkos::Array<real_t, 3> f_C;
  constexpr real_t w_relax(1.25);
  constexpr real_t zero = 0;
  constexpr real_t one = 1;
  constexpr real_t two = 2;
  real_t neighbors(zero);
  
  for ( const ComponentIndex3D dir : {IX,IY,IZ} )
  {
    real_t a(one), b(one), contrib_L(zero), contrib_R(zero);
    get_weight_and_contrib_amr_finest(U, Uintermediate, iCell, iter_space, -1, a, contrib_L, dir, boundarycondition);
    get_weight_and_contrib_amr_finest(U, Uintermediate, iCell, iter_space, +1, b, contrib_R, dir, boundarycondition);
    const real_t f_L = two / (a * (a + b));
    const real_t f_R = two / (b * (a + b));
    f_C[dir] = two / (a * b);
    neighbors += (f_L * contrib_L + f_R * contrib_R) / (size[dir] * size[dir]);             
  }

  const real_t rhs = U.at(iCell, Irhs);
  const real_t sol = U.at(iCell, Isolution);
  const real_t denom =
    f_C[IX] / (size[IX] * size[IX]) +
    f_C[IY] / (size[IY] * size[IY]) +
    f_C[IZ] / (size[IZ] * size[IZ]);

  U.at(iCell, Isolution) += w_relax * ( (neighbors - rhs) / denom - sol);
}

/**
 * @brief Perform a Gauss-Seidel relaxation sweep on leaf cells on a uniform grid.
 * 
 * Applies a Gauss-Seidel update to the solution on leaf cells assuming a uniform grid spacing.
 * This function computes the Laplacian contribution from neighboring cells and updates the
 * solution with a relaxation factor.
 * 
 * @tparam Array_t Type of the array holding solution and related data.
 * 
 * @param[in,out] U The array containing the solution, right-hand side, and residual data for leaf cells.
 * @param[in] Uintermediate The array with intermediate data for corrections or coupling.
 * @param[in] iCell The index of the current leaf cell being updated.
 * @param[in] size The cell size in each spatial direction (uniform grid spacing).
 * @param[in] boundarycondition Boundary conditions applied in each spatial direction.
 */
template< typename Array_t > 
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::gauss_seidel_leaves_uniform(Array_t& U, Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  const auto iter_space = Uintermediate.getShape();
  constexpr real_t w_relax(1.25);
  constexpr real_t zero = 0;
  constexpr real_t two = 2;
  real_t neighbors(zero);

  for ( const ComponentIndex3D dir : {IX,IY,IZ} )
  {
    real_t contrib_L(zero), contrib_R(zero);
    get_contrib_leaves_uniform(U, Uintermediate, iCell, iter_space, -1, contrib_L, dir, boundarycondition);
    get_contrib_leaves_uniform(U, Uintermediate, iCell, iter_space, +1, contrib_R, dir, boundarycondition);
    neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);             
  }
  const real_t rhs = U.at(iCell, Irhs);
  const real_t sol = U.at(iCell, Isolution);
  const real_t denom =
    two / (size[IX] * size[IX]) +
    two / (size[IY] * size[IY]) +
    two / (size[IZ] * size[IZ]);

  U.at(iCell, Isolution) += w_relax * ( (neighbors - rhs) / denom - sol);
}

/**
 * @brief Perform a Gauss-Seidel relaxation sweep on intermediate cells on a uniform grid.
 * 
 * Updates the solution on intermediate cells using Gauss-Seidel iteration with uniform grid spacing.
 * The function computes contributions from neighboring cells and applies a relaxation factor.
 * 
 * @tparam Array_t Type of the array holding solution and related data.
 * 
 * @param[in] U The primary data array (read-only here).
 * @param[in,out] Uintermediate The array containing intermediate-level data.
 * @param[in] iCell The index of the current intermediate cell being updated.
 * @param[in] size The cell size in each spatial direction (uniform grid spacing).
 * @param[in] boundarycondition Boundary conditions applied in each spatial direction.
 */
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::gauss_seidel_intermediate(Array_t& U, Array_t& Uintermediate, const CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  const auto iter_space = Uintermediate.getShape();
  constexpr real_t zero = 0;
  constexpr real_t two = 2;
  real_t neighbors(zero);
  constexpr real_t w_relax(1.25);
  for ( ComponentIndex3D dir : {IX,IY,IZ} )
  {
    real_t contrib_L(zero), contrib_R(zero);
    get_contrib_intermediate_uniform(U, Uintermediate, iCell, iter_space, -1, contrib_L, dir, boundarycondition);
    get_contrib_intermediate_uniform(U, Uintermediate, iCell, iter_space, +1, contrib_R, dir, boundarycondition);
    neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
  }
  const real_t rhs = Uintermediate.at(iCell, Irhs);
  const real_t sol = Uintermediate.at(iCell, Isolution);
  const real_t denom =
    two / (size[IX] * size[IX]) +
    two / (size[IY] * size[IY]) +
    two / (size[IZ] * size[IZ]);
  Uintermediate.at(iCell, Isolution) += w_relax * ( (neighbors - rhs) / denom - sol );
}

/**
 * @brief Perform a Gauss-Seidel sweep on intermediate cells at a given AMR level using red-black ordering.
 * 
 * Applies the Gauss-Seidel relaxation on the intermediate correction cells of an AMR grid level.
 * The function executes either on red or black cells depending on the template parameter.
 * This red-black ordering helps with parallelism and convergence in iterative solvers.
 * 
 * @tparam is_red Compile-time flag to select red (true) or black (false) cells.
 * 
 * @param[in] level The AMR grid level at which to perform the sweep.
 */
template<bool is_red>
void GravitySolver_multigrid::gauss_seidel_intermediate_amr_correction_rb(const level_t level) 
{
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  const auto octs_intermediate = get_subview_octs_intermediate(level);

  auto apply_gauss_seidel = KOKKOS_LAMBDA(const CellIndex& iCell) 
  {
    gauss_seidel_intermediate_amr_correction(Uintermediate, iCell, size, boundarycondition);
  };

  if constexpr (is_red) 
    foreach_cell.foreach_intermediate_red_cell_in_octants("Gauss-Seidel", iter_space, octs_intermediate, apply_gauss_seidel);
  else 
    foreach_cell.foreach_intermediate_black_cell_in_octants("Gauss-Seidel", iter_space, octs_intermediate, apply_gauss_seidel);
}

/**
 * @brief Perform a Gauss-Seidel sweep on leaf cells at the finest AMR level using red-black ordering.
 * 
 * Executes the Gauss-Seidel relaxation on leaf cells of the finest AMR grid level.
 * The sweep is performed either on red or black cells depending on the template parameter.
 * Red-black ordering improves parallelism and convergence in iterative solvers.
 * 
 * @tparam is_red Compile-time flag to select red (true) or black (false) cells.
 * 
 * @param[in] level The AMR grid level at which to perform the sweep.
 */
template< bool is_red >
void GravitySolver_multigrid::gauss_seidel_leaves_amr_finest_rb(const level_t level) 
{
  auto& U = pdata->U;
  const auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  const auto octs = get_subview_octs(level);

  auto apply_gauss_seidel = KOKKOS_LAMBDA(const CellIndex& iCell) 
  {
    gauss_seidel_leaves_amr_finest(U, Uintermediate, iCell, size, boundarycondition);
  };

  if constexpr (is_red) 
    foreach_cell.foreach_red_cell_in_octants("Gauss-Seidel", iter_space, octs, apply_gauss_seidel);
  else 
    foreach_cell.foreach_black_cell_in_octants("Gauss-Seidel", iter_space, octs, apply_gauss_seidel);
}

/**
 * @brief Perform a Gauss-Seidel sweep on leaf cells on a uniform grid with red-black ordering.
 * 
 * Executes the Gauss-Seidel relaxation on leaf cells of a uniform grid at a specified AMR level.
 * The sweep is done on either red or black cells based on the template parameter.
 * This red-black ordering enhances parallel efficiency and convergence properties.
 * 
 * @tparam is_red Compile-time boolean indicating whether to process red cells (true) or black cells (false).
 * 
 * @param[in] level The AMR grid level to perform the sweep on.
 */
template< bool is_red >
void GravitySolver_multigrid::gauss_seidel_leaves_uniform_rb(const level_t level) 
{
  auto& U = pdata->U;
  const auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  const auto octs = get_subview_octs(level);

  auto apply_gauss_seidel = KOKKOS_LAMBDA(const CellIndex& iCell) 
  {
    gauss_seidel_leaves_uniform(U, Uintermediate, iCell, size, boundarycondition);
  };

  if constexpr (is_red) 
    foreach_cell.foreach_red_cell_in_octants("Gauss-Seidel", iter_space, octs, apply_gauss_seidel);
  else 
    foreach_cell.foreach_black_cell_in_octants("Gauss-Seidel", iter_space, octs, apply_gauss_seidel);
}

/**
 * @brief Perform a Gauss-Seidel sweep on intermediate cells with red-black ordering.
 * 
 * Executes the Gauss-Seidel relaxation on intermediate cells at a specified AMR level.
 * The sweep is applied to either red or black intermediate cells depending on the template parameter.
 * This red-black coloring helps improve convergence and parallel performance.
 * 
 * @tparam is_red Compile-time boolean indicating whether to operate on red cells (true) or black cells (false).
 * 
 * @param[in] level The AMR grid level on which to perform the sweep.
 */
template< bool is_red >
void GravitySolver_multigrid::gauss_seidel_intermediate_rb(const level_t level) 
{
  const auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  const auto octs_intermediate = get_subview_octs_intermediate(level);

  auto apply_gauss_seidel = KOKKOS_LAMBDA(const CellIndex& iCell) 
  {
    gauss_seidel_intermediate(U, Uintermediate, iCell, size, boundarycondition);
  };

  if constexpr (is_red) 
    foreach_cell.foreach_intermediate_red_cell_in_octants("Gauss-Seidel", iter_space, octs_intermediate, apply_gauss_seidel);
  else 
    foreach_cell.foreach_intermediate_black_cell_in_octants("Gauss-Seidel", iter_space, octs_intermediate, apply_gauss_seidel);
}

/**
 * @brief Perform multiple Gauss-Seidel smoothing iterations on intermediate cells at an AMR level.
 * 
 * Applies red-black Gauss-Seidel sweeps to correct the intermediate solution at the specified AMR level.
 * Supports distributed MPI execution by exchanging ghost cell data after each half sweep when required.
 * 
 * @param[in] U_ Reference to the user data containing solution arrays.
 * @param[in] nIterations Number of smoothing iterations to perform.
 * @param[in] level The AMR grid level to smooth.
 * @param[in] ghost_comm Ghost communicator used for MPI ghost data exchanges.
 */
void GravitySolver_multigrid::smoothing_intermediate_amr_correction(UserData& U_, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpicomm = foreach_cell.get_amr_mesh().getMpiComm();
  const bool isDistributed = mpicomm.MPI_Comm_size() > 1;
  const bool isMPILevel = isDistributed && (level >= pdata->first_mpi_multigrid_level);
  constexpr bool is_red = true;
  constexpr bool is_black = false;
  for (uint32_t i = 0; i < nIterations; i++) 
  {
    gauss_seidel_intermediate_amr_correction_rb<is_red>(level);
    if (isMPILevel) exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, level, {"solution"}, ghost_comm);

    gauss_seidel_intermediate_amr_correction_rb<is_black>(level);
    if (isMPILevel && i < nIterations - 1) exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, level, {"solution"}, ghost_comm);
  }
}

/**
 * @brief Perform Gauss-Seidel smoothing on uniform grid cells at a specified AMR level.
 * 
 * This function applies red-black Gauss-Seidel sweeps to either leaf cells, intermediate cells, 
 * or both on a uniform grid at the given level. Supports distributed MPI execution with ghost 
 * data exchanges after each sweep where necessary.
 * 
 * @tparam target Specifies the target cells to smooth: LEAVES, INTERMEDIATES, or BOTH.
 * @param[in] U_ Reference to user data containing solution arrays.
 * @param[in] nIterations Number of smoothing iterations to perform.
 * @param[in] level The AMR grid level to smooth.
 * @param[in] ghost_comm Ghost communicator for MPI ghost data exchange.
 */
template <Target target>
void GravitySolver_multigrid::smoothing_uniform(UserData& U_, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm) 
{
  if constexpr (target > Target::BOTH)
    static_assert(target <= Target::BOTH, "Smoothing uniform can only support LEAVES, INTERMEDIATES or BOTH targets");  
  
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpicomm = foreach_cell.get_amr_mesh().getMpiComm();
  const bool isDistributed = mpicomm.MPI_Comm_size() > 1;
  const bool isMPILevel = isDistributed && (level >= pdata->first_mpi_multigrid_level);
  constexpr bool is_red = true;
  constexpr bool is_black = false;
  for (uint32_t i = 0; i < nIterations; i++) 
  {
    if constexpr (target == Target::LEAVES || target == Target::BOTH)
      gauss_seidel_leaves_uniform_rb<is_red>(level);
    if constexpr (target == Target::INTERMEDIATES || target == Target::BOTH)
      gauss_seidel_intermediate_rb<is_red>(level);

    if (isMPILevel) exchange_ghosts_at_level<target>(U_, level, {"solution"}, ghost_comm);

    if constexpr (target == Target::LEAVES || target == Target::BOTH)
      gauss_seidel_leaves_uniform_rb<is_black>(level);
    if constexpr (target == Target::INTERMEDIATES || target == Target::BOTH)
      gauss_seidel_intermediate_rb<is_black>(level);

    if (isMPILevel && i < nIterations - 1) exchange_ghosts_at_level<target>(U_, level, {"solution"}, ghost_comm);
  }
}

/**
 * @brief Perform Gauss-Seidel smoothing on leaf and intermediate cells at the finest AMR level.
 * 
 * Applies red-black Gauss-Seidel sweeps on leaf cells (finest AMR level) and intermediate cells.
 * Supports distributed MPI execution by exchanging ghost cell data after sweeps as needed.
 * 
 * @param[in] U_ Reference to user data containing solution arrays.
 * @param[in] nIterations Number of smoothing iterations to perform.
 * @param[in] level The AMR grid level to smooth.
 * @param[in] ghost_comm Ghost communicator for MPI ghost data exchange.
 */
void GravitySolver_multigrid::smoothing_amr_finest(UserData& U_, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpicomm = foreach_cell.get_amr_mesh().getMpiComm();
  const bool isDistributed = mpicomm.MPI_Comm_size() > 1;
  const bool isMPILevel = isDistributed && (level >= pdata->first_mpi_multigrid_level);
  constexpr bool is_red = true;
  constexpr bool is_black = false;
  for (uint32_t i = 0; i < nIterations; i++) 
  {
    gauss_seidel_leaves_amr_finest_rb<is_red>(level);
    gauss_seidel_intermediate_rb<is_red>(level);

    if (isMPILevel) exchange_ghosts_at_level<Target::BOTH>(U_, level, {"solution"}, ghost_comm);

    gauss_seidel_leaves_amr_finest_rb<is_black>(level);
    gauss_seidel_intermediate_rb<is_black>(level);

    if (isMPILevel && i < nIterations - 1) exchange_ghosts_at_level<Target::BOTH>(U_, level, {"solution"}, ghost_comm);
  }
}

/**
 * @brief Perform a multigrid V-cycle smoothing on a uniform grid at a given level.
 * 
 * This function recursively applies pre- and post-smoothing steps, computes residuals,
 * performs restriction and prolongation operations between levels, and handles MPI ghost
 * data exchanges for distributed memory parallelism. The behavior adapts depending on whether
 * the current level is the coarsest level or not.
 * 
 * @param[in] U_ Reference to the user data containing solution and residual arrays.
 * @param[in] level The current multigrid level on which to perform the V-cycle.
 * @param[in] ghost_comm_minimal Ghost communicator for minimal data exchanges during smoothing and residual calculation.
 * @param[in] ghost_comm_blockwide Ghost communicator for block-wide data exchanges at coarser levels.
 */
void GravitySolver_multigrid::V_cycle_uniform(UserData& U_, const level_t level, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide) 
{  
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpicomm = foreach_cell.get_amr_mesh().getMpiComm();
  const bool isDistributed = mpicomm.MPI_Comm_size() > 1;
  const bool isFirstMPILevel = isDistributed && (level == pdata->first_mpi_multigrid_level);
  const bool isMPILevel = isDistributed && (level >= pdata->first_mpi_multigrid_level);
  
  if (level == pdata->level_coarse)
  {
    smoothing_uniform<Target::BOTH>(U_, pdata->Npre, level, ghost_comm_minimal);
    if (isMPILevel) exchange_ghosts_at_level<Target::BOTH>(U_, level, {"solution"}, ghost_comm_minimal);
    residual_uniform<Target::BOTH>(level);
    if (isMPILevel) exchange_ghosts_at_level<Target::LEAVES>(U_, level, {"res"}, ghost_comm_minimal);
  } 
  else 
  {
    smoothing_uniform<Target::INTERMEDIATES>(U_, pdata->Npre, level, ghost_comm_minimal);
    if (isMPILevel) exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, level, {"solution"}, ghost_comm_minimal);
    residual_uniform<Target::INTERMEDIATES>(level);
  }
  if (isMPILevel) exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, level, {"res"}, ghost_comm_minimal);  

  if (level == pdata->level_coarse) 
    restriction_from_children<Target::BOTH>(level - 1, level, kokkos_array<int>(Iresidual), kokkos_array<int>(Irhs));
  else
    restriction_from_children<Target::INTERMEDIATES>(level - 1, level, kokkos_array<int>(Iresidual), kokkos_array<int>(Irhs));

  if (isFirstMPILevel) 
    reduce_nonMPI_levels(pdata->first_mpi_multigrid_level - 1, kokkos_array<int>(Irhs));
  else if (isMPILevel)
  {
    reduce_ghosts_at_level<Target::INTERMEDIATES>(U_, level - 1, {"rhs"}, ghost_comm_blockwide);
    exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, level - 1, {"rhs"}, ghost_comm_minimal);
  }

  initialise_lhs_from_rhs<Target::ALL_INTERMEDIATES>(level - 1);

  if (level == 1) 
  {
    smoothing_uniform<Target::INTERMEDIATES>(U_, pdata->Npre, level - 1, ghost_comm_minimal);
    if (isMPILevel) exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, level - 1, {"solution"}, ghost_comm_blockwide);
  }
  else V_cycle_uniform(U_, level - 1, ghost_comm_minimal, ghost_comm_blockwide);
    
  if (level == pdata->level_coarse)
    prolongation_from_children<Target::BOTH>(level);
  else 
    prolongation_from_children<Target::INTERMEDIATES>(level);

  if (isMPILevel) exchange_ghosts_at_level<Target::BOTH>(U_, level, {"solution"}, ghost_comm_minimal);

  if (level == pdata->level_coarse)
  {
    smoothing_uniform<Target::BOTH>(U_, pdata->Npost, level, ghost_comm_minimal);
    if (isMPILevel) exchange_ghosts_at_level<Target::LEAVES>(U_, level, {"solution"}, ghost_comm_minimal);
  } 
  else 
    smoothing_uniform<Target::INTERMEDIATES>(U_, pdata->Npost, level, ghost_comm_minimal);
  
  if (isMPILevel) exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, level, {"solution"}, ghost_comm_blockwide);
}


/**
 * @brief Perform a multigrid V-cycle on an Adaptive Mesh Refinement (AMR) hierarchy.
 * 
 * This function recursively applies pre- and post-smoothing steps, computes residuals,
 * performs restriction and prolongation operations between AMR levels, and handles MPI ghost
 * data exchanges for distributed memory parallelism. The behavior differs depending on whether
 * the current level is the finest AMR level or an intermediate level.
 * 
 * @param[in] U_ Reference to the user data containing solution and residual arrays.
 * @param[in] current_level The current multigrid AMR level to process.
 * @param[in] finest_level The finest AMR level in the hierarchy.
 * @param[in] ghost_comm_minimal Ghost communicator for minimal data exchanges during smoothing and residual calculation.
 * @param[in] ghost_comm_blockwide Ghost communicator for block-wide data exchanges at coarser AMR levels.
 */
void GravitySolver_multigrid::V_cycle_amr(UserData& U_, const level_t current_level, const level_t finest_level, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide) 
{  
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpicomm = foreach_cell.get_amr_mesh().getMpiComm();
  const bool isDistributed = mpicomm.MPI_Comm_size() > 1;
  const bool isFirstMPILevel = isDistributed && (current_level == pdata->first_mpi_multigrid_level);
  const bool isMPILevel = isDistributed && (current_level >= pdata->first_mpi_multigrid_level);

  // Full Multigrid

  if ( current_level == finest_level ) 
  {
    smoothing_amr_finest(U_, pdata->Npre, current_level, ghost_comm_minimal);
    if (isMPILevel) exchange_ghosts_at_level<Target::BOTH>(U_, current_level, {"solution"}, ghost_comm_minimal);
    residual_amr_finest(current_level);
    if (isMPILevel) exchange_ghosts_at_level<Target::LEAVES>(U_, current_level, {"res"}, ghost_comm_minimal);
  } 
  else 
  {
    smoothing_intermediate_amr_correction(U_, pdata->Npre, current_level, ghost_comm_minimal);
    if (isMPILevel) exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, current_level, {"solution"}, ghost_comm_minimal);
    residual_intermediate_amr_correction(current_level);
  }
  if (isMPILevel) exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, current_level, {"res"}, ghost_comm_minimal);
  
  if ( current_level == finest_level ) 
    restriction_from_children<Target::BOTH>(current_level - 1, current_level, kokkos_array<int>(Iresidual), kokkos_array<int>(Irhs));
  else
    restriction_from_children<Target::INTERMEDIATES>(current_level - 1, current_level, kokkos_array<int>(Iresidual), kokkos_array<int>(Irhs));


  if (isFirstMPILevel) 
    reduce_nonMPI_levels(pdata->first_mpi_multigrid_level - 1, kokkos_array<int>(Irhs)); 
  else if (isMPILevel)
  {
    reduce_ghosts_at_level<Target::INTERMEDIATES>(U_, current_level - 1, {"rhs"}, ghost_comm_blockwide);
    exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, current_level - 1, {"rhs"}, ghost_comm_minimal);
  }

  initialise_lhs_from_rhs<Target::ALL_INTERMEDIATES>(current_level - 1);

  if ( std::max(0, finest_level - 3) == current_level ) 
  { // TODO: finest - 2 seems to works aswell for spherical symmetry. Check for more realistic cases
    smoothing_intermediate_amr_correction(U_, pdata->Npre, current_level - 1, ghost_comm_minimal);
    if (isMPILevel) exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, current_level - 1, {"solution"}, ghost_comm_blockwide);
  } else V_cycle_amr(U_, current_level - 1, finest_level, ghost_comm_minimal, ghost_comm_blockwide); 
  
  if ( current_level == finest_level ) 
  {
    prolongation_from_children<Target::BOTH>(current_level);
    if (isMPILevel) exchange_ghosts_at_level<Target::BOTH>(U_, current_level, {"solution"}, ghost_comm_minimal);
    smoothing_amr_finest(U_, pdata->Npost, current_level, ghost_comm_minimal);
    if (isMPILevel) exchange_ghosts_at_level<Target::LEAVES>(U_, current_level, {"solution"}, ghost_comm_minimal);
  } 
  else 
  {
    prolongation_from_children<Target::INTERMEDIATES>(current_level);
    if (isMPILevel) exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, current_level, {"solution"}, ghost_comm_minimal);
    smoothing_intermediate_amr_correction(U_, pdata->Npost, current_level, ghost_comm_minimal);
  }   
  if (isMPILevel) exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, current_level, {"solution"}, ghost_comm_blockwide); 
}

/**
 * @brief Count number of octs before a given level.
 * 
 * Computes the total number of octs before a given level.
 * Example values:
 * - nbOcts_before_level(0) = 0
 * - nbOcts_before_level(1) = 1
 * - nbOcts_before_level(2) = 9
 * 
 * @param[in] level The level for which to count the octs before it.
 * @return uint32_t The total number of octs before the specified level.
 */
KOKKOS_INLINE_FUNCTION
uint32_t nbOcts_before_level(const uint32_t level)
{
  return ((1U << (3*level)) - 1) / 7;
}

/**
 * @brief Right hand term of the Poisson equation in the non-cosmological case.
 * 
 * The right hand side of the Poisson equation for gravity in the non-cosmological case
 * is 4 * pi * G * rho.
 * 
 * @param[in] Uin Array to read the density from.
 * @param[in] iCell_Uin Cell index where to read the density.
 * @param[in] rho_mean Average value of rho in the box for periodic cases.
 * @param[in] four_Pi_G Value of 4 * pi * G.
 * @return real_t The right-hand side term value at the given cell.
 */
KOKKOS_INLINE_FUNCTION
real_t GravitySolver_multigrid::b(const UserData::FieldAccessor& Uin, const CellIndex& iCell_Uin, real_t rho_mean, real_t four_Pi_G)
{
  return four_Pi_G*(Uin.at(iCell_Uin, Irho)-rho_mean);
}

/**
 * @brief Right hand term of the Poisson equation in the cosmological case.
 * 
 * The right hand side of the Poisson equation for gravity in cosmological cases.
 * This expression comes from Martel & Shapiro 1998, eq. (38).
 * 
 * @param[in] Uin Array to read the density from.
 * @param[in] iCell_Uin Cell index where to read the density.
 * @param[in] rho_mean Average value of rho in the box for periodic cases.
 * @param[in] aexp Expansion factor.
 * @return real_t The right-hand side term value at the given cell.
 */
KOKKOS_INLINE_FUNCTION
real_t GravitySolver_multigrid::b_cosmo(const UserData::FieldAccessor& Uin, const CellIndex& iCell_Uin, real_t rho_mean, real_t aexp)
{
  return 6.0*aexp*(Uin.at(iCell_Uin, Irho)/rho_mean-1.0);
}


real_t GravitySolver_multigrid::MPI_Allreduce_scalar( real_t local_v )
{
  real_t res;
  MPI_Allreduce( &local_v, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  return res;
}

uint32_t GravitySolver_multigrid::MPI_Allreduce_int_max( uint32_t local_v )
{
  uint32_t res;
  MPI_Allreduce( &local_v, &res, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
  return res;
}

/**
 * @brief Reduce data on non-MPI multigrid levels by performing MPI_Allreduce sums.
 * 
 * This function performs a reduction operation on data fields over non-MPI multigrid levels,
 * combining data across processes where applicable.
 * 
 * @tparam T Type of elements in the field array.
 * @tparam N Number of fields to reduce.
 * @param[in] first_nonMPI_multigrid_level The first multigrid level without MPI distribution.
 * @param[in] iFields Array of field indices to be reduced.
 */
template< typename T, size_t N >
void GravitySolver_multigrid::reduce_nonMPI_levels(const level_t first_nonMPI_multigrid_level, const Kokkos::Array<T, N>& iFields) 
{
  constexpr uint32_t num_vars = N; // number of vars for each cell

  auto& Uintermediate = pdata->Uintermediate;
  ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpi_comm = foreach_cell.get_amr_mesh().getMpiComm();
  const LightOctree& lmesh = foreach_cell.get_amr_mesh().getLightOctree();
  const auto iter_space = Uintermediate.getShape();
  const uint32_t bx=iter_space.bx, by=iter_space.by, bz=iter_space.bz ;
  const uint32_t nbCellsPerBlock = bx * by * bz;
  const uint32_t ncells_1d = 1U << first_nonMPI_multigrid_level; // Total number of octants at level (first_mpi_multigrid_level - 1)
  const uint32_t nbOcts = ncells_1d*ncells_1d*ncells_1d;
  Kokkos::View<real_t*> deviceArray("deviceArray", nbOcts * nbCellsPerBlock * num_vars);

  Kokkos::parallel_for( "Reduce non-MPI levels", Kokkos::RangePolicy<>(0, nbOcts * nbCellsPerBlock * num_vars),
    KOKKOS_LAMBDA( const uint32_t index )
  {
    const uint32_t idx = index / num_vars;
    const uint32_t ivar = index % num_vars;
    const uint32_t iOct = idx / nbCellsPerBlock;
    const uint32_t iz = iOct/(ncells_1d*ncells_1d);
    const uint32_t iy = (iOct - iz*ncells_1d*ncells_1d)/ncells_1d;
    const uint32_t ix = iOct - iy*ncells_1d - iz*ncells_1d*ncells_1d;
    const auto iOct_cell = lmesh.getiOctIntermediateFromCoordinates(ix, iy, iz, first_nonMPI_multigrid_level);

    // Find cell indices
    const uint32_t index_local = idx%nbCellsPerBlock;
    const uint32_t k = index_local/(bx*by);
    const uint32_t j = (index_local - k*bx*by)/bx;
    const uint32_t i = index_local - j*bx - k*bx*by;

    // Finalise
    const CellIndex iCell {iOct_cell, i, j, k, bx, by, bz};
    deviceArray(index) = Uintermediate.at(iCell, iFields[ivar]);
  });

  #ifdef MPI_IS_CUDA_AWARE 
    Kokkos::fence();
    mpi_comm.MPI_Allreduce(deviceArray.data(), deviceArray.data(), nbOcts * nbCellsPerBlock * num_vars, MpiComm::MPI_Op_t::SUM);
    Kokkos::fence();
  #else
    auto hostArray = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), deviceArray);
    mpi_comm.MPI_Allreduce(hostArray.data(), hostArray.data(), nbOcts * nbCellsPerBlock * num_vars, MpiComm::MPI_Op_t::SUM);
    Kokkos::deep_copy(deviceArray, hostArray);
  #endif

  Kokkos::parallel_for( "Reduce non-MPI levels", Kokkos::RangePolicy<>(0, nbOcts * nbCellsPerBlock * num_vars),
    KOKKOS_LAMBDA( const uint32_t index )
  {
    const uint32_t idx = index / num_vars;
    const uint32_t ivar = index % num_vars;
    const uint32_t iOct = idx / nbCellsPerBlock;
    const uint32_t iz = iOct/(ncells_1d*ncells_1d);
    const uint32_t iy = (iOct - iz*ncells_1d*ncells_1d)/ncells_1d;
    const uint32_t ix = iOct - iy*ncells_1d - iz*ncells_1d*ncells_1d;
    auto iOct_cell = lmesh.getiOctIntermediateFromCoordinates(ix, iy, iz, first_nonMPI_multigrid_level);

    // Find cell indices
    const uint32_t index_local = idx%nbCellsPerBlock;
    const uint32_t k = index_local/(bx*by);
    const uint32_t j = (index_local - k*bx*by)/bx;
    const uint32_t i = index_local - j*bx - k*bx*by;

    // Finalise
    const CellIndex iCell {iOct_cell, i, j, k, bx,by,bz, CellIndex::LOCAL_TO_BLOCK};
    Uintermediate.at(iCell, iFields[ivar]) = deviceArray(index);
  });
}

/**
 * @brief Sort and count Leaf and Intermediate (+ ghosts) octants per AMR level.
 * 
 * This function sort and counts leaf and intermediate octants,
 * ghosts, and intermediate ghosts per level in the adaptive mesh refinement (AMR) mesh.
 * The counts are stored in pdata arrays for later use.
 * 
 * @param[in] lmesh The LightOctree object representing the AMR mesh.
 */
void GravitySolver_multigrid::sort_octants_per_level(const LightOctree& lmesh) 
{
  const level_t level_max = lmesh.get_level_max();
  const level_t nlevel = level_max + 1;
  // Count number of octs and ghosts in AMR per level
  const uint32_t numOcts = lmesh.getNumOctants();
  const uint32_t numIntermediateOcts = lmesh.getNumIntermediateOctants();
  const uint32_t numGhosts = lmesh.getNumGhosts();
  const uint32_t numIntermediateGhosts = lmesh.getNumIntermediateGhosts();


  Kokkos::View<uint32_t*> octs_per_level("octs_per_level", numOcts);
  Kokkos::View<uint32_t*> octs_intermediate_per_level("octs_intermediate_per_level", numIntermediateOcts);
  Kokkos::View<uint32_t*> ghosts_per_level("ghosts_per_level", numGhosts);
  Kokkos::View<uint32_t*> ghosts_intermediate_per_level("ghosts_intermediate_per_level", numIntermediateGhosts);

  uint32_t numOcts_tmp(0), numIntermediateOcts_tmp(0), numGhosts_tmp(0), numIntermediateGhosts_tmp(0);
  for (level_t ilevel = 0; ilevel < nlevel; ilevel++) 
  {
    uint32_t numOcts_local(0), numIntermediateOcts_local(0), numGhosts_local(0), numIntermediateGhosts_local(0);
    Kokkos::parallel_scan( "Count number of octs per level", Kokkos::RangePolicy<>(0, numOcts),
      KOKKOS_LAMBDA( const uint32_t iOct, uint32_t& ilist, bool final )
    {
      const level_t level = lmesh.getLevel({iOct, false, false});
      if (level == ilevel)
      {
        if (final) octs_per_level(ilist + numOcts_tmp) = iOct;
        ilist++;
      }
    }, numOcts_local);
    Kokkos::parallel_scan( "Count number of intermediate octs per level", Kokkos::RangePolicy<>(0, numIntermediateOcts),
      KOKKOS_LAMBDA( const uint32_t iOct, uint32_t& ilist, bool final )
    {
      const level_t level = lmesh.getLevel({iOct, false, true});
      if (level == ilevel)
      {
        if (final) octs_intermediate_per_level(ilist + numIntermediateOcts_tmp) = iOct;
        ilist++;
      }
    }, numIntermediateOcts_local);
    Kokkos::parallel_scan( "Count number of ghosts per level", Kokkos::RangePolicy<>(0, numGhosts),
      KOKKOS_LAMBDA( const uint32_t iOct, uint32_t& ilist, bool final )
    {
      const level_t level = lmesh.getLevel({iOct, true, false});
      if (level == ilevel)
      {
        if (final) ghosts_per_level(ilist + numGhosts_tmp) = iOct;
        ilist++;
      }
    }, numGhosts_local);
    Kokkos::parallel_scan( "Count number of intermediate ghosts per level", Kokkos::RangePolicy<>(0, numIntermediateGhosts),
      KOKKOS_LAMBDA( const uint32_t iOct, uint32_t& ilist, bool final )
    {
      const level_t level = lmesh.getLevel({iOct, true, true});
      if (level == ilevel)
      {
        if (final) ghosts_intermediate_per_level(ilist + numIntermediateGhosts_tmp) = iOct;
        ilist++;
      }
    }, numIntermediateGhosts_local);
    numOcts_tmp += numOcts_local;
    numIntermediateOcts_tmp += numIntermediateOcts_local;
    numGhosts_tmp += numGhosts_local;
    numIntermediateGhosts_tmp += numIntermediateGhosts_local;
  }
  DYABLO_ASSERT_KOKKOS_DEBUG( numOcts_tmp == numOcts, "Number of octants per level is not correct");
  DYABLO_ASSERT_KOKKOS_DEBUG( numIntermediateOcts_tmp == numIntermediateOcts, "Number of intermediate octants per level is not correct");
  DYABLO_ASSERT_KOKKOS_DEBUG( numGhosts_tmp == numGhosts, "Number of ghosts per level is not correct");
  DYABLO_ASSERT_KOKKOS_DEBUG( numIntermediateGhosts_tmp == numIntermediateGhosts, "Number of intermediate ghosts per level is not correct");
  pdata->octs_per_level = octs_per_level;
  pdata->octs_intermediate_per_level = octs_intermediate_per_level;
  pdata->ghosts_per_level = ghosts_per_level;
  pdata->ghosts_intermediate_per_level = ghosts_intermediate_per_level;
}

/**
 * @brief Count the number of Leaf and Intermediate (+ ghosts) octants per AMR level.
 * 
 * This function counts how many octants (leaf, intermediate, ghosts, intermediate ghosts)
 * exist per level in the adaptive mesh refinement (AMR) mesh. It uses Kokkos parallel
 * operations to perform atomic increments of counters per level and prints out the counts.
 * 
 * @param[in] lmesh The LightOctree object representing the AMR mesh.
 */
void GravitySolver_multigrid::count_octants_per_level(const LightOctree& lmesh) 
// Count number of Leaf and Intermediate (+ ghosts) octs in AMR
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpi_comm = foreach_cell.get_amr_mesh().getMpiComm();
  const int mpi_rank = mpi_comm.MPI_Comm_rank(); 
  const level_t level_max = lmesh.get_level_max();
  const level_t nlevels = level_max + 1;
  Kokkos::View<uint32_t*> octs_per_level_count("octs_per_level", nlevels+1);
  Kokkos::View<uint32_t*> octs_intermediate_per_level_count("octs_intermediate_per_level", nlevels+1);
  Kokkos::View<uint32_t*> ghosts_per_level_count("ghosts_per_level", nlevels+1);
  Kokkos::View<uint32_t*> ghosts_intermediate_per_level_count("ghosts_intermediate_per_level", nlevels+1);
  Kokkos::deep_copy(octs_per_level_count, 0);
  Kokkos::deep_copy(octs_intermediate_per_level_count, 0);
  Kokkos::deep_copy(ghosts_per_level_count, 0);
  Kokkos::deep_copy(ghosts_intermediate_per_level_count, 0);


  // Count number of octs and ghosts in AMR per level
  const int numOcts = lmesh.getNumOctants();
  const int numIntermediateOcts = lmesh.getNumIntermediateOctants();
  const int numGhosts = lmesh.getNumGhosts();
  const int numIntermediateGhosts = lmesh.getNumIntermediateGhosts();
  Kokkos::parallel_for( "Count number of octs per level", Kokkos::RangePolicy<>(0, numOcts),
    KOKKOS_LAMBDA( const uint32_t iOct )
  {
    const level_t level = lmesh.getLevel({iOct, false, false});
    Kokkos::atomic_fetch_add( &octs_per_level_count(level), 1 );
  }
  );
  Kokkos::parallel_for( "Count number of intermediate octs per level", Kokkos::RangePolicy<>(0, numIntermediateOcts),
    KOKKOS_LAMBDA( const uint32_t iOct )
  {
    const level_t level = lmesh.getLevel({iOct, false, true});
    Kokkos::atomic_fetch_add( &octs_intermediate_per_level_count(level), 1 );
  }
  );
  Kokkos::parallel_for( "Count number of ghosts per level", Kokkos::RangePolicy<>(0, numGhosts),
    KOKKOS_LAMBDA( const uint32_t iOct )
  {
    const level_t level = lmesh.getLevel({iOct, true, false});
    Kokkos::atomic_fetch_add( &ghosts_per_level_count(level), 1 );
  }
  );
  Kokkos::parallel_for( "Count number of intermediate ghosts per level", Kokkos::RangePolicy<>(0, numIntermediateGhosts),
    KOKKOS_LAMBDA( const uint32_t iOct )
  {
    const level_t level = lmesh.getLevel({iOct, true, true});
    Kokkos::atomic_fetch_add( &ghosts_intermediate_per_level_count(level), 1 );
  }
  );

  const auto octs_per_level_count_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), octs_per_level_count);
  const auto octs_intermediate_per_level_count_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), octs_intermediate_per_level_count);
  const auto ghosts_per_level_count_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ghosts_per_level_count);
  const auto ghosts_intermediate_per_level_count_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ghosts_intermediate_per_level_count);


  for(level_t ilevel = 0; ilevel <= level_max; ilevel++)
  { 
    const uint32_t octs = octs_per_level_count_host(ilevel);
    const uint32_t ghosts = ghosts_per_level_count_host(ilevel);
    const uint32_t octs_intermediate = octs_intermediate_per_level_count_host(ilevel);
    const uint32_t ghosts_intermediate = ghosts_intermediate_per_level_count_host(ilevel);
    [[maybe_unused]] const real_t mean_total_nbOctants_per_dimension = Kokkos::floor(Kokkos::cbrt(octs+ghosts+octs_intermediate+ghosts_intermediate));
    [[maybe_unused]] const real_t total_nbOctants_per_dimension = (1U << ilevel);
    DYABLO_ASSERT_KOKKOS_DEBUG( mean_total_nbOctants_per_dimension <= total_nbOctants_per_dimension, "Cannot count more octants per level than there are in the simulation" );
    printf("Rank %d Octree Level %d, octs %u (+ %u) intermediate %u (+ %u)\n", mpi_rank, ilevel, octs, ghosts, octs_intermediate, ghosts_intermediate);
  }

  // Now accumulate
  uint32_t total = 0;
  Kokkos::parallel_scan("Accumulate", Kokkos::RangePolicy<>(0, nlevels + 1) ,
    KOKKOS_LAMBDA (const uint32_t i ,uint32_t& update , bool final ) 
  {
    const uint32_t val = octs_per_level_count(i);
    if (final)
      octs_per_level_count(i) = update ;
    update += val;
  }, total);
  Kokkos::parallel_scan("Accumulate", Kokkos::RangePolicy<>(0, nlevels + 1) ,
    KOKKOS_LAMBDA (const uint32_t i ,uint32_t& update , bool final ) 
  {
    const uint32_t val = octs_intermediate_per_level_count(i);
    if (final)
      octs_intermediate_per_level_count(i) = update ;
    update += val;
  }, total);
  Kokkos::parallel_scan("Accumulate", Kokkos::RangePolicy<>(0, nlevels + 1) ,
    KOKKOS_LAMBDA (const uint32_t i ,uint32_t& update , bool final ) 
  {
    const uint32_t val = ghosts_per_level_count(i);
    if (final)
      ghosts_per_level_count(i) = update ;
    update += val;
  }, total);
  Kokkos::parallel_scan("Accumulate", Kokkos::RangePolicy<>(0, nlevels + 1) ,
    KOKKOS_LAMBDA (const uint32_t i ,uint32_t& update , bool final ) 
  {
    const uint32_t val = ghosts_intermediate_per_level_count(i);
    if (final)
      ghosts_intermediate_per_level_count(i) = update ;
    update += val;
  }, total);

  pdata->octs_per_level_count = octs_per_level_count;
  pdata->octs_intermediate_per_level_count = octs_intermediate_per_level_count;
  pdata->ghosts_per_level_count = ghosts_per_level_count;
  pdata->ghosts_intermediate_per_level_count = ghosts_intermediate_per_level_count;

  /* for(level_t ilevel = 0; ilevel <= level_max; ilevel++)
  { 
    const uint32_t octs = octs_per_level_count_host(ilevel);
    const uint32_t ghosts = ghosts_per_level_count_host(ilevel);
    const uint32_t octs_intermediate = octs_intermediate_per_level_count_host(ilevel);
    const uint32_t ghosts_intermediate = ghosts_intermediate_per_level_count_host(ilevel);
    printf("Accumulated, Rank %d Octree Level %d, octs %u (+ %u) intermediate %u (+ %u)\n", mpi_rank, ilevel, octs, ghosts, octs_intermediate, ghosts_intermediate);
  } */
}

const Kokkos::View<uint32_t*> GravitySolver_multigrid::get_subview_octs(const level_t level) const
{return get_subview_octs(level, level);}
const Kokkos::View<uint32_t*> GravitySolver_multigrid::get_subview_octs_intermediate(const level_t level) const
{return get_subview_octs_intermediate(level, level);}
const Kokkos::View<uint32_t*> GravitySolver_multigrid::get_subview_ghosts(const level_t level) const
{return get_subview_ghosts(level, level);}
const Kokkos::View<uint32_t*> GravitySolver_multigrid::get_subview_ghosts_intermediate(const level_t level) const
{return get_subview_ghosts_intermediate(level, level);}

const Kokkos::View<uint32_t*> GravitySolver_multigrid::get_subview_octs(const level_t level_min, const level_t level_max) const
{
  DYABLO_ASSERT_KOKKOS_DEBUG( level_min < pdata->octs_per_level_count.size(), "Level out of bounds in octs_per_level_count" );
  const auto octs_per_level_count_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pdata->octs_per_level_count);
  return Kokkos::subview(pdata->octs_per_level, std::make_pair(octs_per_level_count_host(level_min), octs_per_level_count_host(level_max + 1)));
}
const Kokkos::View<uint32_t*> GravitySolver_multigrid::get_subview_octs_intermediate(const level_t level_min, const level_t level_max) const
{
  DYABLO_ASSERT_KOKKOS_DEBUG( level_max < pdata->octs_intermediate_per_level_count.size(), "Level out of bounds in octs_intermediate_per_level_count" );
  const auto octs_intermediate_per_level_count_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pdata->octs_intermediate_per_level_count);
  return Kokkos::subview(pdata->octs_intermediate_per_level, std::make_pair(octs_intermediate_per_level_count_host(level_min), octs_intermediate_per_level_count_host(level_max + 1)));
}
const Kokkos::View<uint32_t*> GravitySolver_multigrid::get_subview_ghosts(const level_t level_min, const level_t level_max) const
{
  DYABLO_ASSERT_KOKKOS_DEBUG( level_max < pdata->ghosts_per_level_count.size(), "Level out of bounds in ghosts_per_level_count" );
  const auto ghosts_per_level_count_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pdata->ghosts_per_level_count);
  return Kokkos::subview(pdata->ghosts_per_level, std::make_pair(ghosts_per_level_count_host(level_min), ghosts_per_level_count_host(level_max + 1)));
}
const Kokkos::View<uint32_t*> GravitySolver_multigrid::get_subview_ghosts_intermediate(const level_t level_min, const level_t level_max) const
{
  DYABLO_ASSERT_KOKKOS_DEBUG( level_max < pdata->ghosts_intermediate_per_level_count.size(), "Level out of bounds in ghosts_intermediate_per_level_count" );
  const auto ghosts_intermediate_per_level_count_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pdata->ghosts_intermediate_per_level_count);
  return Kokkos::subview(pdata->ghosts_intermediate_per_level, std::make_pair(ghosts_intermediate_per_level_count_host(level_min), ghosts_intermediate_per_level_count_host(level_max + 1)));
}

//} // namespace

template <Target target, class GhostComm >
void GravitySolver_multigrid::exchange_ghosts_at_level(UserData& U_, const level_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm)
{
  std::vector<UserData::FieldAccessor::FieldInfo> field_info;
  for(uint8_t i=0; i<exchange_vars.size(); i++)
    field_info.push_back( {exchange_vars[i],i} );

  if constexpr (target == Target::LEAVES || target == Target::BOTH) 
  {
    auto Uexchange = U_.getAccessor(field_info);
    ghost_comm.exchange_ghosts_at_level( U_.getAccessor(field_info), level );
  }
  if constexpr (target == Target::INTERMEDIATES || target == Target::BOTH) 
  {
    auto Uexchange = U_.getAccessor_intermediate(field_info);
    ghost_comm.exchange_intermediate_ghosts_at_level( Uexchange, level );
  }
};

template <Target target, class GhostComm >
void GravitySolver_multigrid::reduce_ghosts_at_level(UserData& U_, const level_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm)
{
  std::vector<UserData::FieldAccessor::FieldInfo> field_info;
  for(uint8_t i=0; i<exchange_vars.size(); i++)
    field_info.push_back( {exchange_vars[i],i} );
  
  if constexpr (target == Target::LEAVES || target == Target::BOTH) 
  {
    auto Uexchange = U_.getAccessor(field_info);
    ghost_comm.reduce_ghosts_at_level( Uexchange, level );
   
  }
  if constexpr (target == Target::INTERMEDIATES || target == Target::BOTH) 
  {
    auto Uexchange = U_.getAccessor_intermediate(field_info);
    ghost_comm.reduce_intermediate_ghosts_at_level( Uexchange, level );
  }
};

/**
 * @brief Solves the Poisson equation and updates the gravity field
 * 
 * @param[in,out] U UserData structure to update
 * @param[in] scalar_data ScalarSimulationData structure
*/
void GravitySolver_multigrid::update_gravity_field( UserData& U_, ScalarSimulationData& scalar_data )
{

  pdata->timers.get("GravitySolver_multigrid").start();

  ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpi_comm = foreach_cell.get_amr_mesh().getMpiComm();
  const int mpi_rank = mpi_comm.MPI_Comm_rank();
  auto& amr_mesh = foreach_cell.get_amr_mesh();
  const level_t first_mpi_multigrid_level = pdata->first_mpi_multigrid_level;
  const real_t epsilon = pdata->MG_eps;

  // Get min/max levels
  Kokkos::MinMax<uint32_t>::value_type minmax_result;
  {
    const LightOctree& lmesh = amr_mesh.getLightOctree();
    // Min/max AMR level
    const size_t numOctants = lmesh.getNumOctants();
    Kokkos::parallel_reduce ( " Get min/max level in AMR " , Kokkos::RangePolicy<>(0, numOctants) ,
      KOKKOS_LAMBDA ( const uint32_t iOct , Kokkos::MinMax<uint32_t>::value_type& minmax_result_tmp ) 
    {
      const level_t level_tmp = lmesh.getLevel({iOct, false});
      if ( level_tmp > minmax_result_tmp.max_val ) minmax_result_tmp.max_val = level_tmp;
      if ( level_tmp < minmax_result_tmp.min_val ) minmax_result_tmp.min_val = level_tmp;
    }, Kokkos::MinMax<uint32_t>(minmax_result));
    DYABLO_ASSERT_KOKKOS_DEBUG( minmax_result.max_val <= lmesh.get_level_max(), "Max level found in AMR should not be higher than that stored when building the tree" );
    DYABLO_ASSERT_KOKKOS_DEBUG( minmax_result.min_val == lmesh.get_level_min(), "Min level found in AMR different from coarse level" );
  }

  const level_t local_level_max_found = minmax_result.max_val;
  const level_t global_level_max_found = MPI_Allreduce_int_max(local_level_max_found);

  DYABLO_ASSERT_KOKKOS_DEBUG( first_mpi_multigrid_level <= pdata->level_coarse, "Full coarse level cannot be common to all processes" );

  // Create intermediate storage in LightOctree
  amr_mesh.updateLightOctreeWithIntermediates(first_mpi_multigrid_level);
  // Create intermediate storage and ghostmap in amr_mesh
  amr_mesh.init_intermediates(first_mpi_multigrid_level);
  // Add intermediate ghosts to the LightOctree
  amr_mesh.updateLightOctreeWithIntermediates(first_mpi_multigrid_level);
  const LightOctree& lmesh = amr_mesh.getLightOctree();
  const level_t level_coarse = pdata->level_coarse = lmesh.get_level_min();


  // FIXME: Currently, we need to use Uintermediate.getShape() instead of U.getShape(), as the latter is not updated with new lmesh.
  U_.new_fields({"solution", "rhs",  "res", "mask" });
  U_.new_intermediate_fields( {"rho","gphi", "solution", "rhs", "res", "mask"} ); 
  const std::vector<UserData_fields::FieldAccessor_FieldInfo> fields_info = {
    {"rho", Irho},
    {"gphi", Iphi},
    {"solution", Isolution},
    {"rhs", Irhs},
    { "res", Iresidual },
    { "mask", Imask },
  };
  UserData::FieldAccessor U = U_.getAccessor({fields_info});
  UserData::FieldAccessor Uintermediate = U_.getAccessor_intermediate({fields_info});
  pdata->U = U;
  pdata->Uintermediate = Uintermediate;
 
  // Compute ghost communicators
  const auto iter_space = Uintermediate.getShape();
  GhostCommunicator ghost_comm_minimal(foreach_cell.get_amr_mesh(), iter_space, 1, mpi_comm);
  GhostCommunicator ghost_comm_blockwide(ghost_comm_minimal); // Deep copy constructor
  DYABLO_ASSERT_KOKKOS_DEBUG(
    iter_space.bx == iter_space.by &&
    iter_space.bx == iter_space.bz &&
    iter_space.bx % 2 == 0,
    "Wrong block shape"
  );
  ghost_comm_minimal.init_intermediates(foreach_cell.get_amr_mesh(), iter_space, 1, mpi_comm);
  ghost_comm_blockwide.init_intermediates(foreach_cell.get_amr_mesh(), iter_space, iter_space.bx, mpi_comm);
  ghost_comm_minimal.sort_ghosts_by_levels(lmesh, global_level_max_found);
  ghost_comm_blockwide.sort_ghosts_by_levels(lmesh, global_level_max_found);
  ghost_comm_minimal.sort_intermediate_ghosts_by_levels(lmesh, global_level_max_found);
  ghost_comm_blockwide.sort_intermediate_ghosts_by_levels(lmesh, global_level_max_found);
  
  // Count and list octants per level
  count_octants_per_level(lmesh);
  sort_octants_per_level(lmesh);

  DYABLO_ASSERT_KOKKOS_DEBUG( check_parents(), "Parent check failed" );
  DYABLO_ASSERT_KOKKOS_DEBUG( check_neighbors(), "Neighbors check failed" );

  // Compute rho mean
  real_t rho_mean = 0;
  const real_t xmin(pdata->xmin), ymin(pdata->ymin), zmin(pdata->zmin);
  const real_t xmax(pdata->xmax), ymax(pdata->ymax), zmax(pdata->zmax);
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  
  foreach_cell.reduce_cell("Compute rho_mean", iter_space,
    KOKKOS_LAMBDA(const CellIndex & iCell, real_t & update_rhomean)
  {
    const level_t level = cells.getLevel(iCell);
    const uint32_t nocts1d = 1U << level;
    const Kokkos::Array<real_t, 3> size = {
          1./(nocts1d * iter_space.bx), 
          1./(nocts1d * iter_space.by), 
          1./(nocts1d * iter_space.bz)
    };
    const real_t rhoi = U.at(iCell, Irho);
    update_rhomean += rhoi * size[IX] * size[IY] * size[IZ];
  }, Kokkos::Sum<real_t>(rho_mean));
  const real_t Vtot = (xmax - xmin) * (ymax - ymin) * (zmax - zmin);
  rho_mean = MPI_Allreduce_scalar(rho_mean) / Vtot;
  if (mpi_rank == 0) printf("rhomean = %.5e\n", rho_mean);

  const bool cosmo_run = pdata->cosmo_run;
  real_t aexp = 0;
  if( cosmo_run )
    aexp = scalar_data.get<real_t>("aexp");
  const real_t four_Pi_G = pdata->four_Pi_G;

  // Compute rho on intermediate levels
  for( level_t level = level_coarse + 1; level <= global_level_max_found; level++ )
    restriction_from_children<Target::LEAVES>(level_coarse, level, kokkos_array<int>(Irho), kokkos_array<int>(Irho));
  for( level_t level = level_coarse; level < global_level_max_found; level++ )
    reduce_ghosts_at_level<Target::INTERMEDIATES>(U_, level, {"rho"}, ghost_comm_blockwide);

  // Compute RHS
  foreach_cell.foreach_cell("Set RHS of Laplacian, based on rho", iter_space,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    U.at(iCell, Irhs) = (cosmo_run) ? b_cosmo(U, iCell, rho_mean, aexp) : b(U, iCell, rho_mean, four_Pi_G);
  });
  const auto octs_intermediate = get_subview_octs_intermediate(level_coarse, global_level_max_found);
  foreach_cell.foreach_intermediate_cell_in_octants( "Set RHS of Laplacian, based on rho", iter_space, octs_intermediate,
    KOKKOS_LAMBDA( CellIndex& iCell)
  {
    Uintermediate.at( iCell, Irhs ) = (cosmo_run) ? b_cosmo(Uintermediate, iCell, rho_mean, aexp) : b(Uintermediate, iCell, rho_mean, four_Pi_G);
  }); 


  // If not initial step, read potential from last step
  int step = 0;
  try 
  {
    const int iter = scalar_data.get<int>("iter");
    step = iter;
    if (mpi_rank == 0) printf("Step = %d\n", step);
  } 
  catch (const std::runtime_error& e) 
  {
    std::cerr << "Warning: Could not retrieve 'iter' from scalar_data.\n";
  }

  if (step == 0)
    initialise_lhs_from_rhs<Target::BOTH>(level_coarse);
  else
  {
    copy_multigrid_fields<Target::LEAVES>(level_coarse, level_coarse, kokkos_array<int>(Iphi), kokkos_array<int>(Isolution));
    initialise_lhs_from_rhs<Target::INTERMEDIATES>(level_coarse);
  }

  exchange_ghosts_at_level<Target::BOTH>(U_, level_coarse, {"solution"}, ghost_comm_minimal);

  // Multigrid
  if (mpi_rank == 0) printf("Coarse Multigrid\n");

  real_t residual = std::numeric_limits<real_t>::max();
  residual_uniform<Target::BOTH>(level_coarse);
  residual = residual_norm(level_coarse);
  if (mpi_rank == 0) printf("Level %d Initial residual %.8e\n", level_coarse, residual);

  // Truncation error
  operator_uniform(level_coarse);
  restriction_from_children<Target::BOTH>(level_coarse - 1, level_coarse, kokkos_array<int>(Isolution, Iresidual), kokkos_array<int>(Isolution, Irhs));
  reduce_ghosts_at_level<Target::INTERMEDIATES>(U_, level_coarse - 1, {"rhs", "solution"}, ghost_comm_blockwide);
  exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, level_coarse - 1, {"solution"}, ghost_comm_minimal);
  operator_uniform(level_coarse - 1);
  const real_t truncation = truncation_norm(level_coarse - 1);
  const real_t threshold = epsilon * truncation;
  //printf("threshold %.5e = %.5e * %.5e\n", threshold, epsilon, truncation);

  do
  { 
    fill_multigrid_fields<Target::ALL_INTERMEDIATES>(0, level_coarse - 1, kokkos_array<int>(Irhs), kokkos_array<real_t>(0));
    V_cycle_uniform(U_, level_coarse, ghost_comm_minimal, ghost_comm_blockwide);
    residual_uniform<Target::BOTH>(level_coarse);
    residual = residual_norm(level_coarse);
    if (mpi_rank == 0) printf("Level %d, | Residual = %.8e | Threshold = %.8e |\n", level_coarse, residual, threshold);
  } while (residual > threshold);

  copy_multigrid_fields<Target::ALL>(level_coarse, level_coarse, kokkos_array<int>(Isolution), kokkos_array<int>(Iphi));
  
  if (mpi_rank == 0) printf("AMR Multigrid\n");

  for (level_t ilevel = level_coarse+1; ilevel <= global_level_max_found; ilevel++) 
  {
    prolongation_from_children<Target::BOTH>(ilevel);
    exchange_ghosts_at_level<Target::BOTH>(U_, ilevel, {"solution"}, ghost_comm_minimal);
    compute_mask(U_, std::max(0, ilevel - 4), ilevel, ghost_comm_minimal, ghost_comm_blockwide);

    // Truncation
    operator_amr_finest(ilevel);
    fill_multigrid_fields<Target::ALL_INTERMEDIATES>(ilevel - 1, ilevel - 1, kokkos_array<int>(Isolution, Irhs, Iresidual) , kokkos_array<real_t>(0,0,0));
    restriction_from_children<Target::BOTH>(ilevel - 1, ilevel, kokkos_array<int>(Isolution, Iresidual), kokkos_array<int>(Isolution, Irhs));
    reduce_ghosts_at_level<Target::INTERMEDIATES>(U_, ilevel - 1, {"rhs", "solution"}, ghost_comm_blockwide);
    exchange_ghosts_at_level<Target::INTERMEDIATES>(U_, ilevel - 1, {"solution"}, ghost_comm_minimal);
    operator_intermediate_amr_correction(ilevel - 1);
    const real_t truncation = truncation_norm(ilevel - 1);
    const real_t threshold = epsilon * truncation;

    do
    { 
      fill_multigrid_fields<Target::ALL>(std::max(0, ilevel - 4), ilevel - 1, kokkos_array<int>(Isolution, Irhs, Iresidual), kokkos_array<real_t>(0,0,0));
      V_cycle_amr(U_, ilevel, ilevel, ghost_comm_minimal, ghost_comm_blockwide);
      residual_amr_finest(ilevel);
      residual = residual_norm(ilevel);
      if (mpi_rank == 0) printf("Level %d, | Residual = %.8e | Threshold = %.8e |\n", ilevel, residual, threshold);
    } while (residual > threshold);
    
    copy_multigrid_fields<Target::ALL>(ilevel, ilevel, kokkos_array<int>(Isolution), kokkos_array<int>(Iphi));
  }

  // Update force field in U from potential
  UserData::FieldAccessor Uout = U_.getAccessor({
    {"gx", Igx},
    {"gy", Igy},
    {"gz", Igz},
    {"gphi", Iphi}
  });

  if (mpi_rank == 0) printf("Now compute Force\n");
  gradient_3pt(Uout, Uintermediate); 

  // FIXME: This only removes the keys in the map, not the actual data which is still stored in memory
  for (std::string name : {"solution", "rhs",  "res", "mask" })
    U_.delete_field(name);
  
  U_.erase_intermediate_field();
  amr_mesh.deleteIntermediates();

  pdata->timers.get("GravitySolver_multigrid").stop();
}

}// namespace dyablo

FACTORY_REGISTER( dyablo::GravitySolverFactory, dyablo::GravitySolver_multigrid, "GravitySolver_multigrid" );
