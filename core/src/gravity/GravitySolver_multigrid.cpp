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
  Isolution, Irhs, Iresidual, Imask,
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
  uint32_t level_coarse, Npre, Npost, Ncycles;

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
      configMap.getValue<real_t>("gravity", "MG_eps", 1E-3),
      configMap.getValue<uint32_t>("gravity", "first_mpi_multigrid_level", 2),
      4, // level coarse
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
 * @brief Gradient method.
 * 
 * Three-point gradient 
 * 
 * @param U[in]: The data to read from
 * @param dir[in]: Direction along wich we compute the gradient
*/
template< typename Array_t >
void GravitySolver_multigrid::gradient(Array_t& U, Array_t& Uintermediate)
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;

  foreach_cell.foreach_cell("Gravity_mg::construct_force_field", U.getShape(), 
    KOKKOS_LAMBDA(const CellIndex& iCell)
  { 
    const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
    const real_t phi_C = U.at(iCell, Iphi);

    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      CellIndex::offset_t off_L = {}; off_L[dir] = -1;
      CellIndex::offset_t off_R = {}; off_R[dir] = +1;
      CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U);
      CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U);

      real_t phi_L(0), phi_R(0), a(1), b(1);

      if (CellIndex::BIGGER == iCell_L.status) {
        phi_L = average_8bigger_neighbors(U, Uintermediate, iCell, off_L);
        a = 0.5;
      } else if (CellIndex::SMALLER == iCell_L.status) {
        iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
        phi_L = Uintermediate.at(iCell_L, Iphi);
      } else phi_L = U.at(iCell_L, Iphi);

      if (CellIndex::BIGGER == iCell_R.status) {
        phi_R = average_8bigger_neighbors(U, Uintermediate, iCell, off_R);
        b = 0.5;
      } else if (CellIndex::SMALLER == iCell_R.status) {
        iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
        phi_R = Uintermediate.at(iCell_R, Iphi);
      } else phi_R = U.at(iCell_R, Iphi);
      

      if( BC_ABSORBING == boundarycondition[dir] && iCell_L.is_boundary() ) phi_L = 0;
      if( BC_ABSORBING == boundarycondition[dir] && iCell_R.is_boundary() ) phi_R = 0;

      const Kokkos::Array< VarIndex, 3 > IG = {Igx, Igy, Igz};

      const real_t f_L = b/(a*a + a*b);
      const real_t f_R = -a/(a*b + b*b);
      const real_t f_C = (a-b)/(a*b);

      U.at(iCell, IG[dir]) = (f_L * phi_L + f_R * phi_R + f_C * phi_C)/size[dir];
    }
  });
}

/**
 * @brief Contribution at fine-coarse boundary from coarser neighbours.
 * 
 * At a fine-coarse boundary, perform an interpolation using the 
 * eight neihbouring coarser cells
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param iCell[in]: cell index where the data should be read
 * @param offset[in]: offset applied prior to calling the method to get iCell_U
 * @return the value of the reconstructed neighbour 
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
real_t GravitySolver_multigrid::average_8bigger_neighbors(const Array_t& U, const Array_t& Uintermediate, const CellIndex iCell, CellIndex::offset_t offset) 
{
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
      const int8_t shift_x = 2 * ( l*offset[IX] + i * (2*(iCell.i % 2) - 1) );
      const int8_t shift_y = 2 * ( l*offset[IY] + j * (2*(iCell.j % 2) - 1) );
      const int8_t shift_z = 2 * ( l*offset[IZ] + k * (2*(iCell.k % 2) - 1) );
      const CellIndex::offset_t shift = {shift_x, shift_y, shift_z};
      const CellIndex iCell_coarse = iCell.getNeighbor_ghost(shift, U.getShape());
      if (CellIndex::BIGGER == iCell_coarse.status) {
        result += tmp[counter]*U.at(iCell_coarse, Iphi);
      }
      else {
        const CellIndex iCell_parent = iCell_coarse.getParent(Uintermediate.getShape());
        result += tmp[counter]*Uintermediate.at(iCell_parent, Iphi);
      }
    }
    counter++;
  }
  return result; 
}

/**
 * @brief Contribution at fine-coarse boundary from coarser neighbours.
 * 
 * At a fine-coarse boundary, perform an interpolation using the 
 * eight neihbouring coarser cells
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
 * @return the residual at a given level 
*/
real_t GravitySolver_multigrid::residual_norm(const level_t level)
{
  const auto& U = pdata->U;
  const auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  real_t residual_sqr_leaves = 0;
  real_t residual_sqr_intermediate = 0;
  const auto octs = get_subview_octs(level);
  foreach_cell.reduce_cell_in_octants("Compute residual norm", U.getShape(), octs,
    KOKKOS_LAMBDA(const CellIndex & iCell, real_t & update_residual_sqr)
  {
    const real_t residual_tmp = U.at(iCell, Iresidual);
    update_residual_sqr += residual_tmp * residual_tmp;
  }, Kokkos::Sum<real_t>(residual_sqr_leaves));
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.reduce_intermediate_cell_in_octants("Compute residual norm", U.getShape(), octs_intermediate,
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
 * @brief Initialise solution.
 * 
 * Given the right-hand side of a Poisson equation,  
 * initialise the solution
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
void GravitySolver_multigrid::initialise_intermediate_lhs(const level_t level)
{
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto& iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };

  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Initialise potential", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    Uintermediate.at(iCell, Isolution) = -Uintermediate.at(iCell, Irhs) / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ])) ;
  });
};

/**
 * @brief Restriction operator.
 * 
 * Restriction of mask, and residual to right-hand side of intermediate cells
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
void GravitySolver_multigrid::restriction_from_parents(const level_t level) 
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const int ndim = pdata->ndim;

  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants( "Restrict", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
    real_t residual(0), mask(0);
    const int ns = foreach_sibling( ndim, iCell_c0, Uintermediate.getShape(),
      [&]( const CellIndex& iCell_c )
    {
      if ( iCell_c.iOct.isIntermediate ) {
        residual += Uintermediate.at(iCell_c, Iresidual);
        mask += Uintermediate.at(iCell_c, Imask);
      } 
      else {
        residual += U.at(iCell_c, Iresidual);
        mask += U.at(iCell_c, Imask);
      }
      
    });
    Uintermediate.at( iCell, Irhs ) = residual/ns;
    Uintermediate.at( iCell, Imask ) = mask/ns;
  }); 
}

/**
 * @brief Restriction operator.
 * 
 * Restriction of mask, and residual to right-hand side of intermediate cells
 * Only non-ghost cells contribute to the restriction. 
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
void GravitySolver_multigrid::restriction_from_children(const level_t level) 
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  constexpr real_t inv_ns = 1./8;

  const auto octs = get_subview_octs(level + 1);
  foreach_cell.foreach_cell_in_octants( "Restrict", U.getShape(), octs,
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const CellIndex iCell_p = iCell.getParent(U.getShape());      
    Kokkos::atomic_add( &Uintermediate.at( iCell_p, Irhs ), U.at( iCell, Iresidual ) * inv_ns );
    Kokkos::atomic_add( &Uintermediate.at( iCell_p, Imask ), U.at( iCell, Imask ) * inv_ns );
  }); 
  const auto octs_intermediate = get_subview_octs_intermediate(level + 1);
  foreach_cell.foreach_intermediate_cell_in_octants( "Restrict", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const CellIndex iCell_p = iCell.getParent(Uintermediate.getShape());     
    Kokkos::atomic_add( &Uintermediate.at( iCell_p, Irhs ), Uintermediate.at( iCell, Iresidual ) * inv_ns );
    Kokkos::atomic_add( &Uintermediate.at( iCell_p, Imask ), Uintermediate.at( iCell, Imask ) * inv_ns );
  }); 
}


/**
 * @brief Get neighbouring value.
 * 
 * Get solution of neighbour of intermediate cell 
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param iCell[in]: cell index where the data should be read
 * @param offset[in]: offset applied prior to calling the method to get iCell
 * @return the solution of neighbour 
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
real_t GravitySolver_multigrid::get_neighbor_value(const Array_t& U, const Array_t& Uintermediate, const CellIndex iCell, const CellIndex::offset_t offset)
{
  CellIndex iCell_n = iCell.getNeighbor_ghost_intermediate(offset, Uintermediate.getShape());
  if (iCell_n.level_diff() == 0)
    return Uintermediate.at( iCell_n, Isolution );

  iCell_n = iCell.getNeighbor_ghost(offset, U.getShape());
  return U.at( iCell_n, Isolution );
}

/**
 * @brief Add prolongation (interpolation).
 * 
 * Add a first-order prolongation as linear interpolation 
 * of the solution at a given level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
 * @return the solution at a given level 
*/
void GravitySolver_multigrid::prolongation_from_children(const level_t level) 
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  constexpr real_t f0 = 27.0 / 64;
  constexpr real_t f1 = 9.0 / 64;
  constexpr real_t f2 = 3.0 / 64;
  constexpr real_t f3 = 1.0 / 64;
  const auto octs = get_subview_octs(level);
  foreach_cell.foreach_cell_in_octants( "Prolongation", U.getShape(), octs,
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const int8_t shift_x = 2 * (iCell.i % 2) - 1;
    const int8_t shift_y = 2 * (iCell.j % 2) - 1;
    const int8_t shift_z = 2 * (iCell.k % 2) - 1;
    // Get coarse cell values to interpolate from
    CellIndex iCell_p = iCell.getParent(Uintermediate.getShape());
    iCell_p.status = CellIndex::LOCAL_TO_BLOCK;
    const real_t tmp000 = Uintermediate.at( iCell_p, Isolution );
    const real_t tmp001 = get_neighbor_value(U, Uintermediate, iCell_p, {0, 0, shift_z});
    const real_t tmp010 = get_neighbor_value(U, Uintermediate, iCell_p, {0, shift_y, 0});
    const real_t tmp100 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, 0, 0});
    const real_t tmp011 = get_neighbor_value(U, Uintermediate, iCell_p, {0, shift_y, shift_z});
    const real_t tmp101 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, 0, shift_z});
    const real_t tmp110 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, shift_y, 0});
    const real_t tmp111 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, shift_y, shift_z});
    // Interpolate
    U.at(iCell, Isolution) += f0*tmp000
        + f1 * (tmp001 + tmp010 + tmp100)
        + f2 * (tmp011 + tmp101 + tmp110)
        + f3 * tmp111;
  }); 
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants( "Prolongation", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const int8_t shift_x = 2 * (iCell.i % 2) - 1;
    const int8_t shift_y = 2 * (iCell.j % 2) - 1;
    const int8_t shift_z = 2 * (iCell.k % 2) - 1;
    // Get coarse cell values to interpolate from
    CellIndex iCell_p = iCell.getParent(Uintermediate.getShape());
    iCell_p.status = CellIndex::LOCAL_TO_BLOCK;
    const real_t tmp000 = Uintermediate.at( iCell_p, Isolution );
    const real_t tmp001 = get_neighbor_value(U, Uintermediate, iCell_p, {0, 0, shift_z});
    const real_t tmp010 = get_neighbor_value(U, Uintermediate, iCell_p, {0, shift_y, 0});
    const real_t tmp100 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, 0, 0});
    const real_t tmp011 = get_neighbor_value(U, Uintermediate, iCell_p, {0, shift_y, shift_z});
    const real_t tmp101 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, 0, shift_z});
    const real_t tmp110 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, shift_y, 0});
    const real_t tmp111 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, shift_y, shift_z});
    // Interpolate
    Uintermediate.at(iCell, Isolution) += f0*tmp000
        + f1 * (tmp001 + tmp010 + tmp100)
        + f2 * (tmp011 + tmp101 + tmp110)
        + f3 * tmp111;
  }); 
}

/**
 * @brief Add prolongation (interpolation).
 * 
 * Add a first-order prolongation as linear interpolation 
 * of the solution at a given level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
 * @return the solution at a given level 
*/
void GravitySolver_multigrid::prolongation_from_children_on_intermediate(const level_t level) 
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  constexpr real_t f0 = 27.0 / 64;
  constexpr real_t f1 = 9.0 / 64;
  constexpr real_t f2 = 3.0 / 64;
  constexpr real_t f3 = 1.0 / 64;
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants( "Prolongation", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const int8_t shift_x = 2 * (iCell.i % 2) - 1;
    const int8_t shift_y = 2 * (iCell.j % 2) - 1;
    const int8_t shift_z = 2 * (iCell.k % 2) - 1;
    // Get coarse cell values to interpolate from
    CellIndex iCell_p = iCell.getParent(Uintermediate.getShape());
    iCell_p.status = CellIndex::LOCAL_TO_BLOCK;
    const real_t tmp000 = Uintermediate.at( iCell_p, Isolution );
    const real_t tmp001 = get_neighbor_value(U, Uintermediate, iCell_p, {0, 0, shift_z});
    const real_t tmp010 = get_neighbor_value(U, Uintermediate, iCell_p, {0, shift_y, 0});
    const real_t tmp100 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, 0, 0});
    const real_t tmp011 = get_neighbor_value(U, Uintermediate, iCell_p, {0, shift_y, shift_z});
    const real_t tmp101 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, 0, shift_z});
    const real_t tmp110 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, shift_y, 0});
    const real_t tmp111 = get_neighbor_value(U, Uintermediate, iCell_p, {shift_x, shift_y, shift_z});
    // Interpolate to children
    Uintermediate.at(iCell, Isolution) += f0*tmp000
        + f1 * (tmp001 + tmp010 + tmp100)
        + f2 * (tmp011 + tmp101 + tmp110)
        + f3 * tmp111;
  }); 
}

void GravitySolver_multigrid::prolongation_from_parents(const level_t level) {
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  constexpr real_t f0 = 27.0 / 64;
  constexpr real_t f1 = 9.0 / 64;
  constexpr real_t f2 = 3.0 / 64;
  constexpr real_t f3 = 1.0 / 64;
  const auto octs_intermediate = get_subview_octs_intermediate(level - 1);
  foreach_cell.foreach_intermediate_cell_in_octants( "Prolongation", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    // Get coarse cell values to interpolate from
    const real_t tmp111 = Uintermediate.at( iCell, Isolution );
    const real_t tmp0 = f0 * tmp111;

    const real_t tmp000 = get_neighbor_value(U, Uintermediate, iCell, {-1, -1, -1});
    const real_t tmp001 = get_neighbor_value(U, Uintermediate, iCell, {-1, -1, 0});
    const real_t tmp002 = get_neighbor_value(U, Uintermediate, iCell, {-1, -1, 1});
    const real_t tmp010 = get_neighbor_value(U, Uintermediate, iCell, {-1, 0, -1});
    const real_t tmp011 = get_neighbor_value(U, Uintermediate, iCell, {-1, 0, 0});
    const real_t tmp012 = get_neighbor_value(U, Uintermediate, iCell, {-1, 0, 1});
    const real_t tmp020 = get_neighbor_value(U, Uintermediate, iCell, {-1, 1, -1});
    const real_t tmp021 = get_neighbor_value(U, Uintermediate, iCell, {-1, 1, 0});
    const real_t tmp022 = get_neighbor_value(U, Uintermediate, iCell, {-1, 1, 1});
    const real_t tmp100 = get_neighbor_value(U, Uintermediate, iCell, {0, -1, -1});
    const real_t tmp101 = get_neighbor_value(U, Uintermediate, iCell, {0, -1, 0});
    const real_t tmp102 = get_neighbor_value(U, Uintermediate, iCell, {0, -1, 1});
    const real_t tmp110 = get_neighbor_value(U, Uintermediate, iCell, {0, 0, -1});
    const real_t tmp112 = get_neighbor_value(U, Uintermediate, iCell, {0, 0, 1});
    const real_t tmp120 = get_neighbor_value(U, Uintermediate, iCell, {0, 1, -1});
    const real_t tmp121 = get_neighbor_value(U, Uintermediate, iCell, {0, 1, 0});
    const real_t tmp122 = get_neighbor_value(U, Uintermediate, iCell, {0, 1, 1});
    const real_t tmp200 = get_neighbor_value(U, Uintermediate, iCell, {1, -1, -1});
    const real_t tmp201 = get_neighbor_value(U, Uintermediate, iCell, {1, -1, 0});
    const real_t tmp202 = get_neighbor_value(U, Uintermediate, iCell, {1, -1, 1});
    const real_t tmp210 = get_neighbor_value(U, Uintermediate, iCell, {1, 0, -1});
    const real_t tmp211 = get_neighbor_value(U, Uintermediate, iCell, {1, 0, 0});
    const real_t tmp212 = get_neighbor_value(U, Uintermediate, iCell, {1, 0, 1});
    const real_t tmp220 = get_neighbor_value(U, Uintermediate, iCell, {1, 1, -1});
    const real_t tmp221 = get_neighbor_value(U, Uintermediate, iCell, {1, 1, 0});
    const real_t tmp222 = get_neighbor_value(U, Uintermediate, iCell, {1, 1, 1});
    // Interpolate to children
    const CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
    // foreach_sibling
    if( iCell_c0.iOct.isIntermediate ){
      const CellIndex iCell000 = iCell_c0.getNeighbor_ghost_intermediate({0, 0, 0}, Uintermediate.getShape());
      Uintermediate.at(iCell000, Isolution) += tmp0
          + f1 * (tmp011 + tmp101 + tmp110)
          + f2 * (tmp001 + tmp010 + tmp100)
          + f3 * tmp000;
      const CellIndex iCell001 = iCell_c0.getNeighbor_ghost_intermediate({0, 0, 1}, Uintermediate.getShape());
      Uintermediate.at(iCell001, Isolution) += tmp0
          + f1 * (tmp011 + tmp101 + tmp112)
          + f2 * (tmp001 + tmp012 + tmp102)
          + f3 * tmp002;
      const CellIndex iCell010 = iCell_c0.getNeighbor_ghost_intermediate({0, 1, 0}, Uintermediate.getShape());
      Uintermediate.at(iCell010, Isolution) += tmp0
          + f1 * (tmp011 + tmp121 + tmp110)
          + f2 * (tmp021 + tmp010 + tmp120)
          + f3 * tmp020;
      const CellIndex iCell011 = iCell_c0.getNeighbor_ghost_intermediate({0, 1, 1}, Uintermediate.getShape());
      Uintermediate.at(iCell011, Isolution) += tmp0
              + f1 * (tmp011 + tmp121 + tmp112)
              + f2 * (tmp021 + tmp012 + tmp122)
              + f3 * tmp022;
      const CellIndex iCell100 = iCell_c0.getNeighbor_ghost_intermediate({1, 0, 0}, Uintermediate.getShape());
      Uintermediate.at(iCell100, Isolution) += tmp0
              + f1 * (tmp211 + tmp101 + tmp110)
              + f2 * (tmp201 + tmp210 + tmp100)
              + f3 * tmp200;
      const CellIndex iCell101 = iCell_c0.getNeighbor_ghost_intermediate({1, 0, 1}, Uintermediate.getShape());
      Uintermediate.at(iCell101, Isolution) += tmp0
              + f1 * (tmp211 + tmp101 + tmp112)
              + f2 * (tmp201 + tmp212 + tmp102)
              + f3 * tmp202;
      const CellIndex iCell110 = iCell_c0.getNeighbor_ghost_intermediate({1, 1, 0}, Uintermediate.getShape());
      Uintermediate.at(iCell110, Isolution) += tmp0
              + f1 * (tmp211 + tmp121 + tmp110)
              + f2 * (tmp221 + tmp210 + tmp120)
              + f3 * tmp220;
      const CellIndex iCell111 = iCell_c0.getNeighbor_ghost_intermediate({1, 1, 1}, Uintermediate.getShape());
      Uintermediate.at(iCell111, Isolution) += tmp0
              + f1 * (tmp211 + tmp121 + tmp112)
              + f2 * (tmp221 + tmp212 + tmp122)
              + f3 * tmp222;
    }
    else{
      const CellIndex iCell000 = iCell_c0.getNeighbor_ghost({0, 0, 0}, U.getShape());
      U.at(iCell000, Isolution) += tmp0
          + f1 * (tmp011 + tmp101 + tmp110)
          + f2 * (tmp001 + tmp010 + tmp100)
          + f3 * tmp000;
      const CellIndex iCell001 = iCell_c0.getNeighbor_ghost({0, 0, 1}, U.getShape());
      U.at(iCell001, Isolution) += tmp0
          + f1 * (tmp011 + tmp101 + tmp112)
          + f2 * (tmp001 + tmp012 + tmp102)
          + f3 * tmp002;
      const CellIndex iCell010 = iCell_c0.getNeighbor_ghost({0, 1, 0}, U.getShape());
      U.at(iCell010, Isolution) += tmp0
          + f1 * (tmp011 + tmp121 + tmp110)
          + f2 * (tmp021 + tmp010 + tmp120)
          + f3 * tmp020;
      const CellIndex iCell011 = iCell_c0.getNeighbor_ghost({0, 1, 1}, U.getShape());
      U.at(iCell011, Isolution) += tmp0
              + f1 * (tmp011 + tmp121 + tmp112)
              + f2 * (tmp021 + tmp012 + tmp122)
              + f3 * tmp022;
      const CellIndex iCell100 = iCell_c0.getNeighbor_ghost({1, 0, 0}, U.getShape());
      U.at(iCell100, Isolution) += tmp0
              + f1 * (tmp211 + tmp101 + tmp110)
              + f2 * (tmp201 + tmp210 + tmp100)
              + f3 * tmp200;
      const CellIndex iCell101 = iCell_c0.getNeighbor_ghost({1, 0, 1}, U.getShape());
      U.at(iCell101, Isolution) += tmp0
              + f1 * (tmp211 + tmp101 + tmp112)
              + f2 * (tmp201 + tmp212 + tmp102)
              + f3 * tmp202;
      const CellIndex iCell110 = iCell_c0.getNeighbor_ghost({1, 1, 0}, U.getShape());
      U.at(iCell110, Isolution) += tmp0
              + f1 * (tmp211 + tmp121 + tmp110)
              + f2 * (tmp221 + tmp210 + tmp120)
              + f3 * tmp220;
      const CellIndex iCell111 = iCell_c0.getNeighbor_ghost({1, 1, 1}, U.getShape());
      U.at(iCell111, Isolution) += tmp0
              + f1 * (tmp211 + tmp121 + tmp112)
              + f2 * (tmp221 + tmp212 + tmp122)
              + f3 * tmp222;
    }
  }); 
}

void GravitySolver_multigrid::prolongation_from_parents_on_intermediate(const level_t level) {
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  constexpr real_t f0 = 27.0 / 64;
  constexpr real_t f1 = 9.0 / 64;
  constexpr real_t f2 = 3.0 / 64;
  constexpr real_t f3 = 1.0 / 64;
  const auto octs_intermediate = get_subview_octs_intermediate(level - 1);
  foreach_cell.foreach_intermediate_cell_in_octants( "Prolongation", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    // Get coarse cell values to interpolate from
    const real_t tmp111 = Uintermediate.at( iCell, Isolution );
    const real_t tmp0 = f0 * tmp111;

    const real_t tmp000 = get_neighbor_value(U, Uintermediate, iCell, {-1, -1, -1});
    const real_t tmp001 = get_neighbor_value(U, Uintermediate, iCell, {-1, -1, 0});
    const real_t tmp002 = get_neighbor_value(U, Uintermediate, iCell, {-1, -1, 1});
    const real_t tmp010 = get_neighbor_value(U, Uintermediate, iCell, {-1, 0, -1});
    const real_t tmp011 = get_neighbor_value(U, Uintermediate, iCell, {-1, 0, 0});
    const real_t tmp012 = get_neighbor_value(U, Uintermediate, iCell, {-1, 0, 1});
    const real_t tmp020 = get_neighbor_value(U, Uintermediate, iCell, {-1, 1, -1});
    const real_t tmp021 = get_neighbor_value(U, Uintermediate, iCell, {-1, 1, 0});
    const real_t tmp022 = get_neighbor_value(U, Uintermediate, iCell, {-1, 1, 1});
    const real_t tmp100 = get_neighbor_value(U, Uintermediate, iCell, {0, -1, -1});
    const real_t tmp101 = get_neighbor_value(U, Uintermediate, iCell, {0, -1, 0});
    const real_t tmp102 = get_neighbor_value(U, Uintermediate, iCell, {0, -1, 1});
    const real_t tmp110 = get_neighbor_value(U, Uintermediate, iCell, {0, 0, -1});
    const real_t tmp112 = get_neighbor_value(U, Uintermediate, iCell, {0, 0, 1});
    const real_t tmp120 = get_neighbor_value(U, Uintermediate, iCell, {0, 1, -1});
    const real_t tmp121 = get_neighbor_value(U, Uintermediate, iCell, {0, 1, 0});
    const real_t tmp122 = get_neighbor_value(U, Uintermediate, iCell, {0, 1, 1});
    const real_t tmp200 = get_neighbor_value(U, Uintermediate, iCell, {1, -1, -1});
    const real_t tmp201 = get_neighbor_value(U, Uintermediate, iCell, {1, -1, 0});
    const real_t tmp202 = get_neighbor_value(U, Uintermediate, iCell, {1, -1, 1});
    const real_t tmp210 = get_neighbor_value(U, Uintermediate, iCell, {1, 0, -1});
    const real_t tmp211 = get_neighbor_value(U, Uintermediate, iCell, {1, 0, 0});
    const real_t tmp212 = get_neighbor_value(U, Uintermediate, iCell, {1, 0, 1});
    const real_t tmp220 = get_neighbor_value(U, Uintermediate, iCell, {1, 1, -1});
    const real_t tmp221 = get_neighbor_value(U, Uintermediate, iCell, {1, 1, 0});
    const real_t tmp222 = get_neighbor_value(U, Uintermediate, iCell, {1, 1, 1});
    // Interpolate to children
    const CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
    // foreach_sibling
    if( iCell_c0.iOct.isIntermediate ){
      const CellIndex iCell000 = iCell_c0.getNeighbor_ghost_intermediate({0, 0, 0}, Uintermediate.getShape());
      Uintermediate.at(iCell000, Isolution) += tmp0
          + f1 * (tmp011 + tmp101 + tmp110)
          + f2 * (tmp001 + tmp010 + tmp100)
          + f3 * tmp000;
      const CellIndex iCell001 = iCell_c0.getNeighbor_ghost_intermediate({0, 0, 1}, Uintermediate.getShape());
      Uintermediate.at(iCell001, Isolution) += tmp0
          + f1 * (tmp011 + tmp101 + tmp112)
          + f2 * (tmp001 + tmp012 + tmp102)
          + f3 * tmp002;
      const CellIndex iCell010 = iCell_c0.getNeighbor_ghost_intermediate({0, 1, 0}, Uintermediate.getShape());
      Uintermediate.at(iCell010, Isolution) += tmp0
          + f1 * (tmp011 + tmp121 + tmp110)
          + f2 * (tmp021 + tmp010 + tmp120)
          + f3 * tmp020;
      const CellIndex iCell011 = iCell_c0.getNeighbor_ghost_intermediate({0, 1, 1}, Uintermediate.getShape());
      Uintermediate.at(iCell011, Isolution) += tmp0
              + f1 * (tmp011 + tmp121 + tmp112)
              + f2 * (tmp021 + tmp012 + tmp122)
              + f3 * tmp022;
      const CellIndex iCell100 = iCell_c0.getNeighbor_ghost_intermediate({1, 0, 0}, Uintermediate.getShape());
      Uintermediate.at(iCell100, Isolution) += tmp0
              + f1 * (tmp211 + tmp101 + tmp110)
              + f2 * (tmp201 + tmp210 + tmp100)
              + f3 * tmp200;
      const CellIndex iCell101 = iCell_c0.getNeighbor_ghost_intermediate({1, 0, 1}, Uintermediate.getShape());
      Uintermediate.at(iCell101, Isolution) += tmp0
              + f1 * (tmp211 + tmp101 + tmp112)
              + f2 * (tmp201 + tmp212 + tmp102)
              + f3 * tmp202;
      const CellIndex iCell110 = iCell_c0.getNeighbor_ghost_intermediate({1, 1, 0}, Uintermediate.getShape());
      Uintermediate.at(iCell110, Isolution) += tmp0
              + f1 * (tmp211 + tmp121 + tmp110)
              + f2 * (tmp221 + tmp210 + tmp120)
              + f3 * tmp220;
      const CellIndex iCell111 = iCell_c0.getNeighbor_ghost_intermediate({1, 1, 1}, Uintermediate.getShape());
      Uintermediate.at(iCell111, Isolution) += tmp0
              + f1 * (tmp211 + tmp121 + tmp112)
              + f2 * (tmp221 + tmp212 + tmp122)
              + f3 * tmp222;
    }
  }); 
}


/**
 * @brief Set solution to zero.
 * 
 * Set solution to zero at a given level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
 * @return the residual at a given level 
*/
bool GravitySolver_multigrid::check_parents()
{
  const auto& U = pdata->U;
  const auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  foreach_cell.foreach_cell("Set solution to zero", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    auto level = cells.getLevel(iCell);
    if (level > 0)
      [[maybe_unused]] auto iCell_p = iCell.getParent(U.getShape());
  });
  foreach_cell.foreach_intermediate_cell("Set solution to zero", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    auto level = cells.getLevel(iCell);
    if (level > 0)
      [[maybe_unused]] auto iCell_p = iCell.getParent(Uintermediate.getShape());
  });
  return true;
}

/**
 * @brief Initialise mask.
 * 
 * Initialise mask to 1 at a given level, and -1 elsewhere 
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param finest_level[in]: The grid level
*/
void GravitySolver_multigrid::initialise_mask(const uint32_t finest_level)
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const auto octs = get_subview_octs(0, finest_level);
  foreach_cell.foreach_cell_in_octants("Initialise mask", U.getShape(), octs,
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level < finest_level) U.at(iCell, Imask) = -1;
    else if ( current_level == finest_level )  U.at(iCell, Imask) = 1;
    
  });
  const auto octs_intermediate = get_subview_octs_intermediate(0, finest_level);
  foreach_cell.foreach_intermediate_cell_in_octants("Initialise mask", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level < finest_level) Uintermediate.at(iCell, Imask) = -1;
    else if ( current_level == finest_level )  Uintermediate.at(iCell, Imask) = 1;
  });
}


/**
 * @brief Set solution to zero.
 * 
 * Set solution to zero at a given level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
 * @return the residual at a given level 
*/
void GravitySolver_multigrid::zero_solution(const level_t level)
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const auto octs = get_subview_octs(level);
  foreach_cell.foreach_cell_in_octants("Set solution to zero", U.getShape(), octs,
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    U.at(iCell, Isolution) = 0;
  });
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Set solution to zero", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    Uintermediate.at(iCell, Isolution) = 0;
  });
}

/**
 * @brief Set solution, residual and right-hand side to zero.
 * 
 * Set solution, residual and right-hand side to zero
 * at a given level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
void GravitySolver_multigrid::zero_solution_residual_rhs(const level_t level)
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const auto octs = get_subview_octs(0, level);
  foreach_cell.foreach_cell_in_octants("Set MG fields to zero", U.getShape(), octs,
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    U.at(iCell, Isolution) = 0;
    U.at(iCell, Irhs) = 0;
    U.at(iCell, Iresidual) = 0;
  });
  const auto octs_intermediate = get_subview_octs_intermediate(0, level);
  foreach_cell.foreach_intermediate_cell_in_octants("Set MG fields to zero", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    Uintermediate.at(iCell, Isolution) = 0;
    Uintermediate.at(iCell, Irhs) = 0;
    Uintermediate.at(iCell, Iresidual) = 0;
  });
}

/**
 * @brief Set solution, residual and right-hand side to zero.
 * 
 * Set solution, residual and right-hand side to zero
 * at a given level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
void GravitySolver_multigrid::zero_rhs_mask_intermediate(const level_t level)
{
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Set intermediate rhs and mask to zero", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    Uintermediate.at(iCell, Irhs) = 0;
    Uintermediate.at(iCell, Imask) = 0;
  });
  const auto ghosts_intermediate = get_subview_ghosts_intermediate(level);
  foreach_cell.foreach_intermediate_ghost_cell_in_octants("Set intermediate rhs and mask to zero", Uintermediate.getShape(), ghosts_intermediate,
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    Uintermediate.at(iCell, Irhs) = 0;
    Uintermediate.at(iCell, Imask) = 0;
  });
}


/**
 * @brief Copy solution to potential.
 * 
 * Copy solution to potential at a given level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
void GravitySolver_multigrid::solution_to_potential(const level_t level)
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const auto octs = get_subview_octs(level);
  foreach_cell.foreach_cell_in_octants("Copy solution to potential", U.getShape(), octs,
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    U.at(iCell, Iphi) = U.at(iCell, Isolution);
  });
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Copy solution to potential", Uintermediate.getShape(), octs_intermediate,
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    Uintermediate.at(iCell, Iphi) = Uintermediate.at(iCell, Isolution);
  });
}


template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void get_contrib_intermediate_uniform (Array_t& U, Array_t& Uintermediate, const CellIndex& iCell, int side, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  CellIndex::offset_t offset = {}; offset[dir] = side;
  CellIndex iCell_X = iCell.getNeighbor_ghost_intermediate(offset, Uintermediate.getShape());
  DYABLO_ASSERT_KOKKOS_DEBUG( iCell_X.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
  if ( CellIndex::BIGGER == iCell_X.status ) {
    iCell_X = iCell.getNeighbor_ghost(offset, U.getShape());
    contrib = U.at(iCell_X, Isolution);
  } else contrib = Uintermediate.at(iCell_X, Isolution);
  if ( BC_ABSORBING == boundarycondition[dir] && iCell_X.is_boundary() ) contrib = 0;
};

template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void get_contrib_leaves_uniform (Array_t& U, Array_t& Uintermediate, const CellIndex& iCell, int side, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  CellIndex::offset_t offset = {}; offset[dir] = side;
  CellIndex iCell_X = iCell.getNeighbor_ghost(offset, U.getShape());
  DYABLO_ASSERT_KOKKOS_DEBUG( iCell_X.level_diff() <= 0, "Leaf cell at coarse level cannot have coarser leaf neighbor" );
  if ( CellIndex::SMALLER == iCell_X.status ) {
    iCell_X = iCell.getNeighbor_ghost_intermediate(offset, Uintermediate.getShape());
    contrib = Uintermediate.at(iCell_X, Isolution);
  } else contrib = U.at(iCell_X, Isolution);
  if ( BC_ABSORBING == boundarycondition[dir] && iCell_X.is_boundary() ) contrib = 0;
};

/**
 * @brief Residual on a uniform grid.
 * 
 * Residual of a Poisson equation on a uniform grid
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
void GravitySolver_multigrid::residual_uniform(const level_t level) 
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const auto& iter_space = U.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;

  const auto octs = get_subview_octs(level);
  foreach_cell.foreach_cell_in_octants("Residual", U.getShape(), octs,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    real_t neighbors(0);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t contrib_L(0), contrib_R(0);
      get_contrib_leaves_uniform(U, Uintermediate, iCell, -1, contrib_L, dir, boundarycondition);
      get_contrib_leaves_uniform(U, Uintermediate, iCell, +1, contrib_R, dir, boundarycondition);
      neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
    }
    const real_t laplacian_solution = neighbors - U.at(iCell, Isolution) * ( 2./(size[IX]*size[IX]) + 2./(size[IY]*size[IY]) + 2./(size[IZ]*size[IZ]) );
    U.at(iCell, Iresidual) = U.at(iCell, Irhs) - laplacian_solution;
  });
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Residual", Uintermediate.getShape(), octs_intermediate,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    real_t neighbors(0);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t contrib_L(0), contrib_R(0);
      get_contrib_intermediate_uniform(U, Uintermediate, iCell, -1, contrib_L, dir, boundarycondition);
      get_contrib_intermediate_uniform(U, Uintermediate, iCell, +1, contrib_R, dir, boundarycondition);
      neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
    }
    const real_t laplacian_solution = neighbors - Uintermediate.at(iCell, Isolution) * ( 2./(size[IX]*size[IX]) + 2./(size[IY]*size[IY]) + 2./(size[IZ]*size[IZ]) );
    Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
  });
}


template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void get_weight_and_contrib_intermediate_amr_correction (Array_t& Uintermediate, const CellIndex& iCell, int side, real_t& w, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition, const real_t mask) 
{
  CellIndex::offset_t offset = {}; offset[dir] = side;
  const CellIndex iCell_X = iCell.getNeighbor_ghost_intermediate(offset, Uintermediate.getShape());
  DYABLO_ASSERT_KOKKOS_DEBUG( iCell_X.level_diff() >= 0, "Intermediate L cell cannot have smaller intermediate neighbor" );
  if (iCell_X.status == CellIndex::BIGGER) w = 0.5;
  else if (Uintermediate.at(iCell_X, Imask) <= 0) {
    real_t neighbor_mask = Uintermediate.at(iCell_X, Imask);
    w = mask / (mask - neighbor_mask);
  } //else if (Uintermediate.at(iCell_X, Imask) < 1) w = 1;  // First-order reconstruction 
  else contrib = Uintermediate.at(iCell_X, Isolution);
  if (boundarycondition[dir] == BC_ABSORBING && iCell_X.is_boundary()) contrib = 0;
};
/**
 * @brief Residual on intermediate cells in AMR levels.
 * 
 * Residual of the Poisson equation on intermediate cells in AMR levels 
 * 
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
void GravitySolver_multigrid::residual_intermediate_amr_correction(const level_t level) 
{
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const auto& iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;

  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Residual", Uintermediate.getShape(), octs_intermediate,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    if (Uintermediate.at(iCell, Imask) <= 0) return;

    real_t laplacian_solution(0);
    const real_t central_solution = Uintermediate.at(iCell, Isolution);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t a(1), b(1), contrib_L(0), contrib_R(0);
      const real_t mask = Uintermediate.at(iCell, Imask);
      get_weight_and_contrib_intermediate_amr_correction(Uintermediate, iCell, -1, a, contrib_L, dir, boundarycondition, mask);
      get_weight_and_contrib_intermediate_amr_correction(Uintermediate, iCell, +1, b, contrib_R, dir, boundarycondition, mask);
      const real_t f_L = 2. / (a * (a + b));
      const real_t f_R = 2. / (b * (a + b));
      const real_t f_C = 2. / (a * b);
      laplacian_solution += (f_L*contrib_L + f_R*contrib_R - f_C*central_solution) / (size[dir] * size[dir]);
    }
    Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
  });
}


template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void get_weight_and_contrib_amr_finest (Array_t& U, Array_t& Uintermediate, const CellIndex& iCell, int side, real_t& w, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  CellIndex::offset_t offset = {}; offset[dir] = side;
  CellIndex iCell_X = iCell.getNeighbor_ghost(offset, U.getShape());
  if ( CellIndex::SMALLER == iCell_X.status ) {
    iCell_X = iCell.getNeighbor_ghost_intermediate(offset, Uintermediate.getShape());
    contrib = Uintermediate.at(iCell_X, Isolution);
  } else if (CellIndex::BIGGER == iCell_X.status) {
    contrib = GravitySolver_multigrid::average_8bigger_neighbors(U, Uintermediate, iCell, offset);
    w = 0.5;
  } else contrib = U.at(iCell_X, Isolution);
  if ( BC_ABSORBING == boundarycondition[dir] && iCell_X.is_boundary() ) contrib = 0;
};

/**
 * @brief Residual on a AMR level.
 * 
 * Residual of a Poisson equation on an AMR level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
void GravitySolver_multigrid::residual_amr_finest(const level_t level) 
{
  auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const auto& iter_space = U.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;

  const auto octs = get_subview_octs(level);
  foreach_cell.foreach_cell_in_octants("Residual", U.getShape(), octs,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    real_t laplacian_solution(0);
    const real_t central_solution = U.at(iCell, Isolution);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t a(1), b(1), contrib_L(0), contrib_R(0);
      get_weight_and_contrib_amr_finest(U, Uintermediate, iCell, -1, a, contrib_L, dir, boundarycondition);
      get_weight_and_contrib_amr_finest(U, Uintermediate, iCell, +1, b, contrib_R, dir, boundarycondition);
      const real_t f_L = 2. / (a * (a + b));
      const real_t f_R = 2. / (b * (a + b));
      const real_t f_C = 2. / (a * b);
      laplacian_solution += (f_L*contrib_L + f_R*contrib_R - f_C*central_solution) / (size[dir] * size[dir]);
    }
    U.at(iCell, Iresidual) = U.at(iCell, Irhs) - laplacian_solution;
  });
  const auto octs_intermediate = get_subview_octs_intermediate(level);
  foreach_cell.foreach_intermediate_cell_in_octants("Residual", Uintermediate.getShape(), octs_intermediate,
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    real_t neighbors(0);
    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      real_t contrib_L(0), contrib_R(0);
      get_contrib_intermediate_uniform(U, Uintermediate, iCell, -1, contrib_L, dir, boundarycondition);
      get_contrib_intermediate_uniform(U, Uintermediate, iCell, +1, contrib_R, dir, boundarycondition);
      neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
    }
    const real_t laplacian_solution = neighbors - Uintermediate.at(iCell, Isolution) * ( 2./(size[IX]*size[IX]) + 2./(size[IY]*size[IY]) + 2./(size[IZ]*size[IZ]) );
    Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
  });
}

/**
 * @brief Gauss-Seidel sweep on intermediate at AMR level.
 * 
 * Gauss-Seidel sweep on a correction
 * on intermediate cells at an AMR level
 * 
 * @param Uintermediate[in]: The intermediate data to read from
 * @param is_coloured[in]: function that check if cell is coloured
 * @param level[in]: The grid level
*/
template< typename Array_t > 
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::gauss_seidel_intermediate_amr_correction(Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  if (Uintermediate.at(iCell, Imask) <= 0) 
    return;
  
  Kokkos::Array<real_t, 3> f_C;
  real_t neighbors(0);
  constexpr real_t w_relax(1.);
  for ( const ComponentIndex3D dir : {IX,IY,IZ} )
  {
    real_t a(1), b(1), contrib_L(0), contrib_R(0);
    const real_t mask = Uintermediate.at(iCell, Imask);
    get_weight_and_contrib_intermediate_amr_correction(Uintermediate, iCell, -1, a, contrib_L, dir, boundarycondition, mask);
    get_weight_and_contrib_intermediate_amr_correction(Uintermediate, iCell, +1, b, contrib_R, dir, boundarycondition, mask);
    const real_t f_L = 2. / (a * (a + b));
    const real_t f_R = 2. / (b * (a + b));
    f_C[dir] = 2. / (a * b);
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
 * @brief Gauss-Seidel sweep on leaves at AMR level.
 * 
 * Gauss-Seidel sweep on solution (potential)
 * on leaf cells at an AMR level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param is_coloured[in]: function that check if cell is coloured
 * @param level[in]: The grid level
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::gauss_seidel_leaves_amr_finest(Array_t& U, Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  Kokkos::Array<real_t, 3> f_C;
  real_t neighbors(0);
  constexpr real_t w_relax(1.);
  
  for ( const ComponentIndex3D dir : {IX,IY,IZ} )
  {
    real_t a(1), b(1), contrib_L(0), contrib_R(0);
    get_weight_and_contrib_amr_finest(U, Uintermediate, iCell, -1, a, contrib_L, dir, boundarycondition);
    get_weight_and_contrib_amr_finest(U, Uintermediate, iCell, +1, b, contrib_R, dir, boundarycondition);
    const real_t f_L = 2. / (a * (a + b));
    const real_t f_R = 2. / (b * (a + b));
    f_C[dir] = 2. / (a * b);
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
 * @brief Gauss-Seidel sweep on leaves on a uniform grid.
 * 
 * Gauss-Seidel sweep on leaf cells on a uniform grid
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param is_coloured[in]: function that check if cell is coloured
 * @param level[in]: The grid level
*/
template< typename Array_t > 
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::gauss_seidel_leaves_uniform(Array_t& U, Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  real_t neighbors(0);
  constexpr real_t w_relax(1.);
  for ( const ComponentIndex3D dir : {IX,IY,IZ} )
  {
    real_t contrib_L(0), contrib_R(0);
    get_contrib_leaves_uniform(U, Uintermediate, iCell, -1, contrib_L, dir, boundarycondition);
    get_contrib_leaves_uniform(U, Uintermediate, iCell, +1, contrib_R, dir, boundarycondition);
    neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);             
  }
  const real_t rhs = U.at(iCell, Irhs);
  const real_t sol = U.at(iCell, Isolution);
  const real_t denom =
    2. / (size[IX] * size[IX]) +
    2. / (size[IY] * size[IY]) +
    2. / (size[IZ] * size[IZ]);
    
  U.at(iCell, Isolution) += w_relax * ( (neighbors - rhs) / denom - sol);
}

/**
 * @brief Gauss-Seidel sweep on intermediate cells.
 * 
 * Gauss-Seidel sweep on intermediate cells 
 * without mask
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param is_coloured[in]: function that check if cell is coloured
 * @param level[in]: The grid level
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::gauss_seidel_intermediate(Array_t& U, Array_t& Uintermediate, const CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition) 
{
  real_t neighbors(0);
  constexpr real_t w_relax(1.);
  for ( ComponentIndex3D dir : {IX,IY,IZ} )
  {
    real_t contrib_L(0), contrib_R(0);
    get_contrib_intermediate_uniform(U, Uintermediate, iCell, -1, contrib_L, dir, boundarycondition);
    get_contrib_intermediate_uniform(U, Uintermediate, iCell, +1, contrib_R, dir, boundarycondition);
    neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
  }
  const real_t rhs = Uintermediate.at(iCell, Irhs);
  const real_t sol = Uintermediate.at(iCell, Isolution);
  const real_t denom =
    2. / (size[IX] * size[IX]) +
    2. / (size[IY] * size[IY]) +
    2. / (size[IZ] * size[IZ]);
  Uintermediate.at(iCell, Isolution) += w_relax * ( (neighbors - rhs) / denom - sol );
}

template<bool is_red>
void GravitySolver_multigrid::gauss_seidel_intermediate_amr_correction_rb(const level_t level) 
{
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto& iter_space = Uintermediate.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  const auto octs_intermediate = get_subview_octs_intermediate(level);

  auto apply_gauss_seidel = KOKKOS_LAMBDA(const CellIndex& iCell) {
    gauss_seidel_intermediate_amr_correction(Uintermediate, iCell, size, boundarycondition);
  };

  if (is_red) {
    foreach_cell.foreach_intermediate_red_cell_in_octants("Gauss-Seidel", Uintermediate.getShape(), octs_intermediate, apply_gauss_seidel);
  } else {
    foreach_cell.foreach_intermediate_black_cell_in_octants("Gauss-Seidel", Uintermediate.getShape(), octs_intermediate, apply_gauss_seidel);
  }
}

template< bool is_red >
void GravitySolver_multigrid::gauss_seidel_leaves_amr_finest_rb(const level_t level) 
{
  auto& U = pdata->U;
  const auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto& iter_space = U.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  const auto octs = get_subview_octs(level);

  auto apply_gauss_seidel = KOKKOS_LAMBDA(const CellIndex& iCell) {
    gauss_seidel_leaves_amr_finest(U, Uintermediate, iCell, size, boundarycondition);
  };

  if (is_red) {
    foreach_cell.foreach_red_cell_in_octants("Gauss-Seidel", U.getShape(), octs, apply_gauss_seidel);
  } else {
    foreach_cell.foreach_black_cell_in_octants("Gauss-Seidel", U.getShape(), octs, apply_gauss_seidel);
  }
}

template< bool is_red >
void GravitySolver_multigrid::gauss_seidel_leaves_uniform_rb(const level_t level) 
{
  auto& U = pdata->U;
  const auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto& iter_space = U.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  const auto octs = get_subview_octs(level);

  auto apply_gauss_seidel = KOKKOS_LAMBDA(const CellIndex& iCell) {
    gauss_seidel_leaves_uniform(U, Uintermediate, iCell, size, boundarycondition);
  };

  if (is_red) {
    foreach_cell.foreach_red_cell_in_octants("Gauss-Seidel", U.getShape(), octs, apply_gauss_seidel);
  } else {
    foreach_cell.foreach_black_cell_in_octants("Gauss-Seidel", U.getShape(), octs, apply_gauss_seidel);
  }
}

  
template< bool is_red >
void GravitySolver_multigrid::gauss_seidel_intermediate_rb(const level_t level) 
{
  const auto& U = pdata->U;
  auto& Uintermediate = pdata->Uintermediate;
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const auto& iter_space = U.getShape();
  const uint32_t nocts1d = 1U << level;
  const Kokkos::Array<real_t, 3> size = {
        1./(nocts1d * iter_space.bx), 
        1./(nocts1d * iter_space.by), 
        1./(nocts1d * iter_space.bz)
  };
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;
  const auto octs_intermediate = get_subview_octs_intermediate(level);

  auto apply_gauss_seidel = KOKKOS_LAMBDA(const CellIndex& iCell) {
    gauss_seidel_intermediate(U, Uintermediate, iCell, size, boundarycondition);
  };

  if (is_red) {
    foreach_cell.foreach_intermediate_red_cell_in_octants("Gauss-Seidel", Uintermediate.getShape(), octs_intermediate, apply_gauss_seidel);
  } else {
    foreach_cell.foreach_intermediate_black_cell_in_octants("Gauss-Seidel", Uintermediate.getShape(), octs_intermediate, apply_gauss_seidel);
  }
}

/**
 * @brief Smoothing procedure on intermediate cells at AMR level
 * 
 * Smoothing (several Gauss-Seildel sweeps) on intermediate cells
 * on the correction at AMR level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param nIterations[in]: Number of Gauss-Seidel sweeps
 * @param level[in]: The grid level
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

    if (isMPILevel) 
      exchange_specific_intermediate_ghosts_at_level(U_, level, {"solution"}, ghost_comm);

    gauss_seidel_intermediate_amr_correction_rb<is_black>(level);

    if (isMPILevel)
      exchange_specific_intermediate_ghosts_at_level(U_, level, {"solution"}, ghost_comm);
  }
}

/**
 * @brief Smoothing procedure on intermediate cells at AMR level
 * 
 * Smoothing (several Gauss-Seildel sweeps) on intermediate cells
 * on the correction at AMR level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param nIterations[in]: Number of Gauss-Seidel sweeps
 * @param level[in]: The grid level
*/
void GravitySolver_multigrid::smoothing_uniform(UserData& U_, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpicomm = foreach_cell.get_amr_mesh().getMpiComm();
  const bool isDistributed = mpicomm.MPI_Comm_size() > 1;
  const bool isMPILevel = isDistributed && (level >= pdata->first_mpi_multigrid_level);
  constexpr bool is_red = true;
  constexpr bool is_black = false;
  for (uint32_t i = 0; i < nIterations; i++) 
  {
    gauss_seidel_leaves_uniform_rb<is_red>(level);
    gauss_seidel_intermediate_rb<is_red>(level);

    if (isMPILevel) 
    {
      exchange_specific_leaf_ghosts_at_level(U_, level, {"solution"}, ghost_comm);
      exchange_specific_intermediate_ghosts_at_level(U_, level, {"solution"}, ghost_comm);
    }

    gauss_seidel_leaves_uniform_rb<is_black>(level);
    gauss_seidel_intermediate_rb<is_black>(level);

    if (isMPILevel)
    {
      exchange_specific_leaf_ghosts_at_level(U_, level, {"solution"}, ghost_comm);
      exchange_specific_intermediate_ghosts_at_level(U_, level, {"solution"}, ghost_comm);
    }
  }
}

/**
 * @brief Smoothing procedure on fine AMR level.
 * 
 * Smoothing (several Gauss-Seildel sweeps) the solution (potential)
 * on AMR level at AMR level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
 * @return the residual at a given level 
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

    if (isMPILevel)
    {
      exchange_specific_leaf_ghosts_at_level(U_, level, {"solution"}, ghost_comm);
      exchange_specific_intermediate_ghosts_at_level(U_, level, {"solution"}, ghost_comm);
    }

    gauss_seidel_leaves_amr_finest_rb<is_black>(level);
    gauss_seidel_intermediate_rb<is_black>(level);

    if (isMPILevel)
    {
      exchange_specific_leaf_ghosts_at_level(U_, level, {"solution"}, ghost_comm);
      exchange_specific_intermediate_ghosts_at_level(U_, level, {"solution"}, ghost_comm);
    }
  }
}

/**
 * @brief V cycle of "Multigrid".
 * 
 * V cycle on uniform Multigrid scheme
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
 * @return the residual at a given level 
*/
void GravitySolver_multigrid::V_cycle_uniform(UserData& U_, const level_t level, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide) 
{  
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpicomm = foreach_cell.get_amr_mesh().getMpiComm();
  const bool isDistributed = mpicomm.MPI_Comm_size() > 1;
  const bool isFirstMPILevel = isDistributed && (level == pdata->first_mpi_multigrid_level);
  const bool isMPILevel = isDistributed && (level >= pdata->first_mpi_multigrid_level);
  
  smoothing_uniform(U_, pdata->Npre, level, ghost_comm_minimal);
  residual_uniform(level);

  if (isMPILevel){
    exchange_specific_leaf_ghosts_at_level(U_, level, {"res"}, ghost_comm_minimal);
    exchange_specific_intermediate_ghosts_at_level(U_, level, {"res"}, ghost_comm_minimal);
  }

  zero_rhs_mask_intermediate(level - 1);
  restriction_from_children(level - 1);

  if (isFirstMPILevel) 
    reduce_nonMPI_levels(pdata->first_mpi_multigrid_level, make_array<int, 2>({Irhs, Imask}));
  else if (isMPILevel){
    reduce_specific_intermediate_ghosts_at_level(U_, level - 1, {"rhs", "mask"}, ghost_comm_blockwide);
  }

  initialise_intermediate_lhs(level - 1);
  if (isMPILevel)
    exchange_specific_intermediate_ghosts_at_level(U_, level - 1, {"solution", "rhs", "mask"}, ghost_comm_blockwide);

  if (level == 1) {
      smoothing_uniform(U_, pdata->Npre, level - 1, ghost_comm_blockwide);
  }
  else V_cycle_uniform(U_, level - 1, ghost_comm_minimal, ghost_comm_blockwide);
    
  if (level < pdata->level_coarse)
    prolongation_from_children_on_intermediate(level);
  else 
    prolongation_from_children(level);

  if (isMPILevel){
    exchange_specific_leaf_ghosts_at_level(U_, level, {"solution"}, ghost_comm_blockwide);
    exchange_specific_intermediate_ghosts_at_level(U_, level, {"solution"}, ghost_comm_blockwide);
  } 
  smoothing_uniform(U_, pdata->Npost, level, ghost_comm_blockwide);
}


/**
 * @brief V cycle of "Full Multigrid".
 * 
 * V cycle on AMR Full Multigrid scheme
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
 * @return the residual at a given level 
*/
void GravitySolver_multigrid::V_cycle_amr(UserData& U_, const uint8_t current_level, const uint32_t finest_level, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide) 
{  
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpicomm = foreach_cell.get_amr_mesh().getMpiComm();
  const bool isDistributed = mpicomm.MPI_Comm_size() > 1;
  const bool isFirstMPILevel = isDistributed && (current_level == pdata->first_mpi_multigrid_level);
  const bool isMPILevel = isDistributed && (current_level >= pdata->first_mpi_multigrid_level);

  // Full Multigrid

  if ( current_level == finest_level ) {
    smoothing_amr_finest(U_, pdata->Npre, current_level, ghost_comm_minimal);
    residual_amr_finest(current_level);
  } else {
    smoothing_intermediate_amr_correction(U_, pdata->Npre, current_level, ghost_comm_minimal);
    residual_intermediate_amr_correction(current_level);
  }

  if (isMPILevel){
    exchange_specific_leaf_ghosts_at_level(U_, current_level, {"res"}, ghost_comm_minimal); // TODO: useful here ? Check when only for intermediate levels
    exchange_specific_intermediate_ghosts_at_level(U_, current_level, {"res"}, ghost_comm_minimal);
  }

  zero_rhs_mask_intermediate(current_level - 1);
  restriction_from_children(current_level - 1);

  if (isFirstMPILevel) {
    reduce_nonMPI_levels(pdata->first_mpi_multigrid_level, make_array<int, 2>({Irhs, Imask}));
  } else if (isMPILevel){
    reduce_specific_intermediate_ghosts_at_level(U_, current_level - 1, {"rhs", "mask"}, ghost_comm_blockwide);
  }

  initialise_intermediate_lhs(current_level - 1);
  if (isMPILevel && !isFirstMPILevel){
    exchange_specific_intermediate_ghosts_at_level(U_, current_level - 1, {"solution", "rhs", "mask"}, ghost_comm_blockwide);
  }

  if ( finest_level - 3 == current_level ) { // TODO: finest - 2 seems to works aswell for spherical symmetry. Check for more realistic cases
      smoothing_intermediate_amr_correction(U_, pdata->Npre, pdata->level_coarse, ghost_comm_minimal);
  } else V_cycle_amr(U_, current_level - 1, finest_level, ghost_comm_minimal, ghost_comm_blockwide); 
  
  if ( current_level == finest_level ) {
    prolongation_from_children(current_level);
    if (isMPILevel){
      exchange_specific_leaf_ghosts_at_level(U_, current_level, {"solution"}, ghost_comm_minimal);
      exchange_specific_intermediate_ghosts_at_level(U_, current_level, {"solution"}, ghost_comm_minimal);
    }
    smoothing_amr_finest(U_, pdata->Npost, current_level, ghost_comm_minimal);
  } else {
    prolongation_from_children_on_intermediate(current_level); 
    if (isMPILevel){
      exchange_specific_intermediate_ghosts_at_level(U_, current_level, {"solution"}, ghost_comm_minimal);
    }

    smoothing_intermediate_amr_correction(U_, pdata->Npost, current_level, ghost_comm_minimal);
  }   
    
}

/**
 * @brief Count number of octs before a given level
 * 
 * Total number of octs before a given level
 * nbOcts_before_level(0) = 0
 * nbOcts_before_level(1) = 1
 * nbOcts_before_level(2) = 9
 * 
 * @param level[in]: level
 */
KOKKOS_INLINE_FUNCTION
uint32_t nbOcts_before_level(const uint32_t level)
{
  return ((1U << (3*level)) - 1) / 7;
}

/**
 * @brief Right hand term of the poisson equation in the non-cosmo case
 * 
 * The right handside of the Poisson equation for gravity in non-cosmological case
 * is 4*pi*G*rho
 * 
 * @param Uin[in]: Array to read the density from
 * @param iCell_Uin[in]: cell index where to read the density
 * @param rho_mean[in]: average value of rho in the box for periodic cases
 * @param four_Pi_G[in]: value of four*pi*G
 */
KOKKOS_INLINE_FUNCTION
real_t GravitySolver_multigrid::b(const UserData::FieldAccessor& Uin, const CellIndex& iCell_Uin, real_t rho_mean, real_t four_Pi_G)
{
  return four_Pi_G*(Uin.at(iCell_Uin, Irho)-rho_mean);
}

/**
 * @brief Right hand term of the poisson equation in the cosmo case
 * 
 * The right hand side of the Poisson equation for gravity in cosmological cases.
 * This expression comes from Martel & Shapiro 1998, eq. (38)
 * 
 * @param Uin[in]: Array to read the density from
 * @param iCell_Uin[in]: cell index where to read the density
 * @param rho_mean[in]: average value of rho in the box for periodic cases
 * @param aexp[in]: expansion factor
 * @param size[in]: sizes of the cell at iCell_Uin
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

template< typename T, size_t N >
void GravitySolver_multigrid::reduce_nonMPI_levels(const level_t first_mpi_multigrid_level, const Kokkos::Array<T, N> iFields) 
{
  uint32_t num_vars = N; // number of vars for each cell

  auto& Uintermediate = pdata->Uintermediate;
  ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const MpiComm& mpi_comm = foreach_cell.get_amr_mesh().getMpiComm();
  const LightOctree& lmesh = foreach_cell.get_amr_mesh().getLightOctree();
  const uint32_t bx=Uintermediate.getShape().bx, by=Uintermediate.getShape().by, bz=Uintermediate.getShape().bz ;
  const uint32_t nbCellsPerBlock = bx * by * bz;
  const uint32_t ncells_1d = 1U << (first_mpi_multigrid_level - 1); // Total number of octants at level (first_mpi_multigrid_level - 1)
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
    const auto iOct_cell = lmesh.getiOctIntermediateFromCoordinates(ix, iy, iz, first_mpi_multigrid_level - 1);

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
    {
      auto hostArray = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), deviceArray);
      mpi_comm.MPI_Allreduce(hostArray.data(), hostArray.data(), nbOcts * nbCellsPerBlock * num_vars, MpiComm::MPI_Op_t::SUM);
      Kokkos::deep_copy(deviceArray, hostArray);
    }  
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
    auto iOct_cell = lmesh.getiOctIntermediateFromCoordinates(ix, iy, iz, first_mpi_multigrid_level - 1);

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

void GravitySolver_multigrid::list_octants_per_level(const LightOctree& lmesh) 
// Count number of Leaf and Intermediate (+ ghosts) octs in AMR
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
      if (level == ilevel){
        if (final){
          octs_per_level(ilist + numOcts_tmp) = iOct;
        }
        ilist++;
      }
    }, numOcts_local);
    Kokkos::parallel_scan( "Count number of intermediate octs per level", Kokkos::RangePolicy<>(0, numIntermediateOcts),
      KOKKOS_LAMBDA( const uint32_t iOct, uint32_t& ilist, bool final )
    {
      const level_t level = lmesh.getLevel({iOct, false, true});
      if (level == ilevel){
        if (final){
          octs_intermediate_per_level(ilist + numIntermediateOcts_tmp) = iOct;
        }
        ilist++;
      }
    }, numIntermediateOcts_local);
    Kokkos::parallel_scan( "Count number of ghosts per level", Kokkos::RangePolicy<>(0, numGhosts),
      KOKKOS_LAMBDA( const uint32_t iOct, uint32_t& ilist, bool final )
    {
      const level_t level = lmesh.getLevel({iOct, true, false});
      if (level == ilevel){
        if (final){
          ghosts_per_level(ilist + numGhosts_tmp) = iOct;
        }
        ilist++;
      }
    }, numGhosts_local);
    Kokkos::parallel_scan( "Count number of intermediate ghosts per level", Kokkos::RangePolicy<>(0, numIntermediateGhosts),
      KOKKOS_LAMBDA( const uint32_t iOct, uint32_t& ilist, bool final )
    {
      const level_t level = lmesh.getLevel({iOct, true, true});
      if (level == ilevel){
        if (final){
          ghosts_intermediate_per_level(ilist + numIntermediateGhosts_tmp) = iOct;
        }
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

  /* for(level_t ilevel = 0; ilevel <= level_max; ilevel++){ 
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

template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<T, N> GravitySolver_multigrid::make_array(const Kokkos::Array<T, N>& vals) {
    return vals;
}
//} // namespace

template <class GhostComm >
void GravitySolver_multigrid::exchange_specific_leaf_ghosts(UserData& U_, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm)
{
  std::vector<UserData::FieldAccessor::FieldInfo> field_info;
  for(uint8_t i=0; i<exchange_vars.size(); i++)
    field_info.push_back( {exchange_vars[i],i} );
  auto Uexchange = U_.getAccessor(field_info);
  ghost_comm.exchange_ghosts( Uexchange );
};

template <class GhostComm >
void GravitySolver_multigrid::exchange_specific_intermediate_ghosts(UserData& U_, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm)
{
  std::vector<UserData::FieldAccessor::FieldInfo> field_info;
  for(uint8_t i=0; i<exchange_vars.size(); i++)
    field_info.push_back( {exchange_vars[i],i} );
  auto Uexchange = U_.getAccessor_intermediate(field_info);
  ghost_comm.exchange_intermediate_ghosts( Uexchange );
};

template <class GhostComm >
void GravitySolver_multigrid::exchange_specific_leaf_ghosts_at_level(UserData& U_, const uint8_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm)
{
  std::vector<UserData::FieldAccessor::FieldInfo> field_info;
  for(uint8_t i=0; i<exchange_vars.size(); i++)
    field_info.push_back( {exchange_vars[i],i} );
  auto Uexchange = U_.getAccessor(field_info);
  ghost_comm.exchange_ghosts_at_level( Uexchange, level );
};

template <class GhostComm >
void GravitySolver_multigrid::exchange_specific_intermediate_ghosts_at_level(UserData& U_, const uint8_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm)
{
  std::vector<UserData::FieldAccessor::FieldInfo> field_info;
  for(uint8_t i=0; i<exchange_vars.size(); i++)
    field_info.push_back( {exchange_vars[i],i} );
  auto Uexchange = U_.getAccessor_intermediate(field_info);
  ghost_comm.exchange_intermediate_ghosts_at_level( Uexchange, level );
};

template <class GhostComm >
void GravitySolver_multigrid::reduce_specific_leaf_ghosts_at_level(UserData& U_, const uint8_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm)
{
  std::vector<UserData::FieldAccessor::FieldInfo> field_info;
  for(uint8_t i=0; i<exchange_vars.size(); i++)
    field_info.push_back( {exchange_vars[i],i} );
  auto Uexchange = U_.getAccessor(field_info);
  ghost_comm.reduce_ghosts_at_level( Uexchange, level );
};

template <class GhostComm >
void GravitySolver_multigrid::reduce_specific_intermediate_ghosts_at_level(UserData& U_, const uint8_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm)
{
  std::vector<UserData::FieldAccessor::FieldInfo> field_info;
  for(uint8_t i=0; i<exchange_vars.size(); i++)
    field_info.push_back( {exchange_vars[i],i} );
  auto Uexchange = U_.getAccessor_intermediate(field_info);
  ghost_comm.reduce_intermediate_ghosts_at_level( Uexchange, level );
};

/**
 * @brief Solves the Poisson equation and updates the gravity field
 * 
 * @param U[inout]: UserData structure to update
 * @param scalar_data[in]: ScalarSimulationData structure
*/
void GravitySolver_multigrid::update_gravity_field( UserData& U_, ScalarSimulationData& scalar_data )
{

  pdata->timers.get("GravitySolver_multigrid").start();

  ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpi_comm = foreach_cell.get_amr_mesh().getMpiComm();
  const int mpi_rank = mpi_comm.MPI_Comm_rank();
  auto& amr_mesh = foreach_cell.get_amr_mesh();
  const level_t first_mpi_multigrid_level = pdata->first_mpi_multigrid_level;

  DYABLO_ASSERT_KOKKOS_DEBUG( first_mpi_multigrid_level <= pdata->level_coarse, "Full coarse level cannot be common to all processes" );

  // Create intermediate storage in LightOctree
  amr_mesh.updateLightOctreeWithIntermediates(first_mpi_multigrid_level);
  Kokkos::MinMax<uint32_t>::value_type minmax_result;
  {
    const LightOctree& lmesh = amr_mesh.getLightOctree();
    // Min/max AMR level, and intermediate per level
    const size_t numOctants = lmesh.getNumOctants();
    Kokkos::parallel_reduce ( " Get min/max level in AMR " , Kokkos::RangePolicy<>(0, numOctants) ,
      KOKKOS_LAMBDA ( const uint32_t iOct , Kokkos::MinMax<uint32_t>::value_type& minmax_result_tmp ) {
      const level_t level_tmp = lmesh.getLevel({iOct, false});
      if ( level_tmp > minmax_result_tmp.max_val ) minmax_result_tmp.max_val = level_tmp;
      if ( level_tmp < minmax_result_tmp.min_val ) minmax_result_tmp.min_val = level_tmp;
    } , Kokkos::MinMax<uint32_t>(minmax_result));
    DYABLO_ASSERT_KOKKOS_DEBUG( minmax_result.max_val <= lmesh.get_level_max(),  "Max level found in AMR should not be higher than that stored when building the tree" );
    DYABLO_ASSERT_KOKKOS_DEBUG( minmax_result.min_val == lmesh.get_level_min(), "Min level found in AMR different from coarse level" );
  }

  const level_t local_level_max_found = minmax_result.max_val;
  const uint32_t global_max_level_found = MPI_Allreduce_int_max(local_level_max_found);

  // Create intermediate storage and ghostmap in amr_mesh
  amr_mesh.init_intermediates(first_mpi_multigrid_level);

  // Add intermediate ghosts to the LightOctree
  amr_mesh.updateLightOctreeWithIntermediates(first_mpi_multigrid_level);
  const LightOctree lmesh = amr_mesh.getLightOctree();
  const level_t level_coarse = pdata->level_coarse = lmesh.get_level_min();

  U_.new_fields({"solution", "rhs",  "res", "mask" });
  U_.new_intermediate_fields( {"rho","gphi", "solution", "rhs", "res", "mask"} ); 
  UserData::FieldAccessor U = U_.getAccessor({
    {"rho", Irho},
    {"gphi", Iphi},
    {"solution", Isolution},
    {"rhs", Irhs},
    { "res", Iresidual },
    { "mask", Imask },
    });
  UserData::FieldAccessor Uintermediate = U_.getAccessor_intermediate({
    {"rho", Irho},
    {"gphi", Iphi},
    {"solution", Isolution},
    {"rhs", Irhs},
    { "res", Iresidual },
    { "mask", Imask },
    });
  
  pdata->U = U;
  pdata->Uintermediate = Uintermediate;
 
  // Compute ghost communicators
  const auto& iter_space = U.getShape();
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
  ghost_comm_minimal.sort_ghosts_by_levels(lmesh, U.getShape(), global_max_level_found);
  ghost_comm_blockwide.sort_ghosts_by_levels(lmesh, U.getShape(), global_max_level_found);
  ghost_comm_minimal.sort_intermediate_ghosts_by_levels(lmesh, U.getShape(), global_max_level_found);
  ghost_comm_blockwide.sort_intermediate_ghosts_by_levels(lmesh, U.getShape(), global_max_level_found);
  
  DYABLO_ASSERT_KOKKOS_DEBUG( check_parents(), "Parent check failed" );

  // Count and list octants per level
  count_octants_per_level(lmesh);
  list_octants_per_level(lmesh);

  // Compute rho mean
  real_t rho_mean = 0;
  const real_t xmin(pdata->xmin), ymin(pdata->ymin), zmin(pdata->zmin);
  const real_t xmax(pdata->xmax), ymax(pdata->ymax), zmax(pdata->zmax);
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData(); // Local copy of lmesh????
  
  foreach_cell.reduce_cell("Compute rho_mean", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell, real_t & update_rhomean)
  {
    const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
    const real_t rhoi = U.at(iCell, Irho);
    update_rhomean += rhoi * size[IX] * size[IY] * size[IZ];
  }, Kokkos::Sum<real_t>(rho_mean));
  const real_t Vtot = (xmax - xmin) * (ymax - ymin) * (zmax - zmin);
  rho_mean = MPI_Allreduce_scalar(rho_mean) / Vtot;
  printf("rhomean = %.5e\n", rho_mean);

  const bool cosmo_run = pdata->cosmo_run;
  real_t aexp = 0;
  if( cosmo_run )
    aexp = scalar_data.get<real_t>("aexp");
  const real_t four_Pi_G = pdata->four_Pi_G;  
  // Initialize RHS and solution on leaves

  foreach_cell.foreach_cell("Init RHS and potential", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
    const real_t rhs = (cosmo_run) ? b_cosmo(U, iCell, rho_mean, aexp) : b(U, iCell, rho_mean, four_Pi_G);//U.at(iCell, Irho) - rho_mean;
    U.at(iCell, Irhs) = rhs;
    U.at(iCell, Isolution) = -rhs / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ]));
  });
  exchange_specific_leaf_ghosts(U_, {"solution"}, ghost_comm_blockwide);
  // Initialize RHS on intermediate levels. Solution will be interpolated from coarser levels so no need to initialize
  constexpr int ns = 8; 

  for( uint32_t level = global_max_level_found - 1; level >= level_coarse; level-- )
  {
    const auto octs = get_subview_octs(level+1);
    foreach_cell.foreach_cell_in_octants( "Restrict", U.getShape(), octs, 
    KOKKOS_LAMBDA( CellIndex& iCell)
    {
      const CellIndex iCell_p = iCell.getParent(U.getShape());      
      Kokkos::atomic_add( &Uintermediate.at( iCell_p, Irho ), U.at( iCell, Irho ) / ns );
    });
    const auto octs_intermediate = get_subview_octs_intermediate(level+1);
    foreach_cell.foreach_intermediate_cell_in_octants( "Restrict", Uintermediate.getShape(), octs_intermediate,
    KOKKOS_LAMBDA( CellIndex& iCell)
    {
      const CellIndex iCell_p = iCell.getParent(Uintermediate.getShape());     
      Kokkos::atomic_add( &Uintermediate.at( iCell_p, Irho ), Uintermediate.at( iCell, Irho ) / ns );
    });
    reduce_specific_intermediate_ghosts_at_level(U_, level, {"rho"}, ghost_comm_blockwide);
    exchange_specific_intermediate_ghosts_at_level(U_, level, {"rho"}, ghost_comm_blockwide); // Useful?
  }
  foreach_cell.foreach_intermediate_cell( "Set RHS of Laplacian, based on rho", Uintermediate.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    Uintermediate.at( iCell, Irhs ) = (cosmo_run) ? b_cosmo(Uintermediate, iCell, rho_mean, aexp) : b(Uintermediate, iCell, rho_mean, four_Pi_G); //rho - rho_mean;
  }); 

  residual_uniform(level_coarse);
  solution_to_potential(level_coarse);
  const real_t residual = residual_norm(level_coarse);
  if (mpi_rank == 0) printf("Level %d Residual norm init  V %.8e\n", level_coarse, residual);

  // Multigrid
  if (mpi_rank == 0) printf("Coarse Multigrid\n");
  for(uint32_t i = 0; i < pdata->Ncycles; i++)
  { 
    V_cycle_uniform(U_, level_coarse, ghost_comm_minimal, ghost_comm_blockwide);
    residual_uniform(level_coarse);
    solution_to_potential(level_coarse);
    const real_t residual = residual_norm(level_coarse);
    if (mpi_rank == 0) printf("Level %d Residual norm after V %.8e\n", level_coarse, residual);
  }
  
  if (mpi_rank == 0) printf("AMR Multigrid\n");
  for (level_t ilevel = level_coarse+1; ilevel <= global_max_level_found; ilevel++) 
  {
    zero_solution(ilevel);
    prolongation_from_children(ilevel);
    zero_solution_residual_rhs(ilevel-1);
    solution_to_potential(ilevel);
    initialise_mask(ilevel);
    exchange_specific_leaf_ghosts_at_level(U_, ilevel - 1, {"gphi","solution"}, ghost_comm_minimal); 
    exchange_specific_intermediate_ghosts_at_level(U_, ilevel - 1, {"gphi","solution"}, ghost_comm_minimal);
    exchange_specific_leaf_ghosts_at_level(U_, ilevel, {"gphi", "solution"}, ghost_comm_minimal);
    exchange_specific_intermediate_ghosts_at_level(U_, ilevel, {"gphi", "solution"}, ghost_comm_minimal);
    for(uint32_t i = 0; i < pdata->Ncycles; i++)
    { 
      V_cycle_amr(U_, ilevel, ilevel, ghost_comm_minimal, ghost_comm_blockwide);
      residual_amr_finest(ilevel);
      const real_t residual = residual_norm(ilevel);
      if (mpi_rank == 0) printf("Level %d Residual norm after V %.8e\n", ilevel, residual);
    }
    solution_to_potential(ilevel);
    exchange_specific_leaf_ghosts_at_level(U_, ilevel, {"gphi"}, ghost_comm_minimal);
    exchange_specific_intermediate_ghosts_at_level(U_, ilevel, {"gphi"}, ghost_comm_minimal);
  }

  // Update force field in U from potential

  UserData::FieldAccessor Uout = U_.getAccessor({
    {"gx", Igx},
    {"gy", Igy},
    {"gz", Igz},
    {"gphi", Iphi}
  });

  UserData::FieldAccessor Uoutintermediate = U_.getAccessor_intermediate({
    {"gphi", Iphi},
  });

  printf("Now compute Force\n");
  gradient(Uout, Uoutintermediate);

  // TODO: Delete MG arrays 
  for (std::string name : {"solution", "rhs",  "res", "mask" })
    U_.delete_field(name);
  for (std::string name : {"rho","gphi", "solution", "rhs", "res", "mask"})
    U_.delete_intermediate_field(name);

  // TODO: Delete intermediate octree
  // amr_mesh.deleteIntermediates(); // This function needs to be written

  pdata->timers.get("GravitySolver_multigrid").stop();
}

}// namespace dyablo

FACTORY_REGISTER( dyablo::GravitySolverFactory, dyablo::GravitySolver_multigrid, "GravitySolver_multigrid" );
