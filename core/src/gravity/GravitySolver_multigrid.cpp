#include "GravitySolver_multigrid.h"

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
  bool print_mg_iter;

  uint32_t level_coarse, Npre, Npost, Ncycles;

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
      configMap.getValue<bool>("gravity", "print_mg_iter", false),
      4,
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
 * @brief Old Gradient method.
 * 
 * Old three-point gradient 
 * 
 * @param U[in]: The data to read from
 * @param dir[in]: Direction along wich we compute the gradient
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::gradient0(const Array_t& U)
{
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();

  pdata->foreach_cell.foreach_cell("Gravity_mg::construct_force_field", U.getShape(), 
    KOKKOS_LAMBDA(const CellIndex& iCell)
  { 
    const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
    const real_t phi_C = U.at(iCell, Iphi);

    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      CellIndex::offset_t off_L = {}; off_L[dir] = -1;
      CellIndex::offset_t off_R = {}; off_R[dir] = +1;
      const CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U);
      const CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U);
      const real_t phi_L = get_value( U, iCell_L, Iphi, off_L );
      const real_t phi_R = get_value( U, iCell_R, Iphi, off_R );

      // If neighbor is bigger h (which was dx_small) becomes ( dx_small/2 + dx_big/2 = 3/2*dx_small )
      real_t hl = size[dir];
      if( iCell_L.level_diff()==1 ) hl *= 1.5;
      if( iCell_L.level_diff()==-1 ) hl *= 0.75;
      real_t hr = size[dir];
      if( iCell_R.level_diff()==1 ) hr *= 1.5;
      if( iCell_R.level_diff()==-1 ) hr *= 0.75;
      
      real_t dphi_L = (phi_L - phi_C)/hl;
      real_t dphi_R = (phi_C - phi_R)/hr;

      if( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) dphi_L = 0;
      if( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) dphi_R = 0;

      Kokkos::Array< VarIndex, 3 > IG = {Igx, Igy, Igz};

      U.at(iCell, IG[dir]) = (dphi_L + dphi_R)/2; 
    }
  });
}

/**
 * @brief Gradient method.
 * 
 * Three-point gradient 
 * 
 * @param U[in]: The data to read from
 * @param dir[in]: Direction along wich we compute the gradient
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::gradient(const Array_t& U, const Array_t& Uintermediate)
{
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();

  pdata->foreach_cell.foreach_cell("Gravity_mg::construct_force_field", U.getShape(), 
    KOKKOS_LAMBDA(const CellIndex& iCell)
  { 
    const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);

    for ( ComponentIndex3D dir : {IX,IY,IZ} )
    {
      CellIndex::offset_t off_L = {}; off_L[dir] = -1;
      CellIndex::offset_t off_R = {}; off_R[dir] = +1;
      CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U);
      CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U);

      real_t phi_L(0), phi_R(0), a(1), b(1);

      if (iCell_L.status == CellIndex::BIGGER) {
        phi_L = average_8bigger_neighbors(U, Uintermediate, iCell, off_L);
        a = 0.5;
      } else if (iCell_L.status == CellIndex::SMALLER) {
        iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
        phi_L = Uintermediate.at(iCell_L, Iphi);
      } else phi_L = U.at(iCell_L, Iphi);

      if (iCell_R.status == CellIndex::BIGGER) {
        phi_R = average_8bigger_neighbors(U, Uintermediate, iCell, off_R);
        b = 0.5;
      } else if (iCell_R.status == CellIndex::SMALLER) {
        iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
        phi_R = Uintermediate.at(iCell_R, Iphi);
      } else phi_R = U.at(iCell_R, Iphi);
      

      if( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) phi_L = 0;
      if( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) phi_R = 0;

      Kokkos::Array< VarIndex, 3 > IG = {Igx, Igy, Igz};

      U.at(iCell, IG[dir]) = (phi_L - phi_R)/( size[dir] * (a + b) ); 
    }
  });
}

/**
 * @brief Method returning the value of a given cell.
 * 
 * If the cell index is pointing to a neighbor, then averaging is applied 
 * to the smaller neighbors
 * 
 * @param U[in]: The data to read from
 * @param iCell_U[in]: cell index where the data should be read
 * @param var[in]: variable to read in U
 * @param offset[in]: offset applied prior to calling the method to get iCell_U
 * @return the value of the variable at iCell_U 
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
real_t GravitySolver_multigrid::get_value(const Array_t& U, const CellIndex& iCell_U, VarIndex var, const CellIndex::offset_t& offset)
{
  constexpr int ndim = 3;
  if( iCell_U.is_boundary() )
  {
    return 0; 
    // TODO : When using non-zero fixed value boundary conditions, use value when computing Ax_0 but 0 when computing Ap
  }  
  
  else if( iCell_U.level_diff() >= 0 )
  {
    return U.at(iCell_U, var);
  }
  else
  {
    real_t sum = 0;
    int nbCells =
    foreach_smaller_neighbor<ndim, true>( // TODO : select enable_different_block=false when block-based
      iCell_U, offset, U.getShape(), 
      [&](const ForeachCell::CellIndex& iCell_ghost)
    {
      sum += U.at(iCell_ghost, var);
    });
    return sum/nbCells;
  } 
}

/**
 * @brief Contribution at fine-coarse boundary from coarser neighbours.
 * 
 * At a fine-coarse boundary, perform an interpolation using the 
 * four neihbouring coarser cells
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param iCell[in]: cell index where the data should be read
 * @param offset[in]: offset applied prior to calling the method to get iCell_U
 * @return the value of the reconstructed neighbour 
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
real_t GravitySolver_multigrid::average_4bigger_neighbors(const Array_t& U, const Array_t& Uintermediate, const CellIndex iCell, CellIndex::offset_t offset) {
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
    const CellIndex::offset_t shift = {shift_x, shift_y, shift_z};
    const CellIndex iCell_coarse = iCell.getNeighbor_ghost(shift, U.getShape());
    if (iCell_coarse.status == CellIndex::BIGGER ) result += tmp[counter]*U.at(iCell_coarse, Iphi);
    else {
      const CellIndex iCell_parent = iCell_coarse.getParent(Uintermediate.getShape());
      result += tmp[counter]*Uintermediate.at(iCell_parent, Iphi);
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
 * @param iCell[in]: cell index where the data should be read
 * @param offset[in]: offset applied prior to calling the method to get iCell_U
 * @return the value of the reconstructed neighbour 
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
real_t GravitySolver_multigrid::average_8bigger_neighbors(const Array_t& U, const Array_t& Uintermediate, const CellIndex iCell, CellIndex::offset_t offset) {
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
      if (iCell_coarse.status == CellIndex::BIGGER ) {
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
real_t GravitySolver_multigrid::residual_norm_sqr(const Array_t& U, const Array_t& Uintermediate, const uint32_t level){
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  real_t residual_sqr_leaves = 0;
  real_t residual_sqr_intermediate = 0;
  pdata->foreach_cell.reduce_cell("Compute residual norm", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell, real_t & update_residual_sqr)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (current_level == level) {
      const real_t residual_tmp = U.at(iCell, Iresidual);
      update_residual_sqr += residual_tmp * residual_tmp;
    }
  }, Kokkos::Sum<real_t>(residual_sqr_leaves));
  pdata->foreach_cell.reduce_intermediate_cell("Compute residual norm", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell, real_t & update_residual_sqr)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (current_level == level) {
      const real_t residual_tmp = Uintermediate.at(iCell, Iresidual);
      update_residual_sqr += residual_tmp * residual_tmp;
    }
  }, Kokkos::Sum<real_t>(residual_sqr_intermediate));
  return residual_sqr_leaves + residual_sqr_intermediate;
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::initialise_lhs(const Array_t& U, const Array_t& Uintermediate, const uint32_t level){
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_cell("Write potential", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (level == current_level){
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      U.at(iCell, Isolution) = -U.at(iCell, Irhs) / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ]));
    }
  });
  pdata->foreach_cell.foreach_intermediate_cell("Write potential", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (level == current_level){
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      Uintermediate.at(iCell, Isolution) = -Uintermediate.at(iCell, Irhs) / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ])) ;
    }
  });
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::residual_uniform(const Array_t& U, const Array_t& Uintermediate, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_cell("Residual", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (current_level == level) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t neighbors(0);
      for ( ComponentIndex3D dir : {IX,IY,IZ} )
      {
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() <= 0, "Leaf cell at coarse level cannot have coarser leaf neighbor" );
        CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() <= 0, "Leaf cell at coarse level cannot have coarser leaf neighbor" );
        real_t contrib_L(0), contrib_R(0);
        if ( iCell_L.status == CellIndex::SMALLER ) {
          iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
          contrib_L = Uintermediate.at(iCell_L, Isolution);
        } else contrib_L = U.at(iCell_L, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;
        if ( iCell_R.status == CellIndex::SMALLER ) {
          iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
          contrib_R = Uintermediate.at(iCell_R, Isolution);
        } else contrib_R = U.at(iCell_R, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;
        neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
      }
      const real_t laplacian_solution = neighbors - U.at(iCell, Isolution) * ( 2./(size[IX]*size[IX]) + 2./(size[IY]*size[IY]) + 2./(size[IZ]*size[IZ]) );
      U.at(iCell, Iresidual) = U.at(iCell, Irhs) - laplacian_solution;
    }
  });
  pdata->foreach_cell.foreach_intermediate_cell("Residual intermediate", Uintermediate.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (current_level == level) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t neighbors(0);
      for ( ComponentIndex3D dir : {IX,IY,IZ} )
      {
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
        CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
        real_t contrib_L(0), contrib_R(0);
        if ( iCell_L.status == CellIndex::BIGGER ) {
          iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
          contrib_L = U.at(iCell_L, Isolution);
        } else contrib_L = Uintermediate.at(iCell_L, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;
        if ( iCell_R.status == CellIndex::BIGGER ) {
          iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
          contrib_R = U.at(iCell_R, Isolution);
        } else contrib_R = Uintermediate.at(iCell_R, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;
        neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
      }
      const real_t laplacian_solution = neighbors - Uintermediate.at(iCell, Isolution) * ( 2./(size[IX]*size[IX]) + 2./(size[IY]*size[IY]) + 2./(size[IZ]*size[IZ]) );
      Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
    }
  });
}

/**
 * @brief Residual on intermediate cells in AMR levels.
 * 
 * Residual of the Poisson equation on intermediate cells in AMR levels 
 * 
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::residual_intermediate_amr_correction(const Array_t& Uintermediate, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_intermediate_cell("Residual intermediate", Uintermediate.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (current_level == level)
    if (Uintermediate.at(iCell, Imask) > 0) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t laplacian_solution(0);
      const real_t central_solution = Uintermediate.at(iCell, Isolution);
      for ( ComponentIndex3D dir : {IX,IY,IZ} )
      {
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
        CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
        real_t a(1), b(1), contrib_L(0), contrib_R(0);

        if ( iCell_L.status == CellIndex::BIGGER ) a = 0.5;
        else if (Uintermediate.at(iCell_L, Imask) < 0) { // Second-order reconstruction 
          const real_t cell_mask = Uintermediate.at(iCell, Imask);
          const real_t neighbor_mask = Uintermediate.at(iCell_L, Imask);
          a = cell_mask / (cell_mask - neighbor_mask);
        } // else if (Uintermediate.at(iCell_L, Imask) < 1) a = 1; // First-order reconstruction 
        else contrib_L = Uintermediate.at(iCell_L, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;

        if ( iCell_R.status == CellIndex::BIGGER ) b = 0.5;
        else if (Uintermediate.at(iCell_R, Imask) < 0) { // Second-order reconstruction. Test something else?  
          const real_t cell_mask = Uintermediate.at(iCell, Imask);
          const real_t neighbor_mask = Uintermediate.at(iCell_R, Imask);
          b = cell_mask / (cell_mask - neighbor_mask);
        } //else if (Uintermediate.at(iCell_R, Imask) < 1) b = 1; // First-order reconstruction 
        else contrib_R = Uintermediate.at(iCell_R, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;

        const real_t f_L = 2. / (a * (a + b));
        const real_t f_R = 2. / (b * (a + b));
        const real_t f_C = 2. / (a*b);
        laplacian_solution += (f_L*contrib_L + f_R*contrib_R - f_C*central_solution) / (size[dir] * size[dir]);
      }
      Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
    }
  });
}

/**
 * @brief Residual on a AMR level.
 * 
 * Residual of a Poisson equation on an AMR level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::residual_amr_finest(const Array_t& U, const Array_t& Uintermediate, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_cell("Residual leaves", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (current_level == level) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t laplacian_solution(0);
      const real_t central_solution = U.at(iCell, Isolution);
      for ( ComponentIndex3D dir : {IX,IY,IZ} )
      {
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
        CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
        real_t a(1), b(1), contrib_L(0), contrib_R(0);
        if ( iCell_L.status == CellIndex::SMALLER ) { 
          iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
          contrib_L = Uintermediate.at(iCell_L, Isolution);
        } else if (iCell_L.status == CellIndex::BIGGER) {
            contrib_L = average_8bigger_neighbors(U, Uintermediate, iCell, off_L);
            a = 0.5;
            /* contrib_L = average_4bigger_neighbors(U, Uintermediate, iCell, off_L);
            a = 1.5; */
        } else contrib_L = U.at(iCell_L, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;
        if ( iCell_R.status == CellIndex::SMALLER ) { 
          iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
          contrib_R = Uintermediate.at(iCell_R, Isolution);
        } else if (iCell_R.status == CellIndex::BIGGER) {
            contrib_R = average_8bigger_neighbors(U, Uintermediate, iCell, off_R);
            b = 0.5;
            /* contrib_R = average_4bigger_neighbors(U, Uintermediate, iCell, off_R);
            b = 1.5; */
        } else contrib_R = U.at(iCell_R, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;
        const real_t f_L = 2. / (a * (a + b));
        const real_t f_R = 2. / (b * (a + b));
        const real_t f_C = 2. / (a*b);
        laplacian_solution += (f_L*contrib_L + f_R*contrib_R - f_C*central_solution) / (size[dir] * size[dir]);
      }
      U.at(iCell, Iresidual) = U.at(iCell, Irhs) - laplacian_solution;
    }
  });
  pdata->foreach_cell.foreach_intermediate_cell("Residual intermediate", Uintermediate.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (current_level == level) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t neighbors(0);
      for ( ComponentIndex3D dir : {IX,IY,IZ} )
      {
        real_t contrib_L(0), contrib_R(0);
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
        CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
        if ( iCell_L.status == CellIndex::BIGGER ) {
          iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
          contrib_L = U.at(iCell_L, Isolution);
        } else contrib_L = Uintermediate.at(iCell_L, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;
        if ( iCell_R.status == CellIndex::BIGGER ) {
          iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
          contrib_R = U.at(iCell_R, Isolution);
        } else contrib_R = Uintermediate.at(iCell_R, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;
        neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
      }
      const real_t laplacian_solution = neighbors - Uintermediate.at(iCell, Isolution) * ( 2./(size[IX]*size[IX]) + 2./(size[IY]*size[IY]) + 2./(size[IZ]*size[IZ]) );
      Uintermediate.at(iCell, Iresidual) = Uintermediate.at(iCell, Irhs) - laplacian_solution;
    }
  });
}

/**
 * @brief Restriction operator.
 * 
 * Restriction of mask, and residual to right-hand side of intermediate cells
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::restriction(const Array_t& U, const Array_t& Uintermediate, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_intermediate_cell( "Restrict on intermediate", Uintermediate.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if( current_level == level )
    {
      CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
      real_t residual(0), mask(0);
      const int ns = foreach_sibling( pdata->ndim, iCell_c0, Uintermediate.getShape(),
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
    }
  }); 
}


/**
 * @brief Prolongation (straight injection).
 * 
 * Zeroth-order prolongation as straight injection 
 * of the solution
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::prolongation0_inject(const Array_t& U, const Array_t& Uintermediate, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_intermediate_cell( "Prolongation", Uintermediate.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if ( current_level == level - 1 )
    {
      CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
      const real_t solution = Uintermediate.at( iCell, Isolution );
      //const real_t rhs = Uintermediate.at( iCell, Irhs );
      foreach_sibling( pdata->ndim, iCell_c0, Uintermediate.getShape(),
        [&]( const CellIndex& iCell_c )
      {
        if ( iCell_c.iOct.isIntermediate )  Uintermediate.at(iCell_c, Isolution) = solution;
        else  U.at(iCell_c, Isolution) = solution;
      });
    }
  }); 
}


/**
 * @brief Add Prolongation (straight injection).
 * 
 * Add a zeroth-order prolongation as straight injection
 * of the solution at a given level
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::prolongation0(const Array_t& U, const Array_t& Uintermediate, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_intermediate_cell( "Prolongation", Uintermediate.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if( current_level == level - 1 )
    {
      const real_t correction = Uintermediate.at( iCell, Isolution );
      const CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
      foreach_sibling( pdata->ndim, iCell_c0, Uintermediate.getShape(),
        [&]( const CellIndex& iCell_c )
      {
        if( iCell_c.iOct.isIntermediate ) Uintermediate.at(iCell_c, Isolution) += correction;
        else  U.at(iCell_c, Isolution) += correction;
      });
    }
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
real_t GravitySolver_multigrid::get_neighbor_value(const Array_t& U, const Array_t& Uintermediate, const CellIndex iCell, const CellIndex::offset_t offset){
  CellIndex iCell_n = iCell.getNeighbor_ghost_intermediate(offset, Uintermediate.getShape());
  if (iCell_n.level_diff() != 0){
    iCell_n = iCell.getNeighbor_ghost(offset, U.getShape());
    return U.at( iCell_n, Isolution );
  } else return Uintermediate.at( iCell_n, Isolution );
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
 * @return the residual at a given level 
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::prolongation(const Array_t& U, const Array_t& Uintermediate, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  const real_t f0 = 27.0 / 64;
  const real_t f1 = 9.0 / 64;
  const real_t f2 = 3.0 / 64;
  const real_t f3 = 1.0 / 64;
  pdata->foreach_cell.foreach_intermediate_cell( "Prolongation", Uintermediate.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if( current_level == level - 1 )
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
}


/**
 * @brief Add prolongation (interpolation).
 * 
 * Add a first-order prolongation as linear interpolation 
 * of the solution at a given level on intermediate cells
 * 
 * @param U[in]: The data to read from
 * @param Uintermediate[in]: The intermediate data to read from
 * @param level[in]: The grid level
 * @return the residual at a given level 
*/
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::prolongation_on_intermediate(const Array_t& U, const Array_t& Uintermediate, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  const real_t f0 = 27.0 / 64;
  const real_t f1 = 9.0 / 64;
  const real_t f2 = 3.0 / 64;
  const real_t f3 = 1.0 / 64;
  pdata->foreach_cell.foreach_intermediate_cell( "Prolongation", Uintermediate.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if( current_level == level - 1 )
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::zero_solution(const Array_t& U, const Array_t& Uintermediate, const uint32_t level){
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_cell("Write potential", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    uint32_t current_level = cells.getLevel(iCell);
    if (level == current_level) U.at(iCell, Isolution) = 0;
  });
  pdata->foreach_cell.foreach_intermediate_cell("Write potential", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    uint32_t current_level = cells.getLevel(iCell);
    if (level == current_level) Uintermediate.at(iCell, Isolution) = 0;
  });
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::initialise_mask(const Array_t& U, const Array_t& Uintermediate, const uint32_t finest_level){
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_cell("Write potential", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    uint32_t current_level = cells.getLevel(iCell);
    if (current_level < finest_level) U.at(iCell, Imask) = -1;
    else if ( current_level == finest_level )  U.at(iCell, Imask) = 1;
    
  });
  pdata->foreach_cell.foreach_intermediate_cell("Write potential", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    uint32_t current_level = cells.getLevel(iCell);
    if (current_level < finest_level) Uintermediate.at(iCell, Imask) = -1;
    else if ( current_level == finest_level )  Uintermediate.at(iCell, Imask) = 1;
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::zero_solution_residual_rhs(const Array_t& U, const Array_t& Uintermediate, const uint32_t level){
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_cell("Write potential", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    uint32_t current_level = cells.getLevel(iCell);
    if (level == current_level){
      U.at(iCell, Isolution) = 0;
      U.at(iCell, Irhs) = 0;
      U.at(iCell, Iresidual) = 0;
    }
  });
  pdata->foreach_cell.foreach_intermediate_cell("Write potential", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    uint32_t current_level = cells.getLevel(iCell);
    if (level == current_level){
      Uintermediate.at(iCell, Isolution) = 0;
      Uintermediate.at(iCell, Irhs) = 0;
      Uintermediate.at(iCell, Iresidual) = 0;
    }
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::solution_to_potential(const Array_t& U, const Array_t& Uintermediate, const uint32_t level){
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_cell("Write potential", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    uint32_t current_level = cells.getLevel(iCell);
    if (level == current_level){
      U.at(iCell, Iphi) = U.at(iCell, Isolution);
    }
  });
  pdata->foreach_cell.foreach_intermediate_cell("Write potential", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    uint32_t current_level = cells.getLevel(iCell);
    if (level == current_level){
      Uintermediate.at(iCell, Iphi) = Uintermediate.at(iCell, Isolution);
    }
  });
}


/**
 * @brief Is a red cell.
 * 
 * Check if cell is red
 * 
 * @param iCell[in]: cell index where the data should be read
 * @return a boolean value indicating if cell is red
*/
KOKKOS_INLINE_FUNCTION
bool GravitySolver_multigrid::isRed(const CellIndex iCell) {
  uint32_t idx = iCell.i + iCell.j + iCell.k;
  return (bool)(idx % 2 != 0);
}

/**
 * @brief Is a black cell.
 * 
 * CHeck if cell is black
 * 
 * @param iCell[in]: cell index where the data should be read
 * @return a boolean value indicating if cell is black
*/
KOKKOS_INLINE_FUNCTION
bool GravitySolver_multigrid::isBlack(const CellIndex iCell) {
  uint32_t idx = iCell.i + iCell.j + iCell.k;
  return (bool)(idx % 2 == 0);
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
void GravitySolver_multigrid::gauss_seidel_intermediate_amr_correction(const Array_t& Uintermediate, std::function<bool(CellIndex)> is_coloured, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_intermediate_cell("Gauss-Seidel", Uintermediate.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (current_level == level) 
    if (is_coloured(iCell))
    if (Uintermediate.at(iCell, Imask) > 0) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      Kokkos::Array<real_t, 3> f_C;
      real_t neighbors(0);
      const real_t w_relax(1.);
      for ( const ComponentIndex3D dir : {IX,IY,IZ} )
      {
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate L cell cannot have smaller intermediate neighbor" );
        CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate R cell cannot have smaller intermediate neighbor" );
        real_t a(1), b(1), contrib_L(0), contrib_R(0), f_R(0), f_L(0);
        
        if (iCell_L.status == CellIndex::BIGGER) a = 0.5;
        else if (Uintermediate.at(iCell_L, Imask) < 0) { // Second-order reconstruction. 
          const real_t cell_mask = Uintermediate.at(iCell, Imask);
          const real_t left_neighbor_mask = Uintermediate.at(iCell_L, Imask);
          a = cell_mask / (cell_mask - left_neighbor_mask);
        } // else if (Uintermediate.at(iCell_L, Imask) < 1) a = 1; // First-order reconstruction 
        else contrib_L = Uintermediate.at(iCell_L, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;

        if (iCell_R.status == CellIndex::BIGGER) b = 0.5;
        else if (Uintermediate.at(iCell_R, Imask) < 0) { // Second-order reconstruction.
          const real_t cell_mask = Uintermediate.at(iCell, Imask);
          const real_t right_neighbor_mask = Uintermediate.at(iCell_R, Imask);
          b = cell_mask / (cell_mask - right_neighbor_mask);
        } //else if (Uintermediate.at(iCell_R, Imask) < 1) b = 1;  // First-order reconstruction 
        else contrib_R = Uintermediate.at(iCell_R, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;

        f_L = 2. / (a * (a + b));
        f_R = 2. / (b * (a + b));
        f_C[dir] = 2. / (a*b);
        neighbors += (f_L * contrib_L + f_R * contrib_R) / (size[dir] * size[dir]);                   
      }
      Uintermediate.at(iCell, Isolution) += w_relax * (
        (neighbors - Uintermediate.at(iCell, Irhs)) / ( 
        f_C[IX] / (size[IX]*size[IX]) + f_C[IY] / (size[IY]*size[IY]) + f_C[IZ] / (size[IZ]*size[IZ])
        ) - Uintermediate.at(iCell, Isolution)
      );
    } 
  });
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
void GravitySolver_multigrid::gauss_seidel_leaves_amr_finest(const Array_t& U, const Array_t& Uintermediate, std::function<bool(CellIndex)> is_coloured, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_cell("Gauss-Seidel", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (current_level == level) 
    if (is_coloured(iCell)) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      Kokkos::Array<real_t, 3> f_C;
      real_t neighbors(0);
      const real_t w_relax(1.);
      for ( const ComponentIndex3D dir : {IX,IY,IZ} )
      {
        real_t a(1), b(1), contrib_L(0), contrib_R(0), f_L(0), f_R(0);
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
        CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
        if ( iCell_L.status == CellIndex::SMALLER ) {
          iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
          contrib_L = Uintermediate.at(iCell_L, Isolution);
        } else if (iCell_L.status == CellIndex::BIGGER) {
          contrib_L = average_8bigger_neighbors(U, Uintermediate, iCell, off_L);
          a = 0.5;
          /* contrib_L = average_4bigger_neighbors(U, Uintermediate, iCell, off_L);
          a = 1.5; */
        } else contrib_L = U.at(iCell_L, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;
        if ( iCell_R.status == CellIndex::SMALLER ) {
          iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
          contrib_R = Uintermediate.at(iCell_R, Isolution);
        } else if (iCell_R.status == CellIndex::BIGGER) {
          contrib_R = average_8bigger_neighbors(U, Uintermediate, iCell, off_R);
          b = 0.5;
          /* contrib_R = average_4bigger_neighbors(U, Uintermediate, iCell, off_R);
          b = 1.5; */
        } else contrib_R = U.at(iCell_R, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;
        f_L = 2. / (a * (a + b));
        f_R = 2. / (b * (a + b));
        f_C[dir] = 2. / (a*b);
        neighbors += (f_L * contrib_L + f_R * contrib_R) / (size[dir] * size[dir]);             
      }
      U.at(iCell, Isolution) += w_relax * (
        (neighbors - U.at(iCell, Irhs)) / ( 
        f_C[IX] / (size[IX]*size[IX]) + f_C[IY] / (size[IY]*size[IY]) + f_C[IZ] / (size[IZ]*size[IZ])
        ) - U.at(iCell, Isolution)
      );
    }
  });
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
void GravitySolver_multigrid::gauss_seidel_leaves_uniform(const Array_t& U, const Array_t& Uintermediate, std::function<bool(CellIndex)> is_coloured, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_cell("Gauss-Seidel", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (current_level == level)
    if (is_coloured(iCell)) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t neighbors(0);
      const real_t w_relax(1.);
      for ( const ComponentIndex3D dir : {IX,IY,IZ} )
      {
        real_t contrib_L(0), contrib_R(0);
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
        CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
        if ( iCell_L.level_diff() != 0 ) {
          iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
          contrib_L = Uintermediate.at(iCell_L, Isolution);
        } else contrib_L = U.at(iCell_L, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;
        if ( iCell_R.level_diff() != 0 ) {
          iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
          contrib_R = Uintermediate.at(iCell_R, Isolution);
        } else contrib_R = U.at(iCell_R, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;
        neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);             
      }
      U.at(iCell, Isolution) += w_relax * (
        (neighbors - U.at(iCell, Irhs)) / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ]))
        - U.at(iCell, Isolution)
      );
    }
  });
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
void GravitySolver_multigrid::gauss_seidel_intermediate(const Array_t& U, const Array_t& Uintermediate, std::function<bool(CellIndex)> is_coloured, const uint32_t level) {
  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();
  pdata->foreach_cell.foreach_intermediate_cell("Gauss-Seidel", Uintermediate.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint32_t current_level = cells.getLevel(iCell);
    if (current_level == level)
    if (is_coloured(iCell)) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t neighbors(0);
      const real_t w_relax(1.);
      for ( ComponentIndex3D dir : {IX,IY,IZ} )
      {
        real_t contrib_L(0), contrib_R(0);
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate L cell cannot have smaller intermediate neighbor" );
        CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate R cell cannot have smaller intermediate neighbor" );
        if ( iCell_L.status == CellIndex::BIGGER ) {
          iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
          contrib_L = U.at(iCell_L, Isolution);
        } else contrib_L = Uintermediate.at(iCell_L, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) contrib_L = 0;
        if ( iCell_R.status == CellIndex::BIGGER ) {
          iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
          contrib_R = U.at(iCell_R, Isolution);
        } else contrib_R = Uintermediate.at(iCell_R, Isolution);
        if ( pdata->boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) contrib_R = 0;
        neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
      }
      Uintermediate.at(iCell, Isolution) += w_relax * (
        (neighbors - Uintermediate.at(iCell, Irhs)) / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ]))
        - Uintermediate.at(iCell, Isolution)
      );
    }
  });
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::smoothing_intermediate_amr_correction(const Array_t& Uintermediate, const uint32_t nIterations, const uint32_t level) {
  for (uint32_t i = 0; i < nIterations; i++) {
    gauss_seidel_intermediate_amr_correction(Uintermediate, isRed, level);
    gauss_seidel_intermediate_amr_correction(Uintermediate, isBlack, level);
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::smoothing_uniform(const Array_t& U, const Array_t& Uintermediate, const uint32_t nIterations, const uint32_t level) {
  for (uint32_t i = 0; i < nIterations; i++) {
    gauss_seidel_leaves_uniform(U, Uintermediate, isRed, level);
    gauss_seidel_intermediate(U, Uintermediate, isRed, level);
    gauss_seidel_leaves_uniform(U, Uintermediate, isBlack, level);
    gauss_seidel_intermediate(U, Uintermediate, isBlack, level);
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::smoothing_amr_finest(const Array_t& U, const Array_t& Uintermediate, const uint32_t nIterations, const uint32_t level) {
  for (uint32_t i = 0; i < nIterations; i++) {
    gauss_seidel_leaves_amr_finest(U, Uintermediate, isRed, level);
    gauss_seidel_intermediate(U, Uintermediate, isRed, level);
    gauss_seidel_leaves_amr_finest(U, Uintermediate, isBlack, level);
    gauss_seidel_intermediate(U, Uintermediate, isBlack, level);
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::V_cycle_uniform(const Array_t& U, const Array_t& Uintermediate, const uint32_t level) {
  smoothing_uniform(U, Uintermediate, pdata->Npre, level);
  residual_uniform(U, Uintermediate, level);
  restriction(U, Uintermediate, level - 1);
  initialise_lhs(U, Uintermediate, level - 1);
  if (level == 1) {
    smoothing_uniform(U, Uintermediate, pdata->Npre, level - 1);
  }
  else V_cycle_uniform(U, Uintermediate, level - 1);
  
  prolongation(U, Uintermediate, level);
  smoothing_uniform(U, Uintermediate, pdata->Npost, level);
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
template< typename Array_t >
KOKKOS_INLINE_FUNCTION
void GravitySolver_multigrid::V_cycle_amr(const Array_t& U, const Array_t& Uintermediate, const uint32_t current_level, const uint32_t finest_level) {

  if ( current_level == finest_level ) {
    smoothing_amr_finest(U, Uintermediate, pdata->Npre, current_level);
    residual_amr_finest(U, Uintermediate, current_level);
  } else {
    smoothing_intermediate_amr_correction(Uintermediate, pdata->Npre, current_level);
    residual_intermediate_amr_correction(Uintermediate, current_level);
  }

  restriction(U, Uintermediate, current_level - 1);
  initialise_lhs(U, Uintermediate, current_level - 1);

  //if ( pdata->level_coarse == (current_level - 1) ) {
  if ( finest_level - 3 == current_level ) { // finest - 2 seems to works aswell for spherical case. 
      smoothing_intermediate_amr_correction(Uintermediate, pdata->Npre, pdata->level_coarse);
  } else V_cycle_amr(U, Uintermediate, current_level - 1, finest_level); 
  
  if ( current_level == finest_level ) {
    prolongation(U, Uintermediate, current_level);
    smoothing_amr_finest(U, Uintermediate, pdata->Npost, current_level);
  } else {
    prolongation_on_intermediate(U, Uintermediate, current_level);  
    smoothing_intermediate_amr_correction(Uintermediate, pdata->Npost, current_level);
  }    
  
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


real_t MPI_Allreduce_scalar( real_t local_v )
{
  real_t res;
  MPI_Allreduce( &local_v, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  return res;
}

//} // namespace

/**
 * @brief Solves the Poisson equation and updates the gravity field
 * 
 * @param U[inout]: UserData structure to update
 * @param scalar_data[in]: ScalarSimulationData structure
*/
void GravitySolver_multigrid::update_gravity_field( UserData& U_, ScalarSimulationData& scalar_data )
{

  pdata->timers.get("GravitySolver_multigrid").start();

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


  const ForeachCell::CellMetaData cells = pdata->foreach_cell.getCellMetaData();

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
  pdata->level_coarse = level_coarse;
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


  // Compute rho mean
  real_t rho_mean = 0;
  const uint8_t ndim = pdata->foreach_cell.getDim();
  const real_t xmin(pdata->xmin), ymin(pdata->ymin), zmin(pdata->zmin);
  const real_t xmax(pdata->xmax), ymax(pdata->ymax), zmax(pdata->zmax);
  printf("Before rho mean\n");
  pdata->foreach_cell.reduce_cell("Compute rho_mean", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell, real_t & update_rhomean)
  {
    ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
    real_t rhoi = U.at(iCell, Irho);
    update_rhomean += rhoi * size[IX] * size[IY] * size[IZ];
  }, Kokkos::Sum<real_t>(rho_mean));
  real_t Vtot = (xmax - xmin) * (ymax - ymin) * (zmax - zmin);
  rho_mean = MPI_Allreduce_scalar(rho_mean) / Vtot;
  printf("rhomean = %.5e\n", rho_mean);

  bool cosmo_run = pdata->cosmo_run;
  real_t aexp = 0;
  if( cosmo_run )
    aexp = scalar_data.get<real_t>("aexp");
  real_t four_Pi_G = pdata->four_Pi_G;  
  
  // Initialize RHS and solution on leaves
  pdata->foreach_cell.foreach_cell("Init RHS and potential", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
    const real_t rhs = (cosmo_run) ? b_cosmo(U, iCell, rho_mean, aexp) : b(U, iCell, rho_mean, four_Pi_G);//U.at(iCell, Irho) - rho_mean;
    U.at(iCell, Irhs) = rhs;
    U.at(iCell, Isolution) = -rhs / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ]));
  });

  // Initialize RHS on intermediate levels. Solution will be interpolated from coarser levels so no need to initialize
  for( uint32_t level = max_level_in_amr + 1; level >= level_coarse; level-- )
  {
    pdata->foreach_cell.foreach_intermediate_cell( "average_parent_cell", Uintermediate.getShape(),
    KOKKOS_LAMBDA( CellIndex& iCell)
    {
      const uint32_t current_level = cells.getLevel(iCell);
      if( current_level == level )
      {
        CellIndex iCell_c0 = iCell.getChildren(Uintermediate.getShape());
        real_t rho = 0;
        int ns = foreach_sibling( ndim, iCell_c0, Uintermediate.getShape(),
          [&]( const CellIndex& iCell_c )
        {
          if ( iCell_c.iOct.isIntermediate ) rho += Uintermediate.at(iCell_c, Irho);
          else rho += U.at(iCell_c, Irho);
        });
        rho /= ns;
        Uintermediate.at( iCell, Irho ) = rho;
        Uintermediate.at( iCell, Irhs ) = (cosmo_run) ? b_cosmo(Uintermediate, iCell, rho_mean, aexp) : b(Uintermediate, iCell, rho_mean, four_Pi_G); //rho - rho_mean;
      }
    }); 
  }

  residual_uniform(U, Uintermediate, level_coarse);
  solution_to_potential(U, Uintermediate, level_coarse);
  printf("Level %d Residual norm init  V %f\n", level_coarse, std::sqrt(residual_norm_sqr(U, Uintermediate, level_coarse)));
  
  // Multigrid
  printf("Coarse Multigrid\n");
  for(uint32_t i = 0; i < pdata->Ncycles; i++)
  { 
    V_cycle_uniform(U, Uintermediate, level_coarse);
    residual_uniform(U, Uintermediate, level_coarse);
    solution_to_potential(U, Uintermediate, level_coarse);
    printf("Level %d Residual norm after V %.5e\n", level_coarse, std::sqrt(residual_norm_sqr(U, Uintermediate, level_coarse)));
  }

  printf("AMR Multigrid\n");
  for (uint32_t ilevel = level_coarse+1; ilevel <= max_level_in_amr; ilevel++) {
    zero_solution(U, Uintermediate, ilevel);
    prolongation(U, Uintermediate, ilevel);
    zero_solution_residual_rhs(U, Uintermediate, ilevel-1);
    solution_to_potential(U, Uintermediate, ilevel);
    residual_amr_finest(U, Uintermediate, ilevel);
    initialise_mask(U, Uintermediate, ilevel);

    for(uint32_t i = 0; i < pdata->Ncycles; i++)
    { 
      V_cycle_amr(U, Uintermediate, ilevel, ilevel);
      residual_amr_finest(U, Uintermediate, ilevel);
      solution_to_potential(U, Uintermediate, ilevel);
      printf("Level %d Residual norm after V %.5e\n", ilevel, std::sqrt(residual_norm_sqr(U, Uintermediate, ilevel)));
    }
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

    //gradient0(Uout);
  gradient(Uout, Uoutintermediate);

  for (std::string name : {"solution", "rhs",  "res", "mask" })
    U_.delete_field(name);
  for (std::string name : {"rho","gphi", "solution", "rhs", "res", "mask"})
    U_.delete_intermediate_field(name);

  pdata->timers.get("GravitySolver_multigrid").stop();
}

}// namespace dyablo

FACTORY_REGISTER( dyablo::GravitySolverFactory, dyablo::GravitySolver_multigrid, "GravitySolver_multigrid" );
