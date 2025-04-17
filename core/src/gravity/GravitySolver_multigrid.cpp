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
void GravitySolver_multigrid::gradient(const Array_t& U, const Array_t& Uintermediate)
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
template< typename Array_t >
real_t GravitySolver_multigrid::residual_norm(const Array_t& U, const Array_t& Uintermediate, const level_t level)
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  real_t residual_sqr_leaves = 0;
  real_t residual_sqr_intermediate = 0;
  foreach_cell.reduce_cell("Compute residual norm", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell, real_t & update_residual_sqr)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level == level) {
      const real_t residual_tmp = U.at(iCell, Iresidual);
      update_residual_sqr += residual_tmp * residual_tmp;
    }
  }, Kokkos::Sum<real_t>(residual_sqr_leaves));
  foreach_cell.reduce_intermediate_cell("Compute residual norm", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell, real_t & update_residual_sqr)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level == level) {
      const real_t residual_tmp = Uintermediate.at(iCell, Iresidual);
      update_residual_sqr += residual_tmp * residual_tmp;
    }
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
template< typename Array_t >
void GravitySolver_multigrid::initialise_lhs(const Array_t& U, const Array_t& Uintermediate, const level_t level)
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  foreach_cell.foreach_cell("Initialise potential", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (level == current_level){
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      U.at(iCell, Isolution) = -U.at(iCell, Irhs) / (2. / (size[IX]*size[IX]) + 2. / (size[IY]*size[IY]) + 2. / (size[IZ]*size[IZ]));
    }
  });
  foreach_cell.foreach_intermediate_cell("Initialise potential", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
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
void GravitySolver_multigrid::residual_uniform(const Array_t& U, const Array_t& Uintermediate, const level_t level) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;

  foreach_cell.foreach_cell("Residual", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
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
        if ( CellIndex::SMALLER == iCell_L.status ) {
          iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
          contrib_L = Uintermediate.at(iCell_L, Isolution);
        } else contrib_L = U.at(iCell_L, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_L.is_boundary() ) contrib_L = 0;
        if ( CellIndex::SMALLER == iCell_R.status  ) {
          iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
          contrib_R = Uintermediate.at(iCell_R, Isolution);
        } else contrib_R = U.at(iCell_R, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_R.is_boundary() ) contrib_R = 0;
        neighbors += (contrib_L + contrib_R) / (size[dir] * size[dir]);
      }
      const real_t laplacian_solution = neighbors - U.at(iCell, Isolution) * ( 2./(size[IX]*size[IX]) + 2./(size[IY]*size[IY]) + 2./(size[IZ]*size[IZ]) );
      U.at(iCell, Iresidual) = U.at(iCell, Irhs) - laplacian_solution;
    }
  });
  foreach_cell.foreach_intermediate_cell("Residual", Uintermediate.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
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
        if ( CellIndex::BIGGER == iCell_L.status ) {
          iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
          contrib_L = U.at(iCell_L, Isolution);
        } else contrib_L = Uintermediate.at(iCell_L, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_L.is_boundary() ) contrib_L = 0;
        if ( CellIndex::BIGGER == iCell_R.status ) {
          iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
          contrib_R = U.at(iCell_R, Isolution);
        } else contrib_R = Uintermediate.at(iCell_R, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_R.is_boundary() ) contrib_R = 0;
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
void GravitySolver_multigrid::residual_intermediate_amr_correction(const Array_t& Uintermediate, const level_t level) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;

  foreach_cell.foreach_intermediate_cell("Residual", Uintermediate.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level == level)
    if (Uintermediate.at(iCell, Imask) > 0) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t laplacian_solution(0);
      const real_t central_solution = Uintermediate.at(iCell, Isolution);
      for ( ComponentIndex3D dir : {IX,IY,IZ} )
      {
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        const CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
        const CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate cell cannot have smaller intermediate neighbor" );
        real_t a(1), b(1), contrib_L(0), contrib_R(0);

        if ( CellIndex::BIGGER == iCell_L.status ) a = 0.5;
        else if (Uintermediate.at(iCell_L, Imask) < 0) { // Second-order reconstruction 
          const real_t cell_mask = Uintermediate.at(iCell, Imask);
          const real_t neighbor_mask = Uintermediate.at(iCell_L, Imask);
          a = cell_mask / (cell_mask - neighbor_mask);
        } // else if (Uintermediate.at(iCell_L, Imask) < 1) a = 1; // First-order reconstruction 
        else contrib_L = Uintermediate.at(iCell_L, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_L.is_boundary() ) contrib_L = 0;

        if ( CellIndex::BIGGER == iCell_R.status ) b = 0.5;
        else if (Uintermediate.at(iCell_R, Imask) < 0) { // Second-order reconstruction. Test something else?  
          const real_t cell_mask = Uintermediate.at(iCell, Imask);
          const real_t neighbor_mask = Uintermediate.at(iCell_R, Imask);
          b = cell_mask / (cell_mask - neighbor_mask);
        } //else if (Uintermediate.at(iCell_R, Imask) < 1) b = 1; // First-order reconstruction 
        else contrib_R = Uintermediate.at(iCell_R, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_R.is_boundary() ) contrib_R = 0;

        const real_t f_L = 2. / (a * (a + b));
        const real_t f_R = 2. / (b * (a + b));
        const real_t f_C = 2. / (a * b);
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
void GravitySolver_multigrid::residual_amr_finest(const Array_t& U, const Array_t& Uintermediate, const level_t level) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;

  foreach_cell.foreach_cell("Residual", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
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
        if ( CellIndex::SMALLER == iCell_L.status ) { 
          iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
          contrib_L = Uintermediate.at(iCell_L, Isolution);
        } else if (CellIndex::BIGGER == iCell_L.status) {
            contrib_L = average_8bigger_neighbors(U, Uintermediate, iCell, off_L);
            a = 0.5;
        } else contrib_L = U.at(iCell_L, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_L.is_boundary() ) contrib_L = 0;
        if ( CellIndex::SMALLER == iCell_R.status ) { 
          iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
          contrib_R = Uintermediate.at(iCell_R, Isolution);
        } else if (CellIndex::BIGGER == iCell_R.status) {
            contrib_R = average_8bigger_neighbors(U, Uintermediate, iCell, off_R);
            b = 0.5;
        } else contrib_R = U.at(iCell_R, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_R.is_boundary() ) contrib_R = 0;
        const real_t f_L = 2. / (a * (a + b));
        const real_t f_R = 2. / (b * (a + b));
        const real_t f_C = 2. / (a * b);
        laplacian_solution += (f_L*contrib_L + f_R*contrib_R - f_C*central_solution) / (size[dir] * size[dir]);
      }
      U.at(iCell, Iresidual) = U.at(iCell, Irhs) - laplacian_solution;
    }
  });
  foreach_cell.foreach_intermediate_cell("Residual", Uintermediate.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
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
        if ( CellIndex::BIGGER == iCell_L.status ) {
          iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
          contrib_L = U.at(iCell_L, Isolution);
        } else contrib_L = Uintermediate.at(iCell_L, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_L.is_boundary() ) contrib_L = 0;
        if ( CellIndex::BIGGER == iCell_R.status ) {
          iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
          contrib_R = U.at(iCell_R, Isolution);
        } else contrib_R = Uintermediate.at(iCell_R, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_R.is_boundary() ) contrib_R = 0;
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
void GravitySolver_multigrid::restriction(const Array_t& U, const Array_t& Uintermediate, const level_t level) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const int ndim = pdata->ndim;

  foreach_cell.foreach_intermediate_cell( "Restrict", Uintermediate.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if( current_level == level )
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
    }
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
template< typename Array_t >
void GravitySolver_multigrid::restriction_from_children(const Array_t& U, const Array_t& Uintermediate, const level_t level) 
{
  ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  constexpr real_t inv_ns = 1./8;

  foreach_cell.foreach_cell( "Restrict", U.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if( current_level == level + 1 )
    {
      const CellIndex iCell_p = iCell.getParent(U.getShape());      
      Kokkos::atomic_add( &Uintermediate.at( iCell_p, Irhs ), U.at( iCell, Iresidual ) * inv_ns );
      Kokkos::atomic_add( &Uintermediate.at( iCell_p, Imask ), U.at( iCell, Imask ) * inv_ns );
    }
  }); 
  foreach_cell.foreach_intermediate_cell( "Restrict", Uintermediate.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if( current_level == level + 1 )
    {
      const CellIndex iCell_p = iCell.getParent(Uintermediate.getShape());     
      Kokkos::atomic_add( &Uintermediate.at( iCell_p, Irhs ), Uintermediate.at( iCell, Iresidual ) * inv_ns );
      Kokkos::atomic_add( &Uintermediate.at( iCell_p, Imask ), Uintermediate.at( iCell, Imask ) * inv_ns );
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
template< typename Array_t >
void GravitySolver_multigrid::prolongation(const Array_t& U, const Array_t& Uintermediate, const level_t level) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  constexpr real_t f0 = 27.0 / 64;
  constexpr real_t f1 = 9.0 / 64;
  constexpr real_t f2 = 3.0 / 64;
  constexpr real_t f3 = 1.0 / 64;
  foreach_cell.foreach_cell( "Prolongation", U.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if( current_level == level )
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
    }
  }); 
  foreach_cell.foreach_intermediate_cell( "Prolongation", Uintermediate.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if( current_level == level )
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
    }
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
template< typename Array_t >
void GravitySolver_multigrid::prolongation_on_intermediate(const Array_t& U, const Array_t& Uintermediate, const level_t level) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  constexpr real_t f0 = 27.0 / 64;
  constexpr real_t f1 = 9.0 / 64;
  constexpr real_t f2 = 3.0 / 64;
  constexpr real_t f3 = 1.0 / 64;
  foreach_cell.foreach_intermediate_cell( "Prolongation", Uintermediate.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if( current_level == level )
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
void GravitySolver_multigrid::check_parents(const Array_t& U, const Array_t& Uintermediate)
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  foreach_cell.foreach_cell("Set solution to zero", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    auto level = cells.getLevel(iCell);
    if (level > 0)
      auto iCell_p = iCell.getParent(U.getShape());
  });
  foreach_cell.foreach_intermediate_cell("Set solution to zero", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    auto level = cells.getLevel(iCell);
    if (level > 0)
      auto iCell_p = iCell.getParent(Uintermediate.getShape());
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
void GravitySolver_multigrid::zero_solution(const Array_t& U, const Array_t& Uintermediate, const level_t level)
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  foreach_cell.foreach_cell("Set solution to zero", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (level == current_level) U.at(iCell, Isolution) = 0;
  });
  foreach_cell.foreach_intermediate_cell("Set solution to zero", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
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
void GravitySolver_multigrid::initialise_mask(const Array_t& U, const Array_t& Uintermediate, const uint32_t finest_level)
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  foreach_cell.foreach_cell("Initialise mask", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level < finest_level) U.at(iCell, Imask) = -1;
    else if ( current_level == finest_level )  U.at(iCell, Imask) = 1;
    
  });
  foreach_cell.foreach_intermediate_cell("Initialise mask", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
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
void GravitySolver_multigrid::zero_solution_residual_rhs(const Array_t& U, const Array_t& Uintermediate, const level_t level)
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  foreach_cell.foreach_cell("Set MG fields to zero", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level <= level){
      U.at(iCell, Isolution) = 0;
      U.at(iCell, Irhs) = 0;
      U.at(iCell, Iresidual) = 0;
    }
  });
  foreach_cell.foreach_intermediate_cell("Set MG fields to zero", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level <= level){
      Uintermediate.at(iCell, Isolution) = 0;
      Uintermediate.at(iCell, Irhs) = 0;
      Uintermediate.at(iCell, Iresidual) = 0;
    }
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
void GravitySolver_multigrid::zero_rhs_mask_intermediate(const Array_t& Uintermediate, const level_t level)
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  foreach_cell.foreach_intermediate_cell("Set intermediate rhs and mask to zero", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (level == current_level){
      Uintermediate.at(iCell, Irhs) = 0;
      Uintermediate.at(iCell, Imask) = 0;
    }
  });
  foreach_cell.foreach_intermediate_ghost_cell("Set intermediate rhs and mask to zero", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (level == current_level){
      Uintermediate.at(iCell, Irhs) = 0;
      Uintermediate.at(iCell, Imask) = 0;
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
void GravitySolver_multigrid::solution_to_potential(const Array_t& U, const Array_t& Uintermediate, const level_t level)
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  foreach_cell.foreach_cell("Copy solution to potential", U.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (level == current_level){
      U.at(iCell, Iphi) = U.at(iCell, Isolution);
    }
  });
  foreach_cell.foreach_intermediate_cell("Copy solution to potential", Uintermediate.getShape(),
  KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
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
bool GravitySolver_multigrid::isRed(const CellIndex& iCell) 
{
  const uint32_t idx = iCell.i + iCell.j + iCell.k;
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
bool GravitySolver_multigrid::isBlack(const CellIndex& iCell) 
{
  const uint32_t idx = iCell.i + iCell.j + iCell.k;
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
template< typename Array_t, typename Function >
void GravitySolver_multigrid::gauss_seidel_intermediate_amr_correction(const Array_t& Uintermediate, const level_t level, const Function& is_coloured) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;

  foreach_cell.foreach_intermediate_cell("Gauss-Seidel", Uintermediate.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level == level) 
    if (is_coloured(iCell))
    if (Uintermediate.at(iCell, Imask) > 0) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      Kokkos::Array<real_t, 3> f_C;
      real_t neighbors(0);
      constexpr real_t w_relax(1.);
      for ( const ComponentIndex3D dir : {IX,IY,IZ} )
      {
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        const CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate L cell cannot have smaller intermediate neighbor" );
        const CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate R cell cannot have smaller intermediate neighbor" );
        real_t a(1), b(1), contrib_L(0), contrib_R(0);
        
        if (CellIndex::BIGGER == iCell_L.status) a = 0.5;
        else if (Uintermediate.at(iCell_L, Imask) < 0) { // Second-order reconstruction. 
          const real_t cell_mask = Uintermediate.at(iCell, Imask);
          const real_t left_neighbor_mask = Uintermediate.at(iCell_L, Imask);
          a = cell_mask / (cell_mask - left_neighbor_mask);
        } // else if (Uintermediate.at(iCell_L, Imask) < 1) a = 1; // First-order reconstruction 
        else contrib_L = Uintermediate.at(iCell_L, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_L.is_boundary() ) contrib_L = 0;

        if (CellIndex::BIGGER == iCell_R.status) b = 0.5;
        else if (Uintermediate.at(iCell_R, Imask) < 0) { // Second-order reconstruction.
          const real_t cell_mask = Uintermediate.at(iCell, Imask);
          const real_t right_neighbor_mask = Uintermediate.at(iCell_R, Imask);
          b = cell_mask / (cell_mask - right_neighbor_mask);
        } //else if (Uintermediate.at(iCell_R, Imask) < 1) b = 1;  // First-order reconstruction 
        else contrib_R = Uintermediate.at(iCell_R, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_R.is_boundary() ) contrib_R = 0;

        const real_t f_L = 2. / (a * (a + b));
        const real_t f_R = 2. / (b * (a + b));
        f_C[dir] = 2. / (a * b);
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
template< typename Array_t, typename Function >
void GravitySolver_multigrid::gauss_seidel_leaves_amr_finest(const Array_t& U, const Array_t& Uintermediate, const level_t level, const Function& is_coloured) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;

  foreach_cell.foreach_cell("Gauss-Seidel", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level == level) 
    if (is_coloured(iCell)) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      Kokkos::Array<real_t, 3> f_C;
      real_t neighbors(0);
      constexpr real_t w_relax(1.);
      for ( const ComponentIndex3D dir : {IX,IY,IZ} )
      {
        real_t a(1), b(1), contrib_L(0), contrib_R(0);
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        CellIndex iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
        CellIndex iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
        if ( CellIndex::SMALLER == iCell_L.status ) {
          iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
          contrib_L = Uintermediate.at(iCell_L, Isolution);
        } else if (CellIndex::BIGGER == iCell_L.status) {
          contrib_L = average_8bigger_neighbors(U, Uintermediate, iCell, off_L);
          a = 0.5;
        } else contrib_L = U.at(iCell_L, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_L.is_boundary() ) contrib_L = 0;
        if ( CellIndex::SMALLER == iCell_R.status ) {
          iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
          contrib_R = Uintermediate.at(iCell_R, Isolution);
        } else if (CellIndex::BIGGER == iCell_R.status) {
          contrib_R = average_8bigger_neighbors(U, Uintermediate, iCell, off_R);
          b = 0.5;
        } else contrib_R = U.at(iCell_R, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_R.is_boundary() ) contrib_R = 0;
        const real_t f_L = 2. / (a * (a + b));
        const real_t f_R = 2. / (b * (a + b));
        f_C[dir] = 2. / (a * b);
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
template< typename Array_t, typename Function >
void GravitySolver_multigrid::gauss_seidel_leaves_uniform(const Array_t& U, const Array_t& Uintermediate, const level_t level, const Function& is_coloured) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;

  foreach_cell.foreach_cell("Gauss-Seidel", U.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level == level)
    if (is_coloured(iCell)) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t neighbors(0);
      constexpr real_t w_relax(1.);
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
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_L.is_boundary() ) contrib_L = 0;
        if ( iCell_R.level_diff() != 0 ) {
          iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
          contrib_R = Uintermediate.at(iCell_R, Isolution);
        } else contrib_R = U.at(iCell_R, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_R.is_boundary() ) contrib_R = 0;
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
template< typename Array_t, typename Function >
void GravitySolver_multigrid::gauss_seidel_intermediate(const Array_t& U, const Array_t& Uintermediate, const level_t level, const Function& is_coloured) 
{
  const ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition = pdata->boundarycondition;

  foreach_cell.foreach_intermediate_cell("Gauss-Seidel", Uintermediate.getShape(),
    KOKKOS_LAMBDA(const CellIndex & iCell)
  {
    const uint8_t current_level = cells.getLevel(iCell);
    if (current_level == level)
    if (is_coloured(iCell)) {
      const ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell);
      real_t neighbors(0);
      constexpr real_t w_relax(1.);
      for ( ComponentIndex3D dir : {IX,IY,IZ} )
      {
        real_t contrib_L(0), contrib_R(0);
        CellIndex::offset_t off_L = {}; off_L[dir] = -1;
        CellIndex::offset_t off_R = {}; off_R[dir] = +1;
        CellIndex iCell_L = iCell.getNeighbor_ghost_intermediate(off_L, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_L.level_diff() >= 0, "Intermediate L cell cannot have smaller intermediate neighbor" );
        CellIndex iCell_R = iCell.getNeighbor_ghost_intermediate(off_R, Uintermediate.getShape());
        DYABLO_ASSERT_KOKKOS_DEBUG( iCell_R.level_diff() >= 0, "Intermediate R cell cannot have smaller intermediate neighbor" );
        if ( CellIndex::BIGGER == iCell_L.status ) {
          iCell_L = iCell.getNeighbor_ghost(off_L, U.getShape());
          contrib_L = U.at(iCell_L, Isolution);
        } else contrib_L = Uintermediate.at(iCell_L, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_L.is_boundary() ) contrib_L = 0;
        if ( CellIndex::BIGGER == iCell_R.status ) {
          iCell_R = iCell.getNeighbor_ghost(off_R, U.getShape());
          contrib_R = U.at(iCell_R, Isolution);
        } else contrib_R = Uintermediate.at(iCell_R, Isolution);
        if ( BC_ABSORBING == boundarycondition[dir] && iCell_R.is_boundary() ) contrib_R = 0;
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
void GravitySolver_multigrid::smoothing_intermediate_amr_correction(const Array_t& Uintermediate, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm) 
{
  const bool isMPILevel = (level >= pdata->first_mpi_multigrid_level);
  for (uint32_t i = 0; i < nIterations; i++) {
    gauss_seidel_intermediate_amr_correction(Uintermediate, level, KOKKOS_LAMBDA(const CellIndex& iCell){return isRed(iCell);});
    if (isMPILevel) 
      ghost_comm.exchange_intermediate_ghosts(Uintermediate);

    gauss_seidel_intermediate_amr_correction(Uintermediate, level, KOKKOS_LAMBDA(const CellIndex& iCell){return isBlack(iCell);});
    if (isMPILevel)
      ghost_comm.exchange_intermediate_ghosts(Uintermediate);
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
void GravitySolver_multigrid::smoothing_uniform(const Array_t& U, const Array_t& Uintermediate, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm) 
{
  const bool isMPILevel = (level >= pdata->first_mpi_multigrid_level);
  for (uint32_t i = 0; i < nIterations; i++) {
    gauss_seidel_leaves_uniform(U, Uintermediate, level, KOKKOS_LAMBDA(const CellIndex& iCell){return isRed(iCell);});
    if (isMPILevel)
      ghost_comm.exchange_ghosts(U);

    gauss_seidel_intermediate(U, Uintermediate, level, KOKKOS_LAMBDA(const CellIndex& iCell){return isRed(iCell);});
    if (isMPILevel)
      ghost_comm.exchange_intermediate_ghosts(Uintermediate);

    gauss_seidel_leaves_uniform(U, Uintermediate, level, KOKKOS_LAMBDA(const CellIndex& iCell){return isBlack(iCell);});
    if (isMPILevel)
      ghost_comm.exchange_ghosts(U);

    gauss_seidel_intermediate(U, Uintermediate, level, KOKKOS_LAMBDA(const CellIndex& iCell){return isBlack(iCell);});
    if (isMPILevel)
      ghost_comm.exchange_intermediate_ghosts(Uintermediate);
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
void GravitySolver_multigrid::smoothing_amr_finest(const Array_t& U, const Array_t& Uintermediate, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm) 
{
  const bool isMPILevel = (level >= pdata->first_mpi_multigrid_level);
  for (uint32_t i = 0; i < nIterations; i++) {
    gauss_seidel_leaves_amr_finest(U, Uintermediate, level, KOKKOS_LAMBDA(const CellIndex& iCell){return isRed(iCell);});
    if (isMPILevel)
      ghost_comm.exchange_ghosts(U);

    gauss_seidel_intermediate(U, Uintermediate, level, KOKKOS_LAMBDA(const CellIndex& iCell){return isRed(iCell);});
    if (isMPILevel)
      ghost_comm.exchange_intermediate_ghosts(Uintermediate);

    gauss_seidel_leaves_amr_finest(U, Uintermediate, level, KOKKOS_LAMBDA(const CellIndex& iCell){return isBlack(iCell);});
    if (isMPILevel)
      ghost_comm.exchange_ghosts(U);

    gauss_seidel_intermediate(U, Uintermediate, level, KOKKOS_LAMBDA(const CellIndex& iCell){return isBlack(iCell);});
    if (isMPILevel)
      ghost_comm.exchange_intermediate_ghosts(Uintermediate);
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
void GravitySolver_multigrid::V_cycle_uniform(Array_t& U, Array_t& Uintermediate, const level_t level, const GhostCommunicator& ghost_comm) 
{  
  const bool isFirstMPILevel = (level == pdata->first_mpi_multigrid_level);
  const bool isMPILevel = (level >= pdata->first_mpi_multigrid_level);
  
  smoothing_uniform(U, Uintermediate, pdata->Npre, level, ghost_comm);
  residual_uniform(U, Uintermediate, level);

  if (isMPILevel){
    ghost_comm.exchange_ghosts(U);
    ghost_comm.exchange_intermediate_ghosts(Uintermediate);
  }

  zero_rhs_mask_intermediate(Uintermediate, level - 1);
  restriction_from_children(U, Uintermediate, level - 1);

  if (isFirstMPILevel) 
    reduce_nonMPI_levels(Uintermediate, pdata->first_mpi_multigrid_level, make_array<int, 2>({Irhs, Imask}));
  else if (isMPILevel){
    ghost_comm.reduce_intermediate_ghosts_at_level(Uintermediate, level - 1, make_array<int, 2>({Irhs, Imask}));
    ghost_comm.exchange_intermediate_ghosts(Uintermediate); // Useful?
  }

  initialise_lhs(U, Uintermediate, level - 1);
  if (isMPILevel)
    ghost_comm.exchange_intermediate_ghosts(Uintermediate);

  if (level == 1) {
      smoothing_uniform(U, Uintermediate, pdata->Npre, level - 1, ghost_comm);
  }
  else V_cycle_uniform(U, Uintermediate, level - 1, ghost_comm);
    
  prolongation(U, Uintermediate, level); // Also add case for prolongation_on_intermediates
  if (isMPILevel){
    ghost_comm.exchange_ghosts(U);
    ghost_comm.exchange_intermediate_ghosts(Uintermediate);
  } 
  smoothing_uniform(U, Uintermediate, pdata->Npost, level, ghost_comm);
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
void GravitySolver_multigrid::V_cycle_amr(Array_t& U, Array_t& Uintermediate, const uint8_t current_level, const uint32_t finest_level, const GhostCommunicator& ghost_comm) 
{  
  const bool isFirstMPILevel = (current_level == pdata->first_mpi_multigrid_level);
  const bool isMPILevel = (current_level >= pdata->first_mpi_multigrid_level);

  // Full Multigrid

  if ( current_level == finest_level ) {
    smoothing_amr_finest(U, Uintermediate, pdata->Npre, current_level, ghost_comm);
    residual_amr_finest(U, Uintermediate, current_level);
  } else {
    smoothing_intermediate_amr_correction(Uintermediate, pdata->Npre, current_level, ghost_comm);
    residual_intermediate_amr_correction(Uintermediate, current_level);
  }

  if (isMPILevel){
    ghost_comm.exchange_ghosts(U);
    ghost_comm.exchange_intermediate_ghosts(Uintermediate);
  }

  zero_rhs_mask_intermediate(Uintermediate, current_level - 1);
  restriction_from_children(U, Uintermediate, current_level - 1);

  if (isFirstMPILevel) {
    reduce_nonMPI_levels(Uintermediate, pdata->first_mpi_multigrid_level, make_array<int, 2>({Irhs, Imask}));
  } else if (isMPILevel){
    ghost_comm.reduce_intermediate_ghosts_at_level(Uintermediate, current_level - 1, make_array<int, 2>({Irhs, Imask}));
    ghost_comm.exchange_intermediate_ghosts(Uintermediate); // Useful?
  }

  initialise_lhs(U, Uintermediate, current_level - 1);
  if (isMPILevel && !isFirstMPILevel){
    ghost_comm.exchange_intermediate_ghosts(Uintermediate);
  }

  if ( finest_level - 3 == current_level ) { // TODO: finest - 2 seems to works aswell for spherical symmetry. Check for more realistic cases
      smoothing_intermediate_amr_correction(Uintermediate, pdata->Npre, pdata->level_coarse, ghost_comm);
  } else V_cycle_amr(U, Uintermediate, current_level - 1, finest_level, ghost_comm); 
  
  if ( current_level == finest_level ) {
    prolongation(U, Uintermediate, current_level);
    if (isMPILevel){
      ghost_comm.exchange_ghosts(U);
      ghost_comm.exchange_intermediate_ghosts(Uintermediate);
    }
    smoothing_amr_finest(U, Uintermediate, pdata->Npost, current_level, ghost_comm);
  } else {
    prolongation_on_intermediate(U, Uintermediate, current_level); 
    if (isMPILevel){
      ghost_comm.exchange_intermediate_ghosts(Uintermediate); 
    }

    smoothing_intermediate_amr_correction(Uintermediate, pdata->Npost, current_level, ghost_comm);
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

template< typename Array_t, typename T, std::size_t N >
void GravitySolver_multigrid::reduce_nonMPI_levels(const Array_t& Uintermediate, const level_t first_mpi_multigrid_level, const Kokkos::Array<T, N> iFields) 
{
  uint32_t num_vars = N; // number of vars for each cell

  ForeachCell& foreach_cell = pdata->foreach_cell;
  const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
  const MpiComm& mpi_comm = foreach_cell.get_amr_mesh().getMpiComm();
  const LightOctree& lmesh = foreach_cell.get_amr_mesh().getLightOctree();
  const uint32_t bx=Uintermediate.getShape().bx, by=Uintermediate.getShape().by, bz=Uintermediate.getShape().bz ;
  const uint32_t nbCellsPerBlock = bx * by * bz;
  const uint32_t ncells_1d = 1U << (first_mpi_multigrid_level - 1); // Total number of octants at level (first_mpi_multigrid_level - 1)
  const uint32_t nbOcts = ncells_1d*ncells_1d*ncells_1d;
  Kokkos::View<double*> deviceArray("deviceArray", nbOcts * nbCellsPerBlock * num_vars);

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

template <typename T, size_t N>
KOKKOS_INLINE_FUNCTION
Kokkos::Array<T, N> GravitySolver_multigrid::make_array(const Kokkos::Array<T, N>& vals) {
    return vals;
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

  ForeachCell& foreach_cell = pdata->foreach_cell;
  const MpiComm& mpi_comm = foreach_cell.get_amr_mesh().getMpiComm();
  const int mpi_rank = mpi_comm.MPI_Comm_rank();
  auto& amr_mesh = foreach_cell.get_amr_mesh();
  const level_t first_mpi_multigrid_level = pdata->first_mpi_multigrid_level;
  constexpr level_t min_level_multigrid = 0;

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
  
 
  // Compute ghost communicators
  const auto iter_space = U.getShape();
  GhostCommunicator ghost_comm(foreach_cell.get_amr_mesh(), iter_space, 1, mpi_comm);
  DYABLO_ASSERT_KOKKOS_DEBUG(
    iter_space.bx == iter_space.by &&
    iter_space.bx == iter_space.bz &&
    iter_space.bx % 2 == 0,
    "Wrong block shape"
  );
  ghost_comm.init_intermediates(foreach_cell.get_amr_mesh(), iter_space, iter_space.bx, mpi_comm);

  // Count number of Leaf and Intermediate (+ ghosts) octs in AMR
  {
    const level_t nlevel = global_max_level_found - min_level_multigrid + 1;
    Kokkos::View<int*> octs_per_level("octs_per_level", nlevel);
    Kokkos::View<int*> octs_intermediate_per_level("octs_intermediate_per_level", nlevel);
    Kokkos::View<int*> ghosts_per_level("ghosts_per_level", nlevel);
    Kokkos::View<int*> ghosts_intermediate_per_level("ghosts_intermediate_per_level", nlevel);
    Kokkos::deep_copy(octs_per_level, 0);
    Kokkos::deep_copy(octs_intermediate_per_level, 0);
    Kokkos::deep_copy(ghosts_per_level, 0);
    Kokkos::deep_copy(ghosts_intermediate_per_level, 0);

  
    // Count number of octs and ghosts in AMR per level
    const int numOcts = lmesh.getNumOctants();
    const int numIntermediateOcts = lmesh.getNumIntermediateOctants();
    const int numGhosts = lmesh.getNumGhosts();
    const int numIntermediateGhosts = lmesh.getNumIntermediateGhosts();
    Kokkos::parallel_for( "Count number of octs per level", Kokkos::RangePolicy<>(0, numOcts),
      KOKKOS_LAMBDA( const uint32_t iOct )
      {
        const level_t level = lmesh.getLevel({iOct, false, false});
        Kokkos::atomic_fetch_add( &octs_per_level(level-min_level_multigrid), 1 );
      }
    );
    Kokkos::parallel_for( "Count number of intermediate octs per level", Kokkos::RangePolicy<>(0, numIntermediateOcts),
      KOKKOS_LAMBDA( const uint32_t iOct )
      {
        const level_t level = lmesh.getLevel({iOct, false, true});
        Kokkos::atomic_fetch_add( &octs_intermediate_per_level(level-min_level_multigrid), 1 );
      }
    );
    Kokkos::parallel_for( "Count number of ghosts per level", Kokkos::RangePolicy<>(0, numGhosts),
      KOKKOS_LAMBDA( const uint32_t iOct )
      {
        const level_t level = lmesh.getLevel({iOct, true, false});
        Kokkos::atomic_fetch_add( &ghosts_per_level(level-min_level_multigrid), 1 );
      }
    );
    Kokkos::parallel_for( "Count number of intermediate ghosts per level", Kokkos::RangePolicy<>(0, numIntermediateGhosts),
      KOKKOS_LAMBDA( const uint32_t iOct )
      {
        const level_t level = lmesh.getLevel({iOct, true, true});
        Kokkos::atomic_fetch_add( &ghosts_intermediate_per_level(level-min_level_multigrid), 1 );
      }
    );

    const auto octs_per_level_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), octs_per_level);
    const auto octs_intermediate_per_level_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), octs_intermediate_per_level);
    const auto ghosts_per_level_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ghosts_per_level);
    const auto ghosts_intermediate_per_level_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ghosts_intermediate_per_level);


    for(level_t ilevel = min_level_multigrid; ilevel <= global_max_level_found; ilevel++){ 
      const level_t level_diff = ilevel-min_level_multigrid;
      const uint32_t octs = octs_per_level_host(level_diff);
      const uint32_t ghosts = ghosts_per_level_host(level_diff);
      const uint32_t octs_intermediate = octs_intermediate_per_level_host(level_diff);
      const uint32_t ghosts_intermediate = ghosts_intermediate_per_level_host(level_diff);
      const real_t mean_total_nbOctants_per_dimension = Kokkos::floor(Kokkos::cbrt(octs+ghosts+octs_intermediate+ghosts_intermediate));
      const real_t total_nbOctants_per_dimension = (1U << ilevel);
      DYABLO_ASSERT_KOKKOS_DEBUG( mean_total_nbOctants_per_dimension <= total_nbOctants_per_dimension, "Cannot count more octants per level than there are in the simulation" );
      printf("Rank %d Octree Level %d, octs %u (+ %u) intermediate %u (+ %u)\n", mpi_rank, ilevel, octs, ghosts, octs_intermediate, ghosts_intermediate);
    }
  }
  
  check_parents(U, Uintermediate);
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
  ghost_comm.exchange_ghosts(U);
  // Initialize RHS on intermediate levels. Solution will be interpolated from coarser levels so no need to initialize
  constexpr int ns = 8;

  for( uint32_t level = global_max_level_found - 1; level >= level_coarse; level-- )
  {
    foreach_cell.foreach_cell( "Restrict", U.getShape(),
    KOKKOS_LAMBDA( CellIndex& iCell)
    {
      const uint8_t current_level = cells.getLevel(iCell);
      if( current_level == level + 1 )
      {
        const CellIndex iCell_p = iCell.getParent(U.getShape());      
        Kokkos::atomic_add( &Uintermediate.at( iCell_p, Irho ), U.at( iCell, Irho ) / ns );
      }
    }); 
    foreach_cell.foreach_intermediate_cell( "Restrict", Uintermediate.getShape(),
    KOKKOS_LAMBDA( CellIndex& iCell)
    {
      const uint8_t current_level = cells.getLevel(iCell);
      if( current_level == level + 1 )
      {
        const CellIndex iCell_p = iCell.getParent(Uintermediate.getShape());     
        Kokkos::atomic_add( &Uintermediate.at( iCell_p, Irho ), Uintermediate.at( iCell, Irho ) / ns );
      }
    });
    ghost_comm.reduce_intermediate_ghosts_at_level(Uintermediate, level, make_array<int, 1>({Irho}));
    ghost_comm.exchange_intermediate_ghosts(Uintermediate);
  }
  foreach_cell.foreach_intermediate_cell( "Set RHS of Laplacian, based on rho", Uintermediate.getShape(),
  KOKKOS_LAMBDA( CellIndex& iCell)
  {
    Uintermediate.at( iCell, Irhs ) = (cosmo_run) ? b_cosmo(Uintermediate, iCell, rho_mean, aexp) : b(Uintermediate, iCell, rho_mean, four_Pi_G); //rho - rho_mean;
  }); 
  ghost_comm.exchange_intermediate_ghosts(Uintermediate);

  residual_uniform(U, Uintermediate, level_coarse);
  solution_to_potential(U, Uintermediate, level_coarse);
  const real_t residual = residual_norm(U, Uintermediate, level_coarse);
  if (mpi_rank == 0) printf("Level %d Residual norm init  V %.8e\n", level_coarse, residual);

  // Multigrid
  if (mpi_rank == 0) printf("Coarse Multigrid\n");
  for(uint32_t i = 0; i < pdata->Ncycles; i++)
  { 
    V_cycle_uniform(U, Uintermediate, level_coarse, ghost_comm);
    residual_uniform(U, Uintermediate, level_coarse);
    solution_to_potential(U, Uintermediate, level_coarse);
    const real_t residual = residual_norm(U, Uintermediate, level_coarse);
    if (mpi_rank == 0) printf("Level %d Residual norm after V %.8e\n", level_coarse, residual);
  }
  
  if (mpi_rank == 0) printf("AMR Multigrid\n");
  for (level_t ilevel = level_coarse+1; ilevel <= global_max_level_found; ilevel++) {
    zero_solution(U, Uintermediate, ilevel);
    prolongation(U, Uintermediate, ilevel);
    zero_solution_residual_rhs(U, Uintermediate, ilevel-1);
    solution_to_potential(U, Uintermediate, ilevel);
    initialise_mask(U, Uintermediate, ilevel);
    ghost_comm.exchange_ghosts(U);
    ghost_comm.exchange_intermediate_ghosts(Uintermediate);

    for(uint32_t i = 0; i < pdata->Ncycles; i++)
    { 
      V_cycle_amr(U, Uintermediate, ilevel, ilevel, ghost_comm);
      residual_amr_finest(U, Uintermediate, ilevel);
      solution_to_potential(U, Uintermediate, ilevel);
      const real_t residual = residual_norm(U, Uintermediate, ilevel);
      if (mpi_rank == 0) printf("Level %d Residual norm after V %.8e\n", ilevel, residual);
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
