#pragma once

#include <memory>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "gravity/GravitySolver_base.h"
#include "mpi/GhostCommunicator.h"



class Timers;
class ConfigMap;

namespace dyablo {

/**
 * @brief Class solving the poisson equation for gravity using
 * a multigrid scheme 
 */
class GravitySolver_multigrid : public GravitySolver{
public: 
  using level_t = uint8_t;
  GravitySolver_multigrid(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers );
  ~GravitySolver_multigrid();
  void update_gravity_field( UserData& U, ScalarSimulationData& scalar_data);

  // MPI
  template <typename T, size_t N> KOKKOS_INLINE_FUNCTION Kokkos::Array<T, N> make_array(const Kokkos::Array<T, N>& vals);
  static real_t MPI_Allreduce_scalar( real_t local_v );
  static uint32_t MPI_Allreduce_int_max( uint32_t local_v );
  template< typename Array_t, typename T, size_t N > void reduce_nonMPI_levels(const Array_t& Uintermediate, const level_t first_mpi_multigrid_level, const Kokkos::Array<T, N> iFields);
  
  // Octants per level
  void count_octants_per_level(const LightOctree& lmesh);
  void list_octants_per_level(const LightOctree& lmesh);
  const Kokkos::View<uint32_t*> get_subview_octs(const level_t level) const;
  const Kokkos::View<uint32_t*> get_subview_octs(const level_t level_min, const level_t level_max) const;
  const Kokkos::View<uint32_t*> get_subview_octs_intermediate(const level_t level) const;
  const Kokkos::View<uint32_t*> get_subview_octs_intermediate(const level_t level_min, const level_t level_max) const;
  const Kokkos::View<uint32_t*> get_subview_ghosts(const level_t level) const;
  const Kokkos::View<uint32_t*> get_subview_ghosts(const level_t level_min, const level_t level_max) const;
  const Kokkos::View<uint32_t*> get_subview_ghosts_intermediate(const level_t level) const;
  const Kokkos::View<uint32_t*> get_subview_ghosts_intermediate(const level_t level_min, const level_t level_max) const;


  // RHS
  KOKKOS_INLINE_FUNCTION static real_t b(const UserData::FieldAccessor& Uin, const ForeachCell::CellIndex& iCell_Uin, real_t rho_mean, real_t four_Pi_G);
  KOKKOS_INLINE_FUNCTION static real_t b_cosmo(const UserData::FieldAccessor& Uin, const ForeachCell::CellIndex& iCell_Uin, real_t rho_mean, real_t aexp);

  // Mesh
  template< typename Array_t >  void check_parents(const Array_t& U, const Array_t& Uintermediate);
  KOKKOS_INLINE_FUNCTION static bool isRed(const ForeachCell::CellIndex& iCell);
  KOKKOS_INLINE_FUNCTION static bool isBlack(const ForeachCell::CellIndex& iCell);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static real_t get_value(const Array_t& U, const ForeachCell::CellIndex& iCell_U, VarIndex var, const ForeachCell::CellIndex::offset_t& offset, const int ndim);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static real_t average_4bigger_neighbors(const Array_t& U, const Array_t& Uintermediate, const ForeachCell::CellIndex iCell, ForeachCell::CellIndex::offset_t offset);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static real_t average_8bigger_neighbors(const Array_t& U, const Array_t& Uintermediate, const ForeachCell::CellIndex iCell, ForeachCell::CellIndex::offset_t offset);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static real_t get_neighbor_value(const Array_t& U, const Array_t& Uintermediate, const ForeachCell::CellIndex iCell, const ForeachCell::CellIndex::offset_t offset);
  template< typename Array_t > void restriction_from_parents(const Array_t& U, const Array_t& Uintermediate, const level_t level);
  template< typename Array_t > void restriction_from_children(const Array_t& U, const Array_t& Uintermediate, const level_t level);
  template< typename Array_t > void prolongation_from_children(const Array_t& U, const Array_t& Uintermediate, const level_t level);
  template< typename Array_t > void prolongation_from_children_on_intermediate(const Array_t& U, const Array_t& Uintermediate, const level_t level);
  template< typename Array_t > void zero_solution(const Array_t& U, const Array_t& Uintermediate, const level_t level);
  template< typename Array_t > void initialise_mask(const Array_t& U, const Array_t& Uintermediate, const uint32_t finest_level);
  template< typename Array_t > void zero_solution_residual_rhs(const Array_t& U, const Array_t& Uintermediate, const level_t level);
  template< typename Array_t > void zero_rhs_mask_intermediate(const Array_t& Uintermediate, const level_t level);
  template< typename Array_t > void solution_to_potential(const Array_t& U, const Array_t& Uintermediate, const level_t level);

  template< typename Array_t > void gradient0(const Array_t& U);
  template< typename Array_t > void gradient(const Array_t& U, const Array_t& Uintermediate);

  // Multigrid 

  template< typename Array_t > void V_cycle_uniform(Array_t& U, Array_t& Uintermediate, const level_t level, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide);
  template< typename Array_t > void V_cycle_amr(Array_t& U, Array_t& Uintermediate, const uint8_t current_level, const uint32_t finest_level, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide);
  
  // Laplacian
  template< typename Array_t > void initialise_intermediate_lhs(const Array_t& Uintermediate, const level_t level);
  template< typename Array_t > real_t residual_norm(const Array_t& U, const Array_t& Uintermediate, const level_t level);
  template< typename Array_t > void residual_amr_finest(const Array_t& U, const Array_t& Uintermediate, const level_t level);
  template< typename Array_t > void residual_uniform(const Array_t& U, const Array_t& Uintermediate, const level_t level);
  template< typename Array_t > void residual_intermediate_amr_correction(const Array_t& Uintermediate, const level_t level);  
  template< typename Array_t, typename Function > void gauss_seidel_intermediate_amr_correction(const Array_t& Uintermediate, const level_t level, const Function& is_coloured); 
  template< typename Array_t, typename Function > void gauss_seidel_leaves_amr_finest(const Array_t& U, const Array_t& Uintermediate, const level_t level, const Function& is_coloured); 
  template< typename Array_t, typename Function > void gauss_seidel_leaves_uniform(const Array_t& U, const Array_t& Uintermediate, const level_t level, const Function& is_coloured);
  template< typename Array_t, typename Function > void gauss_seidel_intermediate(const Array_t& U, const Array_t& Uintermediate, const level_t level, const Function& is_coloured);
  template< typename Array_t > void smoothing_intermediate_amr_correction(const Array_t& Uintermediate, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm_minimal);
  template< typename Array_t > void smoothing_uniform(const Array_t& U, const Array_t& Uintermediate, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm_minimal);                 
  template< typename Array_t > void smoothing_amr_finest(const Array_t& U, const Array_t& Uintermediate, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm_minimal);
  struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
