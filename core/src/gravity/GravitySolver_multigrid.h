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

enum class Target {
  LEAVES,
  INTERMEDIATES,
  BOTH,
  GHOST_LEAVES,
  GHOST_INTERMEDIATES,
  BOTH_GHOSTS,
  ALL_LEAVES,
  ALL_INTERMEDIATES,
  ALL,
};

class GravitySolver_multigrid : public GravitySolver{
public: 
  using level_t = uint8_t;
  using Shape_t = UserData_fields::FieldView_t::Shape_t;
  GravitySolver_multigrid(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers );
  ~GravitySolver_multigrid();
  void update_gravity_field( UserData& U, ScalarSimulationData& scalar_data);

  // MPI
  template <Target target, class GhostComm > void exchange_ghosts_at_level(UserData& U_, const uint8_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm);
  template <Target target, class GhostComm > void reduce_ghosts_at_level(UserData& U_, const uint8_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm);

  template <typename T, typename... Ts>
  KOKKOS_INLINE_FUNCTION
  static constexpr Kokkos::Array<T, sizeof...(Ts)> kokkos_array(const Ts... vals) {
    return Kokkos::Array<T, sizeof...(Ts)>{static_cast<T>(vals)...};
  }
  
  static real_t MPI_Allreduce_scalar( real_t local_v );
  static uint32_t MPI_Allreduce_int_max( uint32_t local_v );
  template< typename T, size_t N > void reduce_nonMPI_levels(const level_t first_mpi_multigrid_level, const Kokkos::Array<T, N>& iFields);
  
  // Octants per level
  void count_octants_per_level(const LightOctree& lmesh);
  void sort_octants_per_level(const LightOctree& lmesh);
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
  bool check_parents();
  void compute_mask(UserData& U_, const level_t level_min, const level_t level_max, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide);
  template<size_t N> KOKKOS_INLINE_FUNCTION static void accumulate_up_tree(const UserData_fields::FieldAccessor& U, ForeachCell::CellIndex& iCell, const Shape_t& iter_space, const level_t level_stop, const level_t level_start, const Kokkos::Array<real_t, N>& factor_init, const real_t factor_decay, const Kokkos::Array<int, N>& iFields);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static real_t average_8bigger_neighbors(const Array_t& U, const Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Shape_t& iter_space, ForeachCell::CellIndex::offset_t offset);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static real_t get_neighbor_value(const Array_t& U, const Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Shape_t& iter_space, const ForeachCell::CellIndex::offset_t offset);
  template< Target target, typename Integer_t, size_t N > void restriction_from_children(const level_t level_stop, const level_t level_start, const Kokkos::Array<Integer_t, N>& iFields_children, const Kokkos::Array<Integer_t, N>& iFields_parents);
  template <Target target> void prolongation_from_children(const level_t level);
  template <Target target, typename T, size_t N> void copy_multigrid_fields(const level_t level_min, const level_t level_max, const Kokkos::Array<T, N>& iFields_src, const Kokkos::Array<T, N>& iFields_dst);
  template <Target target, typename Integer_t, typename Real_t, size_t N> void fill_multigrid_fields(const level_t min_level, const level_t max_level, const Kokkos::Array<Integer_t, N>& iFields, const Kokkos::Array<Real_t, N>& values);

  template< typename Array_t > KOKKOS_INLINE_FUNCTION static void get_weight_and_contrib_gradient_3pt (Array_t& U, Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Shape_t& iter_space, const int side, real_t& w, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition); 
  template< typename Array_t > void gradient_3pt(Array_t& U, Array_t& Uintermediate);
  template< typename Array_t > void gradient_5pt(Array_t& U, Array_t& Uintermediate);

  // Multigrid 

  void V_cycle_uniform(UserData& U_, const level_t level, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide);
  void V_cycle_amr(UserData& U_, const level_t current_level, const level_t finest_level, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide);
  
  // Laplacian
  void operator_amr_finest(const level_t level);
  void operator_uniform(const level_t level);
  void operator_intermediate_amr_correction(const level_t level);  
  template <Target target> void initialise_lhs_from_rhs(const level_t level);
  real_t residual_norm(const level_t level);
  real_t truncation_norm(const level_t level);
  void residual_amr_finest(const level_t level);
  template <Target target> void residual_uniform(const level_t level);
  void residual_intermediate_amr_correction(const level_t level);  
  template< bool is_red > void gauss_seidel_intermediate_amr_correction_rb(const level_t level); 
  template< bool is_red > void gauss_seidel_leaves_amr_finest_rb(const level_t level); 
  template< bool is_red > void gauss_seidel_leaves_uniform_rb(const level_t level);
  template< bool is_red > void gauss_seidel_intermediate_rb(const level_t level);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static void gauss_seidel_intermediate_amr_correction(Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition); 
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static void gauss_seidel_leaves_amr_finest(Array_t& U, Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition); 
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static void gauss_seidel_leaves_uniform(Array_t& U, Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static void gauss_seidel_intermediate(Array_t& U, Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Kokkos::Array<real_t, 3>& size, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition); 
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static void get_weight_and_contrib_amr_finest (Array_t& U, Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Shape_t& iter_space, const int side, real_t& w, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition); 
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static void get_weight_and_contrib_intermediate_amr_correction (Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Shape_t& iter_space, const int side, real_t& w, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition, const real_t mask);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static void get_contrib_leaves_uniform (Array_t& U, Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Shape_t& iter_space, const int side, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition); 
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static void get_contrib_intermediate_uniform (Array_t& U, Array_t& Uintermediate, const ForeachCell::CellIndex& iCell, const Shape_t& iter_space, const int side, real_t& contrib, const ComponentIndex3D dir, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition); 
  void smoothing_intermediate_amr_correction(UserData& U_, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm_minimal);
  template <Target target> void smoothing_uniform(UserData& U_, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm_minimal);                 
  void smoothing_amr_finest(UserData& U_, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm_minimal);
  struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
