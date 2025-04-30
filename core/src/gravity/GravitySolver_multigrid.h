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
  template <class GhostComm > void exchange_specific_ghosts(UserData& U_, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm);
  template <class GhostComm > void exchange_specific_intermediate_ghosts(UserData& U_, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm);
  template <class GhostComm > void exchange_specific_ghosts_at_level(UserData& U_, const uint8_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm);
  template <class GhostComm > void exchange_specific_intermediate_ghosts_at_level(UserData& U_, const uint8_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm);
  template <class GhostComm > void reduce_specific_ghosts_at_level(UserData& U_, const uint8_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm);
  template <class GhostComm > void reduce_specific_intermediate_ghosts_at_level(UserData& U_, const uint8_t level, const std::vector< std::string >& exchange_vars, GhostComm& ghost_comm);
  template <typename T, size_t N> KOKKOS_INLINE_FUNCTION Kokkos::Array<T, N> make_array(const Kokkos::Array<T, N>& vals);
  static real_t MPI_Allreduce_scalar( real_t local_v );
  static uint32_t MPI_Allreduce_int_max( uint32_t local_v );
  template< typename T, size_t N > void reduce_nonMPI_levels(const level_t first_mpi_multigrid_level, const Kokkos::Array<T, N> iFields);
  
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
  void check_parents();
  KOKKOS_INLINE_FUNCTION static bool isRed(const ForeachCell::CellIndex& iCell);
  KOKKOS_INLINE_FUNCTION static bool isBlack(const ForeachCell::CellIndex& iCell);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static real_t get_value(const Array_t& U, const ForeachCell::CellIndex& iCell_U, VarIndex var, const ForeachCell::CellIndex::offset_t& offset, const int ndim);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static real_t average_4bigger_neighbors(const Array_t& U, const Array_t& Uintermediate, const ForeachCell::CellIndex iCell, ForeachCell::CellIndex::offset_t offset);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static real_t average_8bigger_neighbors(const Array_t& U, const Array_t& Uintermediate, const ForeachCell::CellIndex iCell, ForeachCell::CellIndex::offset_t offset);
  template< typename Array_t > KOKKOS_INLINE_FUNCTION static real_t get_neighbor_value(const Array_t& U, const Array_t& Uintermediate, const ForeachCell::CellIndex iCell, const ForeachCell::CellIndex::offset_t offset);
  void restriction_from_parents(const level_t level);
  void restriction_from_children(const level_t level);
  void prolongation_from_children(const level_t level);
  void prolongation_from_children_on_intermediate(const level_t level);
  void zero_solution(const level_t level);
  void initialise_mask(const uint32_t finest_level);
  void zero_solution_residual_rhs(const level_t level);
  void zero_rhs_mask_intermediate(const level_t level);
  void solution_to_potential(const level_t level);

  template< typename Array_t > void gradient(Array_t& U, Array_t& Uintermediate);

  // Multigrid 

  void V_cycle_uniform(UserData& U_, const level_t level, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide);
  void V_cycle_amr(UserData& U_, const uint8_t current_level, const uint32_t finest_level, const GhostCommunicator& ghost_comm_minimal, const GhostCommunicator& ghost_comm_blockwide);
  
  // Laplacian
  void initialise_intermediate_lhs(const level_t level);
  real_t residual_norm(const level_t level);
  void residual_amr_finest(const level_t level);
  void residual_uniform(const level_t level);
  void residual_intermediate_amr_correction(const level_t level);  
  template< typename Function > void gauss_seidel_intermediate_amr_correction(const level_t level, const Function& is_coloured); 
  template< typename Function > void gauss_seidel_leaves_amr_finest(const level_t level, const Function& is_coloured); 
  template< typename Function > void gauss_seidel_leaves_uniform(const level_t level, const Function& is_coloured);
  template< typename Function > void gauss_seidel_intermediate(const level_t level, const Function& is_coloured);
  void smoothing_intermediate_amr_correction(UserData& U_, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm_minimal);
  void smoothing_uniform(UserData& U_, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm_minimal);                 
  void smoothing_amr_finest(UserData& U_, const uint32_t nIterations, const level_t level, const GhostCommunicator& ghost_comm_minimal);
  struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
