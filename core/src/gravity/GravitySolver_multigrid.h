#pragma once

#include <memory>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "gravity/GravitySolver_base.h"

class Timers;
class ConfigMap;

namespace dyablo {

/**
 * @brief Class solving the poisson equation for gravity using
 * a multigrid scheme 
 */
class GravitySolver_multigrid : public GravitySolver{
public: 
  GravitySolver_multigrid(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers );
  ~GravitySolver_multigrid();
  void update_gravity_field( UserData& U, ScalarSimulationData& scalar_data);

  // Mesh
  static bool isRed(const ForeachCell::CellIndex iCell);
  static bool isBlack(const ForeachCell::CellIndex iCell);
  template< typename Array_t >  real_t get_value(const Array_t& U, const ForeachCell::CellIndex& iCell_U, VarIndex var, const ForeachCell::CellIndex::offset_t& offset);
  template< typename Array_t >  real_t average_4bigger_neighbors(const Array_t& U, const Array_t& Uintermediate, const ForeachCell::CellIndex iCell, ForeachCell::CellIndex::offset_t offset);
  template< typename Array_t >  real_t average_8bigger_neighbors(const Array_t& U, const Array_t& Uintermediate, const ForeachCell::CellIndex iCell, ForeachCell::CellIndex::offset_t offset);
  template< typename Array_t >  real_t get_neighbor_value(const Array_t& U, const Array_t& Uintermediate, const ForeachCell::CellIndex iCell, const ForeachCell::CellIndex::offset_t offset);
  template< typename Array_t >  void restriction(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t >  void prolongation0_inject(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t >  void prolongation0(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t >  void prolongation(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t >  void prolongation_on_intermediate(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t >  void zero_solution(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t >  void initialise_mask(const Array_t& U, const Array_t& Uintermediate, const uint32_t finest_level);
  template< typename Array_t >  void zero_solution_residual_rhs(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t >  void solution_to_potential(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);

  // Multigrid 

  template< typename Array_t >  void V_cycle_uniform(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t >  void V_cycle_amr(const Array_t& U, const Array_t& Uintermediate, const uint32_t current_level, const uint32_t finest_level);
  
  // Laplacian
  template< typename Array_t >  void initialise_lhs(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t > real_t residual_norm_sqr(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t >  void residual_amr_finest(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t >  void residual_uniform(const Array_t& U, const Array_t& Uintermediate, const uint32_t level);
  template< typename Array_t >  void residual_intermediate_amr_correction(const Array_t& Uintermediate, const uint32_t level);  
  template< typename Array_t >  void gauss_seidel_intermediate_amr_correction(const Array_t& Uintermediate, std::function<bool(ForeachCell::CellIndex)> is_coloured, const uint32_t level); 
  template< typename Array_t >  void gauss_seidel_leaves_amr_finest(const Array_t& U, const Array_t& Uintermediate, std::function<bool(ForeachCell::CellIndex)> is_coloured, const uint32_t level); 
  template< typename Array_t >  void gauss_seidel_leaves_uniform(const Array_t& U, const Array_t& Uintermediate, std::function<bool(ForeachCell::CellIndex)> is_coloured, const uint32_t level);
  template< typename Array_t >  void gauss_seidel_intermediate(const Array_t& U, const Array_t& Uintermediate, std::function<bool(ForeachCell::CellIndex)> is_coloured, const uint32_t level);
  template< typename Array_t >  void smoothing_intermediate_amr_correction(const Array_t& Uintermediate, const uint32_t nIterations, const uint32_t level);
  template< typename Array_t >  void smoothing_uniform(const Array_t& U, const Array_t& Uintermediate, const uint32_t nIterations, const uint32_t level);                 
  template< typename Array_t >  void smoothing_amr_finest(const Array_t& U, const Array_t& Uintermediate, const uint32_t nIterations, const uint32_t level);
   struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
