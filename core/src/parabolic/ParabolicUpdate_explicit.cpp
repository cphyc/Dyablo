#include "parabolic/ParabolicUpdate_base.h"

#include "mpi/GhostCommunicator.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "parabolic/ParabolicTerm_thermal_conduction.h"
#include "parabolic/ParabolicTerm_viscosity.h"

namespace dyablo {
namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using GlobalArray = ForeachCell::CellArray_global;
using PatchArray = ForeachCell::CellArray_patch;
using CellIndex = ForeachCell::CellIndex;

/**
 * @brief Computes the primitive variables at a given cell index in an array
 *        and stores it in another array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array passed to the method
 * @tparam CellIndex the type of index passed to the method
 * @param params the hydro parameters for the conversion
 * @param Ugroup The array where the conservative variables are stored
 * @param iCell_Ugroup the index where to look and where to store
 * @param Qgroup the array where the primitive variables are written
 * 
 * @note Ugroup and Qgroup should have the same sizes and properties
 */
template< 
  int ndim, 
  typename State,
  typename Array_t, 
  typename CellIndex >
KOKKOS_INLINE_FUNCTION
void compute_primitives(real_t gamma0,   const Array_t& Ugroup, 
                        const CellIndex& iCell_Ugroup, const Array_t& Qgroup)
{
  using ConsState = typename State::ConsState;
  using PrimState = typename State::PrimState;

  ConsState uLoc{};
  getConservativeState<ndim>(Ugroup, iCell_Ugroup, uLoc);
  PrimState qLoc = consToPrim<ndim>(uLoc, gamma0);
  setPrimitiveState<ndim>(Qgroup, iCell_Ugroup, qLoc);
}

}// namespace
}

#include "hyperbolic/CopyGhostBlockCellData.h"

namespace dyablo {
/**
 * @brief Kernel implementing an explicit update for a parabolic term
 * 
 * @param configMap The configuration map of the run
 * @param foreach_cell Traversal used for the calculation of the kernel
 * @param timers handles to time kernel application
 * @param term_type type of parabolic term being solved
 */
class ParabolicUpdate_explicit : public ParabolicUpdate {
public:
  ParabolicUpdate_explicit(
    ConfigMap&        configMap,
    ForeachCell&      foreach_cell,
    Timers&           timers, 
    ParabolicTermType term_type)
   : foreach_cell(foreach_cell),
     configMap(configMap),
     timers(timers),
     term_type(term_type),
     gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
     bc_manager(configMap) { 

     }

  void update( UserData& U, ScalarSimulationData& scalar_data) {
    real_t dt = scalar_data.get<real_t>("dt");
    int ndim = foreach_cell.getDim();
    if (ndim == 2)
      update_parabolic<2>(U, dt);
    else
      update_parabolic<3>(U, dt);
  }
  
  template <int ndim>
  void update_parabolic(UserData &U,
                        real_t dt) 
  {
    if (term_type == PARABOLIC_THERMAL_CONDUCTION)
      update_aux<ndim, ParabolicTerm_thermal_conduction>(U, dt);
    else if (term_type == PARABOLIC_VISCOSITY)
      update_aux<ndim, ParabolicTerm_viscosity>(U, dt);
    else
      throw std::runtime_error("Error, parabolic term not implemented");
  }

  template <
    int ndim, 
    typename ParabolicTerm>
  void update_aux(UserData &U, 
                  real_t dt) 
  {
    using PatchArray = ForeachCell::CellArray_patch;
    using State      = typename ParabolicTerm::State;
    using ConsState  = typename ParabolicTerm::ConsState;
    using PrimState  = typename ParabolicTerm::PrimState;


    // 1- Allocating temporary arrays
    // Note : Should that be 2, 2, 2 ? or 1, 1, 1 ? or parabolic-term-type dependent ?
    PatchArray::Ref Ugroup_ = foreach_cell.reserve_patch_tmp("Ugroup", 2, 2, (ndim == 3)?2:0, State::N);
    PatchArray::Ref Qgroup_ = foreach_cell.reserve_patch_tmp("Qgroup", 2, 2, (ndim == 3)?2:0, State::N);
    GhostedArray rhs = foreach_cell.allocate_ghosted_array("rhs", State::N);

    ParabolicTerm parabolic_term{configMap};

    std::string kernel_name = "Parabolic[explicit] " + named_enum<ParabolicTermType>::to_string(term_type);
    timers.get(kernel_name).start();

    real_t gamma0 = this->gamma0;
    BoundaryConditions bc_manager = this->bc_manager;

    auto fields_info = parabolic_term.getFieldsInfo();
    UserData::FieldAccessor Uin = U.getAccessor( fields_info );
    auto fields_info_next = fields_info;
    for( auto& p : fields_info_next )
      p.name += "_next";
    UserData::FieldAccessor Uout = U.getAccessor( fields_info_next );

    // Iteration over all the patches
    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
    foreach_cell.foreach_patch(kernel_name,
      PATCH_LAMBDA( const ForeachCell::Patch& patch) 
    {
      // Allocating temporary patchs
      PatchArray Ugroup = patch.allocate_tmp(Ugroup_);
      PatchArray Qgroup = patch.allocate_tmp(Qgroup_);

      // 1- Copy non ghosted array Uin into temporary ghosted Ugroup with two ghosts
      patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
      {
          copyGhostBlockCellData<ndim, State>(
          Uin, iCell_Ugroup, 
          cellmetadata, 
          bc_manager,
          Ugroup);
      });

      // 2- Convert to primitives
      patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
      { 
        compute_primitives<ndim, State>(gamma0, Ugroup, iCell_Ugroup, Qgroup);
      });

      // 3- Calculating rhs term and applying parabolic term
      patch.foreach_cell(Uout.getShape(), CELL_LAMBDA(const CellIndex& iCell_Uout)
      {
        ForeachCell::SearchMode_local search_local( ForeachCell::SearchMode_local::ASSERT );
        CellIndex iCell_Qgroup = Qgroup.getShape().convert_index(iCell_Uout, search_local);
        parabolic_term.template compute_rhs<ndim>(Uin, Ugroup, Qgroup, rhs, iCell_Uout, iCell_Qgroup, cellmetadata);
      });
    });

    // 4- Communicating rhs values and adding up everything
    const int ghost_count = 2;
    GhostCommunicator ghost_comm(foreach_cell.get_amr_mesh(), rhs.getShape(), ghost_count );
    ghost_comm.reduce_ghosts( rhs );

    // 5- Updating Uout
    foreach_cell.foreach_cell("Parabolic explicit update", Uout.getShape(), CELL_LAMBDA(const CellIndex& iCell_Uout) {
      ConsState u, rhs_val;
      getConservativeState<ndim>(Uout, iCell_Uout, u);
      getConservativeState<ndim>(rhs, iCell_Uout, rhs_val);
      setConservativeState<ndim>(Uout, iCell_Uout, u+dt*rhs_val); 
    });

    timers.get(kernel_name).stop();
  }

  // Attributes
  ForeachCell &foreach_cell;
  ConfigMap   &configMap;
  Timers      &timers;

  ParabolicTermType term_type;
  real_t gamma0;

  BoundaryConditions bc_manager;
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::ParabolicUpdateFactory, 
                 dyablo::ParabolicUpdate_explicit, 
                 "ParabolicUpdate_explicit")
