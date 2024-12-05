#include "HyperbolicUpdate_base.h"
#include "RiemannSolvers.h"
#include "HyperbolicUpdate_utils.h"

#include "boundary_conditions/BoundaryConditions.h"
#include "mpi/GhostCommunicator.h"

namespace dyablo {
namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using GlobalArray = ForeachCell::CellArray_global;
using PatchArray = ForeachCell::CellArray_patch;
using CellIndex = ForeachCell::CellIndex;
using FieldArray = UserData::FieldAccessor;

}// namespace
}// namespace dyablo

#include "hydro/CopyGhostBlockCellData.h"

namespace dyablo {


template<typename State_>
class HyperbolicUpdate_RK2: public HyperbolicUpdate {
public:
  using State = State_;
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  HyperbolicUpdate_RK2(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    timers(timers),
    params(configMap),
    bc_manager(configMap),
    gravity_type(configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE)),
    well_balanced(configMap.getValue<bool>("hydro", "well_balanced", false))
  {
    if (gravity_type & GRAVITY_CONSTANT) {
      gx = configMap.getValue<real_t>("gravity", "gx",  0.0);
      gy = configMap.getValue<real_t>("gravity", "gy",  0.0);
      gz = configMap.getValue<real_t>("gravity", "gz",  0.0);
    } 

    DYABLO_ASSERT_HOST_RELEASE((std::is_same_v<PrimState, PrimHydroState> || !well_balanced), 
                               "Well balanced scheme not implemented for MHD solvers yet !");
  }

  /**
   * @brief Solves hydro for one step using the RK2 method
   * 
   * @param Uin the input global array
   * @param Uout the output global array
   * @param dt the timestep
   */
  void update( UserData& U, ScalarSimulationData& scalar_data) 
  {
    real_t dt = scalar_data.get<real_t>("dt");
    uint32_t ndim = foreach_cell.getDim();
    if (ndim == 2)
      update_aux<2>(U, dt);
    else if(ndim == 3)
      update_aux<3>(U, dt);  
    else
       DYABLO_ASSERT_HOST_RELEASE(false, "invalid ndim = " << ndim);
  }

  template<int ndim>
  void update_aux(UserData& U, real_t dt)
  {
    Timers& timers = this->timers; 
    ForeachCell& foreach_cell = this->foreach_cell;
    int nb_ghosts = 2;
    GhostCommunicator ghost_comm(foreach_cell.get_amr_mesh(), U.getShape(), nb_ghosts );
      
    auto fields_info = ConsState::getFieldsInfo();
    UserData::FieldAccessor Uin = U.getAccessor( fields_info );
    auto fields_info_next = fields_info;
    for( auto& p : fields_info_next )
      p.name += "_next";
    UserData::FieldAccessor Uout = U.getAccessor( fields_info_next );

    auto fm_cons = ConsState::getFieldManager();

    // Temporary Ustar array
    auto Ustar = foreach_cell.allocate_ghosted_array("U*", fm_cons);
    
    // Two update stages and one correction stage
    timers.get("HyperbolicUpdate_RK2").start();
    rk_update<ndim>(Uin, Ustar, dt);
    ghost_comm.exchange_ghosts(Ustar);
    rk_update<ndim>(Ustar, Uout, dt);
    rk_correct<ndim>(Uin, Uout);

    timers.get("HyperbolicUpdate_RK2").stop();
  }



  template<int ndim, typename Array_in_t, typename Array_out_t>
  void rk_update(const Array_in_t& Uin, 
                 const Array_out_t& Uout, 
                 real_t dt) 
  {
    auto foreach_cell = this->foreach_cell;
    auto gravity_type = this->gravity_type;
    auto params = this->params;

    auto fm_prim = PrimState::getFieldManager().get_id2index();
    auto fm_cons = ConsState::getFieldManager().get_id2index();
    
    // Create abstract temporary ghosted arrays for patches 
    PatchArray::Ref Ugroup_ = foreach_cell.reserve_patch_tmp("Ugroup", 2, 2, (ndim == 3)?2:0, fm_cons, State::N);
    PatchArray::Ref Qgroup_ = foreach_cell.reserve_patch_tmp("Qgroup", 2, 2, (ndim == 3)?2:0, fm_prim, State::N);
    
    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    auto bc_manager = this->bc_manager;

    bool has_gravity = gravity_type!=GRAVITY_NONE;
    bool gravity_use_field = gravity_type&GRAVITY_FIELD;
    [[maybe_unused]] bool gravity_use_scalar = gravity_type==GRAVITY_CST_SCALAR;
    real_t gx = this->gx, gy = this->gy, gz = this->gz;

    GravityInfo ginfo {gx, gy, gz, 
                       has_gravity && gravity_use_scalar,
                       this->well_balanced};

    DYABLO_ASSERT_HOST_RELEASE( !has_gravity || ( gravity_use_field != gravity_use_scalar ) ,
      "If gravity is on it must either use the force field from U or a constant scalar force field"  );

    // Iterate over patches
    foreach_cell.foreach_patch( "HyperbolicUpdate_RK2::update",
      PATCH_LAMBDA( const ForeachCell::Patch& patch )
    {
      PatchArray Ugroup = patch.allocate_tmp(Ugroup_);
      PatchArray Qgroup = patch.allocate_tmp(Qgroup_);

      // Copy non ghosted array Uin into temporary ghosted Ugroup with two ghosts
      patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
      {
          copyGhostBlockCellData<ndim, State>(
          Uin, iCell_Ugroup, 
          cellmetadata, 
          bc_manager,
          Ugroup);
      });

      patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
      { 
        compute_primitives<ndim, State>(params, Ugroup, iCell_Ugroup, Qgroup);
      });
      
      patch.foreach_cell( Uout.getShape(), CELL_LAMBDA(const CellIndex& iCell_Uout)
      {
        auto size = cellmetadata.getCellSize(iCell_Uout);
        typename State::ConsState u0{};
        getConservativeState<ndim>(Uin, iCell_Uout, u0);
        setConservativeState<ndim>(Uout, iCell_Uout, u0);
        euler_update<ndim, State>(params, IX, iCell_Uout, Uin, Qgroup, dt, size[IX], bc_manager, ginfo, Uout);
        euler_update<ndim, State>(params, IY, iCell_Uout, Uin, Qgroup, dt, size[IY], bc_manager, ginfo, Uout);
        if(ndim==3) 
          euler_update<ndim, State>(params, IZ, iCell_Uout, Uin, Qgroup, dt, size[IZ], bc_manager, ginfo, Uout);
      });
    });

    clean_negative_primitive_values<ndim, State>(foreach_cell, Uout, params.gamma0, params.smallr, params.smallp);

  }

  template<int ndim>
  void rk_correct(const FieldArray& Uin, 
                  const FieldArray& Uout) 
  {
    foreach_cell.foreach_patch("HyperbolicUpdate_RK2::correct",
      PATCH_LAMBDA( const ForeachCell::Patch& patch )
    {
      patch.foreach_cell( Uout.getShape(), CELL_LAMBDA(const CellIndex &iCell_Uout) {
        auto iCell_Uin = Uin.getShape().convert_index(iCell_Uout);
        
        ConsState uin{}, uout{};
        getConservativeState<ndim>(Uin,  iCell_Uin,  uin);
        getConservativeState<ndim>(Uout, iCell_Uout, uout);
        const ConsState res = 0.5 * (uin + uout);
        setConservativeState<ndim>( Uout, iCell_Uout, res);
      });
    });
  }

private:
  ForeachCell& foreach_cell;

  Timers& timers;  

  RiemannParams params;
  BoundaryConditions bc_manager;
  
  GravityType gravity_type;
  real_t gx, gy, gz;
  bool well_balanced;
};

}


FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HyperbolicUpdate_RK2<dyablo::HydroState>, 
                  "HydroUpdate_RK2")

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HyperbolicUpdate_RK2<dyablo::MHDState>, 
                  "MHDUpdate_RK2")

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HyperbolicUpdate_RK2<dyablo::GLMMHDState>, 
                  "GLMMHDUpdate_RK2")

