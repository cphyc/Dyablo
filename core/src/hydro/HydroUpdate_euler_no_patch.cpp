#include "HydroUpdate_base.h"
#include "RiemannSolvers.h"
#include "HydroUpdate_utils.h"

#include "boundary_conditions/BoundaryConditions.h"

namespace dyablo {
namespace{
using CellIndex     = ForeachCell::CellIndex;
using FieldAccessor = UserData::FieldAccessor;
using offset_t      = typename CellIndex::offset_t;

enum VarIndex_gravity {IGX, IGY, IGZ};

/**
 * Applies corrector step for gravity
 * @param Uin Initial values before update
 * @param iCell_Uin Position insides Uin/Uout (non ghosted)
 * @param dt time step
 * @param use_field Get gravity field from Uin
 * @param gx, gy, gz, scalar values when use_field == false
 * @param Uout Updated array after hydro without gravity
 **/
template<int ndim, typename State>
KOKKOS_INLINE_FUNCTION
void apply_gravity_correction( const FieldAccessor& Uin,
                               const FieldAccessor& Uin_g,
                               const CellIndex& iCell_Uin,
                               real_t dt,
                               bool use_field,
                               real_t gx, real_t gy, real_t gz,
                               const FieldAccessor& Uout ){
  if(use_field)
  {
    gx = Uin_g.at(iCell_Uin, IGX);
    gy = Uin_g.at(iCell_Uin, IGY);
    if (ndim == 3)
      gz = Uin_g.at(iCell_Uin, IGZ);
  }

  real_t rhoOld = Uin.at(iCell_Uin, State::Irho);
  
  real_t rhoNew = Uout.at(iCell_Uin, State::Irho);
  real_t rhou = Uout.at(iCell_Uin, State::Irho_vx);
  real_t rhov = Uout.at(iCell_Uin, State::Irho_vy);
  real_t ekin_old = rhou*rhou + rhov*rhov;
  real_t rhow;
  
  if (ndim == 3) {
    rhow = Uout.at(iCell_Uin, State::Irho_vz);
    ekin_old += rhow*rhow;
  }
  
  ekin_old = 0.5 * ekin_old / rhoNew;

  rhou += 0.5 * dt * gx * (rhoOld + rhoNew);
  rhov += 0.5 * dt * gy * (rhoOld + rhoNew);

  Uout.at(iCell_Uin, State::Irho_vx) = rhou;
  Uout.at(iCell_Uin, State::Irho_vy) = rhov;
  if (ndim == 3) {
    rhow += 0.5 * dt * gz * (rhoOld + rhoNew);
    Uout.at(iCell_Uin, State::Irho_vz) = rhow;
  }

  // Energy correction should be included in case of self-gravitation ?
  real_t ekin_new = rhou*rhou + rhov*rhov;
  if (ndim == 3)
    ekin_new += rhow*rhow;
  
  ekin_new = 0.5 * ekin_new / rhoNew;
  Uout.at(iCell_Uin, State::Ie_tot) += (ekin_new - ekin_old);
}

}// namespace
}// namespace dyablo

namespace dyablo {

/**
 * @brief Euler update algorithm
 * 
 * @tparam State the type of state to treat
 */
template<typename State_>
class HydroUpdate_euler_no_patch : public HydroUpdate {
public:
  using State     = State_;
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

private:
  struct ValueSlope {
    PrimState q;
    PrimState slope;
    real_t size;
  };

public:
  HydroUpdate_euler_no_patch(
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
      gx = configMap.getValue<real_t>("gravity", "gx", 0.0);
      gy = configMap.getValue<real_t>("gravity", "gy", 0.0);
      gz = configMap.getValue<real_t>("gravity", "gz", 0.0);
    } 

    DYABLO_ASSERT_HOST_RELEASE((std::is_same_v<PrimState, PrimHydroState> || !well_balanced), 
                               "Well balanced scheme not implemented for MHD solvers yet !");
  }

  ~HydroUpdate_euler_no_patch() {}

  /**
   * @brief Solves hydro for one step using the euler method
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
    else if (ndim == 3)
      update_aux<3>(U, dt);  
    else
      DYABLO_ASSERT_HOST_RELEASE(false, "invalid ndim = " << ndim);
  }

  template< 
    int ndim >
  void update_aux( UserData& U, real_t dt) 
  {

    const RiemannParams& params = this->params; 
    auto bc_manager = this->bc_manager;
    Timers& timers = this->timers; 
    ForeachCell& foreach_cell = this->foreach_cell;

    auto fields_info = ConsState::getFieldsInfo();
    FieldAccessor Uin = U.getAccessor( fields_info );
    //FieldAccessor Uin_g;
    //if( gravity_use_field )
    //  Uin_g = U.getAccessor( {{"gx",IGX}, {"gy",IGY}, {"gz",IGZ}} );
    auto fields_info_next = fields_info;
    for( auto& p : fields_info_next )
      p.name += "_next";
    FieldAccessor Uout = U.getAccessor( fields_info_next );
    
    timers.get("HydroUpdate_euler").start();

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    // Initializing output array
    foreach_cell.foreach_cell( "HydroUpdate_euler_no_patch::init",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell) {
        ConsState uC;
        getConservativeState<ndim>(Uin, iCell, uC);
        setConservativeState<ndim>(Uout, iCell, uC);
      });

    // Setting the ghosts to 0 to accumulate fluxes
    foreach_cell.foreach_ghost_cell( "HydroUpdate_euler_no_patch::resetting_ghosts",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell) {
        ConsState empty_state{};
        setConservativeState<ndim>(Uout, iCell, empty_state);
      });

    // Iterate over cells
    foreach_cell.foreach_cell( "HydroUpdate_euler_no_patch::update",
      Uout.getShape(),
      CELL_LAMBDA(const CellIndex &iCell)
    {
      // Returns a value at a position and its associated slope
      auto get_value_and_slope = [&](const CellIndex &iCell, ComponentIndex3D dir) {
        // Getting stencil
        offset_t off_m{}, off_p{};
        off_m[dir] = -1;
        off_p[dir] =  1;

        CellIndex iCell_m = iCell.getNeighbor_ghost(off_m, Uout.getShape());
        CellIndex iCell_p = iCell.getNeighbor_ghost(off_p, Uout.getShape());

        // Reading conservative states
        ConsState uL{}, uC{}, uR{};

        if (iCell.is_boundary())
          uC = bc_manager.template getBoundaryValue<ndim, State>(Uin, iCell, cellmetadata);
        else
          getConservativeState<ndim>(Uin, iCell, uC);

        // Getting left value
        int Ldiff = iCell_m.level_diff();
        if (iCell_m.is_boundary())
          uL = bc_manager.template getBoundaryValue<ndim, State>(Uin, iCell_m, cellmetadata);
        else if (Ldiff < 0) {
          foreach_smaller_neighbor<ndim>(iCell_m, off_m, Uin.getShape(),
            [&](const CellIndex& iCell_neigh) {
              ConsState uloc;
              getConservativeState<ndim>(Uin, iCell, uloc);
              uL += uloc;
            });
          uL *= (ndim == 2 ? 0.5 : 0.25);
        }
        else
          getConservativeState<ndim>(Uin, iCell_m, uL);

        // Getting right value
        int Rdiff = iCell_p.level_diff();
        if (iCell_p.is_boundary())
          uR = bc_manager.template getBoundaryValue<ndim, State>(Uin, iCell_p, cellmetadata);
        else if (Rdiff < 0) {
          foreach_smaller_neighbor<ndim>(iCell_p, off_p, Uin.getShape(),
            [&](const CellIndex& iCell_neigh) {
              ConsState uloc;
              getConservativeState<ndim>(Uin, iCell, uloc);
              uR += uloc;
            });
          uR *= (ndim == 2 ? 0.5 : 0.25);
        }
        else
          getConservativeState<ndim>(Uin, iCell_p, uR);

        // Converting to primitive states
        const PrimState qL = consToPrim<ndim>(uL, params.gamma0);
        const PrimState qC = consToPrim<ndim>(uC, params.gamma0);
        const PrimState qR = consToPrim<ndim>(uR, params.gamma0);      

        // Getting the length right and left
        const real_t sizes[] = {0.75, 1.0, 1.5};
        const real_t dL = sizes[Ldiff+1];
        const real_t dR = sizes[Rdiff+1];  

        // Computing minmod slope for the direction
        PrimState slope{};
        auto dqp = (qR - qC) / dR;
        auto dqm = (qC - qL) / dL;
        state_foreach_var([](real_t& res, real_t dvp, real_t dvm) {
          if (dvp * dvm <= 0.0)
            res = 0.0;
          else
            res = fabs(dvp) > fabs(dvm) ? dvm : dvp;
        }, slope, dqp, dqm);

        return ValueSlope{qC, 
                          slope, 
                          cellmetadata.getCellSize(iCell)[dir]};
      }; // get_value_and_slope

      // Adding a state to a view without race conditions
      auto atomic_add_state = [&](const CellIndex &iC, ConsState flux) {
        Kokkos::atomic_add(&Uout.at(iC, ConsState::Irho), flux.rho);
        Kokkos::atomic_add(&Uout.at(iC, ConsState::Ie_tot), flux.e_tot);
        Kokkos::atomic_add(&Uout.at(iC, ConsState::Irho_vx), flux.rho_u);
        Kokkos::atomic_add(&Uout.at(iC, ConsState::Irho_vy), flux.rho_v);
        if (ndim == 3)
          Kokkos::atomic_add(&Uout.at(iC, ConsState::Irho_vz), flux.rho_w);
      };


      auto process_dir = [&](const CellIndex &iCell, ComponentIndex3D dir) {
        // Getting stencil
        offset_t off_m{}, off_p{};
        off_m[dir] = -1;
        off_p[dir] =  1;

        const CellIndex iCell_m = iCell.getNeighbor_ghost(off_m, Uin.getShape());
        const CellIndex iCell_p = iCell.getNeighbor_ghost(off_p, Uin.getShape());

        // Getting centered value and slope
        ValueSlope qsC = get_value_and_slope(iCell, dir);

        constexpr real_t dim_fac = (ndim == 2 ? 0.5 : 0.25);

        // Left side
        int Ldiff = iCell_m.level_diff();
        if (Ldiff >= 0) {
          ValueSlope qsL = get_value_and_slope(iCell_m, dir);

          // Reconstructing
          PrimState qL = qsL.q + 0.5 * qsL.slope;
          PrimState qC = qsC.q - 0.5 * qsC.slope;

          //printf("On density slope : %lf %lf; %lf %lf; %lf %lf\n", qsL.q.rho, qsC.q.rho, qsL.slope.rho, qsC.slope.rho, qL.rho, qC.rho); 
          // Solving
          qL = swapComponents(qL, dir);
          qC = swapComponents(qC, dir);
          ConsState flux = riemann_hydro(qL, qC, params);
          flux = swapComponents(flux, dir);

          // And adding flux to output cell, and neighbor if bigger
          atomic_add_state(iCell, flux * dt / qsC.size);
          if (Ldiff == 1)
            atomic_add_state(iCell_m, flux * -dim_fac * dt / qsL.size);
        } // If smaller we skip

        // Right side
        int Rdiff = iCell_p.level_diff();
        if (Rdiff >= 0) {
          ValueSlope qsR = get_value_and_slope(iCell_p, dir);

          // Reconstructing
          PrimState qC = qsC.q + 0.5 * qsC.slope;
          PrimState qR = qsR.q - 0.5 * qsR.slope;

          // Solving
          qC = swapComponents(qC, dir);
          qR = swapComponents(qR, dir);
          ConsState flux = riemann_hydro(qC, qR, params);
          flux = swapComponents(flux, dir);

          // And adding flux to output cell, and neighbor if bigger
          atomic_add_state(iCell, flux * -1.0 * dt / qsC.size);

          // Adding our flux to the neighbor if it is bigger
          if (Rdiff == 1)
            atomic_add_state(iCell_p, flux * dim_fac * dt / qsR.size);
        } // If smaller we skip
      };


      // Actual work
      process_dir(iCell, IX);
      process_dir(iCell, IY);
      if (ndim == 3)
        process_dir(iCell, IZ);

    });

    // Reducing the ghosts to accumulate the flux in the data arrays 
    GhostCommunicator ghost_comm(std::shared_ptr<AMRmesh>(&foreach_cell.get_amr_mesh(), [](AMRmesh*){}));
    U.reduce_ghosts(ghost_comm);

    // Checking that we have not overshot in density and pressure
    clean_negative_primitive_values<ndim, State>(foreach_cell, Uout, params.gamma0, params.smallr, params.smallp);

    timers.get("HydroUpdate_euler").stop();
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

} // namespace dyablo

FACTORY_REGISTER( dyablo::HydroUpdateFactory, 
                  dyablo::HydroUpdate_euler_no_patch<dyablo::HydroState>, 
                  "HydroUpdate_euler_no_patch")

