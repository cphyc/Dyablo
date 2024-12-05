#include "HyperbolicUpdate_base.h"

namespace dyablo {


// Should we keep that name ?
class HydroUpdate_gravity : public HyperbolicUpdate {
private:
  ForeachCell& foreach_cell;  
  Timers& timers;
  int ndim;
  GravityType gravity_type;
  real_t gx, gy, gz; 

public:
  HydroUpdate_gravity(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    timers(timers),
    ndim(configMap.getValue<int>("mesh", "ndim", 3)),
    gravity_type(configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE)) 
  {
    if (gravity_type & GRAVITY_CONSTANT) {
      gx = configMap.getValue<real_t>("gravity", "gx", 0.0);
      gy = configMap.getValue<real_t>("gravity", "gy", 0.0);
      gz = configMap.getValue<real_t>("gravity", "gz", 0.0);
    } 
  }

  void update( UserData& U, ScalarSimulationData& scalar_data)
  {
    using FieldAccessor = UserData::FieldAccessor;
    using FieldInfo = UserData::FieldAccessor::FieldInfo;
    using CellIndex = ForeachCell::CellIndex;

    bool use_field = gravity_type&GRAVITY_FIELD;

    enum VarIndex_gravity{ Ie_tot, Irho, Irho_vx, Irho_vy, Irho_vz, IGX, IGY, IGZ };
    
    std::vector<FieldInfo> Uin_fields = {{"rho", Irho}};
    if(use_field)
    {
      Uin_fields.push_back({"gx" , IGX});
      Uin_fields.push_back({"gy" , IGY});
      if(ndim==3)
        Uin_fields.push_back({"gz" , IGZ});
    }   
    std::vector<FieldInfo> Uout_fields = {
      {"e_tot_next"   , Ie_tot}, 
      {"rho_next"   , Irho}, 
      {"rho_vx_next", Irho_vx}, 
      {"rho_vy_next", Irho_vy}
    };
    if(ndim==3)
      Uout_fields.push_back({"rho_vz_next", Irho_vz});

    FieldAccessor Uin = U.getAccessor( Uin_fields );
    FieldAccessor Uout = U.getAccessor( Uout_fields );

    real_t dt = scalar_data.get<real_t>("dt");
    int ndim = this->ndim;

    real_t gx_ = this->gx;
    real_t gy_ = this->gy;
    real_t gz_ = this->gz;
    foreach_cell.foreach_cell("HydroUpdate_gravity", Uin.getShape(), 
      KOKKOS_LAMBDA(const CellIndex& iCell)
    { 
      real_t gx = gx_;
      real_t gy = gy_;
      real_t gz = gz_;
      if(use_field)
      {
        gx = Uin.at(iCell, IGX);
        gy = Uin.at(iCell, IGY);
        if (ndim == 3)
          gz = Uin.at(iCell, IGZ);
      }

      real_t rhoOld = Uin.at(iCell, Irho);
      
      real_t rhoNew = Uout.at(iCell, Irho);
      real_t rhou = Uout.at(iCell, Irho_vx);
      real_t rhov = Uout.at(iCell, Irho_vy);
      real_t ekin_old = rhou*rhou + rhov*rhov;
      real_t rhow;
      
      if (ndim == 3) {
        rhow = Uout.at(iCell, Irho_vz);
        ekin_old += rhow*rhow;
      }
      
      ekin_old = 0.5 * ekin_old / rhoNew;

      rhou += 0.5 * dt * gx * (rhoOld + rhoNew);
      rhov += 0.5 * dt * gy * (rhoOld + rhoNew);

      Uout.at(iCell, Irho_vx) = rhou;
      Uout.at(iCell, Irho_vy) = rhov;
      if (ndim == 3) {
        rhow += 0.5 * dt * gz * (rhoOld + rhoNew);
        Uout.at(iCell, Irho_vz) = rhow;
      }

      // Energy correction should be included in case of self-gravitation ?
      real_t ekin_new = rhou*rhou + rhov*rhov;
      if (ndim == 3)
        ekin_new += rhow*rhow;
      
      ekin_new = 0.5 * ekin_new / rhoNew;
      Uout.at(iCell, Ie_tot) += (ekin_new - ekin_old);
    });
  }

};

} // namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_gravity, 
                  "HydroUpdate_gravity")
