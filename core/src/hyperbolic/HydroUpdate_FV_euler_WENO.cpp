#include "states/State_hydro.h"

#include "hyperbolic/HyperbolicPolicy_Hydro.h"

#include "hyperbolic/Hyperbolic_euler_WENO.h"

namespace dyablo{

class HydroUpdate_FV_euler_WENO
  : public Hyperbolic_euler_WENO<HyperbolicPolicy_Hydro>
{
public:
  using Hyperbolic_euler_WENO<HyperbolicPolicy_Hydro>::Hyperbolic_euler_WENO;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_FV_euler_WENO, 
                  "HydroUpdate_FV_euler_WENO")