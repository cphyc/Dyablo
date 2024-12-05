#include "states/State_hydro.h"

#include "hyperbolic/policy/HyperbolicPolicy_legacy.h"

#include "hyperbolic/scheme/Hyperbolic_euler.h"

namespace dyablo{

class HydroUpdate_FV_euler_legacy 
  : public Hyperbolic_euler<HyperbolicPolicy_legacy<HydroState>>
{
public:
  using Hyperbolic_euler<HyperbolicPolicy_legacy<HydroState>>::Hyperbolic_euler;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_FV_euler_legacy, 
                  "HydroUpdate_FV_euler_legacy")