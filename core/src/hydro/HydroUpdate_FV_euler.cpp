#include "states/State_hydro.h"

#include "hydro/HyperbolicPolicy_Hydro.h"

#include "hydro/Hyperbolic_euler.h"

namespace dyablo{

class HydroUpdate_FV_euler 
  : public Hyperbolic_euler<HyperbolicPolicy_Hydro>
{
public:
  using Hyperbolic_euler<HyperbolicPolicy_Hydro>::Hyperbolic_euler;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_FV_euler, 
                  "HydroUpdate_FV_euler")