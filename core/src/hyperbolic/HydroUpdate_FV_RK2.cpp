#include "states/State_hydro.h"

#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"

#include "hyperbolic/scheme/Hyperbolic_RK2.h"

namespace dyablo{

class HydroUpdate_FV_RK2
  : public Hyperbolic_RK2<HyperbolicPolicy_Hydro>
{
public:
  using Hyperbolic_RK2<HyperbolicPolicy_Hydro>::Hyperbolic_RK2;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_FV_RK2, 
                  "HydroUpdate_FV_RK2")