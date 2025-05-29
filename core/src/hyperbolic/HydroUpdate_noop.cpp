#include "states/State_hydro.h"

#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"

#include "hyperbolic/scheme/Hyperbolic_noop.h"

namespace dyablo{

class HydroUpdate_noop 
  : public Hyperbolic_noop<HyperbolicPolicy_Hydro>
{
public:
  using Hyperbolic_noop<HyperbolicPolicy_Hydro>::Hyperbolic_noop;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_noop, 
                  "HydroUpdate_noop")