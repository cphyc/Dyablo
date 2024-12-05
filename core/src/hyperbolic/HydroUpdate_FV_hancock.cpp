#include "states/State_hydro.h"

#include "hyperbolic/HyperbolicPolicy_Hydro.h"

#include "hyperbolic/Hyperbolic_hancock.h"

namespace dyablo{

class HydroUpdate_FV_hancock 
  : public Hyperbolic_hancock<HyperbolicPolicy_Hydro>
{
public:
  using Hyperbolic_hancock<HyperbolicPolicy_Hydro>::Hyperbolic_hancock;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_FV_hancock, 
                  "HydroUpdate_FV_hancock")