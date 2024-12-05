#include "states/State_hydro.h"

#include "hydro/HyperbolicPolicy_Hydro.h"

#include "hydro/Hyperbolic_euler_nopatch.h"

namespace dyablo {

class HydroUpdate_FV_euler_nopatch 
  : public HyperbolicUpdate_euler_nopatch<HyperbolicPolicy_Hydro>
{
public:
  using HyperbolicUpdate_euler_nopatch<HyperbolicPolicy_Hydro>::HyperbolicUpdate_euler_nopatch;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::HydroUpdate_FV_euler_nopatch, 
                  "HydroUpdate_FV_euler_nopatch")