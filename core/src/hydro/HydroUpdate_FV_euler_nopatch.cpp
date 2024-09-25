#include "states/State_hydro.h"

#include "hydro/FiniteVolumePolicy_Hydro.h"

#include "hydro/FiniteVolume_euler_nopatch.h"

namespace dyablo {

class HydroUpdate_FV_euler_nopatch 
  : public HydroUpdate_euler_nopatch<FiniteVolumePolicy_Hydro>
{
public:
  using HydroUpdate_euler_nopatch<FiniteVolumePolicy_Hydro>::HydroUpdate_euler_nopatch;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HydroUpdateFactory, 
                  dyablo::HydroUpdate_FV_euler_nopatch, 
                  "HydroUpdate_FV_euler_nopatch")