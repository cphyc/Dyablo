#include "states/State_hydro.h"

#include "hydro/FiniteVolumePolicy_legacy.h"
#include "hydro/FiniteVolumePolicy_Slope.h"

#include "hydro/FiniteVolume_euler.h"

namespace dyablo{

class HydroUpdate_FV_euler 
  : public FiniteVolume_euler<FiniteVolumePolicy_legacy<HydroState>>
{
public:
  using FiniteVolume_euler<FiniteVolumePolicy_legacy<HydroState>>::FiniteVolume_euler;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HydroUpdateFactory, 
                  dyablo::HydroUpdate_FV_euler, 
                  "HydroUpdate_FV_euler")