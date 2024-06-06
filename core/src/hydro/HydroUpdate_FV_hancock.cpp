#include "states/State_hydro.h"

#include "hydro/FiniteVolumePolicy_Hydro.h"

#include "hydro/FiniteVolume_hancock.h"

namespace dyablo{

class HydroUpdate_FV_hancock 
  : public FiniteVolume_hancock<FiniteVolumePolicy_Hydro>
{
public:
  using FiniteVolume_hancock<FiniteVolumePolicy_Hydro>::FiniteVolume_hancock;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HydroUpdateFactory, 
                  dyablo::HydroUpdate_FV_hancock, 
                  "HydroUpdate_FV_hancock")