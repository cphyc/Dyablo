#include "states/State_hydro.h"

#include "hydro/FiniteVolumePolicy_Hydro.h"

#include "hydro/FiniteVolume_RK2.h"

namespace dyablo{

class HydroUpdate_FV_RK2
  : public FiniteVolume_RK2<FiniteVolumePolicy_Hydro>
{
public:
  using FiniteVolume_RK2<FiniteVolumePolicy_Hydro>::FiniteVolume_RK2;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HydroUpdateFactory, 
                  dyablo::HydroUpdate_FV_RK2, 
                  "HydroUpdate_FV_RK2")