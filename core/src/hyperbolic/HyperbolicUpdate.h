#pragma once

#include "scheme/HyperbolicUpdate_base.h"
#include "states/State_forward.h"

namespace dyablo {

template<typename State> class HyperbolicUpdate_hancock_oneneighbor;

class HydroUpdate_euler;
class HydroUpdate_euler_nopatch;
class HydroUpdate_RK2;
class HydroUpdate_hancock;
class HydroUpdate_gravity;

} //namespace dyablo 


template<>
inline bool dyablo::HyperbolicUpdateFactory::init()
{ 
  DECLARE_REGISTERED(dyablo::HyperbolicUpdate_hancock_oneneighbor<dyablo::HydroState>);
  DECLARE_REGISTERED(dyablo::HydroUpdate_euler);
  DECLARE_REGISTERED(dyablo::HydroUpdate_RK2);
  DECLARE_REGISTERED(dyablo::HydroUpdate_euler_nopatch);
  DECLARE_REGISTERED(dyablo::HydroUpdate_hancock);
  DECLARE_REGISTERED(dyablo::HydroUpdate_gravity);

  return true;
}
