#pragma once

#include "HyperbolicUpdate_base.h"
#include "states/State_forward.h"

namespace dyablo {


class HydroUpdate_legacy;

template<typename State> class HyperbolicUpdate_hancock_oneneighbor;
template<typename State> class HyperbolicUpdate_hancock;
template<typename State> class HyperbolicUpdate_euler;
template<typename State> class HyperbolicUpdate_RK2;


class HydroUpdate_FV_euler;
class HydroUpdate_FV_euler_nopatch;
class HydroUpdate_FV_euler_legacy;
class HydroUpdate_FV_RK2;
class HydroUpdate_FV_hancock;
class HydroUpdate_gravity;

} //namespace dyablo 


template<>
inline bool dyablo::HyperbolicUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::HydroUpdate_legacy);
  
  DECLARE_REGISTERED(dyablo::HyperbolicUpdate_hancock_oneneighbor<dyablo::HydroState>);
  DECLARE_REGISTERED(dyablo::HyperbolicUpdate_hancock_oneneighbor<dyablo::MHDState>);
  DECLARE_REGISTERED(dyablo::HyperbolicUpdate_hancock<dyablo::HydroState>);
  DECLARE_REGISTERED(dyablo::HyperbolicUpdate_hancock<dyablo::MHDState>);
  DECLARE_REGISTERED(dyablo::HyperbolicUpdate_euler<dyablo::HydroState>);
  DECLARE_REGISTERED(dyablo::HyperbolicUpdate_euler<dyablo::MHDState>);
  DECLARE_REGISTERED(dyablo::HyperbolicUpdate_euler<dyablo::GLMMHDState>);
  DECLARE_REGISTERED(dyablo::HyperbolicUpdate_RK2<dyablo::HydroState>);
  DECLARE_REGISTERED(dyablo::HyperbolicUpdate_RK2<dyablo::MHDState>);
  DECLARE_REGISTERED(dyablo::HyperbolicUpdate_RK2<dyablo::GLMMHDState>);
  DECLARE_REGISTERED(dyablo::HydroUpdate_FV_euler);
  DECLARE_REGISTERED(dyablo::HydroUpdate_FV_RK2);
  DECLARE_REGISTERED(dyablo::HydroUpdate_FV_euler_nopatch);
  DECLARE_REGISTERED(dyablo::HydroUpdate_FV_euler_legacy);
  DECLARE_REGISTERED(dyablo::HydroUpdate_FV_hancock);
  DECLARE_REGISTERED(dyablo::HydroUpdate_gravity);

  return true;
}
