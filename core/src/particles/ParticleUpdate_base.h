#pragma once

#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "user_data/UserData.h"
#include "ScalarSimulationData.h"

namespace dyablo {


class ParticleUpdate{
public: 
  // ParticleUpdate(
  //               const ConfigMap& configMap,
  //               ForeachCell&& params,
  //               Timers& timers );
  virtual ~ParticleUpdate(){}
  virtual void update( UserData& U, ScalarSimulationData& scalar_data ) = 0;
};

using ParticleUpdateFactory = RegisteringFactory< ParticleUpdate, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
