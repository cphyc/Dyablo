#pragma once

#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "user_data/UserData.h"
#include "ScalarSimulationData.h"

namespace dyablo {

class SourceUpdate{
public: 
  virtual ~SourceUpdate(){}
  
  virtual void update( UserData &U,
                       ScalarSimulationData& scalar_data) = 0;
};

using SourceUpdateFactory = RegisteringFactory< SourceUpdate, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
