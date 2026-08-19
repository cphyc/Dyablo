#pragma once

#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "user_data/UserData.h"
#include "ScalarSimulationData.h"
#include "enums.h"

namespace dyablo {

class ParabolicUpdate {
public: 
  virtual ~ParabolicUpdate(){}

  virtual void update(UserData &U,
                      ScalarSimulationData& scalar_data) = 0;
};

using ParabolicUpdateFactory = RegisteringFactory< ParabolicUpdate, 
  ConfigMap&,       /*configMap*/
  ForeachCell&,     /*foreach_cell*/
  Timers&,          /*timers*/
  ParabolicTermType /*term_type*/>;  

} //namespace dyablo 
