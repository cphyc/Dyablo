#pragma once 

#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "user_data/UserData.h"

namespace dyablo{


class InitialConditions{
public:
  // InitialConditions(
  //     ConfigMap& configMap,
  //     ForeachCell& foreach_cell,  
  //     Timers& timers);
  virtual void init( UserData& U ) = 0;
  virtual ~InitialConditions(){}
};

using InitialConditionsFactory = RegisteringFactory<InitialConditions, 
  ConfigMap&,
  ForeachCell&, 
  Timers&>;


} // namespace dyablo