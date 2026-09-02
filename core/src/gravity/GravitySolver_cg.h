#pragma once

#include "GravitySolver_base.h"

#include <memory>

class Timers;
class ConfigMap;

namespace dyablo {

/**
 * @brief Class solving the poisson equation for gravity using
 * a matrix-free implementation of the preconditionned-conjugate gradient 
 */
class GravitySolver_cg : public GravitySolver{
public: 
  GravitySolver_cg(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers );
  ~GravitySolver_cg();
  void update_gravity_field( UserData& U, ScalarSimulationData& scalar_data);

   struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
