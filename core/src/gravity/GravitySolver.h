#pragma once

#include "GravitySolver_base.h" // IWYU pragma: export
#include "plugins_lib.h"

template<>
inline bool dyablo::GravitySolverFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}

