#pragma once

#include "InitialConditions_base.h" // IWYU pragma: export
#include "plugins_lib.h"

template<>
bool dyablo::InitialConditionsFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}
