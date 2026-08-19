#pragma once

#include "MapUserData_base.h" // IWYU pragma: export
#include "plugins_lib.h"

template<>
inline bool dyablo::MapUserDataFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}