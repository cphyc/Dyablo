#pragma once

#include "HyperbolicUpdate_base.h" // IWYU pragma: export

#include "plugins_lib.h"

template<>
inline bool dyablo::HyperbolicUpdateFactory::init()
{ 
  dyablo::load_dyablo_plugins_lib();
  return true;
}
