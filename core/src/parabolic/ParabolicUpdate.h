#pragma once

#include "ParabolicUpdate_base.h" // IWYU pragma: export
#include "plugins_lib.h"

template<>
inline bool dyablo::ParabolicUpdateFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}
