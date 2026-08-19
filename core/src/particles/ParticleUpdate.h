#pragma once

#include "ParticleUpdate_base.h" // IWYU pragma: export
#include "plugins_lib.h"

template<>
inline bool dyablo::ParticleUpdateFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}
