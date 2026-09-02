#pragma once

#include "IOManager_base.h" // IWYU pragma: export
#include "plugins_lib.h"

template<>
inline bool dyablo::IOManagerFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}