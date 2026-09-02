#pragma once

#include "SourceUpdate_base.h" // IWYU pragma: export
#include "plugins_lib.h"

template<>
inline bool dyablo::SourceUpdateFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}

