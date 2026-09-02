#pragma once

#include "Compute_dt_base.h" // IWYU pragma: export
#include "plugins_lib.h"

template<>
inline bool dyablo::Compute_dtFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}