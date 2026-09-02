#pragma once

#include "DerivedFields_base.h" // IWYU pragma: export
#include "plugins_lib.h"

template<>
bool dyablo::DerivedFieldsFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}

