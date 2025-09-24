#pragma once

#include "source_terms/SourceUpdate_base.h"
#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"
#include "hyperbolic/policy/HyperbolicPolicy_GLMMHD.h"

template<>
inline bool dyablo::SourceUpdateFactory::init()
{
  dyablo::load_dyablo_plugins_lib();
  return true;
}

