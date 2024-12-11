#pragma once

#include "source_terms/SourceUpdate_base.h"

namespace dyablo {

class SourceUpdate_Cooling_FF;

} //namespace dyablo 


template<>
inline bool dyablo::SourceUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::SourceUpdate_Cooling_FF);

  return true;
}

