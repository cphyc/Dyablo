#pragma once

#include "source_terms/SourceUpdate_base.h"

namespace dyablo {

class SourceUpdate_Cooling_FF;
class SourceUpdate_GLM;

class SourceUpdate_cooling_grackle_table;

} //namespace dyablo 


template<>
inline bool dyablo::SourceUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::SourceUpdate_Cooling_FF);
  DECLARE_REGISTERED(dyablo::SourceUpdate_GLM);

  DECLARE_REGISTERED(dyablo::SourceUpdate_cooling_grackle_table);

  return true;
}

