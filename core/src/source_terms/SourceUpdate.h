#pragma once

#include "source_terms/SourceUpdate_base.h"

namespace dyablo {

class SourceUpdate_Cooling_FF;
class SourceUpdate_GLM;
class SourceUpdate_Ionization_Bunny;
class SourceUpdate_Ionization_Chem;

} //namespace dyablo 


template<>
inline bool dyablo::SourceUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::SourceUpdate_Cooling_FF);
  DECLARE_REGISTERED(dyablo::SourceUpdate_GLM);
  DECLARE_REGISTERED(dyablo::SourceUpdate_Ionization_Bunny);
  DECLARE_REGISTERED(dyablo::SourceUpdate_Ionization_Chem);

  return true;
}

