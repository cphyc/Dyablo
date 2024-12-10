#include "states/State_hydro.h"

#include "hyperbolic/policy/HyperbolicPolicy_GLMMHD.h"

#include "hyperbolic/scheme/Hyperbolic_hancock.h"

namespace dyablo{

class GLMMHDUpdate_hancock 
  : public Hyperbolic_hancock<HyperbolicPolicy_GLMMHD>
{
public:
  using Hyperbolic_hancock<HyperbolicPolicy_GLMMHD>::Hyperbolic_hancock;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::GLMMHDUpdate_hancock, 
                  "GLMMHDUpdate_hancock")