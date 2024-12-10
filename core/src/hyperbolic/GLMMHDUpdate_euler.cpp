#include "states/State_hydro.h"

#include "hyperbolic/policy/HyperbolicPolicy_GLMMHD.h"

#include "hyperbolic/scheme/Hyperbolic_euler.h"

namespace dyablo{

class GLMMHDUpdate_euler 
  : public Hyperbolic_euler<HyperbolicPolicy_GLMMHD>
{
public:
  using Hyperbolic_euler<HyperbolicPolicy_GLMMHD>::Hyperbolic_euler;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::GLMMHDUpdate_euler, 
                  "GLMMHDUpdate_euler")