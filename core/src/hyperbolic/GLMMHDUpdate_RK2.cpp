#include "states/State_hydro.h"

#include "hyperbolic/policy/HyperbolicPolicy_GLMMHD.h"

#include "hyperbolic/scheme/Hyperbolic_RK2.h"

namespace dyablo{

class GLMMHDUpdate_RK2
  : public Hyperbolic_RK2<HyperbolicPolicy_GLMMHD>
{
public:
  using Hyperbolic_RK2<HyperbolicPolicy_GLMMHD>::Hyperbolic_RK2;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::GLMMHDUpdate_RK2, 
                  "GLMMHDUpdate_RK2")