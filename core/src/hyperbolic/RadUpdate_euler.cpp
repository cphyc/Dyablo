#include "hyperbolic/policy/HyperbolicPolicy_Rad.h"

#include "hyperbolic/scheme/Hyperbolic_euler.h"

namespace dyablo{

class RadUpdate_euler 
  : public Hyperbolic_euler<HyperbolicPolicy_Rad>
{
public:
  using Hyperbolic_euler<HyperbolicPolicy_Rad>::Hyperbolic_euler;
};

} //namespace dyablo

FACTORY_REGISTER( dyablo::HyperbolicUpdateFactory, 
                  dyablo::RadUpdate_euler, 
                  "RadUpdate_euler_M1")