#include "InitialConditions_analytical.h"
#include "AnalyticalFormula_tools.h"

namespace dyablo{

struct AnalyticalFormula_Zeldovitch_pancake : public AnalyticalFormula_base{
  
  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;
  const real_t xmin, xmax;
  real_t omegab;
  real_t omegam;
  real_t vmax;
  real_t fact;

  AnalyticalFormula_Zeldovitch_pancake( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),
    xmin( configMap.getValue<real_t>("mesh","xmin", 0) ),
    xmax( configMap.getValue<real_t>("mesh","xmax", 1) ),
    omegab(configMap.getValue<real_t>("cosmology","omegab", 0.999)),
    omegam(configMap.getValue<real_t>("cosmology","omegam", 1.0))
  {
    DYABLO_ASSERT_HOST_RELEASE(ndim == 3, "Initial conditions only for 3D");

    real_t a_cross = configMap.getValue<real_t>("zeldovitch_pancake", "aCross", 0.1);
    real_t a_start = configMap.getValue<real_t>("cosmology", "aStart", 0.01);

    real_t zc = 1/a_cross - 1;
    real_t zi = 1/a_start - 1;
    this->vmax = (1/(M_PI)) * (1+zc)/std::pow(1+zi, 1.5);
    this->fact = (1+zc)/(1+zi);
  }


  KOKKOS_INLINE_FUNCTION
  bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const 
  {
    const real_t gamma0 = this->gamma0;
    const real_t smallr = this->smallr;
    const real_t smallp = this->smallp;
    const real_t error_max = this->error_max;
    return AnalyticalFormula_tools::auto_refine( *this, gamma0, smallr, smallp, error_max,
                                                  x, y, z, dx, dy, dz );
  }

  KOKKOS_INLINE_FUNCTION
  ConsHydroState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    PrimHydroState res {};

    res.rho = (this->omegab/this->omegam) * (1 - fact*cos(2*M_PI*x));  // equation 105 bryan et al  // ressayer la formule ramses
    //res.rho = (this->omegab/this->omegam) * (1 + fact*cos(2*M_PI*x)); //equation 28 ramses elle marche moins bien
    res.p = this->smallp;

    // 0 at edges and center of box,
    // velocity vmax towards center at 1/4, 3/4 box
    // https://arxiv.org/abs/astro-ph/0111367
    // u = -1/2pi * (1+zc)/(1+zi)^3/2 * sin( 2pi x )     (22)
    res.u   = vmax * sin( x * 2 * M_PI );
    res.v   = 0.0;
    res.w   = 0.0;

    ConsHydroState cons_res = primToCons<3>(res, gamma0);
    return cons_res; 
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_Zeldovitch_pancake>, 
                 "zeldovitch_pancake");

