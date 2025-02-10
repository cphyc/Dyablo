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
  real_t omegav;
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
    omegab(configMap.getValue<real_t>("cosmology","omegab", 0.049)),
    omegam(configMap.getValue<real_t>("cosmology","omegam", 0.3)),
    omegav(configMap.getValue<real_t>("cosmology","omegav", 0.7))
  {
    DYABLO_ASSERT_HOST_RELEASE(ndim == 3, "Initial conditions only for 3D");

    real_t across = configMap.getValue<real_t>("zeldovitch_pancake", "aCross", 0.1);
    real_t astart = configMap.getValue<real_t>("cosmology", "aStart", 0.01);
      
    real_t omegak = 1.0 - omegam - omegav;
    real_t etastart    = sqrt(omegam / astart + omegav * astart * astart + omegak);
    real_t etacross    = sqrt(omegam / across + omegav * across * across + omegak);

    real_t dladt = astart * etastart;

    real_t fomega;
    real_t dcross;
    real_t dplus;

    if (omegam >= 1.0 && omegav <= 0.0)
      fomega = 1.0;
    else
    {
      {
        auto ddplus = [&](real_t a)
        {
          if (a <= 0.0)
            return 0.0;

          real_t eta = sqrt(omegam / a + omegav * a * a + 1.0 - omegam - omegav);
          return 2.5 / (eta * eta * eta);
        };

        //------------------------------ DPLUS
        // UGLY trapezoid rule integration
        const real_t Np = 10000;
        const real_t da = astart / Np;
        real_t sum      = 0.0;
        for (int i = 0; i < Np; ++i)
        {
          sum += 0.5 * (ddplus(i * da) + ddplus((i + 1) * da));
        }
        sum *= da;

        dplus = etastart / astart * sum;

        //------------------------------ DCROSS
        const real_t db = across / Np;
        sum      = 0.0;
        for (int i = 0; i < Np; ++i)
        {
          sum += 0.5 * (ddplus(i * db) + ddplus((i + 1) * db));
        }
        sum *= db;

        dcross = etacross / across * sum;
      }
      fomega = (2.5 / dplus - 1.5 * omegam / astart - omegak) / (etastart * etastart);
    }
    this->fact = dplus / dcross;
    this->vmax = dplus / dcross * fomega * dladt / sqrt(omegam) / (M_PI); 
  
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
    res.rho = (this->omegab/this->omegam) / (1.0 + fact*cos( z * 2 * M_PI));
    res.p = this->smallp;

    res.u   = 0.0;
    res.v   = 0.0;
    res.w   = vmax * sin( z * 2 * M_PI );

    ConsHydroState cons_res = primToCons<3>(res, gamma0);
    return cons_res;
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_Zeldovitch_pancake>, 
                 "zeldovitch_pancake");

