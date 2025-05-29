#include "InitialConditions_analytical.h"

#include "AnalyticalFormula_tools.h"
#include "states/State_forward.h"
#include "utils/units/Units.h"

namespace dyablo{

struct AnalyticalFormula_T_rho_grid : public AnalyticalFormula_base{
     // blast problem parameters
    const int ndim;
    const real_t Tmin;
    const real_t Tmax;
    const real_t rhomin;
    const real_t rhomax;
    const real_t zmin;
    const real_t zmax;
    const real_t gamma0, smallr, smallc, smallp;
    const real_t scale_l;
    const real_t scale_d;
    const real_t scale_t;
    const real_t scale_v;
    
    AnalyticalFormula_T_rho_grid( ConfigMap& configMap ) :
        ndim( configMap.getValue<int>("mesh", "ndim", 3) ),
        // Length are scaled by quadrant width (0.5,0.5,0.5 is center of quadrant when blast_n* != 1)
        Tmin( configMap.getValue<real_t>("T_rho_grid", "Tmin") ),
        Tmax( configMap.getValue<real_t>("T_rho_grid", "Tmax") ),
        rhomin( configMap.getValue<real_t>("T_rho_grid", "rhomin") ),
        rhomax( configMap.getValue<real_t>("T_rho_grid", "rhomax") ),
        zmin( configMap.getValue<real_t>("T_rho_grid", "zmin") ),
        zmax( configMap.getValue<real_t>("T_rho_grid", "zmax") ),
        // Number of quadrants in each direction
        gamma0 ( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
        smallr ( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
        smallc ( configMap.getValue<real_t>("hydro","smallc", 1e-10) ),
        smallp ( smallc*smallc / gamma0 ),
        scale_l( configMap.getValue<real_t>("units", "length") ),
        scale_d( configMap.getValue<real_t>("units", "density") ),
        scale_t( configMap.getValue<real_t>("units", "time") ),
        scale_v( scale_l / scale_t)
    {}

    KOKKOS_INLINE_FUNCTION
    bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    {
        real_t gamma0 = this->gamma0;
        real_t smallr = this->smallr;
        real_t smallp = this->smallp;
        return false;
    }

    KOKKOS_INLINE_FUNCTION
    ConsHydroState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    {
        // Quadrant size
        real_t T = Tmin * pow(Tmax / Tmin, x);
        real_t rho = rhomin * pow(rhomax / rhomin, y);
        real_t redshift = zmin + (zmax - zmin) * z;
        real_t scale_T2 = Units::PROTON_MASS / Units::KBOLTZ * SQR(scale_v);

        real_t P = T / (gamma0 - 1) / scale_T2 * rho;

        PrimHydroState q;
        q.rho = rho;
        q.p = P;

        return dyablo::primToCons<3>(q, gamma0);
    } 
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_T_rho_grid>, 
                "T_rho_grid");