#include "InitialConditions_analytical.h"

#include "AnalyticalFormula_tools.h"
#include "states/State_forward.h"
#include "ionization/Ionization_utils.h"


namespace dyablo{

struct AnalyticalFormula_rad_blast : public AnalyticalFormula_base{
    using State = RadState;
    using PrimState = typename State::PrimState;
    using ConsState = typename State::ConsState;

     // Blast problem parameters
    const int ndim;
    const real_t error_max;
    const real_t xmin, xmax;
    const real_t ymin, ymax;
    const real_t zmin, zmax;    
    const real_t gamma0, smallr, smallc, smallp;
    const real_t clight_fraction;
    const int levelMin, bx;
    RadType rad_type;
    const real_t box_size;
    const real_t temperature, temperature_bb, gas_density, xe_start, zre_start;
    real_t rhostar, pstar; 
    
    AnalyticalFormula_rad_blast( ConfigMap& configMap ) :
        ndim( configMap.getValue<int>("mesh", "ndim", 3) ),
        error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),      
        xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ), xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
        ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ), ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
        zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ), zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
        gamma0 ( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
        smallr ( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
        smallc ( configMap.getValue<real_t>("hydro","smallc", 1e-10) ),
        smallp ( smallc*smallc / gamma0 ),
        clight_fraction(configMap.getValue<real_t>("cosmology", "clight_fraction", 0.1)),
        levelMin(configMap.getValue<real_t>("amr", "level_min", 6)),
        bx(configMap.getValue<real_t>("amr", "bx", 4)),
        rad_type(configMap.getValue<RadType>("ionization", "mode", STROMGREN)),
        box_size(configMap.getValue<real_t>("rad", "box_size", 6.6)),
        temperature(configMap.getValue<real_t>("rad", "temperature", 1e4)),
        temperature_bb(configMap.getValue<real_t>("ionization", "temp_black_body", 1e5)),
        gas_density(configMap.getValue<real_t>("ionization", "gas_density", 1000.0)),
        xe_start(configMap.getValue<real_t>("rad", "xe_start", 1.2e-3)),
        zre_start(configMap.getValue<real_t>("ionization", "zre_start", -1000.0))
    {

        using namespace Units;

        real_t omegam = 0.0;
        real_t rstar = box_size * (Kilo * parsec) ; // box size in m 
        real_t dx = rstar/ (pow(2, levelMin) * bx); // Cell size (m)
        real_t tstar = 1.0 * Mega * 365.0*24.0*3600.0; // sec
        real_t vstar = rstar/tstar; //m/s
        real_t ctilde = clight_fraction * SPEEDOFLIGHT / vstar;
        rhostar = gas_density * PROTON_MASS; // 1000 atomes/m3 par défaut
        pstar = rhostar * vstar * vstar;

        // Compute sigma_n, sigma_e and typical energy
        auto s = computeSigma(this->temperature_bb);

        configMap.getValue<real_t>( "cosmology", "omegam", omegam );
        configMap.getValue<real_t>( "cosmology", "dx", dx );
        configMap.getValue<real_t>( "cosmology", "rhostar", rhostar );
        configMap.getValue<real_t>( "cosmology", "tstar", tstar );
        configMap.getValue<real_t>( "cosmology", "vstar", vstar );
        configMap.getValue<real_t>( "cosmology", "astart", 1.0 );
        configMap.getValue<real_t>( "cosmology", "ctilde", ctilde);

        configMap.getValue<real_t>( "ionization", "sigma_n_c", s.sn * clight_fraction * SPEEDOFLIGHT );
        configMap.getValue<real_t>( "ionization", "sigma_e_c", s.se * clight_fraction * SPEEDOFLIGHT );
        configMap.getValue<real_t>( "ionization", "typical_energy", s.etyp);
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
    ConsState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    {        
        using namespace Units;

        ConsState res{};

        real_t temp = this->temperature * Kelvin;

        res.fx_rad = 0.0;
        res.fy_rad = 0.0;
        res.fz_rad = 0.0;

        res.xe = this->xe_start;
        res.zre = this->zre_start;   

        if(rad_type==BUNNY){
            
            res.e_rad = 7e-3; // dummy value
            res.rho = 13.0;
            res.e_tot = 17.0/(gamma0-1.0);           
            res.temp = temp;
        }

        // Stromgren and Iliev tests 2/5
        else if(rad_type==STROMGREN){

            res.e_rad = 0.0;
            res.rho_u = 0.0;
            res.rho_v = 0.0;
            res.rho_w = 0.0;
            res.rho = gas_density * PROTON_MASS / rhostar;

            real_t p_0 = (gamma0 - 1.0) * 1.5 * gas_density * KBOLTZ * temp / pstar;
            res.e_tot = p_0/(gamma0-1.0);

            res.temp = temp;
        }

        // Iliev test 3/7
        else if(rad_type==SHADOW){

            res.e_rad = 0.0;
            res.rho_u = 0.0;
            res.rho_v = 0.0;
            res.rho_w = 0.0;

            real_t radius = 0.8/box_size; // 0.8 kpc par rapport à 6.6 kpc. On veut une valeur entre 0 et 1
            real_t pos_x = 5.0/box_size;   // 5kpc par rapport à 6.6kpc. On veut une valeur entre 0 et 1
            real_t pos_y = 0.5;  // au milieu
            real_t pos_z = 0.5;  // au milieu

            real_t r2 = (x-pos_x)*(x-pos_x) + (y-pos_y)*(y-pos_y);
            if( this->ndim == 3 ) r2 += (z-pos_z)*(z-pos_z);

            real_t factor = 1.0;
            if (r2 < radius*radius) {
                factor = 200.0;
            }

            res.temp = temp / factor;
            res.rho = gas_density * factor * PROTON_MASS / rhostar;

            real_t p_0 = (gamma0 - 1.0) * 1.5 * gas_density * KBOLTZ * temp / pstar;
            res.e_tot = p_0/(gamma0-1.0);
        } 

        return res;
    }
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_rad_blast>, 
                 "rad_blast");
