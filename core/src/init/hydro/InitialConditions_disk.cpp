#include "../InitialConditions_analytical.h"
#include "AnalyticalFormula_base_hydro.hpp"

namespace dyablo{

/**
 * Kelvin-Helmholtz instability test.
 * Based on Lecoanet et al "A validated non-linear kelvin-helmholtz benchmark for numerical 
 * hydrodynamics", 2016, Monthly Notices of the Royal Astronomy Society
 **/
struct AnalyticalFormula_disk : public AnalyticalFormula_base_hydro{

  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;

  // Physical parameters
  const real_t scale_length; // Scale length for the disk
  const real_t scale_height; // Scale height for the disk
  const real_t gas_mass;     // Total mass of the gas
  const real_t disk_temperature; // Initial temperature of the gas [T/µ, K]
  const real_t disk_metallicity;  // Initial metallicity of the gas
  const real_t halo_temperature; // Initial temperature of the halo gas [T/µ, K]
  const real_t halo_metallicity;  // Initial metallicity of the halo gas

  const std::string velocity_profile_file;
  const real_t r_max;
  const real_t z_max;

  real_t rho_0;

  const real_t pi = 3.14159265358979323846;

  Kokkos::View<real_t*> r_profile;
  Kokkos::View<real_t*> v_profile;

  real_t min_density;

  AnalyticalFormula_disk( ConfigMap& configMap ) :
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "epsilon_coarsen", 0.8)),
    
    scale_length( configMap.getValue<real_t>("disk", "scale_length") ),
    scale_height( configMap.getValue<real_t>("disk", "scale_height") ),
    gas_mass(     configMap.getValue<real_t>("disk", "gas_mass") ),
    disk_temperature( configMap.getValue<real_t>("disk", "disk_temperature") ),
    disk_metallicity( configMap.getValue<real_t>("disk", "disk_metallicity") ),
    halo_temperature( configMap.getValue<real_t>("disk", "halo_temperature") ),
    halo_metallicity( configMap.getValue<real_t>("disk", "halo_metallicity") ),
    velocity_profile_file(configMap.getValue<std::string>("disk", "velocity_profile_file")),
    r_max(        configMap.getValue<real_t>("disk", "r_max") ),
    z_max(        configMap.getValue<real_t>("disk", "z_max") )

  {
    DYABLO_ASSERT_HOST_RELEASE(ndim == 3, "Initial conditions only for 3D");

    // Read in velocity profile
    std::ifstream file( velocity_profile_file );
    DYABLO_ASSERT_HOST_RELEASE( file.is_open(),
      "Could not open velocity profile file: " << velocity_profile_file
    );
    std::string line;
    std::vector<real_t> r_vals;
    std::vector<real_t> v_vals;
    while( std::getline(file, line) )
    {
      std::istringstream iss(line);
      real_t r,v;
      iss >> r >> v;
      r_vals.push_back(r);
      v_vals.push_back(v);
    }
    file.close();

    // Copy to Kokkos views
    const size_t npoints = r_vals.size();
    r_profile = Kokkos::View<real_t*>("r_profile", npoints);
    v_profile = Kokkos::View<real_t*>("v_profile", npoints);

    auto r_profile_host = Kokkos::create_mirror_view( r_profile );
    auto v_profile_host = Kokkos::create_mirror_view( v_profile );

    for( size_t i=0; i<npoints; i++ ) {
      r_profile_host(i) = r_vals[i];
      v_profile_host(i) = v_vals[i];
    }
    Kokkos::deep_copy( r_profile, r_profile_host );
    Kokkos::deep_copy( v_profile, v_profile_host );

    {
      auto dens_units = 1e9 * Units::SOLAR_MASS() / Units::kpc().pow<3>();
      min_density = (1e-6 * Units::PROTON_MASS() / Units::cm3()).convert_to( dens_units );
    }

    rho_0 = gas_mass/(4.*pi*scale_height*scale_length*scale_length);
  }


  KOKKOS_INLINE_FUNCTION
  ConsHydroState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    ConsHydroState res;
    // rho(r,z)=rho_0*exp(-r/r_d)*exp(-abs(z)/z_d)
    real_t r = fmax( smallr, SQRT( x*x + y*y ) );
    real_t z_abs = FABS(z);

    real_t rho = fmax(
      rho_0 * std::exp( -r/scale_length ) * std::exp( -z_abs/scale_height ),
      min_density
    );


    // Find rotational velocity from profile
    real_t v = 0.0;
    {
      // Linear interpolation in profile
      size_t npoints = r_profile.extent(0);
      if( r <= r_profile(0) )
        v = v_profile(0);
      else if( r >= r_profile(npoints-1) )
        v = v_profile(npoints-1);
      else {
        for( size_t i=0; i<npoints-1; i++ ) {
          if( r >= r_profile(i) && r < r_profile(i+1) ) {
            real_t frac = (r - r_profile(i)) / (r_profile(i+1) - r_profile(i));
            v = v_profile(i) + frac * ( v_profile(i+1) - v_profile(i) );
            break;
          }
        }
      }
    }

    // Set average density + null velocity in the halo
    real_t p;
    if (r > r_max || z_abs > z_max) {
      rho = min_density;
      v = 0;
      p = (halo_temperature * Units::K() * Units::KBOLTZ() / Units::PROTON_MASS()).convert_to( (Units::km() / Units::s()).pow<2>()) * rho;
    } else {
      p = (disk_temperature * Units::K() * Units::KBOLTZ() / Units::PROTON_MASS()).convert_to( (Units::km() / Units::s()).pow<2>()) * rho;
    }

    // Compute vx, vy, vz
    real_t theta = atan2( y, x );
    res.rho   = rho;
    res.rho_u = -rho * v * sin(theta);
    res.rho_v = +rho * v * cos(theta);
    res.rho_w = 0.0;       // No vertical velocity
    res.e_tot = 0.5 * rho * v * v + p / (gamma0-1.0);
    return res;
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory,
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_disk>,
                 "disk");
