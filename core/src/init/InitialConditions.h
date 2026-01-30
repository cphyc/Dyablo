#pragma once

#include "init/InitialConditions_base.h"

namespace dyablo{

class InitialConditions_uniform;

template< typename AnalyticalFormula >
class InitialConditions_analytical;

// Restart
class InitialConditions_restart;
class InitialConditions_tiled_restart;

// Hydrodynamics
class AnalyticalFormula_blast;
class AnalyticalFormula_implode;
class AnalyticalFormula_riemann2d;
class AnalyticalFormula_KelvinHelmholtz;
class AnalyticalFormula_RayleighTaylor;
class AnalyticalFormula_sod;
class AnalyticalFormula_Zeldovitch_pancake;
class InitialConditions_zeldovitch_particles;
class AnalyticalFormula_double_mach;
class AnalyticalFormula_disk;

// MHD
class AnalyticalFormula_OrszagTang;
class AnalyticalFormula_MHD_blast;
class AnalyticalFormula_MHD_rotor;
class AnalyticalFormula_MHD_RayleighTaylor;

// Particles
class InitialConditions_simple_particles;
class InitialConditions_simple_particles_from_file;
class InitialConditions_particle_grid;
class InitialConditions_particles_uniform;

// Cosmology
class InitialConditions_grafic_fields;

// Convection
class AnalyticalFormula_C91;
class AnalyticalFormula_tri_layer;


} // namespace dyablo



template<>
bool dyablo::InitialConditionsFactory::init()
{
  DECLARE_REGISTERED( dyablo::InitialConditions_uniform);

#ifdef DYABLO_USE_HDF5
  DECLARE_REGISTERED( dyablo::InitialConditions_restart );
  DECLARE_REGISTERED( dyablo::InitialConditions_tiled_restart );
#endif

  DECLARE_REGISTERED( dyablo::InitialConditions_simple_particles );
  DECLARE_REGISTERED( dyablo::InitialConditions_simple_particles_from_file );
  DECLARE_REGISTERED( dyablo::InitialConditions_particle_grid );
  DECLARE_REGISTERED( dyablo::InitialConditions_particles_uniform );
  
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_blast> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_implode> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_riemann2d> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_KelvinHelmholtz> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_RayleighTaylor> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_double_mach> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_disk> );

  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_OrszagTang > );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_blast> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_rotor> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_RayleighTaylor> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_sod> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_Zeldovitch_pancake> );
  DECLARE_REGISTERED( dyablo::InitialConditions_zeldovitch_particles );

  DECLARE_REGISTERED( dyablo::InitialConditions_grafic_fields );

  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_C91> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_tri_layer> );

  return true;
}
