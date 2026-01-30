#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

#include "utils/units/Units.h"
#include "Cosmo.h"


namespace dyablo{

/**
 * Create particles along a cartesian grid, initial particle velocities are set using 
 * the rho_vx, rho_vy, rho_vz fields
 * Configured in .ini in the [particle_grid] section
 * - nx, ny, nz, the number of particles in each direction
 * - total_mass the cumulated mass of all the particles, each particle has the same mass
 * - dt_perturb : dt used to displace particles (using particle velocities)
 * 
**/
class InitialConditions_zeldovitch_particles : public InitialConditions{
    ForeachCell& foreach_cell;
    ForeachParticle foreach_particle;
    uint32_t nx, ny, nz;
    const real_t xmin, xmax;
    const real_t ymin, ymax;
    const real_t zmin, zmax;
    real_t total_mass;
    real_t dt_perturb;
    real_t vfact;
    std::string particle_array_name;

public:
  InitialConditions_zeldovitch_particles(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : foreach_cell(foreach_cell),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ), xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
    ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ), ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
    zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ), zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
    
    particle_array_name(configMap.getValue<std::string>("particle_grid", "particle_array_name", "particles"))
  {
    AMRmesh& pmesh = foreach_cell.get_amr_mesh();
    uint32_t default_nx = foreach_cell.blockSize()[IX]*pmesh.get_coarse_grid_size()[IX];
    uint32_t default_ny = foreach_cell.blockSize()[IY]*pmesh.get_coarse_grid_size()[IY];
    uint32_t default_nz = foreach_cell.blockSize()[IZ]*pmesh.get_coarse_grid_size()[IZ];
    this->nx = configMap.getValue<uint32_t>("particle_grid", "nx", default_nx);
    this->ny = configMap.getValue<uint32_t>("particle_grid", "ny", default_ny);
    this->nz = configMap.getValue<uint32_t>("particle_grid", "nz", default_nz);
    
    real_t default_total_mass=1.0, default_dt_perturb=0.0, default_vfact=1.0;
    bool cosmo = configMap.hasValue("cosmology", "astart");
    if( cosmo )
    {
      real_t astart = configMap.getValue<real_t>( "cosmology", "astart" );
      real_t omegam = configMap.getValue<real_t>( "cosmology", "omegam" );
      real_t omegab = configMap.getValue<real_t>( "cosmology", "omegab" );
      real_t omegav = configMap.getValue<real_t>( "cosmology", "omegav" );
      
      auto eta = [&omegam, &omegav](real_t a)
      {
        return sqrt(omegam/a+omegav*a*a+1.0-omegam-omegav);
      };

      auto dplus = [&eta]( real_t a )
      {
        auto ddplus = [&eta]( real_t a )
        {
          if( a==0 ) return 0.0;
          return 2.5 / (eta(a)*eta(a)*eta(a));
        };
        constexpr int max_iter = 20;
        real_t precision = 1e-10;
        real_t romberg_res = Impl::romberg<max_iter>(ddplus, 0, a, precision );
        return eta(a)/a*romberg_res;
      };

      auto dladt = [&eta]( real_t a )
      {
        return a*eta(a);
      };

      auto fomega = [&]( real_t a )
      {
        if (omegam==1.0 && omegav==0.0)
          return 1.0;
        
        real_t omegak = 1-omegam-omegav;
        return (2.5 / dplus(a) - 1.5 * omegam / a - omegak ) / (eta(a)*eta(a));
      };
      
      using Inv_Time = decltype( 1/Units::s() );
      real_t H0 = configMap.getValue_in_code_unit<Inv_Time>("cosmology", "H0");
      real_t four_pi_G = configMap.getValue<real_t>( "gravity", "4_Pi_G" );

      real_t rhoc = 3.0 * H0 * H0 /( 2 * four_pi_G);
      real_t Vbox = (xmax-xmin) * (ymax-ymin) * (zmax -zmin);

      default_total_mass = ((omegam-omegab) * rhoc * Vbox);
      default_dt_perturb = 1/ (fomega(astart)*dladt(astart))/ H0;

      real_t across = configMap.getValue<real_t>("zeldovitch_pancake", "aCross");

      real_t dplus_ratio = dplus(astart) / dplus(across);
      default_vfact = (dplus_ratio * (xmax-xmin) / (2*M_PI) * fomega(astart)*dladt(astart) * H0);
    }

    this->total_mass = configMap.getValue<real_t>("particle_grid", "total_mass", default_total_mass);
    this->dt_perturb = configMap.getValue<real_t>("particle_grid", "dt_perturb", default_dt_perturb);
    this->vfact = configMap.getValue<real_t>("zeldovitch_pancake", "vfact", default_vfact);
  }

  void init( UserData& U )
  {
    int mpi_size = GlobalMpiSession::get_comm_world().MPI_Comm_size();
    int mpi_rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();

    uint32_t nx = this->nx;
    uint32_t ny = this->ny;
    uint32_t nz = this->nz;
    uint64_t nbpart_global = nx*ny*nz;
    uint64_t ipart_global_begin = nbpart_global*mpi_rank/mpi_size;
    uint64_t ipart_global_end = nbpart_global*(mpi_rank+1)/mpi_size;
    uint32_t nbpart_local = ipart_global_end-ipart_global_begin;

    U.new_ParticleArray(particle_array_name, nbpart_local);

    real_t Lx = this->xmax-this->xmin;
    real_t Ly = this->ymax-this->ymin;
    real_t Lz = this->zmax-this->zmin;
    real_t dx = (Lx)/this->nx;
    real_t dy = (Ly)/this->ny;
    real_t dz = (Lz)/this->nz;
    real_t xmin = this->xmin;
    real_t ymin = this->ymin;
    real_t zmin = this->zmin;
    real_t vfact = this->vfact;

    {
      UserData::ParticleArray_t P = U.getParticleArray( particle_array_name );
      foreach_particle.foreach_particle("InitialConditions_zeldovitch_particles::init_pos", P,
      KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart) 
      {
        uint64_t ipart_global = ipart_global_begin + iPart;

        // TODO : maybe add particles in morton order?
        uint32_t i = ipart_global%nx;
        uint32_t j = (ipart_global/nx)%ny;
        uint32_t k = (ipart_global/nx)/ny;

        P.pos(iPart, IX) = xmin + i*dx + 0.5*dx;
        P.pos(iPart, IY) = ymin + j*dy + 0.5*dy;
        P.pos(iPart, IZ) = zmin + k*dz + 0.5*dz;
      });
    }

    U.distributeParticles(particle_array_name);

    real_t mass = this->total_mass / nbpart_global;
    real_t dt_perturb = this->dt_perturb;

    U.new_ParticleAttribute(particle_array_name, "vx");
    U.new_ParticleAttribute(particle_array_name, "vy");
    U.new_ParticleAttribute(particle_array_name, "vz");
    U.new_ParticleAttribute(particle_array_name, "mass");
    
    enum VarIndex_particle_grid { IVX, IVY, IVZ, IMASS };
      

    {
      UserData::ParticleArray_t P = U.getParticleArray( particle_array_name );
      auto Pout = U.getParticleAccessor( particle_array_name,
                                        { {"vx", IVX}, 
                                          {"vy", IVY},
                                          {"vz", IVZ},
                                          {"mass", IMASS} });

      ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();


      
      foreach_particle.foreach_particle("InitialConditions_zeldovitch_particles::init_attributes", P,
      KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart) 
      {
        Pout.at( iPart, IMASS ) = mass;
        
        real_t x = P.pos( iPart, IX );

        Pout.at( iPart, IVX ) = vfact * sin(2*M_PI*x/Lx);
        P.pos( iPart, IX ) = P.pos( iPart, IX ) + Pout.at( iPart, IVX ) * dt_perturb;

        // Compute periodic position
        P.pos(iPart, IX) = fmod( (P.pos(iPart, IX) - xmin) + (Lx) , Lx) + xmin;

      });
    }

    U.distributeParticles(particle_array_name);
  }  
}; 

} // namespace dyablo


FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_zeldovitch_particles, 
                 "zeldovitch_particles");

