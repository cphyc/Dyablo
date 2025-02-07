#include "InitialConditions_base.h"
#include "utils/io/FortranBinaryReader.h"
#include "utils/units/Units.h"
#include "ionization/Ionization_utils.h"

namespace dyablo {

class InitialConditions_grafic_fields : public InitialConditions
{
  struct grafic_header
  {
    int32_t nx,ny,nz;
    float dx;
    float xo,yo,zo;
    float astart;
    float om,ov,H0;
  };

public:
  InitialConditions_grafic_fields(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : 
    foreach_cell(foreach_cell),
    gamma0(configMap.getValue<real_t>("hydro", "gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro", "smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro", "smallc", 1e-10)),
    smallp(smallc * smallc / gamma0),
    clight_fraction(configMap.getValue<real_t>("cosmology", "clight_fraction", 0.1)),
    xe_start(configMap.getValue<real_t>("rad", "xe_start", 1.2e-3)),
    zre_start(configMap.getValue<real_t>("ionization", "zre_start", -1000.0)),
    temperature_bb(configMap.getValue<real_t>("ionization", "temp_black_body", 1e5))
  {
    {
      const AMRmesh& pmesh = foreach_cell.get_amr_mesh();
      int bx = foreach_cell.blockSize()[IX];
      int cell_level = pmesh.get_level_min() + std::log2(bx);
      this->graficDir = configMap.getValue<std::string>("grafic", "inputDir", std::string("data/IC_")+std::to_string(cell_level));
    }

    // Read header from grafic file
    grafic_header header;
    std::string grafic_filename = this->graficDir + "/ic_deltab";
    std::ifstream grafic_file(grafic_filename , std::ios::in|std::ios::binary );
    DYABLO_ASSERT_HOST_RELEASE(grafic_file, "Could not open grafic file : `" + grafic_filename + "`");
    FortranBinaryReader::read_record( grafic_file, &header, 1 );

    printHeader(header, graficDir);

    // Check if values are the same as .ini (or create new entries)
    auto set_or_check = [&]( std::string section, std::string param, auto val )
    {
      using T = decltype(val);
      if( configMap.hasValue(section, param) )
      {
        DYABLO_ASSERT_HOST_RELEASE( configMap.getValue< T >(section, param) == val, 
          ".ini parameter does not match grafic file : \n"
          << ".ini " << section << "/" << param << " : " << configMap.getValue< T >(section, param) << "\n"
          << grafic_filename << " : " << val
           )
      }
      else
        configMap.getValue< T >(section, param, val);
    };

    {
      // Mean mass for a raw cell used for refinement
      double mass0 = 1.0/(header.nx*header.ny*header.nz);
      std::cout << "COSMO mass0=" << mass0 << std::endl;

      real_t mass_coarsen_factor = configMap.getValue<real_t>("cosmology", "mass_coarsen_factor", 0.1);
      real_t mass_refine_factor = configMap.getValue<real_t>("cosmology", "mass_refine_factor", 1.0);

      set_or_check("amr", "mass_coarsen", mass0*mass_coarsen_factor);
      set_or_check("amr", "mass_refine", mass0*mass_refine_factor);
    }

    DYABLO_ASSERT_HOST_RELEASE( 0 == header.nx%foreach_cell.blockSize()[IX], "Block size (x) is not compatible with grafic file" );
    DYABLO_ASSERT_HOST_RELEASE( 0 == header.ny%foreach_cell.blockSize()[IY], "Block size (y) is not compatible with grafic file" );
    DYABLO_ASSERT_HOST_RELEASE( 0 == header.nz%foreach_cell.blockSize()[IZ], "Block size (z) is not compatible with grafic file" );
    set_or_check( "amr", "coarse_oct_resolution_x", header.nx/foreach_cell.blockSize()[IX] );
    set_or_check( "amr", "coarse_oct_resolution_y", header.ny/foreach_cell.blockSize()[IY] );
    set_or_check( "amr", "coarse_oct_resolution_z", header.nz/foreach_cell.blockSize()[IZ] );
    this->xmin = header.xo;
    set_or_check( "mesh", "xmin", header.xo );
    this->ymin = header.yo;
    set_or_check( "mesh", "ymin", header.yo );
    this->zmin = header.zo;
    set_or_check( "mesh", "zmin", header.zo );

    this->astart = header.astart;
    set_or_check( "cosmology", "astart", header.astart );
    this->omegam = header.om;
    set_or_check( "cosmology", "omegam", omegam );
    set_or_check( "cosmology", "omegav", header.ov );
    this->omegab = configMap.getValue<real_t>("cosmology", "omegab", 0.049);

    using namespace Units;

    this->H0 = header.H0 * (Kilo * meter) / second / (Mega * parsec); // Hubble constant (s-1)
    set_or_check( "cosmology", "H0", H0 );
    real_t dx = header.dx * (Mega * parsec); // Cell size (m)
    set_or_check( "cosmology", "dx", dx );

    rhoc = 3. * H0 * H0 / (8. * M_PI * NEWTON_G); // comoving critical density (kg/m3)
    real_t rstar = header.nx * header.dx * (Mega * parsec); // box size in m 
    tstar = 2. / H0 / sqrt(omegam); // sec
    vstar = rstar / tstar; //m/s
    rhostar = rhoc * omegam;
    pstar = rhostar * vstar * vstar;

    set_or_check( "cosmology", "vstar", vstar );
    set_or_check( "cosmology", "rhostar", rhostar );
    set_or_check( "cosmology", "tstar", tstar );
    set_or_check( "cosmology", "ctilde", clight_fraction * 3e5 * (Kilo*meter/second) * astart/vstar);

    real_t cosmo_z = 1. / astart - 1.;
    this->temp = 317.5 * (cosmo_z * cosmo_z) / (151.0 * 151.0);

    // Compute sigma_n, sigma_e and typical energy
    auto s = computeSigma(this->temperature_bb);
    set_or_check( "ionization", "sigma_n_c", s.sn * clight_fraction * SPEEDOFLIGHT );
    set_or_check( "ionization", "sigma_e_c", s.se * clight_fraction * SPEEDOFLIGHT );
    set_or_check( "ionization", "typical_energy", s.etyp);
  }

  void init( UserData& U )
  {
    ForeachCell& foreach_cell = this->foreach_cell;

    // Read header from ic_deltab
    grafic_header header;
    {
      // Open grafic file ic_deltab
      std::string grafic_filename = this->graficDir + "/ic_deltab";
      std::ifstream grafic_file(grafic_filename , std::ios::in|std::ios::binary );
      DYABLO_ASSERT_HOST_RELEASE(grafic_file, "Could not open grafic file : `" + grafic_filename + "`");
      FortranBinaryReader::read_record( grafic_file, &header, 1 );
    }

    int32_t nx = header.nx;
    int32_t ny = header.ny;
    int32_t nz = header.nz;    

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    auto fill_field = [&](const std::string& grafic_file_name, const std::string& dyablo_field_name)
    {
      Kokkos::View< float***, Kokkos::LayoutLeft > grafic_field_device ( 
          grafic_file_name,
          nx, ny, nz
      );

      {
        // Open grafic file
        std::string grafic_filename = this->graficDir + "/" + grafic_file_name;
        std::ifstream grafic_file(grafic_filename , std::ios::in|std::ios::binary );
        DYABLO_ASSERT_HOST_RELEASE(grafic_file, "Could not open grafic file : `" + grafic_filename + "`");
        // Read header from grafic file
        grafic_header header;
        FortranBinaryReader::read_record( grafic_file, &header, 1 );
        // Read array from grafic file
        auto grafic_field_host = Kokkos::create_mirror_view( grafic_field_device );
        for(uint32_t z=0; z<(uint32_t)nz; z++)
          FortranBinaryReader::read_record( grafic_file, &grafic_field_host(0,0,z), nx*ny );
        Kokkos::deep_copy( grafic_field_device, grafic_field_host );
      }

      enum VarIndex { Ifield };
      UserData::FieldAccessor U_field = U.getAccessor({{dyablo_field_name, Ifield}});

      real_t xmin = this->xmin;
      real_t ymin = this->ymin;
      real_t zmin = this->zmin;

      foreach_cell.foreach_cell( "InitialConditions_grafic_fields::init_field", U_field.getShape(),
        KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
      {
        auto c = cellmetadata.getCellCenter( iCell );
        auto s = cellmetadata.getCellSize( iCell );
        uint32_t ix = (c[IX] - xmin) / s[IX];
        uint32_t iy = (c[IY] - ymin) / s[IY];
        uint32_t iz = (c[IZ] - zmin) / s[IZ];

        U_field.at( iCell, Ifield ) = grafic_field_device(ix,iy,iz);
      });
    };

    // Sequentially read density and velocities from grafic file
    U.new_fields({"rho","e_tot","rho_vx","rho_vy","rho_vz", "e_rad", "fx_rad", "fy_rad", "fz_rad", "xe", "zre", "temp"});
    fill_field( "ic_deltab", "rho" );
    fill_field( "ic_velbx", "rho_vx" );
    fill_field( "ic_velby", "rho_vy" );
    fill_field( "ic_velbz", "rho_vz" );

    enum VarIndex_hydro {ID, IE, IU, IV, IW, IDR, IUR, IVR, IWR, IXE, IZR, ITemp};
    UserData::FieldAccessor Uinout = U.getAccessor({
        {"rho", ID},
        {"e_tot", IE},
        {"rho_vx", IU},
        {"rho_vy", IV},
        {"rho_vz", IW},
        {"e_rad",  IDR},
        {"fx_rad", IUR},
        {"fy_rad", IVR},
        {"fz_rad", IWR},
        {"xe", IXE},
	      {"zre", IZR},
        {"temp", ITemp},
    });

    using namespace Units;

    // Parameters
    real_t gamma0 = this->gamma0;
    real_t omegab = this->omegab;
    real_t omegam = this->omegam;
    real_t astart = this->astart;
    real_t smallp = this->smallp;
    real_t rhoc = this->rhoc;
    real_t pstar = this->pstar;
    real_t vstar = this->vstar;
    real_t xe_start = this->xe_start;
    real_t zre_start = this->zre_start;
    real_t temp = this->temp;

    foreach_cell.foreach_cell( "InitialConditions_grafic_fields::compute_conservative", Uinout.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      real_t cosmo_density = Uinout.at( iCell, ID );
      real_t cosmo_velx = Uinout.at( iCell, IU );
      real_t cosmo_vely = Uinout.at( iCell, IV );
      real_t cosmo_velz = Uinout.at( iCell, IW );

      real_t u = cosmo_velx * 1e3 * astart / vstar;
      real_t v = cosmo_vely * 1e3 * astart / vstar;
      real_t w = cosmo_velz * 1e3 * astart / vstar;
      real_t u2 = u * u + v * v + w * w;

      real_t rho = (cosmo_density + 1.0) * omegab / omegam;
      real_t rho_u = rho * u;
      real_t rho_v = rho * v;
      real_t rho_w = rho * w;

      // Physical baryon density in kg/m3
      real_t cosmo_rhob = (cosmo_density + 1.0) * omegab * rhoc / (astart * astart * astart);

      // Physical pressure
      real_t cosmo_pressure = (gamma0 - 1.0) * 1.5 * (cosmo_rhob * (1. - Units::YHE) / PROTON_MASS * (1. + Units::yHE)) * Units::KBOLTZ * temp;
      real_t p = fmax( cosmo_pressure/pstar * (astart * astart * astart * astart * astart), smallp );
      real_t e_tot = rho*u2/2.0 + p/(gamma0-1.0);

      // Hydro
      Uinout.at( iCell, ID ) = rho;
      Uinout.at( iCell, IE ) = e_tot;
      Uinout.at( iCell, IU ) = rho_u;
      Uinout.at( iCell, IV ) = rho_v;
      Uinout.at( iCell, IW ) = rho_w;

      // Rad
      Uinout.at( iCell, IDR ) = 1e-18;
      Uinout.at( iCell, IUR ) = 0.0;
      Uinout.at( iCell, IVR ) = 0.0;
      Uinout.at( iCell, IWR ) = 0.0;
      Uinout.at( iCell, IXE ) = xe_start;
      Uinout.at( iCell, IZR ) = zre_start;
      Uinout.at( iCell, ITemp ) = temp;

    });

  }

/**
 * Print header
*/
void printHeader(grafic_header hdr, std::string graficDir){

 std::cout << "GRAFIC : reading initial Conditions from " << graficDir << std::endl
           << "nx = " << hdr.nx << std::endl
           << "ny = " << hdr.ny << std::endl
           << "nz = " << hdr.nz << std::endl
           << "x0 = " << hdr.xo << std::endl
           << "y0 = " << hdr.yo << std::endl
           << "z0 = " << hdr.zo << std::endl
           << "astart = " << hdr.astart << std::endl
           << "om = " << hdr.om << std::endl
           << "ov = " << hdr.ov << std::endl
           << "H0 = " << hdr.H0 << " [km/s/Mpc]" << std::endl
           << "dx = " << hdr.dx << " [Mpc]" << std::endl
           << "box size = " << hdr.nx * hdr.dx << " [Mpc]" << std::endl
           << "box size = " << hdr.nx * hdr.dx * hdr.H0/100.0 << " [Mpc/h]" << std::endl
      //     << "mass0 = " << mass0 << std::endl
           << "##########################" << std::endl;
}

private:
  std::string graficDir;
  ForeachCell& foreach_cell;
  
  // Cosmo params
  real_t xmin, ymin, zmin;
  real_t omegab;
  real_t astart, omegam;
  real_t H0;
  real_t rhoc, pstar, vstar, rhostar, tstar;


  // Hydro params
  real_t gamma0;
  real_t smallr,smallc,smallp;

  real_t clight_fraction;

  real_t xe_start, zre_start, temperature_bb, temp;

};

} //namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_grafic_fields, 
                 "grafic_fields");