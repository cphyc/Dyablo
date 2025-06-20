#include "SourceUpdate_base.h"
#include "utils/units/Units.h"

namespace dyablo{

/**
 * @brief Ionization 'Bunny' source term
 */
class SourceUpdate_Ionization_Stromgren : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;

  real_t dN_photons; // Number of photons per cell [1/cm^3/s]

  // Units
  real_t unit_time;
  real_t unit_density;
  real_t unit_length;
  real_t unit_photon_number;

public:
  SourceUpdate_Ionization_Stromgren(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  :  foreach_cell(foreach_cell),
     timers(timers),
     dN_photons(configMap.getValue<real_t>("source_terms", "dN_photons", 1e50)),
     unit_time( configMap.getValue<real_t>("units", "time", 1.0) ),
     unit_density( configMap.getValue<real_t>("units", "density", 1.0) ),
     unit_length( configMap.getValue<real_t>("units", "length", 1.0) ),
     unit_photon_number( configMap.getValue<real_t>("units", "photon_number", 1.0) )
  { }

  void update( UserData &U,
               ScalarSimulationData& scalar_data)
  {
    uint32_t ndim = foreach_cell.getDim();

    ForeachCell& foreach_cell = this->foreach_cell;

    timers.get("SourceUpdate_Ionization_Stromgren").start();

    enum VarIndex {IDR,IUR,IVR,IWR};

    UserData::FieldAccessor Uout = U.getAccessor(
      {
        {"e_rad_next",   IDR},
        {"fx_rad_next",  IUR},
        {"fy_rad_next",  IVR},
        {"fz_rad_next",  IWR}
      });

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    const real_t dt = scalar_data.get<real_t>("dt"); 
    const real_t dN_photons = 2.8e42;

    foreach_cell.foreach_cell( "SourceUpdate_Ionization_Stromgren", Uout.getShape(),
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell_Uout)
    {
      auto pos = cells.getCellCenter(iCell_Uout);
      auto cell_size = cells.getCellSize(iCell_Uout);
      auto cell_volume = cell_size[IX] * cell_size[IY] * (ndim == 3 ? cell_size[IZ] : 1);

      real_t x = pos[IX];
      real_t y = pos[IY];
      real_t z = pos[IZ];

      real_t x1=0.5,y1=0.5,z1=0.5;
      real_t r1=SQR((x-x1) / cell_size[IX]) + SQR((y-y1) / cell_size[IY]) + SQR((z-z1) / cell_size[IZ]);

      if(r1 <= 2*2){
        // std::cout << "SOURCE : " << 7.7e+44 / (cell_size[IX] * cell_size[IY] * cell_size[IZ] * pow(unit_length, 3)) * dt * unit_time << std::endl;
        Uout.at(iCell_Uout,IDR) += dN_photons / (cell_volume * pow(unit_length, ndim)) * dt * unit_time;
        Uout.at(iCell_Uout,IUR) += 0; // F_photons/SQRT(3.);
        Uout.at(iCell_Uout,IVR) += 0; // F_photons/SQRT(3.);
        Uout.at(iCell_Uout,IWR) += 0; // F_photons/SQRT(3.);
      }
    });

    timers.get("SourceUpdate_Ionization_Stromgren").stop();
  }
};


} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory,
                  dyablo::SourceUpdate_Ionization_Stromgren,
                  "SourceUpdate_Ionization_Stromgren" );