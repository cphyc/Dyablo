#include "SourceUpdate_base.h"

namespace dyablo{

/**
 * @brief Ionization 'Bunny' source term
 */
class SourceUpdate_Ionization_Bunny : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;
public:
  SourceUpdate_Ionization_Bunny(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  :  foreach_cell(foreach_cell),
     timers(timers)
  { }

  void update( UserData &U,
               ScalarSimulationData& scalar_data)
  {
    uint32_t ndim = foreach_cell.getDim();

    ForeachCell& foreach_cell = this->foreach_cell;

    timers.get("SourceUpdate_Ionization_Bunny").start();

    enum VarIndex {IDR,IUR,IVR,IWR};

    UserData::FieldAccessor Uout = U.getAccessor( 
      {
        {"e_rad_next",   IDR}, 
        {"fx_rad_next",  IUR},
        {"fy_rad_next",  IVR},
        {"fz_rad_next",  IWR}
      });

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    foreach_cell.foreach_cell( "SourceUpdate_Ionization_Bunny", Uout.getShape(), 
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell_Uout) 
    {
      auto pos = cells.getCellCenter(iCell_Uout);

      real_t x = pos[IX];
      real_t y = pos[IY];
      real_t z = pos[IZ];

      real_t x1=0.25,y1=0.25,z1=0.5;
      real_t r1=(x-x1)*(x-x1)+(y-y1)*(y-y1)+(z-z1)*(z-z1);
      real_t x2=0.25,y2=0.75,z2=0.5;
      real_t r2=(x-x2)*(x-x2)+(y-y2)*(y-y2)+(z-z2)*(z-z2);
      
      if(r1<(0.05*0.05)){
        real_t rhoNew= 1.0;
        Uout.at(iCell_Uout,IDR)=rhoNew;
        Uout.at(iCell_Uout,IUR)=rhoNew*0.1/SQRT(2.);
        Uout.at(iCell_Uout,IVR)=rhoNew*0.1/SQRT(2.);
        if (ndim == 3)
          Uout.at(iCell_Uout,IWR)=0.;
      }

      if(r2<(0.05*0.05)){
        real_t rhoNew= 1.0;
        Uout.at(iCell_Uout,IDR)=rhoNew;
        Uout.at(iCell_Uout,IUR)=rhoNew*0.1/SQRT(2.);
        Uout.at(iCell_Uout,IVR)=-rhoNew*0.1/SQRT(2.);
        if (ndim == 3)
          Uout.at(iCell_Uout,IWR)=0.;
      }
    });

    timers.get("SourceUpdate_Ionization_Bunny").stop();
  }
};


} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory, 
                  dyablo::SourceUpdate_Ionization_Bunny, 
                  "SourceUpdate_Ionization_Bunny" );