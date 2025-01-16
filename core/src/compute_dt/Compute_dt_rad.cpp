#include "Compute_dt_base.h"

#include "utils_hydro.h"

#include "states/State_forward.h"

namespace dyablo {


class Compute_dt_rad : public Compute_dt
{
public:
  Compute_dt_rad(   ConfigMap& configMap,
                        ForeachCell& foreach_cell,
                        Timers& timers )
  : foreach_cell(foreach_cell),
    ctilde_a0(configMap.getValue<real_t>( "cosmology", "ctilde" ) / configMap.getValue<real_t>( "cosmology", "astart" ))
  {
    real_t default_cfl = 0.5;
    if (configMap.hasValue("hydro", "cfl")) {
      std::cout << "WARNING : hydro/cfl is deprecated in .ini, use dt/hydro_cfl instead !" << std::endl;
      default_cfl = configMap.getValue<real_t>("hydro", "cfl");
    }
    this->cfl = configMap.getValue<real_t>("dt", "hydro_cfl", default_cfl);
  }

  void compute_dt( const UserData& U, ScalarSimulationData& scalar_data )
  {
    real_t aexp = scalar_data.get<real_t>("aexp");
    real_t ctilde = this->ctilde_a0 * aexp;

    auto cells = foreach_cell.getCellMetaData();

    // TODO : We don't have to iterate on every cell for this
    real_t inv_dt;
    foreach_cell.reduce_cell( "compute_dt_rad", U.getShape(),
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& inv_dt_update )
    {
      auto cell_size = cells.getCellSize(iCell);
      real_t dx = cell_size[IX];
      real_t dy = cell_size[IY];
      real_t dz = cell_size[IZ];

	    inv_dt_update = FMAX( inv_dt_update, ctilde/dx+ ctilde/dy + ctilde/dz );
      
    }, Kokkos::Max<real_t>(inv_dt) );

    real_t dt_local = cfl / inv_dt;

    DYABLO_ASSERT_HOST_RELEASE(dt_local>0, "invalid dt = " << dt_local);

    real_t dt;
    auto communicator = foreach_cell.get_amr_mesh().getMpiComm();
    communicator.MPI_Allreduce(&dt_local, &dt, 1, MpiComm::MPI_Op_t::MIN);

    scalar_data.set<real_t>("dt", dt);
  }
  

private:
  ForeachCell& foreach_cell;
  real_t ctilde_a0;
  real_t cfl;
  
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_rad, "Compute_dt_rad" );
