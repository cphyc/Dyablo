#include "SourceUpdate_base.h"

namespace dyablo{

/**
 * @brief GLM parabolic cleaning term. 
 * 
 * This plugin participates in the divergence cleaning of MHD runs
 * by applying the parabolic source term of GLMMHD equations.
 * @ref Dedner et al., "Hyperbolic Divergence Cleaning for the MHD Equations", 2002
 * @ref Mignone et al., "High-order conservative finite difference GLM–MHD schemes for cell-centered MHD", 2010 

 */
class SourceUpdate_GLM : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;
  real_t flux;

  real_t base_c_h; // Hyperbolic cleaning speed
  real_t c_r;      // Proxy ratio of parabolic over cleaning speed : c_r = c_p^2/c_h
public:
  SourceUpdate_GLM(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  :  foreach_cell(foreach_cell),
     timers(timers)
  { 
    const real_t cfl = configMap.getValue<real_t>("dt", "hydro_cfl", 0.8);
    const uint32_t ndim = configMap.getValue<uint32_t>("mesh", "ndim", 3);
    const real_t xmin = configMap.getValue<real_t>("mesh", "xmin", 0.0);
    const real_t xmax = configMap.getValue<real_t>("mesh", "xmax", 1.0);
    const real_t ymin = configMap.getValue<real_t>("mesh", "ymin", 0.0);
    const real_t ymax = configMap.getValue<real_t>("mesh", "ymax", 1.0);
    const real_t zmin = configMap.getValue<real_t>("mesh", "zmin", 0.0);
    const real_t zmax = configMap.getValue<real_t>("mesh", "zmax", 1.0);

    const uint32_t level_max = configMap.getValue<uint32_t>("amr", "level_max", 10);

    const uint32_t bx = configMap.getValue<uint32_t>("amr", "bx", 0);
    const uint32_t by = configMap.getValue<uint32_t>("amr", "by", 0);
    const uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", 1);

    const real_t Lx = xmax - xmin;
    const real_t Ly = ymax - ymin;
    const real_t Lz = zmax - zmin; 

    const real_t min_dx = Lx / ((1 << level_max) * bx);
    const real_t min_dy = Ly / ((1 << level_max) * by);
    const real_t min_dz = Lz / ((1 << level_max) * bz); 

    real_t min_dh = FMIN(min_dx, min_dy);
    if (ndim == 3)
      min_dh = FMIN(min_dh, min_dz);

    // This value should be divided by dt in the kernels where it appears !
    base_c_h   = 0.5*cfl * min_dh;
    c_r = configMap.getValue<real_t>("hydro", "c_r", 0.1);
  }

  ~SourceUpdate_GLM() {}

  template<int ndim>
  void update_aux( UserData &U,
                   real_t dt) {
    ForeachCell& foreach_cell = this->foreach_cell;

    timers.get("GLM Parabolic Cleaning").start();

    enum VarIndex {IPSI};

    UserData::FieldAccessor Uout = U.getAccessor( {{"psi_next", VarIndex::IPSI}} );

    const real_t c_h = base_c_h / dt;
    const real_t c_p = sqrt(c_r * c_h);
    const real_t diff_factor = c_h*c_h/(c_p*c_p);

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    foreach_cell.foreach_cell( "GLMMHD_Parabolic_Source::update", Uout.getShape(), 
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell_Uout) {
      Uout.at(iCell_Uout, VarIndex::IPSI) *= exp(-dt * diff_factor);
    });

    timers.get("GLM Parabolic Cleaning").stop();
  }

  void update( UserData &U,
               ScalarSimulationData& scalar_data)
  {
    uint32_t ndim = foreach_cell.getDim();
    real_t dt = scalar_data.get<real_t>("dt");
    if (ndim == 2)
      update_aux<2>(U, dt);
    else
      update_aux<3>(U, dt);
  }
};


} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory, 
                  dyablo::SourceUpdate_GLM, 
                  "SourceUpdate_GLM" );