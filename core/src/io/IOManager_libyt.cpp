/**
 * @file IOManager_libyt.cpp
 * @brief libyt in-situ analysis IO plugin for Dyablo (oct-based AMR path).
 *
 * Exposes Dyablo's block-AMR data to yt via libyt's oct path. The
 * DataArrayBlock layout [cell, var, leaf] (x-fastest, LayoutLeft) is
 * passed zero-copy as a strided NumPy array inside Python.
 *
 * Configuration (add to .ini file):
 *   [output]
 *   backend=IOManager_libyt
 *   write_variables=rho,rho_vx,rho_vy,rho_vz,e_tot
 *
 *   [libyt]
 *   script=inline_script          # Python script loaded at init (no .py)
 *   function=yt_inline            # function called each snapshot
 *
 * The libyt script receives:
 *   libyt.oct_grids[i]  — dict with 'level' and 'pos' for each local leaf
 *   libyt.oct_data[field_name]  — strided numpy array [leaf, var, cell]
 */

#include "io/IOManager_base.h"

#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ScalarSimulationData.h"
#include "amr/LightOctree.h"
#include "amr/LightOctree_hashmap.h"
#include "foreach_cell/ForeachCell.h"
#include "kokkos_shared.h"
#include "user_data/UserData.h"
#include "utils/monitoring/Timers.h"

// libyt headers
#include "libyt.h"
#include "yt_type_oct.h"

namespace dyablo {

class IOManager_libyt : public IOManager {
public:
    IOManager_libyt(ConfigMap& configMap, ForeachCell& foreach_cell, Timers& timers);
    ~IOManager_libyt() override;

    void save_snapshot(const UserData& U, ScalarSimulationData& scalar_data) override;

private:
    ForeachCell& foreach_cell_;
    real_t xmin_, xmax_, ymin_, ymax_, zmin_, zmax_;
    std::set<std::string> write_varnames_;
    std::string script_; // owned; passed as c_str() to libyt
    std::string function_; // Python function called per snapshot
};

// ---------------------------------------------------------------------------
// Constructor: initialise libyt once for the lifetime of this IO manager
// ---------------------------------------------------------------------------
IOManager_libyt::IOManager_libyt(ConfigMap& configMap,
                                  ForeachCell& foreach_cell,
                                  Timers& timers)
    : foreach_cell_(foreach_cell),
      xmin_(configMap.getValue<real_t>("mesh", "xmin", 0.0)),
      xmax_(configMap.getValue<real_t>("mesh", "xmax", 1.0)),
      ymin_(configMap.getValue<real_t>("mesh", "ymin", 0.0)),
      ymax_(configMap.getValue<real_t>("mesh", "ymax", 1.0)),
      zmin_(configMap.getValue<real_t>("mesh", "zmin", 0.0)),
      zmax_(configMap.getValue<real_t>("mesh", "zmax", 1.0)),
      script_(configMap.getValue<std::string>("libyt", "script", "inline_script")),
      function_(configMap.getValue<std::string>("libyt", "function", "yt_inline"))
{
    // Parse comma-separated write_variables
    std::string wv = configMap.getValue<std::string>("output", "write_variables", "rho");
    std::stringstream ss(wv);
    std::string var;
    while (std::getline(ss, var, ','))
        write_varnames_.insert(var);

    yt_param_libyt param_libyt;
    param_libyt.verbose = YT_VERBOSE_INFO;
    param_libyt.script  = script_.c_str();
    param_libyt.check_data = false;

    int dummy_argc = 0;
    char** dummy_argv = nullptr;
    if (yt_initialize(dummy_argc, dummy_argv, &param_libyt) != YT_SUCCESS)
        throw std::runtime_error("IOManager_libyt: yt_initialize() failed");
}

IOManager_libyt::~IOManager_libyt()
{
    yt_finalize();
}

// ---------------------------------------------------------------------------
// save_snapshot: called once per output step
// ---------------------------------------------------------------------------
void IOManager_libyt::save_snapshot(const UserData& U_,
                                     ScalarSimulationData& scalar_data)
{
    // -----------------------------------------------------------------------
    // 1. Gather mesh metadata
    // -----------------------------------------------------------------------
    auto& amr_mesh  = foreach_cell_.get_amr_mesh();
    const LightOctree& lmesh = amr_mesh.getLightOctree();
    const auto& storage = lmesh.getStorage();

    const uint32_t bx = foreach_cell_.blockSize()[IX];
    const uint32_t by = foreach_cell_.blockSize()[IY];
    const uint32_t bz = foreach_cell_.blockSize()[IZ];
    const uint32_t NML = bx * by * bz;

    const uint32_t nbOcts_local  = amr_mesh.getNumOctants();
    const uint64_t nbOcts_global = amr_mesh.getGlobalNumOctants();

    const int level_min = storage.level_min; (void)level_min;
    const auto& cgs = storage.coarse_grid_size;

    const real_t t  = scalar_data.get<real_t>("time");

    // Collect the fields we actually want to write (intersection with UserData)
    std::set<std::string> enabled = U_.getEnabledFields();
    std::vector<std::string> fields_to_write;
    for (const auto& name : write_varnames_)
        if (enabled.count(name))
            fields_to_write.push_back(name);

    const int n_vars = static_cast<int>(fields_to_write.size());
    if (n_vars == 0)
        return; // nothing to do

    // -----------------------------------------------------------------------
    // 2. Build a contiguous host DataArrayBlock
    //    Layout: [cell, var, leaf] — strides (1, NML, n_vars*NML)
    // -----------------------------------------------------------------------
    std::vector<double> data_block(static_cast<size_t>(NML) * n_vars * nbOcts_local, 0.0);

    for (int ivar = 0; ivar < n_vars; ++ivar) {
        // getField deep-copies one column of the device storage
        auto field      = U_.getField(fields_to_write[ivar]);
        auto host_field = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field.U);

        for (uint32_t iOct = 0; iOct < nbOcts_local; ++iOct) {
            for (uint32_t cell = 0; cell < NML; ++cell) {
                data_block[cell
                           + static_cast<size_t>(ivar) * NML
                           + static_cast<size_t>(iOct) * n_vars * NML] =
                    static_cast<double>(host_field(cell, 0, iOct));
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3. Fill leaf metadata (level + centre position)
    // -----------------------------------------------------------------------
    std::vector<yt_oct> octs_local(nbOcts_local);
    for (uint32_t iOct = 0; iOct < nbOcts_local; ++iOct) {
        LightOctree::OctantIndex idx{iOct, false};
        auto center = lmesh.getCenter(idx); // normalized [0,1]
        octs_local[iOct].level  = lmesh.getLevel(idx);
        octs_local[iOct].pos[0] = xmin_ + center[IX] * (xmax_ - xmin_);
        octs_local[iOct].pos[1] = ymin_ + center[IY] * (ymax_ - ymin_);
        octs_local[iOct].pos[2] = zmin_ + center[IZ] * (zmax_ - zmin_);
    }

    // -----------------------------------------------------------------------
    // 4. yt_set_Parameters — domain geometry, units, cosmology
    // -----------------------------------------------------------------------
    yt_param_yt param_yt;
    param_yt.frontend          = "libyt_oct";
    param_yt.fig_basename      = "dyablo";
    param_yt.length_unit       = 1.0;
    param_yt.mass_unit         = 1.0;
    param_yt.time_unit         = 1.0;
    param_yt.velocity_unit     = 1.0;
    param_yt.current_time      = static_cast<double>(t);
    param_yt.dimensionality    = 3;
    param_yt.domain_left_edge[0]  = xmin_; param_yt.domain_left_edge[1]  = ymin_; param_yt.domain_left_edge[2]  = zmin_;
    param_yt.domain_right_edge[0] = xmax_; param_yt.domain_right_edge[1] = ymax_; param_yt.domain_right_edge[2] = zmax_;
    // Coarsest grid: coarse_grid_size * block_dims cells per direction
    param_yt.domain_dimensions[0] = static_cast<int>(cgs[IX] * bx);
    param_yt.domain_dimensions[1] = static_cast<int>(cgs[IY] * by);
    param_yt.domain_dimensions[2] = static_cast<int>(cgs[IZ] * bz);
    param_yt.refine_by         = 2;
    param_yt.num_grids         = 0; // patch path not used
    param_yt.num_grids_local   = 0;
    param_yt.num_fields        = n_vars;
    param_yt.current_redshift  = 0.0;
    param_yt.cosmological_simulation = 0;
    param_yt.periodicity[0] = param_yt.periodicity[1] = param_yt.periodicity[2] = 0;

    if (yt_set_Parameters(&param_yt) != YT_SUCCESS)
        throw std::runtime_error("IOManager_libyt: yt_set_Parameters() failed");

    // -----------------------------------------------------------------------
    // 5. yt_get_FieldsPtr — fill field metadata
    // -----------------------------------------------------------------------
    yt_field* field_list = nullptr;
    if (yt_get_FieldsPtr(&field_list) != YT_SUCCESS)
        throw std::runtime_error("IOManager_libyt: yt_get_FieldsPtr() failed");

    for (int i = 0; i < n_vars; ++i) {
        field_list[i].field_name        = fields_to_write[i].c_str();
        field_list[i].field_type        = "cell-centered";
        field_list[i].contiguous_in_x   = true; // x-fastest (Fortran/LayoutLeft)
        field_list[i].field_dtype       = YT_DOUBLE;
        field_list[i].var_index         = i;    // column in DataArrayBlock
        // attribute: field is dimensionless code unit by default
        field_list[i].field_unit        = "code_units";
    }

    // -----------------------------------------------------------------------
    // 6. yt_set_OctParameters — block shape + leaf counts
    // -----------------------------------------------------------------------
    yt_param_oct param_oct;
    param_oct.num_leaves_total  = static_cast<long>(nbOcts_global);
    param_oct.num_leaves_local  = static_cast<int>(nbOcts_local);
    param_oct.block_dimensions[0] = static_cast<int>(bx);
    param_oct.block_dimensions[1] = static_cast<int>(by);
    param_oct.block_dimensions[2] = static_cast<int>(bz);

    if (yt_set_OctParameters(&param_oct) != YT_SUCCESS)
        throw std::runtime_error("IOManager_libyt: yt_set_OctParameters() failed");

    // -----------------------------------------------------------------------
    // 7. yt_get_OctsPtr — fill leaf metadata
    // -----------------------------------------------------------------------
    yt_oct* octs_ptr = nullptr;
    if (yt_get_OctsPtr(&octs_ptr) != YT_SUCCESS)
        throw std::runtime_error("IOManager_libyt: yt_get_OctsPtr() failed");

    for (uint32_t i = 0; i < nbOcts_local; ++i)
        octs_ptr[i] = octs_local[i];

    // -----------------------------------------------------------------------
    // 8. yt_get_OctDataPtr — register the DataArrayBlock pointer
    // -----------------------------------------------------------------------
    if (yt_get_OctDataPtr(data_block.data(), YT_DOUBLE) != YT_SUCCESS)
        throw std::runtime_error("IOManager_libyt: yt_get_OctDataPtr() failed");

    // -----------------------------------------------------------------------
    // 9. yt_commit — builds Python data structures
    // -----------------------------------------------------------------------
    if (yt_commit() != YT_SUCCESS)
        throw std::runtime_error("IOManager_libyt: yt_commit() failed");

    // -----------------------------------------------------------------------
    // 10. Run inline analysis, then free
    // -----------------------------------------------------------------------
    if (yt_run_Function(function_.c_str()) != YT_SUCCESS)
        throw std::runtime_error("IOManager_libyt: yt_run_Function() failed");

    if (yt_free() != YT_SUCCESS)
        throw std::runtime_error("IOManager_libyt: yt_free() failed");
}

} // namespace dyablo

FACTORY_REGISTER(dyablo::IOManagerFactory, dyablo::IOManager_libyt, "IOManager_libyt");
