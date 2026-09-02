#pragma once

#include "user_data/UserData.h"
#include "user_data/FieldAccessor.h"
#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

#include "states/State_hydro.h"
#include "boundary_conditions/BoundaryConditions.h"
#include "utils/config/named_enum.h"

/**
 * Boundary conditions for thermal conduction
 */
enum ThermalBoundaryMode {
  TBM_TEMPERATURE, 
  TBM_TEMP_GRADIENT
};

template<>
inline named_enum<ThermalBoundaryMode>::init_list named_enum<ThermalBoundaryMode>::names() 
{
  return {
    {ThermalBoundaryMode::TBM_TEMPERATURE,   "temperature"},
    {ThermalBoundaryMode::TBM_TEMP_GRADIENT, "temperature_gradient"}
  };
}

/**
 * In the case we have an analytical kappa, this enum allows to pick which 
 * way kappa is calculated.
 */
enum KappaMode {
  KM_NONE,
  KM_TRI_LAYER, 
  KM_SPITZER,
};

template<>
inline named_enum<KappaMode>::init_list named_enum<KappaMode>::names() 
{
  return {
    {KappaMode::KM_NONE,      "none"},
    {KappaMode::KM_TRI_LAYER, "tri_layer"},
    {KappaMode::KM_SPITZER,   "spitzer"},
  };
}

namespace dyablo {

namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using GlobalArray  = ForeachCell::CellArray_global;
using PatchArray   = ForeachCell::CellArray_patch;
using CellIndex    = ForeachCell::CellIndex;
using CellMetaData = ForeachCell::CellMetaData;
using pos_t        = Kokkos::Array<real_t, 3>;

enum VarIndex_thermal {
  IRHO, 
  IPRS
};
}

/**
 * @brief Parabolic term solving for thermal conduction : dE/dt + div(Kappa grad T) = 0
 */
class ParabolicTerm_thermal_conduction {
public:
  using State     = HydroState; 
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  ParabolicTerm_thermal_conduction(ConfigMap &configMap)
    : bc_manager(configMap),
      ndim (configMap.getValue<int>("mesh", "ndim", 2) ),
      gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
      kappa_cst(configMap.getValue<real_t>("thermal_conduction", "kappa", 0.0)),
      diffusivity_mode(configMap.getValue<DiffusivityMode>("thermal_conduction", "diffusivity_mode", DM_CONSTANT)),
      bctc_min{
        configMap.getValue<ThermalBoundaryMode>("thermal_conduction", "bctc_xmin", TBM_TEMP_GRADIENT),
        configMap.getValue<ThermalBoundaryMode>("thermal_conduction", "bctc_ymin", TBM_TEMP_GRADIENT),
        configMap.getValue<ThermalBoundaryMode>("thermal_conduction", "bctc_zmin", TBM_TEMP_GRADIENT)
      },
      bctc_max{
        configMap.getValue<ThermalBoundaryMode>("thermal_conduction", "bctc_xmax", TBM_TEMP_GRADIENT),
        configMap.getValue<ThermalBoundaryMode>("thermal_conduction", "bctc_ymax", TBM_TEMP_GRADIENT),
        configMap.getValue<ThermalBoundaryMode>("thermal_conduction", "bctc_zmax", TBM_TEMP_GRADIENT)        
      },
      bctc_min_val{
        configMap.getValue<real_t>("thermal_conduction", "bctc_xmin_val", 0.0),
        configMap.getValue<real_t>("thermal_conduction", "bctc_ymin_val", 0.0),
        configMap.getValue<real_t>("thermal_conduction", "bctc_zmin_val", 0.0)
      },
      bctc_max_val{
        configMap.getValue<real_t>("thermal_conduction", "bctc_xmax_val", 0.0),
        configMap.getValue<real_t>("thermal_conduction", "bctc_ymax_val", 0.0),
        configMap.getValue<real_t>("thermal_conduction", "bctc_zmax_val", 0.0)        
      },
      min_pos {
        configMap.getValue<real_t>("mesh", "xmin"),
        configMap.getValue<real_t>("mesh", "ymin"),
        configMap.getValue<real_t>("mesh", "zmin")
      },
      max_pos {
        configMap.getValue<real_t>("mesh", "xmax"),
        configMap.getValue<real_t>("mesh", "ymax"),
        configMap.getValue<real_t>("mesh", "zmax")
      }
      {
        if (diffusivity_mode == DM_ANALYTICAL) {
          kappa_mode = configMap.getValue<KappaMode>("thermal_conduction", "kappa_mode", KM_NONE);

          if (kappa_mode == KM_TRI_LAYER) {
            tr_thick = configMap.getValue<real_t>("tri_layer", "transition_thickness", 0.0);
            z1       = configMap.getValue<real_t>("tri_layer", "z1", 0.0);
            z2       = configMap.getValue<real_t>("tri_layer", "z2", 1.0);
            K1       = configMap.getValue<real_t>("tri_layer", "K1", 1.0);
            K2       = configMap.getValue<real_t>("tri_layer", "K2", 1.0);
          }
        }
        else 
          kappa_mode = KM_NONE;
      };

  /**
   * @brief Computes the heat flux (Kappa grad T) at the boundary 
   * @param q The primitive state in the first cell of the domain
   * @param dh The size of the cell in the boundary direction
   * @param dir The current axis
   * @param pos The position of the boundary
   * @param min_bound if the boundary is the "left" or "right" boundary
   */
  KOKKOS_INLINE_FUNCTION
  real_t getBoundaryHeatFlux(const PrimHydroState   q,
                             const real_t           dh,
                             const ComponentIndex3D dir,
                             const pos_t            pos,
                             const bool             min_bound) const {
    const real_t Tc = q.p / q.rho;
    real_t kappa = kappa_cst;
    real_t gradT = 0.0;

    if (min_bound) {
      kappa = 0.5 * (compute_kappa(pos, bctc_min_val[dir]) + compute_kappa(pos, Tc));
      if (bctc_min[dir] == TBM_TEMPERATURE)
        gradT = 2.0 * (Tc - bctc_min_val[dir]) / dh;
      else if (bctc_min[dir] == TBM_TEMP_GRADIENT)
        gradT = bctc_min_val[dir];
    }
    else {
      kappa = 0.5 * (compute_kappa(pos, bctc_max_val[dir]) + compute_kappa(pos, Tc));
      if (bctc_max[dir] == TBM_TEMPERATURE)
        gradT = 2.0 * (bctc_max_val[dir] - Tc) / dh;
      else if (bctc_max[dir] == TBM_TEMP_GRADIENT)
        gradT = bctc_max_val[dir];
    }

    return kappa * gradT;
  }

  KOKKOS_INLINE_FUNCTION
  real_t compute_kappa(const pos_t& pos, const real_t T) const {
    real_t kappa = kappa_cst;

    // Analytical calculations 
    if (diffusivity_mode == DM_ANALYTICAL) {
      if (kappa_mode == KM_TRI_LAYER) {
        const real_t d = (ndim == 2 ? pos[IY] : pos[IZ]);
        const real_t th = tr_thick;
        const real_t k1 = kappa_cst * K1;
        const real_t k2 = kappa_cst * K2;

        const real_t tr1 = (tanh((d-z1)/th) + 1.0) * 0.5;
        const real_t tr2 = (tanh((z2-d)/th) + 1.0) * 0.5;
        const real_t tr = tr1*tr2;

        kappa = k2 * (1.0-tr) + k1 * tr;
      }
      else if (kappa_mode == KM_SPITZER) {
        kappa = kappa_cst * Kokkos::pow(T, 2.5);
      }
    }

    return kappa;
  }

  static std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo()
  {
    return ConsHydroState::getFieldsInfo();
  }


  template <int ndim, typename Uin_t>
  KOKKOS_INLINE_FUNCTION
  void compute_rhs(const Uin_t&        Uin,
                   const PatchArray&   Ugroup,
                   const PatchArray&   Qgroup,
                   const GhostedArray& rhs,
                   const CellIndex&    iCell_Uout,
                   const CellIndex&    iCell_Qgroup,
                   const CellMetaData& cellmetadata) const
  {
    // Aliases
    using offset_t = CellIndex::offset_t;
    
    // TODO : Move this to a more appropriate place with EOS
    auto compute_temperature = [](const PrimHydroState &q) -> real_t {
      return q.p / q.rho;
    };

    // Getting cell info
    ForeachCell::SearchMode_local search_local( ForeachCell::SearchMode_local::ASSERT );
    ForeachCell::SearchMode_neighbor search_neighbor( cellmetadata.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

    CellIndex iCell_Uin    = Uin.getShape().convert_index(iCell_Uout, search_local);
    auto iCell_rhs = rhs.getShape().convert_index(iCell_Uout, search_local);
    auto size = cellmetadata.getCellSize(iCell_Uin);
    auto pos  = cellmetadata.getCellCenter(iCell_Uin);
    real_t V  = size[IX] * size[IY] * (ndim == 3 ? size[IZ] : 1.0);

    PrimHydroState qC;
    getPrimitiveState<ndim>(Qgroup, iCell_Qgroup, qC);
    auto TC = compute_temperature(qC);
    auto kappaC = compute_kappa(pos, TC);

    // Relative sizes as function of level difference
    constexpr real_t S[3] {0.75, 1.0, 1.5};

    // Interface area array
    const real_t A[3] = {ndim == 3 ? size[IY]*size[IZ] : size[IY],
                         ndim == 3 ? size[IX]*size[IZ] : size[IX],
                         size[IX]*size[IY]};

    auto compute_thermal_flux = [&](ComponentIndex3D dir) {
      offset_t offsetm{0}, offsetp{0};
      offsetm[dir] = -1;
      offsetp[dir] =  1;
      real_t area = A[dir];

      // Getting neighbor element
      auto iiL = iCell_Uin.getNeighbor(offsetm, search_neighbor);
      auto iiR = iCell_Uin.getNeighbor(offsetp, search_neighbor);

      // And level differences
      int ldiff_L = iiL.level_diff();
      int ldiff_R = iiR.level_diff();

      // And relative sizes
      real_t SL = S[ldiff_L+1];
      real_t SR = S[ldiff_R+1];

      const real_t one_over_dh = area/V;
      const real_t dim_fac = (ndim == 2 ? 0.25 : 0.125);

      // LEFT :
      // Only one neighbor
      if (iiL.is_boundary()) {
        auto pos_bc = pos;
        pos_bc[dir] = min_pos[dir];
        const real_t FL = one_over_dh * getBoundaryHeatFlux(qC, size[dir], dir, pos_bc, true);
        Kokkos::atomic_add(&rhs.at(iCell_rhs, ConsHydroState::VarIndex::Ie_tot), -FL);
      }
      else if (ldiff_L >= 0) {
        PrimHydroState qL; 
        auto pos = cellmetadata.getCellCenter(iCell_Qgroup + offsetm);
        getPrimitiveState<ndim>(Qgroup, iCell_Qgroup + offsetm, qL);

        const real_t TL = compute_temperature(qL);
        auto kappa = 0.5 * (kappaC + compute_kappa(pos, TL));

        const real_t FL = one_over_dh * kappa * (TC - TL) / (SL * size[dir]);
        Kokkos::atomic_add(&rhs.at(iCell_rhs, ConsHydroState::VarIndex::Ie_tot), -FL);

        if (ldiff_L > 0)
          Kokkos::atomic_add(&rhs.at(iiL, ConsHydroState::VarIndex::Ie_tot), FL*dim_fac);
      }

      // RIGHT :
      // Only one neighbor
      if (iiR.is_boundary()) {
        auto pos_bc = pos;
        pos_bc[dir] = max_pos[dir];
        const real_t FR = one_over_dh * getBoundaryHeatFlux(qC, size[dir], dir, pos_bc, false);
        Kokkos::atomic_add(&rhs.at(iCell_rhs, ConsHydroState::VarIndex::Ie_tot), FR);
      }
      else if (ldiff_R >= 0) {
        PrimHydroState qR;
        auto pos = cellmetadata.getCellCenter(iCell_Qgroup + offsetp);
        getPrimitiveState<ndim>(Qgroup, iCell_Qgroup + offsetp, qR);

        const real_t TR = compute_temperature(qR);
        auto kappa = 0.5 * (kappaC + compute_kappa(pos, TR));

        const real_t FR = one_over_dh * kappa * (TR - TC) / (SR * size[dir]);
        Kokkos::atomic_add(&rhs.at(iCell_rhs, ConsHydroState::VarIndex::Ie_tot), FR);

        if (ldiff_R > 0)
          Kokkos::atomic_add(&rhs.at(iiR, ConsHydroState::VarIndex::Ie_tot), -FR*dim_fac);
      }
    };

    compute_thermal_flux(IX);
    compute_thermal_flux(IY);
    if (ndim == 3) 
      compute_thermal_flux(IZ);
  }

private:
  BoundaryConditions bc_manager;

  int ndim;
  real_t gamma0;
  real_t kappa_cst;
  DiffusivityMode diffusivity_mode;
  KappaMode kappa_mode;

  ThermalBoundaryMode bctc_min[3], bctc_max[3];
  real_t bctc_min_val[3], bctc_max_val[3];
  real_t min_pos[3], max_pos[3];

  // Tri-Layer parameters
  real_t tr_thick, z1, z2, K1, K2;
};

}