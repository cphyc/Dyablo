#pragma once

#include "states/State_Ops.h"
#include "HyperbolicPolicy_base.h"
#include "HyperbolicPolicy_Slope.h"
#include "HyperbolicPolicy_BoundaryConditions.h"

namespace dyablo{

/**
 * @brief Structure holding conservative Radiative transfert variables
 **/ 
struct HyperbolicPolicy_RadState {
  enum VarIndex : dyablo::VarIndex
  {
    Ie_rad, 
    Ifx_rad,
    Ify_rad,
    Ifz_rad
  }; 

  static FieldManager getFieldManager()
  {
    return FieldManager( {VarIndex::Ie_rad, VarIndex::Ifx_rad, VarIndex::Ify_rad, VarIndex::Ifz_rad } );
  } 

  real_t e_rad = 0;
  real_t fx_rad = 0;
  real_t fy_rad = 0;
  real_t fz_rad = 0;
};

DECLARE_STATE_TYPE( HyperbolicPolicy_RadState, 4 );
DECLARE_STATE_GET( HyperbolicPolicy_RadState, 0, e_rad );
DECLARE_STATE_GET( HyperbolicPolicy_RadState, 1, fx_rad );
DECLARE_STATE_GET( HyperbolicPolicy_RadState, 2, fy_rad );
DECLARE_STATE_GET( HyperbolicPolicy_RadState, 3, fz_rad );

class HyperbolicPolicy_State_Rad
{
private:
  int ndim;
  using CellIndex = ForeachCell::CellIndex;
  using FieldAccessor = UserData::FieldAccessor;
public:
  using PrimState = HyperbolicPolicy_RadState;
  using ConsState = HyperbolicPolicy_RadState;

  HyperbolicPolicy_State_Rad( ConfigMap& configMap )
  : ndim(configMap.getValue<int>("mesh", "ndim", 3))
  {}

  using ConsVarIndex = ConsState::VarIndex;

  FieldAccessor getUin( UserData& U ) const
  {
    std::vector<FieldAccessor::FieldInfo> Uin_fieldinfo { 
      {"e_rad",   ConsVarIndex::Ie_rad}, 
      {"fx_rad",  ConsVarIndex::Ifx_rad},
      {"fy_rad",  ConsVarIndex::Ify_rad},
      {"fz_rad",  ConsVarIndex::Ifz_rad}
    };

    return U.getAccessor( Uin_fieldinfo );
  }

  FieldAccessor getUout( UserData& U ) const
  {
    std::vector<FieldAccessor::FieldInfo> Uout_fieldinfo { 
      {"e_rad_next",   ConsVarIndex::Ie_rad}, 
      {"fx_rad_next",  ConsVarIndex::Ifx_rad},
      {"fy_rad_next",  ConsVarIndex::Ify_rad},
      {"fz_rad_next",  ConsVarIndex::Ifz_rad} 
    };
    return U.getAccessor( Uout_fieldinfo );
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getConsState( const Array_t& U, const CellIndex& iCell ) const
  {
    ConsState u;
    u.e_rad  = U.at(iCell, ConsState::VarIndex::Ie_rad );
    u.fx_rad = U.at(iCell, ConsState::VarIndex::Ifx_rad );
    u.fy_rad = U.at(iCell, ConsState::VarIndex::Ify_rad );
    u.fz_rad = (ndim == 3 ? U.at(iCell, ConsState::VarIndex::Ifz_rad ) : 0.0);
    return u;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    U.at(iCell, ConsState::VarIndex::Ie_rad) = u.e_rad;
    U.at(iCell, ConsState::VarIndex::Ifx_rad) = u.fx_rad;
    U.at(iCell, ConsState::VarIndex::Ify_rad) = u.fy_rad;
    if (ndim == 3)
      U.at(iCell, ConsState::VarIndex::Ifz_rad) = u.fz_rad;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void atomic_addConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::Ie_rad), u.e_rad);
    Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::Ifx_rad), u.fx_rad);
    Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::Ify_rad), u.fy_rad);
    if (ndim == 3)
      Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::Ifz_rad), u.fz_rad);
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  PrimState getPrimState( const Array_t& Q, const CellIndex& iCell ) const
  {
    return getConsState( Q, iCell );
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setPrimState( const Array_t& Q, const CellIndex& iCell, const PrimState& q ) const
  {
    return setPrimState( Q, iCell, q );
  }

  KOKKOS_INLINE_FUNCTION
  PrimState consToPrim( const ConsState& U ) const
  {
    return U;
  }

  KOKKOS_INLINE_FUNCTION
  ConsState primToCons( const PrimState& Q ) const
  {
    return Q;
  }
};

class HyperbolicPolicy_RiemannSolver_Rad_M1
{
private:
  real_t ctilde_a0;

public:
  using State = HyperbolicPolicy_State_Rad;
  using PrimState = State::PrimState;
  using ConsState = State::ConsState;

  HyperbolicPolicy_RiemannSolver_Rad_M1( ConfigMap& configMap )
  : ctilde_a0(configMap.getValue<real_t>( "cosmology", "ctilde" ) / configMap.getValue<real_t>( "cosmology", "astart" ))
  {}

  HyperbolicPolicy_RiemannSolver_Rad_M1()
  {}

  struct PolicyScalarData{
    real_t aexp;
  };

  static PolicyScalarData getScalarData( const ScalarSimulationData& scalar_data )
  {
    return PolicyScalarData{scalar_data.get<real_t>("aexp")};
  }

  KOKKOS_INLINE_FUNCTION
  ConsState riemann_solver( PrimState qL, PrimState qR, ComponentIndex3D dir, const PolicyScalarData &scalar_data ) const
  {
    qL = swapComponents(qL, dir);
    qR = swapComponents(qR, dir);
    real_t ctilde = this->ctilde_a0 * scalar_data.aexp;
    ConsState flux = riemann_M1(qL, qR, ctilde);
    flux = swapComponents(flux, dir);
    return flux;
  }

private:

  KOKKOS_INLINE_FUNCTION
  PrimState swapComponents(const PrimState &q, ComponentIndex3D comp) const
  {
    switch( comp )
    {
      case IX:
        return q;
      case IY:
        return PrimState{q.e_rad, q.fy_rad, q.fx_rad, q.fz_rad};
      case IZ:
        return PrimState{q.e_rad, q.fz_rad, q.fy_rad, q.fx_rad};
      default:
        DYABLO_ASSERT_KOKKOS_DEBUG(false, "invalid component");
        return PrimState{};
    }
  }

  /* Already implemented : PrimState == ConsState
  KOKKOS_INLINE_FUNCTION
  ConsState swapComponents(const ConsState &u, ComponentIndex3D comp) const;
  {
    ...
  }
  */

  KOKKOS_INLINE_FUNCTION
  ConsState riemann_M1( PrimState qleft, PrimState qright, real_t ctilde ) const
  {
    // RHD part

    // Maximum wave speed
    real_t cmax = ctilde;

    // Conservative variables
    ConsState uleft, uright;

    real_t dl = qleft.e_rad;
    real_t ul = qleft.fx_rad;
    real_t vl = qleft.fy_rad;
    real_t wl = qleft.fz_rad;

    real_t dr = qright.e_rad;
    real_t ur = qright.fx_rad;
    real_t vr = qright.fy_rad;
    real_t wr = qright.fz_rad;

    // Mass density
    uleft.e_rad = dl;//qleft .e_rad;
    uright.e_rad = dr;//qright.e_rad;

    // Normal momentum
    uleft.fx_rad = ul;//qleft .fx_rad;
    uright.fx_rad = ur;//qright.fx_rad;

    // Transverse momentum
    uleft.fy_rad = vl;//qleft .fy_rad;
    uright.fy_rad = vr;//qright.fy_rad;

    uleft.fz_rad = wl;//qleft .fz_rad;
    uright.fz_rad = wr;//qright.fz_rad;

    //===============================
    // Compute left and right fluxes
    //===============================
    ConsState fleft, fright;
    // photon number density
    fleft.e_rad = uleft.fx_rad;
    fright.e_rad = uright.fx_rad;

    auto Eddington = [&](real_t fx, real_t fy, real_t fz, real_t ee, real_t c, int i, int j) {
          real_t c2e = ee * c * c; // 2 flop
          real_t ff = 0.;
          real_t arg, chi, res = 0.;
          real_t n[3];
          n[0] = 0.;
          n[1] = 0.;
          n[2] = 0.;
          if(ee > 0)
          {
            ff = SQRT(fx * fx + fy * fy + fz * fz); // 6 flop
            
            if(ff > 0)
              {
                n[0] = fx / ff;
                n[1] = fy / ff;
                n[2] = fz / ff;
              }
              ff = ff / (c * ee); // 2flop
          }
          arg = FMAX(4. - 3.*ff * ff, 0.); // 4 flop
          chi = FMAX((3. + 4.*ff * ff) / (5. + 2.*SQRT(arg)),1./3.); // 7 flops
          if(i == j) res = (1. - chi) / 2.*c2e; // 1 flops on average
          arg = (3.*chi - 1.) / 2.*c2e;
          res += arg * n[i] * n[j];
          return res;
        };
    
    // Normal momentum
    fleft.fx_rad = Eddington(uleft.fx_rad,uleft.fy_rad,uleft.fz_rad,uleft.e_rad,cmax,0,0);
    fright.fx_rad= Eddington(uright.fx_rad,uright.fy_rad,uright.fz_rad,uright.e_rad,cmax,0,0);

    // Transverse momentum
    fleft.fy_rad = Eddington(uleft.fx_rad,uleft.fy_rad,uleft.fz_rad,uleft.e_rad,cmax,1,0);
    fright.fy_rad= Eddington(uright.fx_rad,uright.fy_rad,uright.fz_rad,uright.e_rad,cmax,1,0);

    fleft.fz_rad = Eddington(uleft.fx_rad,uleft.fy_rad,uleft.fz_rad,uleft.e_rad,cmax,2,0);
    fright.fz_rad= Eddington(uright.fx_rad,uright.fy_rad,uright.fz_rad,uright.e_rad,cmax,2,0);

    //==============================
    // Compute Lax-Friedrich fluxes
    //==============================
    return HALF_F * ( fleft + fright - cmax*(uright - uleft) );
  }

};

class HyperbolicPolicy_Rad_impl
  : public HyperbolicPolicy_State_Rad,
    public HyperbolicPolicy_RiemannSolver_Rad_M1,
    public HyperbolicPolicy_Slope_dynamic<HyperbolicPolicy_State_Rad>,
    public HyperbolicPolicy_BoundaryConditions_PeriodicOnly<HyperbolicPolicy_State_Rad>
{
private:
  using CellIndex     = ForeachCell::CellIndex;
  using CellMetaData  = ForeachCell::CellMetaData;
  using State = HyperbolicPolicy_State_Rad;
public:
  using PrimState = State::PrimState;
  using ConsState = State::ConsState;

  HyperbolicPolicy_Rad_impl( ConfigMap& configMap )
  : HyperbolicPolicy_State_Rad(configMap),
    HyperbolicPolicy_RiemannSolver_Rad_M1(configMap),
    HyperbolicPolicy_Slope_dynamic<HyperbolicPolicy_State_Rad>(configMap),
    HyperbolicPolicy_BoundaryConditions_PeriodicOnly(configMap)
  {}
};

using HyperbolicPolicy_Rad = HyperbolicPolicy_base< HyperbolicPolicy_Rad_impl >;

} //namespace dyablo