#pragma once

#include "states/State_Ops.h"
#include "HyperbolicPolicy_base.h"
#include "HyperbolicPolicy_Slope.h"
#include "HyperbolicPolicy_BoundaryConditions.h"

namespace dyablo{

/**
 * @brief Structure holding conservative MHD variables
 * 
 * @note Important here : p denotes the gas pressure and not the total pressure
 **/ 
struct HyperbolicPolicy_ConsGLMMHDState {
  enum VarIndex : dyablo::VarIndex
  {
    Irho,
    Ie_tot,
    Irho_vx,
    Irho_vy,
    Irho_vz,
    IBx,
    IBy,
    IBz,
    Ipsi
  }; 

  static FieldManager getFieldManager()
  {
    return FieldManager( {VarIndex::Irho, VarIndex::Ie_tot, VarIndex::Irho_vx, VarIndex::Irho_vy, VarIndex::Irho_vz, VarIndex::IBx, VarIndex::IBy, VarIndex::IBz, VarIndex::Ipsi } );
  } 

  real_t rho = 0;
  real_t e_tot = 0;
  real_t rho_u = 0;
  real_t rho_v = 0;
  real_t rho_w = 0;
  real_t Bx = 0;
  real_t By = 0;
  real_t Bz = 0;
  real_t psi = 0;
};

DECLARE_STATE_TYPE( HyperbolicPolicy_ConsGLMMHDState, 9 );
DECLARE_STATE_GET( HyperbolicPolicy_ConsGLMMHDState, 0, rho );
DECLARE_STATE_GET( HyperbolicPolicy_ConsGLMMHDState, 1, e_tot );
DECLARE_STATE_GET( HyperbolicPolicy_ConsGLMMHDState, 2, rho_u );
DECLARE_STATE_GET( HyperbolicPolicy_ConsGLMMHDState, 3, rho_v );
DECLARE_STATE_GET( HyperbolicPolicy_ConsGLMMHDState, 4, rho_w );
DECLARE_STATE_GET( HyperbolicPolicy_ConsGLMMHDState, 5, Bx );
DECLARE_STATE_GET( HyperbolicPolicy_ConsGLMMHDState, 6, By );
DECLARE_STATE_GET( HyperbolicPolicy_ConsGLMMHDState, 7, Bz );
DECLARE_STATE_GET( HyperbolicPolicy_ConsGLMMHDState, 8, psi );

/**
 * @brief Structure holding primitive MHD variables
 */
struct HyperbolicPolicy_PrimGLMMHDState {
  enum VarIndex : dyablo::VarIndex
  {
    Irho,
    Ip,
    Iu,
    Iv,
    Iw,
    IBx,
    IBy,
    IBz,
    Ipsi
  };

  static FieldManager getFieldManager()
  {
    return FieldManager( {VarIndex::Irho, VarIndex::Ip, VarIndex::Iu, VarIndex::Iv, VarIndex::Iw, VarIndex::IBx, VarIndex::IBy, VarIndex::IBz, VarIndex::Ipsi } );
  }

  real_t rho = 0;
  real_t p = 0;
  real_t u = 0;
  real_t v = 0;
  real_t w = 0;
  real_t Bx = 0;
  real_t By = 0;
  real_t Bz = 0;
  real_t psi = 0;
};

DECLARE_STATE_TYPE( HyperbolicPolicy_PrimGLMMHDState, 9 );
DECLARE_STATE_GET( HyperbolicPolicy_PrimGLMMHDState, 0, rho );
DECLARE_STATE_GET( HyperbolicPolicy_PrimGLMMHDState, 1, p );
DECLARE_STATE_GET( HyperbolicPolicy_PrimGLMMHDState, 2, u );
DECLARE_STATE_GET( HyperbolicPolicy_PrimGLMMHDState, 3, v );
DECLARE_STATE_GET( HyperbolicPolicy_PrimGLMMHDState, 4, w );
DECLARE_STATE_GET( HyperbolicPolicy_PrimGLMMHDState, 5, Bx );
DECLARE_STATE_GET( HyperbolicPolicy_PrimGLMMHDState, 6, By );
DECLARE_STATE_GET( HyperbolicPolicy_PrimGLMMHDState, 7, Bz );
DECLARE_STATE_GET( HyperbolicPolicy_PrimGLMMHDState, 8, psi );


class HyperbolicPolicy_State_GLMMHD
{
private:
  int ndim;
  real_t gamma0;

  using CellIndex = ForeachCell::CellIndex;
  using FieldAccessor = UserData::FieldAccessor;
public:
  using PrimState = HyperbolicPolicy_PrimGLMMHDState;
  using ConsState = HyperbolicPolicy_ConsGLMMHDState;

  HyperbolicPolicy_State_GLMMHD( ConfigMap& configMap )
  : ndim(configMap.getValue<int>("mesh", "ndim", 3)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4))
  {}

  using ConsVarIndex = HyperbolicPolicy_ConsGLMMHDState::VarIndex;

  FieldAccessor getUin( UserData& U ) const
  {
    std::vector<FieldAccessor::FieldInfo> Uin_fieldinfo { 
      {"rho",     ConsVarIndex::Irho}, 
      {"e_tot",   ConsVarIndex::Ie_tot},
      {"rho_vx",  ConsVarIndex::Irho_vx},
      {"rho_vy",  ConsVarIndex::Irho_vy},
      {"rho_vz",  ConsVarIndex::Irho_vz},
      {"Bx",      ConsVarIndex::IBx},
      {"By",      ConsVarIndex::IBy},
      {"Bz",      ConsVarIndex::IBz},
      {"psi",     ConsVarIndex::Ipsi}
    };

    return U.getAccessor( Uin_fieldinfo );
  }

  FieldAccessor getUout( UserData& U ) const
  {
    std::vector<FieldAccessor::FieldInfo> Uout_fieldinfo { 
      {"rho_next",     ConsVarIndex::Irho}, 
      {"e_tot_next",   ConsVarIndex::Ie_tot},
      {"rho_vx_next",  ConsVarIndex::Irho_vx},
      {"rho_vy_next",  ConsVarIndex::Irho_vy},
      {"rho_vz_next",  ConsVarIndex::Irho_vz},
      {"Bx_next",      ConsVarIndex::IBx},
      {"By_next",      ConsVarIndex::IBy},
      {"Bz_next",      ConsVarIndex::IBz},
      {"psi_next",     ConsVarIndex::Ipsi}
    };
    return U.getAccessor( Uout_fieldinfo );
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getConsState( const Array_t& U, const CellIndex& iCell ) const
  {
    ConsState u;
    u.rho   = U.at(iCell, ConsState::VarIndex::Irho );
    u.e_tot = U.at(iCell, ConsState::VarIndex::Ie_tot );
    u.rho_u = U.at(iCell, ConsState::VarIndex::Irho_vx );
    u.rho_v = U.at(iCell, ConsState::VarIndex::Irho_vy );
    u.rho_w = (ndim == 3 ? U.at(iCell, ConsState::VarIndex::Irho_vz ) : 0.0);
    u.Bx    = U.at(iCell, ConsState::VarIndex::IBx);
    u.By    = U.at(iCell, ConsState::VarIndex::IBy);
    u.Bz    = U.at(iCell, ConsState::VarIndex::IBz);
    u.psi   = U.at(iCell, ConsState::VarIndex::Ipsi);
    return u;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    U.at(iCell, ConsState::VarIndex::Irho) = u.rho;
    U.at(iCell, ConsState::VarIndex::Ie_tot) = u.e_tot;
    U.at(iCell, ConsState::VarIndex::Irho_vx) = u.rho_u;
    U.at(iCell, ConsState::VarIndex::Irho_vy) = u.rho_v;
    if (ndim == 3)
      U.at(iCell, ConsState::VarIndex::Irho_vz) = u.rho_w;
    U.at(iCell, ConsState::VarIndex::IBx) = u.Bx;
    U.at(iCell, ConsState::VarIndex::IBy) = u.By;
    U.at(iCell, ConsState::VarIndex::IBz) = u.Bz;
    U.at(iCell, ConsState::VarIndex::Ipsi) = u.psi;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void atomic_addConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::Irho), u.rho);
    Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::Ie_tot), u.e_tot);
    Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::Irho_vx), u.rho_u);
    Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::Irho_vy), u.rho_v);
    if (ndim == 3)
      Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::Irho_vz), u.rho_w);
    Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::IBx), u.Bx);
    Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::IBy), u.By);
    Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::IBz), u.Bz);
    Kokkos::atomic_add(&U.at(iCell, ConsState::VarIndex::Ipsi), u.psi);
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  PrimState getPrimState( const Array_t& Q, const CellIndex& iCell ) const
  {
    PrimState q;
    q.rho = Q.at(iCell, PrimState::VarIndex::Irho );
    q.p   = Q.at(iCell, PrimState::VarIndex::Ip );
    q.u   = Q.at(iCell, PrimState::VarIndex::Iu );
    q.v   = Q.at(iCell, PrimState::VarIndex::Iv );
    q.w   = (ndim == 3 ? Q.at(iCell, PrimState::VarIndex::Iw ) : 0.0);
    q.Bx  = Q.at(iCell, PrimState::VarIndex::IBx );
    q.By  = Q.at(iCell, PrimState::VarIndex::IBy );
    q.Bz  = Q.at(iCell, PrimState::VarIndex::IBz );
    q.psi = Q.at(iCell, PrimState::VarIndex::Ipsi );
    return q;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setPrimState( const Array_t& Q, const CellIndex& iCell, const PrimState& q ) const
  {
    Q.at(iCell, PrimState::VarIndex::Irho) = q.rho;
    Q.at(iCell, PrimState::VarIndex::Ip) = q.p;
    Q.at(iCell, PrimState::VarIndex::Iu) = q.u;
    Q.at(iCell, PrimState::VarIndex::Iv) = q.v;
    if (ndim == 3)
      Q.at(iCell, PrimState::VarIndex::Iw) = q.w;
    Q.at(iCell, PrimState::VarIndex::IBx) = q.Bx;
    Q.at(iCell, PrimState::VarIndex::IBy) = q.By;
    Q.at(iCell, PrimState::VarIndex::IBz) = q.Bz;
    Q.at(iCell, PrimState::VarIndex::Ipsi) = q.psi;
  }

  KOKKOS_INLINE_FUNCTION
  PrimState consToPrim( const ConsState& U ) const
  {
    real_t gamma0 = this->gamma0;

    const real_t Ek = 0.5 * (U.rho_u*U.rho_u+U.rho_v*U.rho_v+U.rho_w*U.rho_w)/U.rho;
    const real_t hB2 = 0.5 * (U.Bx*U.Bx+U.By*U.By+U.Bz*U.Bz);
    const real_t p = (U.e_tot - Ek - hB2) * (gamma0-1.0);
    return {U.rho, 
            p, 
            U.rho_u/U.rho, 
            U.rho_v/U.rho, 
            (ndim == 3 ? U.rho_w/U.rho : 0.0),
            U.Bx,
            U.By,
            U.Bz,
            U.psi};
  }

  KOKKOS_INLINE_FUNCTION
  ConsState primToCons( const PrimState& Q ) const
  {
    real_t gamma0 = this->gamma0;

    const real_t Ek = 0.5 * Q.rho * (Q.u*Q.u+Q.v*Q.v+Q.w*Q.w);
    const real_t hB2 = 0.5 * (Q.Bx*Q.Bx+Q.By*Q.By+Q.Bz*Q.Bz);
    const real_t E  = Ek + hB2 + Q.p / (gamma0-1.0);
    return {Q.rho, 
            E, 
            Q.rho*Q.u, 
            Q.rho*Q.v, 
            (ndim ==3 ? Q.rho*Q.w : 0.0),
            Q.Bx,
            Q.By,
            Q.Bz,
            Q.psi};
  }
};

class HyperbolicPolicy_RiemannSolver_GLMMHD_hlld
{
private:
  struct Rparams {
    real_t gamma0;
    real_t smallr;
    real_t smallp;
    real_t smalle;
    real_t c_h;
  } rparams;

public:
  using State = HyperbolicPolicy_State_GLMMHD;
  using PrimState = State::PrimState;
  using ConsState = State::ConsState;

  HyperbolicPolicy_RiemannSolver_GLMMHD_hlld( ConfigMap& configMap )
  : rparams( 
    {
      .gamma0 = configMap.getValue<real_t>("hydro", "gamma0", 1.4),
      .smallr = configMap.getValue<real_t>("hydro", "smallr", 1e-10),
      .smallp = configMap.getValue<real_t>("hydro", "smallp", 1e-10),
      .smalle = configMap.getValue<real_t>("hydro", "smalle", 1e-5)
    })
  {
    // Computing the value of c_h according to the smallest cell size and hydro-cfl
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
    rparams.c_h = 0.5*cfl * min_dh;
  }

  template < typename ScalarData_t >
  KOKKOS_INLINE_FUNCTION
  ConsState riemann_solver( PrimState qL, PrimState qR, ComponentIndex3D dir, const ScalarData_t &scalar_data ) const
  {
    qL = swapComponents(qL, dir);
    qR = swapComponents(qR, dir);
    ConsState flux = riemann_hlld(qL, qR, scalar_data);
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
        return PrimState{q.rho, q.p, q.v, q.u, q.w, q.By, q.Bx, q.Bz, q.psi};
      case IZ:
        return PrimState{q.rho, q.p, q.w, q.v, q.u, q.Bz, q.By, q.Bx, q.psi};
      default:
        DYABLO_ASSERT_KOKKOS_DEBUG(false, "invalid component");
        return PrimState{};
    }
  }

  KOKKOS_INLINE_FUNCTION
  ConsState swapComponents(const ConsState &u, ComponentIndex3D comp) const
  {
    switch( comp )
    {
      case IX:
        return u;
      case IY:
        return ConsState{u.rho, u.e_tot, u.rho_v, u.rho_u, u.rho_w, u.By, u.Bx, u.Bz, u.psi};
      case IZ:
        return ConsState{u.rho, u.e_tot, u.rho_w, u.rho_v, u.rho_u, u.Bz, u.By, u.Bx, u.psi};
      default:
        DYABLO_ASSERT_KOKKOS_DEBUG(false, "invalid component");
        return ConsState{};
    }
  }

  template < typename ScalarData_t >
  KOKKOS_INLINE_FUNCTION
  ConsState riemann_hlld( PrimState qleft, PrimState qright, const ScalarData_t &scalar_data ) const
  {
    real_t gamma0 = rparams.gamma0;
    real_t smallr = rparams.smallr;
    real_t smallp = rparams.smallp;
    real_t smalle = rparams.smalle;
    real_t c_h    = rparams.c_h / scalar_data.dt;

    // GLM MHD evaluation of Bx
    const real_t Bx   = 0.5 * (qleft.Bx + qright.Bx) - 0.5/c_h * (qright.psi - qleft.psi);
    const real_t psi  = 0.5*(qleft.psi + qright.psi) - c_h * 0.5 * (qright.Bx - qleft.Bx); 
    const real_t Bsgn = (Bx < 0.0 ? -1.0 : 1.0);

    // Left variables
    real_t rl = fmax(qleft.rho, smallr);
    real_t pl = fmax(qleft.p, rl*smallp);
    real_t ul =      qleft.u;
    real_t vl =      qleft.v;
    real_t wl =      qleft.w;
    real_t Byl =     qleft.By;
    real_t Bzl =     qleft.Bz;
    real_t B2l =     Bx*Bx+Byl*Byl+Bzl*Bzl;
    real_t pTl =     pl + 0.5 * B2l;
    real_t El  =     pl / (gamma0-1.0) + 0.5*rl*(ul*ul+vl*vl+wl*wl) + 0.5*B2l;


    // Right variables
    real_t rr = fmax(qright.rho, smallr);
    real_t pr = fmax(qright.p, rr*smallp);
    real_t ur =      qright.u;
    real_t vr =      qright.v;
    real_t wr =      qright.w;
    real_t Byr =     qright.By;
    real_t Bzr =     qright.Bz;
    real_t B2r =     Bx*Bx+Byr*Byr+Bzr*Bzr;
    real_t pTr =     pr + 0.5 * B2r;
    real_t Er  =     pr / (gamma0-1.0) + 0.5*rr*(ur*ur+vr*vr+wr*wr) + 0.5*B2r;

    auto computeFastMagnetoAcousticSpeed = [&](const PrimState &q) {
      const real_t gp = gamma0 * q.p;
      const real_t B2 = Bx*Bx + q.By*q.By + q.Bz*q.Bz;
      
      return sqrt(0.5 * (gp + B2 + sqrt((gp + B2)*(gp + B2) - 4.0*gp*Bx*Bx)) / q.rho);
    };
    
    
    real_t cfl = computeFastMagnetoAcousticSpeed(qleft);
    real_t cfr = computeFastMagnetoAcousticSpeed(qright);
    
    // HLL Wave speed
    real_t Sl = fmin(ul, ur) - fmax(cfl, cfr);
    real_t Sr = fmax(ul, ur) + fmax(cfl, cfr);

    // Lagrangian speed of sound
    const real_t rCl = rl*(ul-Sl);
    const real_t rCr = rr*(Sr-ur);

    // Entropy wave speed
    const real_t uS = (rCr*ur + rCl*ul - pTr + pTl) / (rCr+rCl);
    
    // Single Star state
    const real_t pTS = (rCr*pTl + rCl*pTr - rCr*rCl*(ur-ul)) / (rCr+rCl); 

    // Single star densities
    const real_t rlS = rl * (Sl-ul)/(Sl-uS);
    const real_t rrS = rr * (Sr-ur)/(Sr-uS);

    // Single star velocities
    const real_t econvl = rl*(Sl-ul)*(Sl-uS)-Bx*Bx;
    const real_t econvr = rr*(Sr-ur)*(Sr-uS)-Bx*Bx;

    const real_t uconvl = (uS-ul) / econvl;
    const real_t uconvr = (uS-ur) / econvr;
    const real_t Bconvl = (rCl*rCl/rl - Bx*Bx) / econvl;
    const real_t Bconvr = (rCr*rCr/rr - Bx*Bx) / econvr;

    real_t vlS, vrS, wlS, wrS, BylS, ByrS, BzlS, BzrS;

    // Switching to two state on the left ?
    if (FABS(econvl) < smalle*Bx*Bx) {
      vlS = vl;
      wlS = wl;
      BylS = Byl;
      BzlS = Bzl;
    }
    else {
      vlS = vl - Bx*Byl * uconvl;
      wlS = wl - Bx*Bzl * uconvl;
      BylS = Byl * Bconvl;
      BzlS = Bzl * Bconvl;
    }

    // Switching to two state on the right ?
    if (FABS(econvr) < smalle*Bx*Bx) {
      vrS  = vr;
      wrS  = wr;
      ByrS = Byr;
      BzrS = Bzr;
    }
    else {
      vrS = vr - Bx*Byr * uconvr;
      wrS = wr - Bx*Bzr * uconvr;
      ByrS = Byr * Bconvr;
      BzrS = Bzr * Bconvr;
    }

    // Single star total energy
    const real_t udotBl = ul*Bx+vl*Byl+wl*Bzl;
    const real_t udotBr = ur*Bx+vr*Byr+wr*Bzr;
    const real_t uSdotBSl = uS*Bx+vlS*BylS+wlS*BzlS;
    const real_t uSdotBSr = uS*Bx+vrS*ByrS+wrS*BzrS;

    const real_t ElS = ((Sl-ul)*El - pTl*ul + pTS*uS + Bx*(udotBl - uSdotBSl)) / (Sl-uS);
    const real_t ErS = ((Sr-ur)*Er - pTr*ur + pTS*uS + Bx*(udotBr - uSdotBSr)) / (Sr-uS);

    // Alfven wave speeds
    const real_t srlS = sqrt(rlS);
    const real_t srrS = sqrt(rrS);
    const real_t SlS = uS - fabs(Bx) / srlS;
    const real_t SrS = uS + fabs(Bx) / srrS;

    // Double Star state
    const real_t den_fac = 1.0 / (srlS + srrS);
    const real_t vSS = (srlS*vlS + srrS*vrS + (ByrS-BylS)*Bsgn) * den_fac;
    const real_t wSS = (srlS*wlS + srrS*wrS + (BzrS-BzlS)*Bsgn) * den_fac;
    const real_t BySS = (srlS*ByrS + srrS*BylS + srlS*srrS*(vrS-vlS)*Bsgn) * den_fac;
    const real_t BzSS = (srlS*BzrS + srrS*BzlS + srlS*srrS*(wrS-wlS)*Bsgn) * den_fac; 

    const real_t uSSdotBSS = uS*Bx + vSS*BySS + wSS*BzSS;

    const real_t ElSS = ElS - srlS * (uSdotBSl - uSSdotBSS) * Bsgn;
    const real_t ErSS = ErS + srrS * (uSdotBSr - uSSdotBSS) * Bsgn;

    // Lambda to compute a flux from a primitive state
    auto computeFlux = [&](const PrimState &q,  const real_t e_tot) -> ConsState {
      ConsState res{};

      res.rho   = q.rho * q.u;
      res.rho_u = q.rho * q.u * q.u + q.p - q.Bx*q.Bx;
      res.rho_v = q.rho * q.u * q.v - q.Bx*q.By;
      res.rho_w = q.rho * q.u * q.w - q.Bx*q.Bz;
      res.Bx    = q.psi;
      res.By    = q.By*q.u - q.Bx*q.v;
      res.Bz    = q.Bz*q.u - q.Bx*q.w;
      res.e_tot = (e_tot + q.p) * q.u - q.Bx*(q.Bx*q.u+q.By*q.v+q.Bz*q.w);
      res.psi   = sqrt(c_h) * Bx; // Warning : Bx not q.Bx !!!
      
      return res;
    };

    // Disjunction of cases
    PrimState q; 
    real_t e_tot;
    if (Sl > 0.0) { // qL
      q = qleft;
      e_tot = El;
      q.p = pTl;
    }
    else if (SlS > 0.0) { // qL*
      q.rho = rlS;
      q.u   = uS;
      q.v   = vlS;
      q.w   = wlS;
      q.Bx  = Bx;
      q.By  = BylS;
      q.Bz  = BzlS;

      q.p = pTS;
      e_tot = ElS;
    }
    else if (uS > 0.0) { // qL**
      q.rho = rlS;
      q.u   = uS;
      q.v   = vSS;
      q.w   = wSS;
      q.Bx  = Bx;
      q.By  = BySS;
      q.Bz  = BzSS;

      q.p   = pTS;
      e_tot = ElSS;
    }
    else if (SrS > 0.0) { // qR**
      q.rho = rrS;
      q.u   = uS;
      q.v   = vSS;
      q.w   = wSS;
      q.Bx  = Bx;
      q.By  = BySS;
      q.Bz  = BzSS;

      q.p   = pTS;
      e_tot = ErSS;
    }
    else if (Sr > 0.0) { // qR*
      q.rho = rrS;
      q.u   = uS;
      q.v   = vrS;
      q.w   = wrS;
      q.Bx  = Bx;
      q.By  = ByrS;
      q.Bz  = BzrS;

      q.p = pTS;
      e_tot = ErS;
    }
    else { // SR < 0.0; qR
      q = qright;
      e_tot = Er;
      q.p = pTr;
    }
    q.psi = psi;

    return computeFlux(q, e_tot);
  }

};

class HyperbolicPolicy_BoundaryConditions_GLMMHD
{
private:
  using HyperbolicPolicy_State = HyperbolicPolicy_State_GLMMHD;
  using CellIndex     = ForeachCell::CellIndex;
  using CellMetaData  = ForeachCell::CellMetaData;
  using offset_t      = CellIndex::offset_t;

  Kokkos::Array<BoundaryConditionType, 3> bc_min, bc_max;
  Kokkos::Array<MagneticBoundaryConditionType, 3> bcmag_min, bcmag_max;
  struct Rparams {
    real_t gamma0;
    real_t smallr;
    real_t smallp;
    real_t smallc;
    real_t psi_out;
    real_t c_h;
  } rparams;

public:
  using PrimState = typename HyperbolicPolicy_State::PrimState;
  using ConsState = typename HyperbolicPolicy_State::ConsState;

  static std::string name() {return "default";}

  HyperbolicPolicy_BoundaryConditions_GLMMHD( ConfigMap& configMap )
  : bc_min{
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING),
    },
    bc_max{
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING)
    },
    bcmag_min{
      configMap.getValue<MagneticBoundaryConditionType>("mesh","magnetic_boundary_type_xmin", BCMAG_SAME_AS_HYDRO),
      configMap.getValue<MagneticBoundaryConditionType>("mesh","magnetic_boundary_type_ymin", BCMAG_SAME_AS_HYDRO),
      configMap.getValue<MagneticBoundaryConditionType>("mesh","magnetic_boundary_type_zmin", BCMAG_SAME_AS_HYDRO),
    },
    bcmag_max{
      configMap.getValue<MagneticBoundaryConditionType>("mesh","magnetic_boundary_type_xmax", BCMAG_SAME_AS_HYDRO),
      configMap.getValue<MagneticBoundaryConditionType>("mesh","magnetic_boundary_type_ymax", BCMAG_SAME_AS_HYDRO),
      configMap.getValue<MagneticBoundaryConditionType>("mesh","magnetic_boundary_type_zmax", BCMAG_SAME_AS_HYDRO)
    },
    rparams( 
    {
      configMap.getValue<real_t>("hydro", "gamma0", 1.4),
      configMap.getValue<real_t>("hydro", "smallr", 1e-10),
      configMap.getValue<real_t>("hydro", "smallp", 1e-10),
      configMap.getValue<real_t>("hydro", "smallc", 1e-10),
      configMap.getValue<real_t>("hydro", "psi_out", 0.0)
    })
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
    rparams.c_h = 0.5*cfl * min_dh;
  }

  template < typename Array_t, typename Policy_t, typename ScalarData_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue_impl( const Policy_t      &policy, 
                                   const Array_t       &U, 
                                   const CellIndex     &iCell_boundary, 
                                   const CellMetaData  &metadata, 
                                   const ScalarData_t  &scalar_data) const 
  {
    CellIndex iCell_inside;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_inside, offset);

    auto sign = [](int x){return (x>0)-(x<0);};

    CellIndex::offset_t symmetric_offset {
      (int16_t)(-offset[IX] + sign(offset[IX])), 
      (int16_t)(-offset[IY] + sign(offset[IY])), 
      (int16_t)(-offset[IZ] + sign(offset[IZ]))
    }; 

    CellIndex iCell_sym = iCell_inside.getNeighbor(symmetric_offset);
    ConsState u_sym = policy.getConsState( U, iCell_sym );    
    ConsState res = u_sym;

    if ( (offset[IX] > 0 && bc_max[IX] == BC_REFLECTING)
      || (offset[IX] < 0 && bc_min[IX] == BC_REFLECTING) )
    {
        res.rho_u = -u_sym.rho_u;
        res.Bx    = -u_sym.Bx;
    }
    if ( (offset[IY] > 0 && bc_max[IY] == BC_REFLECTING)
      || (offset[IY] < 0 && bc_min[IY] == BC_REFLECTING) )
    {
        res.rho_v = -u_sym.rho_v;
        res.By    = -u_sym.By;
    }
    if ( (offset[IZ] > 0 && bc_max[IZ] == BC_REFLECTING)
      || (offset[IZ] < 0 && bc_min[IZ] == BC_REFLECTING) )
    {
        res.rho_w = -u_sym.rho_w;
        res.Bz    = -u_sym.Bz;
    }

    // Now defining values for boundary conditions
    if ( (offset[IX] > 0 && bcmag_max[IX] == BCMAG_NORMAL_FIELD)
      || (offset[IX] < 0 && bcmag_min[IX] == BCMAG_NORMAL_FIELD) )
    {
      PrimState q = policy.consToPrim(res);
      q.By = 0.0;
      q.Bz = 0.0;
      res = policy.primToCons(q); // We go through consToPrim -> primToCons to correctly repercuss the variation on the energy
    }
    else if ( (offset[IX] > 0 && bcmag_max[IX] == BCMAG_PERFECT_CONDUCTOR)
           || (offset[IX] < 0 && bcmag_min[IX] == BCMAG_PERFECT_CONDUCTOR) )
    {
      PrimState q = policy.consToPrim(res);
      q.Bx = 0.0;
      res = policy.primToCons(q);
    }

    if ( (offset[IY] > 0 && bcmag_max[IY] == BCMAG_NORMAL_FIELD)
      || (offset[IY] < 0 && bcmag_min[IY] == BCMAG_NORMAL_FIELD) )
    {
      PrimState q = policy.consToPrim(res);
      q.Bx = 0.0;
      q.Bz = 0.0;
      res = policy.primToCons(q); 
    }
    else if ( (offset[IY] > 0 && bcmag_max[IY] == BCMAG_PERFECT_CONDUCTOR)
           || (offset[IY] < 0 && bcmag_min[IY] == BCMAG_PERFECT_CONDUCTOR) )
    {
      PrimState q = policy.consToPrim(res);
      q.By = 0.0;
      res = policy.primToCons(q);
    }

    if ( (offset[IZ] > 0 && bcmag_max[IZ] == BCMAG_NORMAL_FIELD)
      || (offset[IZ] < 0 && bcmag_min[IZ] == BCMAG_NORMAL_FIELD) )
    {
      PrimState q = policy.consToPrim(res);
      q.Bx = 0.0;
      q.By = 0.0;
      res = policy.primToCons(q); 
    }
    else if ( (offset[IZ] > 0 && bcmag_max[IZ] == BCMAG_PERFECT_CONDUCTOR)
           || (offset[IZ] < 0 && bcmag_min[IZ] == BCMAG_PERFECT_CONDUCTOR) )
    {
      PrimState q = policy.consToPrim(res);
      q.Bz = 0.0;
      res = policy.primToCons(q);
    }

    res.psi = rparams.psi_out;

    return res;
  }

  template < typename Array_t, typename Policy_t, typename ScalarData_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux_impl( const Policy_t      &policy, 
                                  const Array_t       &U, 
                                  const CellIndex     &iCell_boundary, 
                                  const CellMetaData  &metadata, 
                                  const ScalarData_t  &scalar_data) const 
  {
    CellIndex iCell_ref;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_ref, offset);

    ConsState u_in = policy.getConsState( U, iCell_ref );
    PrimState q_in = policy.consToPrim(u_in);

    
    bool dir_IX = offset[IX] == -1 || offset[IX] == 1;
    bool dir_IY = offset[IY] == -1 || offset[IY] == 1;
    bool dir_IZ = offset[IZ] == -1 || offset[IZ] == 1;
    DYABLO_ASSERT_KOKKOS_DEBUG( (int)dir_IX + (int)dir_IY + (int)dir_IZ == 1
                              , "offset is not compatible with getBoundaryFlux" );

    ComponentIndex3D dir = IZ;
    if( dir_IX )
      dir = IX;
    else if( dir_IY )
      dir = IY;
    else if( dir_IZ )
      dir = IZ;
    else
      DYABLO_ASSERT_KOKKOS_DEBUG(false, "Internal error! Should not happen");

    bool reflecting = (offset[dir] > 0 && bc_max[dir] == BC_REFLECTING)
                  ||  (offset[dir] < 0 && bc_min[dir] == BC_REFLECTING);
    bool absorbing  = (offset[dir] > 0 && bc_max[dir] == BC_ABSORBING)
                  ||  (offset[dir] < 0 && bc_min[dir] == BC_ABSORBING);
    bool mag_perfect_conductor = (offset[dir] > 0 && bcmag_max[dir] == BCMAG_PERFECT_CONDUCTOR)
                              || (offset[dir] < 0 && bcmag_min[dir] == BCMAG_PERFECT_CONDUCTOR);
    bool mag_normal_field      = (offset[dir] > 0 && bcmag_max[dir] == BCMAG_NORMAL_FIELD)
                              || (offset[dir] < 0 && bcmag_min[dir] == BCMAG_NORMAL_FIELD);
     
    real_t v_in[3] = {q_in.u, q_in.v, q_in.w};
    real_t v_normal = v_in[dir];

    ConsState flux_out {};
    /**
     * In the reflecting case, the values in the "ghosts" are supposed to be reflecting the
     * ones inside the domain, hence reconstruction yields u_norm = 0 at the boundary, simplifying
     * the calculation of the flux to only the pressure gradient term in the flux.
     */
    if( reflecting )
    {
      flux_out.rho_u = ((dir==IX) ? q_in.p : 0);
      flux_out.rho_v = ((dir==IY) ? q_in.p : 0);
      flux_out.rho_w = ((dir==IZ) ? q_in.p : 0);
    }
    /**
     * In the absorbing case, the values in the ghosts are supposed to be interpolated from the ones 
     * inside the domain to provide a null gradient through the boundary. Hence we can take the
     * reconstructed value at the boundary as the riemann-problem solution.
     */
    else if( absorbing )
    {
      real_t f_rho = q_in.rho*v_normal;

      flux_out.rho = f_rho;
      flux_out.rho_u = f_rho*q_in.u + ((dir==IX) ? q_in.p : 0);
      flux_out.rho_v = f_rho*q_in.v + ((dir==IY) ? q_in.p : 0);
      flux_out.rho_w = f_rho*q_in.w + ((dir==IZ) ? q_in.p : 0);
      flux_out.e_tot = (q_in.p + u_in.e_tot) * v_normal;
    }


    /**
     * Magnetic boundaries
     * 
     * Perfect conductor : Bz = 0
     */
    if ( mag_perfect_conductor )
    {
      const real_t c_h = rparams.c_h / scalar_data.dt;
      const real_t Bx  = 0.5 * q_in.Bx - 0.5 / c_h * (rparams.psi_out - q_in.psi);
      flux_out.Bx  = 0.5*(q_in.psi + rparams.psi_out); // Should that be 0 ?
      flux_out.By  = q_in.By * q_in.u;
      flux_out.Bz  = q_in.Bz * q_in.u;
      flux_out.psi = c_h*c_h*Bx;
    }
    /**
     * Normal field : Bx=By=0
     */
    else if ( mag_normal_field )
    {
      const real_t c_h = rparams.c_h / scalar_data.dt;
      const real_t Bx  = q_in.Bx - 0.5 / c_h * (rparams.psi_out - q_in.psi); // 0.5 * (2*Bx)
      flux_out.Bx  = 0.5*(q_in.psi + rparams.psi_out) - c_h * 0.5 * q_in.Bx;
      flux_out.By  = q_in.By * q_in.u;
      flux_out.Bz  = q_in.Bz * q_in.u;
      flux_out.psi = c_h*c_h*Bx;
    }

    return flux_out;
  }
};

class HyperbolicPolicy_GLMMHD_impl
  : public HyperbolicPolicy_State_GLMMHD,
    public HyperbolicPolicy_RiemannSolver_GLMMHD_hlld,
    public HyperbolicPolicy_Slope_dynamic<HyperbolicPolicy_State_GLMMHD>,
    public HyperbolicPolicy_BoundaryConditions_GLMMHD
{
private:
  using CellIndex     = ForeachCell::CellIndex;
  using CellMetaData  = ForeachCell::CellMetaData;
  using State = HyperbolicPolicy_State_GLMMHD;
public:
  using PrimState = State::PrimState;
  using ConsState = State::ConsState;

  HyperbolicPolicy_GLMMHD_impl( ConfigMap& configMap )
  : HyperbolicPolicy_State_GLMMHD(configMap),
    HyperbolicPolicy_RiemannSolver_GLMMHD_hlld(configMap),
    HyperbolicPolicy_Slope_dynamic<HyperbolicPolicy_State_GLMMHD>(configMap),
    HyperbolicPolicy_BoundaryConditions_GLMMHD(configMap)
  {}

  struct PolicyScalarData {
    real_t dt;
  };

  static PolicyScalarData getScalarData( const ScalarSimulationData& scalar_data )
  {
    return {
      scalar_data.get<real_t>("dt")
    };
  }

  template < typename Array_t, typename Policy_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux( const Policy_t &policy, const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata, const PolicyScalarData &scalar_data) const
  {
    return HyperbolicPolicy_BoundaryConditions_GLMMHD::getBoundaryFlux_impl(policy,U,iCell_boundary,metadata,scalar_data);
  }

  template < typename Array_t, typename Policy_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Policy_t &policy, const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata, const PolicyScalarData &scalar_data ) const 
  {
    return HyperbolicPolicy_BoundaryConditions_GLMMHD::getBoundaryValue_impl(policy,U,iCell_boundary,metadata,scalar_data);
  }
};

using HyperbolicPolicy_GLMMHD = HyperbolicPolicy_base< HyperbolicPolicy_GLMMHD_impl >;

} //namespace dyablo