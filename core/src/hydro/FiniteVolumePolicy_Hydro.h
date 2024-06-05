#pragma once

#include "states/State_hydro.h"
#include "FiniteVolumePolicy_base.h"
#include "FiniteVolumePolicy_Slope.h"

namespace dyablo{

class FiniteVolumePolicy_State_Hydro
{
private:
  int ndim;
  real_t gamma0;

  using CellIndex = ForeachCell::CellIndex;
  using FieldAccessor = UserData::FieldAccessor;
public:
  using PrimState = PrimHydroState;
  using ConsState = ConsHydroState;

  FiniteVolumePolicy_State_Hydro( ConfigMap& configMap )
  : ndim(configMap.getValue<int>("mesh", "ndim", 3)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4))
  {}

  FieldAccessor getUin( UserData& U ) const
  {
    return U.getAccessor( ConsState::getFieldsInfo() );
  }

  FieldAccessor getUout( UserData& U ) const
  {
    auto fields_info_next = ConsState::getFieldsInfo();
    for( auto& p : fields_info_next )
      p.name += "_next";
    return U.getAccessor( fields_info_next );
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getConsState( const Array_t& U, const CellIndex& iCell ) const
  {
    ConsState u;
    u.rho   = U.at(iCell, ConsHydroState::VarIndex::Irho );
    u.e_tot = U.at(iCell, ConsHydroState::VarIndex::Ie_tot );
    u.rho_u = U.at(iCell, ConsHydroState::VarIndex::Irho_vx );
    u.rho_v = U.at(iCell, ConsHydroState::VarIndex::Irho_vy );
    u.rho_w = (ndim == 3 ? U.at(iCell, ConsHydroState::VarIndex::Irho_vz ) : 0.0);
    return u;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    U.at(iCell, ConsHydroState::VarIndex::Irho) = u.rho;
    U.at(iCell, ConsHydroState::VarIndex::Ie_tot) = u.e_tot;
    U.at(iCell, ConsHydroState::VarIndex::Irho_vx) = u.rho_u;
    U.at(iCell, ConsHydroState::VarIndex::Irho_vy) = u.rho_v;
    if (ndim == 3)
      U.at(iCell, ConsHydroState::VarIndex::Irho_vz) = u.rho_w;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void atomic_addConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    Kokkos::atomic_add(&U.at(iCell, ConsHydroState::VarIndex::Irho), u.rho);
    Kokkos::atomic_add(&U.at(iCell, ConsHydroState::VarIndex::Ie_tot), u.e_tot);
    Kokkos::atomic_add(&U.at(iCell, ConsHydroState::VarIndex::Irho_vx), u.rho_u);
    Kokkos::atomic_add(&U.at(iCell, ConsHydroState::VarIndex::Irho_vy), u.rho_v);
    if (ndim == 3)
      Kokkos::atomic_add(&U.at(iCell, ConsHydroState::VarIndex::Irho_vz), u.rho_w);
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  PrimState getPrimState( const Array_t& Q, const CellIndex& iCell ) const
  {
    PrimState q;
    q.rho = Q.at(iCell, PrimHydroState::VarIndex::Irho );
    q.p   = Q.at(iCell, PrimHydroState::VarIndex::Ip );
    q.u   = Q.at(iCell, PrimHydroState::VarIndex::Iu );
    q.v   = Q.at(iCell, PrimHydroState::VarIndex::Iv );
    q.w   = (ndim == 3 ? Q.at(iCell, PrimHydroState::VarIndex::Iw ) : 0.0);
    return q;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setPrimState( const Array_t& Q, const CellIndex& iCell, const PrimState& q ) const
  {
    Q.at(iCell, PrimHydroState::VarIndex::Irho) = q.rho;
    Q.at(iCell, PrimHydroState::VarIndex::Ip) = q.p;
    Q.at(iCell, PrimHydroState::VarIndex::Iu) = q.u;
    Q.at(iCell, PrimHydroState::VarIndex::Iv) = q.v;
    if (ndim == 3)
      Q.at(iCell, PrimHydroState::VarIndex::Iw) = q.w;
  }

  KOKKOS_INLINE_FUNCTION
  PrimState consToPrim( const ConsState& U ) const
  {
    real_t gamma0 = this->gamma0;

    const real_t Ek = 0.5 * (U.rho_u*U.rho_u+U.rho_v*U.rho_v+U.rho_w*U.rho_w)/U.rho;
    const real_t p = (U.e_tot - Ek) * (gamma0-1.0);
    return {U.rho, 
            p, 
            U.rho_u/U.rho, 
            U.rho_v/U.rho, 
            (ndim == 3 ? U.rho_w/U.rho : 0.0)};
  }

  KOKKOS_INLINE_FUNCTION
  ConsState primToCons( const PrimState& Q ) const
  {
    real_t gamma0 = this->gamma0;

    const real_t Ek = 0.5 * Q.rho * (Q.u*Q.u+Q.v*Q.v+Q.w*Q.w);
    const real_t E  = Ek + Q.p / (gamma0-1.0);
    return {Q.rho, 
            E, 
            Q.rho*Q.u, 
            Q.rho*Q.v, 
            (ndim ==3 ? Q.rho*Q.w : 0.0)};
  }
};

class FiniteVolumePolicy_RiemannSolver_Hydro_hllc
{
private:
  struct Rparams {
    real_t gamma0;
    real_t smallr;
    real_t smallp;
    real_t smallc;
  } rparams;

public:
  using State = FiniteVolumePolicy_State_Hydro;
  using PrimState = State::PrimState;
  using ConsState = State::ConsState;

  FiniteVolumePolicy_RiemannSolver_Hydro_hllc( ConfigMap& configMap )
  : rparams( 
    {
      .gamma0 = configMap.getValue<real_t>("hydro", "gamma0", 1.4),
      .smallr = configMap.getValue<real_t>("hydro", "smallr", 1e-10),
      .smallp = configMap.getValue<real_t>("hydro", "smallp", 1e-10),
      .smallc = configMap.getValue<real_t>("hydro", "smallc", 1e-10) 
    })
  {}

  KOKKOS_INLINE_FUNCTION
  ConsState riemann_solver( PrimState qL, PrimState qR, ComponentIndex3D dir ) const
  {
    qL = swapComponents(qL, dir);
    qR = swapComponents(qR, dir);
    ConsState flux = riemann_hllc(qL, qR);
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
        return PrimState{q.rho, q.p, q.v, q.u, q.w};
      case IZ:
        return PrimState{q.rho, q.p, q.w, q.v, q.u};
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
        return ConsState{u.rho, u.e_tot, u.rho_v, u.rho_u, u.rho_w};
      case IZ:
        return ConsState{u.rho, u.e_tot, u.rho_w, u.rho_v, u.rho_u};
      default:
        DYABLO_ASSERT_KOKKOS_DEBUG(false, "invalid component");
        return ConsState{};
    }
  }

  KOKKOS_INLINE_FUNCTION
  ConsState riemann_hllc( PrimState qleft, PrimState qright ) const
  {
    real_t gamma0 = rparams.gamma0;
    real_t smallr = rparams.smallr;
    real_t smallp = rparams.smallp;
    real_t smallc = rparams.smallc;

    const real_t entho = 1 / (gamma0 - 1);

    // Left variables
    real_t rl = fmax(qleft.rho, smallr);
    real_t pl = fmax(qleft.p, rl*smallp);
    real_t ul =      qleft.u;
    real_t vl =      qleft.v;
    real_t wl =      qleft.w;

    real_t ecinl = 0.5*rl*(ul*ul+vl*vl+wl*wl);
    real_t etotl = pl*entho+ecinl;
    real_t ptotl = pl;


    // Right variables
    real_t rr = fmax(qright.rho, smallr);
    real_t pr = fmax(qright.p, rr*smallp);
    real_t ur =      qright.u;
    real_t vr =      qright.v;
    real_t wr =      qright.w;

    real_t ecinr = 0.5*rr*(ur*ur+vr*vr+wr*wr);
    real_t etotr = pr*entho+ecinr;
    real_t ptotr = pr;
    
    // Find the largest eigenvalues in the normal direction to the interface
    real_t cfastl = SQRT(fmax(gamma0*pl/rl,smallc*smallc));
    real_t cfastr = SQRT(fmax(gamma0*pr/rr,smallc*smallc));

    // Compute HLL wave speed
    real_t SL = fmin(ul,ur) - fmax(cfastl,cfastr);
    real_t SR = fmax(ul,ur) + fmax(cfastl,cfastr);

    // Compute lagrangian sound speed
    real_t rcl = rl*(ul-SL);
    real_t rcr = rr*(SR-ur);
    
    // Compute acoustic star state
    real_t ustar    = (rcr*ur   +rcl*ul   +  (ptotl-ptotr))/(rcr+rcl);
    real_t ptotstar = (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);

    // Left star region variables
    real_t rstarl    = rl*(SL-ul)/(SL-ustar);
    real_t etotstarl = ((SL-ul)*etotl-ptotl*ul+ptotstar*ustar)/(SL-ustar);
    
    // Right star region variables
    real_t rstarr    = rr*(SR-ur)/(SR-ustar);
    real_t etotstarr = ((SR-ur)*etotr-ptotr*ur+ptotstar*ustar)/(SR-ustar);
    
    // Sample the solution at x/t=0
    real_t ro, uo, ptoto, etoto;
    //real_t p_out;
    if (SL > 0) {
      ro=rl;
      uo=ul;
      ptoto=ptotl;
      etoto=etotl;
      //p_out=pl;
    } else if (ustar > 0) {
      ro=rstarl;
      uo=ustar;
      ptoto=ptotstar;
      etoto=etotstarl;
      //p_out=ptotstar;
    } else if (SR > 0) {
      ro=rstarr;
      uo=ustar;
      ptoto=ptotstar;
      etoto=etotstarr;
      //p_out=ptotstar;
    } else {
      ro=rr;
      uo=ur;
      ptoto=ptotr;
      etoto=etotr;
      //p_out=pr;
    }
      
    // Compute the Godunov flux
    ConsState flux;
    flux.rho   = ro*uo;
    flux.rho_u = ro*uo*uo+ptoto;
    flux.e_tot = (etoto+ptoto)*uo;
    if (flux.rho > 0) {
      flux.rho_v = flux.rho*qleft.v;
      flux.rho_w = flux.rho*qleft.w;
    } else {
      flux.rho_v = flux.rho*qright.v;
      flux.rho_w = flux.rho*qright.w;
    }

    return flux;
  }

};

class FiniteVolumePolicy_BoundaryConditions_Hydro
{
private:
  using FiniteVolumePolicy_State = FiniteVolumePolicy_State_Hydro;
  using CellIndex     = ForeachCell::CellIndex;
  using CellMetaData  = ForeachCell::CellMetaData;
  using offset_t      = CellIndex::offset_t;

  Kokkos::Array<BoundaryConditionType, 3> bc_min, bc_max;
  struct Rparams {
    real_t gamma0;
    real_t smallr;
    real_t smallp;
    real_t smallc;
  } rparams;
  const FiniteVolumePolicy_State_Hydro& state;

public:
  using PrimState = FiniteVolumePolicy_State::PrimState;
  using ConsState = FiniteVolumePolicy_State::ConsState;

  FiniteVolumePolicy_BoundaryConditions_Hydro( ConfigMap& configMap, const FiniteVolumePolicy_State& state )
  : bc_min{
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING)
    },
    bc_max{
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING),
      configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING)
    },
    rparams( 
    {
      .gamma0 = configMap.getValue<real_t>("hydro", "gamma0", 1.4),
      .smallr = configMap.getValue<real_t>("hydro", "smallr", 1e-10),
      .smallp = configMap.getValue<real_t>("hydro", "smallp", 1e-10),
      .smallc = configMap.getValue<real_t>("hydro", "smallc", 1e-10) 
    }),
    state(state)
  {}

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata) const 
  {
    const FiniteVolumePolicy_State& policy = this->state;

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
    }
    if ( (offset[IY] > 0 && bc_max[IY] == BC_REFLECTING)
      || (offset[IY] < 0 && bc_min[IY] == BC_REFLECTING) )
    {
        res.rho_v = -u_sym.rho_v;
    }
    if ( (offset[IZ] > 0 && bc_max[IZ] == BC_REFLECTING)
      || (offset[IZ] < 0 && bc_min[IZ] == BC_REFLECTING) )
    {
        res.rho_w = -u_sym.rho_w;
    }

    return res;
  }

  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux( const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata) const
  {
    CellIndex iCell_ref;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_ref, offset);

    const FiniteVolumePolicy_State& policy = this->state;

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
    
    // TODO : don't break encapsulation here
    real_t gamma0 = rparams.gamma0;
    real_t smallr = rparams.smallr;
    real_t smallp = rparams.smallp;
    real_t smallc = rparams.smallc;
    
    real_t r_in = q_in.rho;
    real_t p_in = q_in.p;
    real_t v_in[3] = {q_in.u, q_in.v, q_in.w};
    real_t v_normal = v_in[dir];

    // Left variables
    real_t rl = fmax(r_in, smallr);
    real_t pl = fmax(p_in, rl*smallp);   

    // Right variables
    real_t rr = fmax(r_in, smallr);
    real_t pr = fmax(p_in, rr*smallp);

    real_t ul=0, ur=0;
    if( reflecting )
    {
       ul = offset[dir] * v_normal;
       ur = - ul;
    }
    else if( absorbing )
    {
       ul = v_normal;
       ur = v_normal;
    }
    
    real_t ptotl = pl;
    real_t ptotr = pr;
      
    // Find the largest eigenvalues in the normal direction to the interface
    real_t cfastl = SQRT(fmax(gamma0*pl/rl,smallc*smallc));
    real_t cfastr = SQRT(fmax(gamma0*pr/rr,smallc*smallc));

    // Compute HLL wave speed
    real_t SL = fmin(ul,ur) - fmax(cfastl,cfastr);
    real_t SR = fmax(ul,ur) + fmax(cfastl,cfastr);

    // Compute lagrangian sound speed
    real_t rcl = rl*(ul-SL);
    real_t rcr = rr*(SR-ur);
      
    // Compute acoustic star state
    real_t ptotstar = (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);

    ConsHydroState flux_out {};
    if( reflecting )
    {
      flux_out.rho_u = ((dir==IX) ? ptotstar : 0);
      flux_out.rho_v = ((dir==IY) ? ptotstar : 0);
      flux_out.rho_w = ((dir==IZ) ? ptotstar : 0);
    }
    else if( absorbing )
    {
      real_t f_rho = r_in*v_normal;

      flux_out.rho = f_rho;
      flux_out.rho_u = f_rho*q_in.u + ((dir==IX) ? ptotstar : 0);
      flux_out.rho_v = f_rho*q_in.v + ((dir==IY) ? ptotstar : 0);
      flux_out.rho_w = f_rho*q_in.w + ((dir==IZ) ? ptotstar : 0);
      flux_out.e_tot = (ptotstar + u_in.e_tot) * v_normal;
    }

    return flux_out;
  }
};

class FiniteVolumePolicy_Hydro_impl
  : public FiniteVolumePolicy_State_Hydro,
    public FiniteVolumePolicy_RiemannSolver_Hydro_hllc,
    public FiniteVolumePolicy_Slope_dynamic<FiniteVolumePolicy_State_Hydro>,
    public FiniteVolumePolicy_BoundaryConditions_Hydro
{
public:
  FiniteVolumePolicy_Hydro_impl( ConfigMap& configMap )
  : FiniteVolumePolicy_State_Hydro(configMap),
    FiniteVolumePolicy_RiemannSolver_Hydro_hllc(configMap),
    FiniteVolumePolicy_Slope_dynamic<FiniteVolumePolicy_State_Hydro>(configMap),
    FiniteVolumePolicy_BoundaryConditions_Hydro(configMap, *this)
  {}

};

using FiniteVolumePolicy_Hydro = FiniteVolumePolicy_base< FiniteVolumePolicy_Hydro_impl >;

} //namespace dyablo