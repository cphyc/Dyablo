#pragma once

#include "real_type.h"
#include "kokkos_shared.h"
#include "State_Ops.h"

namespace dyablo {

/**
 * @brief Structure holding conservative radiative-hydrodynamics variables
 **/ 
struct ConsRadState {

  enum VarIndex : dyablo::VarIndex
  {
    Irho,
    Ie_tot,
    Irho_vx,
    Irho_vy,
    Irho_vz,
    IDR, 
    IUR,
    IVR,
    IWR,
    IXE,
    IZR,
    ITemp
  };  

  static std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo()
  {
    return  { {"rho",    VarIndex::Irho}, 
              {"e_tot",  VarIndex::Ie_tot},
              {"rho_vx",  VarIndex::Irho_vx},
              {"rho_vy",  VarIndex::Irho_vy},
              {"rho_vz",  VarIndex::Irho_vz},
              {"e_rad",  VarIndex::IDR},
              {"fx_rad", VarIndex::IUR},
              {"fy_rad", VarIndex::IVR},
              {"fz_rad", VarIndex::IWR},
              {"xe", VarIndex::IXE},
	            {"zre", VarIndex::IZR},
              {"temp", VarIndex::ITemp}

    };
  }

  static FieldManager getFieldManager()
  {
    return FieldManager( {
      VarIndex::Irho, 
      VarIndex::Ie_tot, 
      VarIndex::Irho_vx, 
      VarIndex::Irho_vy, 
      VarIndex::Irho_vz, 
      VarIndex::IDR, 
      VarIndex::IUR, 
      VarIndex::IVR,
      VarIndex::IWR,
      VarIndex::IXE,
      VarIndex::IZR,
      VarIndex::ITemp
    } );
  }

  real_t rho = 0;
  real_t e_tot = 0;
  real_t rho_u = 0;
  real_t rho_v = 0;
  real_t rho_w = 0;
  real_t e_rad = 0;
  real_t fx_rad = 0;
  real_t fy_rad = 0;
  real_t fz_rad = 0;
  real_t xe=0;
  real_t zre=0;
  real_t temp=0;
};

DECLARE_STATE_TYPE( ConsRadState, 12 );
DECLARE_STATE_GET( ConsRadState, 0, rho );
DECLARE_STATE_GET( ConsRadState, 1, e_tot );
DECLARE_STATE_GET( ConsRadState, 2, rho_u );
DECLARE_STATE_GET( ConsRadState, 3, rho_v );
DECLARE_STATE_GET( ConsRadState, 4, rho_w );
DECLARE_STATE_GET( ConsRadState, 5, e_rad );
DECLARE_STATE_GET( ConsRadState, 6, fx_rad );
DECLARE_STATE_GET( ConsRadState, 7, fy_rad );
DECLARE_STATE_GET( ConsRadState, 8, fz_rad );
DECLARE_STATE_GET( ConsRadState, 9, xe );
DECLARE_STATE_GET( ConsRadState, 10, zre );
DECLARE_STATE_GET( ConsRadState, 11, temp );


/**
 * @brief Structure holding primitive magneto-hydrodynamics variables
 */
struct PrimRadState {


enum VarIndex : dyablo::VarIndex
  {
    Irho,
    Ip,
    Iu,
    Iv,
    Iw,
    IDR, 
    IUR,
    IVR,
    IWR,
    IXE,
    IZR,
    ITemp
  }; 

  static std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo()
  {
    return  { {"rho",VarIndex::Irho}, 
              {"e_tot",  VarIndex::Ip},
              {"rho_vx",  VarIndex::Iu},
              {"rho_vy",  VarIndex::Iv},
              {"rho_vz",  VarIndex::Iw},
              {"e_rad", VarIndex::IDR},
              {"fx_rad", VarIndex::IUR},
              {"fy_rad", VarIndex::IVR},
              {"fz_rad", VarIndex::IWR},
              {"xe", VarIndex::IXE},
	            {"zre", VarIndex::IZR},
              {"temp", VarIndex::ITemp}
    };
  }

  static FieldManager getFieldManager()
  {
    return FieldManager( {
      VarIndex::Irho, 
      VarIndex::Ip, 
      VarIndex::Iu, 
      VarIndex::Iv, 
      VarIndex::Iw, 
      VarIndex::IDR, 
      VarIndex::IUR, 
      VarIndex::IVR,
      VarIndex::IWR,
      VarIndex::IXE,
      VarIndex::IZR,
      VarIndex::ITemp
    } );
  }


  real_t rho = 0;
  real_t p = 0;
  real_t u = 0;
  real_t v = 0;
  real_t w = 0;
  real_t e_rad = 0;
  real_t fx_rad = 0;
  real_t fy_rad = 0;
  real_t fz_rad = 0;
  real_t xe =0;
  real_t zre =0;
  real_t temp = 0;
};

DECLARE_STATE_TYPE( PrimRadState, 12 );
DECLARE_STATE_GET( PrimRadState, 0, rho );
DECLARE_STATE_GET( PrimRadState, 1, p );
DECLARE_STATE_GET( PrimRadState, 2, u );
DECLARE_STATE_GET( PrimRadState, 3, v );
DECLARE_STATE_GET( PrimRadState, 4, w );
DECLARE_STATE_GET( PrimRadState, 5, e_rad );
DECLARE_STATE_GET( PrimRadState, 6, fx_rad );
DECLARE_STATE_GET( PrimRadState, 7, fy_rad );
DECLARE_STATE_GET( PrimRadState, 8, fz_rad );
DECLARE_STATE_GET( PrimRadState, 9, xe );
DECLARE_STATE_GET( PrimRadState, 10, zre );
DECLARE_STATE_GET( PrimRadState, 11, temp );


/**
 * @brief Structure grouping the primitive and conservative Rad state as well
 *        as information on the number of fields to store per state
 */
struct RadState {
  using PrimState = PrimRadState;
  using ConsState = ConsRadState;
  static constexpr size_t N = 12;
};

/**
* @brief Returns a conservative state at a given cell index in an array
* 
* @tparam ndim the number of dimensions
* @tparam Array_t the type of array where we are looking up
* @tparam CellIndex the type of cell index used

* @param U the array in which we are getting the state
* @param iCell the index of the cell 
* @return the hydro state at position iCell in U
*/
template< int ndim, 
          typename Array_t, 
          typename CellIndex >
KOKKOS_INLINE_FUNCTION
void getConservativeState( const Array_t& U, const CellIndex& iCell, ConsRadState &res )
{
  res.rho   = U.at(iCell, ConsRadState::VarIndex::Irho);
  res.e_tot = U.at(iCell, ConsRadState::VarIndex::Ie_tot);
  res.rho_u = U.at(iCell, ConsRadState::VarIndex::Irho_vx);
  res.rho_v = U.at(iCell, ConsRadState::VarIndex::Irho_vy);
  res.rho_w = (ndim == 3 ? U.at(iCell, ConsRadState::VarIndex::Irho_vz) : 0.0);

  res.e_rad = U.at(iCell, ConsRadState::VarIndex::IDR);
  res.fx_rad = U.at(iCell, ConsRadState::VarIndex::IUR);
  res.fy_rad = U.at(iCell, ConsRadState::VarIndex::IVR);
  res.fz_rad = (ndim == 3 ? U.at(iCell, ConsRadState::VarIndex::IWR) : 0.0);

  res.xe = U.at(iCell, ConsRadState::VarIndex::IXE);
  res.zre = U.at(iCell, ConsRadState::VarIndex::IZR);
  res.temp = U.at(iCell, ConsRadState::VarIndex::ITemp);

}

/**
 * @brief Returns a primitive hydro state at a given cell index in an array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array where we are looking up
 * @tparam CellIndex the type of cell index used
 *
 * @param U the array in which we are getting the state
 * @param iCell the index of the cell 
 * @return the hydro state at position iCell in U
 */
template< int ndim,
          typename Array_t, 
          typename CellIndex >
KOKKOS_INLINE_FUNCTION
void getPrimitiveState( const Array_t& U, const CellIndex& iCell, PrimRadState &res )
{
  res.rho = U.at(iCell, PrimRadState::VarIndex::Irho);
  res.p   = U.at(iCell, PrimRadState::VarIndex::Ip);
  res.u   = U.at(iCell, PrimRadState::VarIndex::Iu);
  res.v   = U.at(iCell, PrimRadState::VarIndex::Iv);
  res.w   = (ndim == 3 ? U.at(iCell, PrimRadState::VarIndex::Iw) : 0.0);

  res.e_rad  = U.at(iCell, PrimRadState::VarIndex::IDR);
  res.fx_rad  = U.at(iCell, PrimRadState::VarIndex::IUR);
  res.fy_rad  = U.at(iCell, PrimRadState::VarIndex::IVR);
  res.fz_rad  = (ndim == 3 ? U.at(iCell, PrimRadState::VarIndex::IWR) : 0.0);
  res.xe = U.at(iCell, PrimRadState::VarIndex::IXE);
  res.zre = U.at(iCell, PrimRadState::VarIndex::IZR);
  res.temp = U.at(iCell, PrimRadState::VarIndex::ITemp);
}

/**
 * @brief Stores a primitive hydro state in an array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array in which the primitive value is stored
 * @tparam CellIndex the type of cell index used
 * 
 * @param U the array where we are storing the state
 * @param iCell the index of cell
 * @param u the value to store in the array
 */
template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void setPrimitiveState( const Array_t& U, const CellIndex& iCell, PrimRadState u) {
  U.at(iCell, PrimRadState::VarIndex::Irho) = u.rho;
  U.at(iCell, PrimRadState::VarIndex::Ip) = u.p;
  U.at(iCell, PrimRadState::VarIndex::Iu) = u.u;
  U.at(iCell, PrimRadState::VarIndex::Iv) = u.v;

  U.at(iCell, PrimRadState::VarIndex::IDR) = u.e_rad;
  U.at(iCell, PrimRadState::VarIndex::IUR) = u.fx_rad;
  U.at(iCell, PrimRadState::VarIndex::IVR) = u.fy_rad;
  if (ndim == 3) {
    U.at(iCell, PrimRadState::VarIndex::Iw) = u.w;
    U.at(iCell, PrimRadState::VarIndex::IWR) = u.fz_rad;
  }

   U.at(iCell, PrimRadState::VarIndex::IXE) = u.xe;
   U.at(iCell, PrimRadState::VarIndex::IZR) = u.zre;
   U.at(iCell, PrimRadState::VarIndex::ITemp) = u.temp;
}

/**
 * @brief Stores a conservative hydro state in an array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array in which the primitive value is stored
 * @tparam CellIndex the type of cell index used
 * 
 * @param U the array where we are storing the state
 * @param iCell the index of cell
 * @param u the value to store in the array
 */
template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void setConservativeState( const Array_t& U, const CellIndex& iCell, ConsRadState u) {
  U.at(iCell, ConsRadState::VarIndex::Irho) = u.rho;
  U.at(iCell, ConsRadState::VarIndex::Ie_tot) = u.e_tot;
  U.at(iCell, ConsRadState::VarIndex::Irho_vx) = u.rho_u;
  U.at(iCell, ConsRadState::VarIndex::Irho_vy) = u.rho_v;
  
  U.at(iCell, ConsRadState::VarIndex::IDR) = u.e_rad;
  U.at(iCell, ConsRadState::VarIndex::IUR) = u.fx_rad;
  U.at(iCell, ConsRadState::VarIndex::IVR) = u.fy_rad;
  if (ndim == 3) {
    U.at(iCell, ConsRadState::VarIndex::Irho_vz) = u.rho_w;
    U.at(iCell, ConsRadState::VarIndex::IWR) = u.fz_rad;
  }

  U.at(iCell, ConsRadState::VarIndex::IXE) = u.xe;
  U.at(iCell, ConsRadState::VarIndex::IZR) = u.zre;
  U.at(iCell, ConsRadState::VarIndex::ITemp) = u.temp;

}

/**
 * @brief Converts from a Rad conservative state to a Rad primitive state
 * 
 * @tparam ndim the number of dimensions
 * 
 * @param U the initial conservative state
 * @param gamma0 adiabatic index
 * @return the primitive version of U
 */
template<int ndim>
KOKKOS_INLINE_FUNCTION
PrimRadState consToPrim(const ConsRadState &U, real_t gamma0) {
  const real_t Ek = 0.5 * (U.rho_u*U.rho_u
                          +U.rho_v*U.rho_v
                          +(ndim == 3 ? U.rho_w*U.rho_w : 0.0))/U.rho;

  const real_t p = (U.e_tot - Ek) * (gamma0-1.0);
  return {U.rho, 
          p, 
          U.rho_u/U.rho, 
          U.rho_v/U.rho, 
          (ndim == 3 ? U.rho_w/U.rho : 0.0),
          U.e_rad,
          U.fx_rad,
          U.fy_rad,
          (ndim == 3 ? U.fz_rad : 0.0),
          U.xe,
	        U.zre,
          U.temp
          };
}

/**
 * @brief Converts from a Rad primitive state to a Rad conservative state
 * 
 * @tparam ndim the number of dimensions
 * 
 * @param Q the initial primitive state
 * @param gamma0 adiabatic index
 * @return the conservative version of Q
 */
template<int ndim>
KOKKOS_INLINE_FUNCTION
ConsRadState primToCons(const PrimRadState &Q, real_t gamma0) {
  const real_t Ek = 0.5 * Q.rho * (Q.u*Q.u
                                  +Q.v*Q.v
                                  +(ndim == 3 ? Q.w*Q.w : 0.0));
                                  
  const real_t E  = Ek + Q.p / (gamma0-1.0);
  return {Q.rho, 
          E, 
          Q.rho*Q.u, 
          Q.rho*Q.v, 
          (ndim ==3 ? Q.rho*Q.w : 0.0),
	        Q.e_rad,
          Q.fx_rad,
          Q.fy_rad,
          (ndim == 3 ? Q.fz_rad : 0.0),
          Q.xe,
	        Q.zre,
          Q.temp
          };
}

/**
 * @brief Swaps a component in velocity and magnetic field with the X component. 
 *        The Riemann problem is always solved by considering an interface on the 
 *        X-axis. So when solving it for other components, those should be swapped 
 *        before and after solving the Riemann problem.
 *  
 * @param Q (IN/OUT) the primitive Rad state to modify
 * @param comp the component to swap with X
 */
KOKKOS_INLINE_FUNCTION
PrimRadState swapComponents(const PrimRadState &q, ComponentIndex3D comp) {
  switch( comp )
  {
    case IX:
      return q;
    case IY:
      return PrimRadState{q.rho, q.p, q.v, q.u, q.w, q.e_rad,q.fy_rad, q.fx_rad, q.fz_rad, q.xe, q.zre, q.temp};
    case IZ:
      return PrimRadState{q.rho, q.p, q.w, q.v, q.u, q.e_rad,q.fz_rad, q.fy_rad, q.fx_rad, q.xe, q.zre, q.temp};
    default:
      assert(false);
      return PrimRadState{};
  }
}

/**
 * @brief Swaps a component in velocity and magnetic field with the X component. 
 *        The Riemann problem is always solved by considering an interface on the 
 *        X-axis. So when solving it for other components, those should be swapped 
 *        before and after solving the Riemann problem.
 *  
 * @param Q (IN/OUT) the primitive Rad state to modify
 * @param comp the component to swap with X
 */
KOKKOS_INLINE_FUNCTION
ConsRadState swapComponents(const ConsRadState &u, ComponentIndex3D comp) {
  switch( comp )
  {
    case IX:
      return u;
    case IY:
      return ConsRadState{u.rho, u.e_tot, u.rho_v, u.rho_u, u.rho_w, u.e_rad,u.fy_rad, u.fx_rad, u.fz_rad, u.xe, u.zre, u.temp};
    case IZ:
      return ConsRadState{u.rho, u.e_tot, u.rho_w, u.rho_v, u.rho_u, u.e_rad,u.fz_rad, u.fy_rad, u.fx_rad, u.xe, u.zre, u.temp};
    default:
      assert(false);
      return ConsRadState{};
  }
}
} // namespace dyablo

