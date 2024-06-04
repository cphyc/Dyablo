#pragma once

#include "UserData.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo{

/**
 * Interface for FiniteVolumePolicy used in Finite Volume solvers
 * to verify that FiniteVolumePolicy implements all methods correctly
 * We recommand that you wrap your FiniteVolumePolicy implementation 
 * in this template interface to check every interface methods
 **/
template< typename FiniteVolumePolicy_impl >
class FiniteVolumePolicy_base : public FiniteVolumePolicy_impl
{
public:
  /// Structure containing primitive variables
  using PrimState = typename FiniteVolumePolicy_impl::PrimState;
  /// Structure containing conservative variables
  using ConsState = typename FiniteVolumePolicy_impl::ConsState;
  using CellIndex = ForeachCell::CellIndex;
  using CellMetaData = ForeachCell::CellMetaData;
  using FieldAccessor = UserData::FieldAccessor;  

  /// FiniteVolumePolicy needs to be constructible from a ConfigMap
  FiniteVolumePolicy_base( ConfigMap& configMap )
  : FiniteVolumePolicy_impl(configMap)
  {}

  /**
   * Extract a FieldAccessor from U for input fields 
   * This must allow reading current timestep fields with getConsState()
   **/
  FieldAccessor getUin( UserData& U ) const
  {
    return FiniteVolumePolicy_impl::getUin(U);
  }

  /**
   * Extract a FieldAccessor from U for updated fields
   * This must allow reading/settings fields with get/setConsState()
   **/
  FieldAccessor getUout( UserData& U ) const
  {
    return FiniteVolumePolicy_impl::getUout(U);
  }

  /**
   * @brief Returns the conservative state at a given cell index in an array
   * 
   * @tparam Array_t the type of array where we are looking up
   * 
   * @param U the array in which we are getting the state
   * @param iCell the index of the cell 
   * @return the conservative state at position iCell in U : U(iCell)
   **/
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getConsState( const Array_t& U, const CellIndex& iCell ) const
  {
    return FiniteVolumePolicy_impl::getConsState(U, iCell);
  }

  /**
   * @brief Write the conservative state to an array
   * 
   * U(iCell) := u
   * 
   * @tparam Array_t the type of array in which the primitive value is stored
   * 
   * @param U the array where we are storing the state
   * @param iCell the index of cell
   * @param u the State value to store in the array
   **/
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    FiniteVolumePolicy_impl::setConsState(U, iCell, u);
  }

  /**
   * @brief Atomically accumulate a conservative state to an array
   * 
   * U(iCell) += u
   * 
   * @tparam Array_t the type of array in which the primitive value is stored
   * 
   * @param U the array where we are storing the state
   * @param iCell the index of cell
   * @param u the State value to accumulate with the existing value in the array
   **/
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void atomic_addConsState( const Array_t& U, const CellIndex& iCell, const ConsState& u ) const
  {
    FiniteVolumePolicy_impl::atomic_addConsState(U, iCell, u);
  }

  /**
   * @brief Returns the primitive state at a given cell index in an array
   * 
   * @tparam Array_t the type of array where we are looking up
   * 
   * @param Q the array in which we are getting the state
   * @param iCell the index of the cell 
   * @return the conservative state at position iCell in Q : Q(iCell)
   **/
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  PrimState getPrimState( const Array_t& Q, const CellIndex& iCell ) const
  {
    return FiniteVolumePolicy_impl::getPrimState(Q, iCell);
  }

  /**
   * @brief Write the primitive state to an array
   * 
   * Q(iCell) := q
   * 
   * @tparam Array_t the type of array in which the primitive value is stored
   * 
   * @param Q the array where we are storing the state
   * @param iCell the index of cell
   * @param q the State value to store in the array
   **/
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  void setPrimState( const Array_t& Q, const CellIndex& iCell, const PrimState& q ) const
  {
    FiniteVolumePolicy_impl::setPrimState(Q, iCell, q);
  }

  /***
   * @brief Converts from conservative state to primitive state
   * 
   * @param u the initial conservative state
   ***/
  KOKKOS_INLINE_FUNCTION
  PrimState consToPrim( const ConsState& u ) const
  {
    return FiniteVolumePolicy_impl::consToPrim(u);
  }

  /***
   * @brief Converts from primitive state to conservative state
   * 
   * @param u the initial primitive state
   ***/
  KOKKOS_INLINE_FUNCTION
  ConsState primToCons( const PrimState& q ) const
  {
    return FiniteVolumePolicy_impl::primToCons(q);
  }

  /** 
   * @brief Riemann solver
   * 
   * Compute flux at interface using states at both sides of the interface
   *
   * @param qleft left state (primitive variables)
   * @param qright right state (primitive variables)
   * @param dir x, y or z direction
   * @return output flux
   */
  KOKKOS_INLINE_FUNCTION
  ConsState riemann_solver( PrimState qL, PrimState qR, ComponentIndex3D dir ) const
  {
    return FiniteVolumePolicy_impl::riemann_solver(qL, qR, dir);
  }

  /**
   * \brief Get field values for a cell outside of the simulation domain
   * 
   * This method is usually called by kernels when trying to access neighbors outside
   * of the simulation domain.
   * Returns the state of a virtual cell outside of the domain.
   * The value returned may be determined by the values inside the domain 
   * (e.g. reflecting/absorbing boundary conditions)
   * 
   * \param U array containing conservative variables
   * \param iCell_boundary cell index outside of the domain (iCell_boundary.is_boundary() must be true)
   * \param metadata CellMetaData object allowing for mesh queries on size/position
   * \return The conservative state defined at the given boundary position 
   */
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata) const 
  {
    return FiniteVolumePolicy_impl::getBoundaryValue(U, iCell_boundary, metadata);
  }
  
  /**
   * \brief Get flux value for a neighboring cell outside of the simulation domain
   * 
   * This method is called when trying to compute the flux with neighbors outside
   * of the simulation domain.
   * Returns the flux at the interface on domain boundary.
   * The value returned may be determined by the values inside the domain 
   * (e.g. reflecting/absorbing boundary conditions)
   * 
   * \param U array containing conservative variables
   * \param iCell_boundary cell index outside of the domain (iCell_boundary.is_boundary() must be true), 
   *          the cell must have an interface with a cell inside the domain
   * \param metadata CellMetaData object allowing for mesh queries on size/position
   * \return The flux at the boundary interface 
   */
  template < typename Array_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux( const Array_t &U, const CellIndex &iCell_boundary, const CellMetaData &metadata) const
  {
    return FiniteVolumePolicy_impl::getBoundaryFlux(U, iCell_boundary, metadata);
  }
};

template< class T >
struct is_FiniteVolumePolicy 
  : std::false_type
{};

template< class U >
struct is_FiniteVolumePolicy<FiniteVolumePolicy_base<U>> 
  : std::true_type
{};

template< class T >
inline constexpr bool is_FiniteVolumePolicy_v 
  = is_FiniteVolumePolicy<T>::value;

} // namespace dyablo