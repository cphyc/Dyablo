#pragma once

#include <variant>

namespace dyablo{

namespace dyablo_std_variant {

template< typename... T >
using dyablo_variant = std::variant<T...>;

template< class T, class... Types >
KOKKOS_INLINE_FUNCTION
constexpr bool dyablo_variant_holds_alternative( const dyablo_variant<Types...>& v ) noexcept
{
  return std::holds_alternative<T>( v );
}

template< class T, class... Types >
KOKKOS_INLINE_FUNCTION
constexpr T& dyablo_variant_get( std::variant<Types...>& v ) noexcept
{
  T* res = std::get_if<T>(&v);
  DYABLO_ASSERT_KOKKOS_DEBUG( res != nullptr, "Bad variant access" );
  return *res;
}

template< class T, class... Types >
KOKKOS_INLINE_FUNCTION
constexpr const T& dyablo_variant_get( const std::variant<Types...>& v ) noexcept
{
  const T* res = std::get_if<T>(&v);
  DYABLO_ASSERT_KOKKOS_DEBUG( res != nullptr, "Bad variant access" );
  return *res;
}

// template< class R, class Visitor, typename... Ts >
// KOKKOS_INLINE_FUNCTION
// constexpr R dyablo_variant_visit( const Visitor& vis, const dyablo_variant<Ts...>& var ) noexcept
// {
//   return std::visit( vis, var );
// }



} // namespace dyablo_std_variant

using namespace dyablo_std_variant;

template< class R, class Visitor, class var_t, typename T0, typename... Ts >
KOKKOS_INLINE_FUNCTION
constexpr R dyablo_variant_visit_aux( const Visitor& vis, const var_t& var ) noexcept
{
  if( dyablo_variant_holds_alternative<T0>(var) )
    return vis( dyablo_variant_get<T0>(var) );
  else
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( sizeof...(Ts) > 0, "Internal Error : bad variant" );
    if constexpr (sizeof...(Ts) > 0)
      return dyablo_variant_visit_aux<R, Visitor, var_t, Ts...>(vis, var);
    else 
      return R(); 
  }
}

template< class R, class Visitor, typename... Ts >
KOKKOS_INLINE_FUNCTION
constexpr R dyablo_variant_visit( const Visitor& vis, const dyablo_variant<Ts...>& var ) noexcept
{
  return dyablo_variant_visit_aux<R, Visitor, dyablo_variant<Ts...>, Ts...>(vis, var);
}


} // namespace dyablo