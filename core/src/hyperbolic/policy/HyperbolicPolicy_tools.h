#pragma once

namespace dyablo {

/// Apply f(int index, auto elt) to tuple elements until f returns true;
template< typename F, typename Tuple_t, size_t... Is >
KOKKOS_INLINE_FUNCTION 
bool tuple_foreach_until( F f, const Tuple_t& t, std::index_sequence<Is...>)
{
  return ( f(Is, std::get<Is>(t)) || ... );
}

/// Find the index n of the tuple element where std::get<n>(t).name() == name
template< typename T0, typename... Ts >
size_t tuple_find_name(std::string name, int N=0)
{
  if( T0::name() == name )
    return N;
  else
  {
    if constexpr( sizeof...(Ts) == 0 )
    {
      DYABLO_ASSERT_HOST_RELEASE( false, "Could not find tuple element '" << name << "'" );
      return -1;
    }
    else
      return tuple_find_name<Ts...>(name, N+1);
  }
}

/***
 * Apply function f() to the n-th element of the tuple, 
 * like f(std::get<n>(t)) but with non-constexpr n
 * @param f T -> Res, can take every type in Ts... as input and always return the same type Res
 * @param t a tuple containing types compatible with f()
 * @returns same return type as f()
 ***/
template< typename F, typename... Ts >
KOKKOS_INLINE_FUNCTION 
auto tuple_apply_nth( int n, F f, const std::tuple<Ts...>& t)
{
  using ret_type = std::invoke_result_t< F, std::tuple_element_t<0, std::tuple<Ts...> > >;
  static_assert( ( std::is_same_v< ret_type, std::invoke_result_t< F, Ts > > && ... ), "Return type mismatch for F" );

  ret_type res;

  // apply f if i==n, stop when that happens
  auto apply_nth = [&](int i, const auto& ti)
  {
    if( i==n ) 
      res = f(ti);
    return i==n;
  };

  [[maybe_unused]] bool found = tuple_foreach_until( apply_nth, t, std::index_sequence_for<Ts...>{} );
  DYABLO_ASSERT_KOKKOS_DEBUG( found, "Internal error : could not find corresponding tuple element" );
  
  return res;
}

} // namespace dyablo