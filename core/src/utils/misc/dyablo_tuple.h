/***
 * This is a version of std::tuple that is meant to be portable with kokkos
 * dyablo_tuple only defines what is actually used in dyablo
 * An std::tuple implementation is available by defining DYABLO_TUPLE_STD
 * If DYABLO_TUPLE_STD is defined, be careful to avoid std::tuple features that are not defined
 ***/

#pragma once

/// Use std::tuple implementation
#define DYABLO_TUPLE_STD

#ifdef DYABLO_TUPLE_STD

#include <tuple>
#include <Kokkos_Core.hpp>

namespace dyablo {

/// Note : please try to run new code that uses dyablo_tuple with the custom implementation
template<typename... Ts>
using dyablo_tuple = std::tuple<Ts...>;

template<typename... Ts>
KOKKOS_INLINE_FUNCTION
constexpr auto make_dyablo_tuple( Ts... vs )
{
    return std::make_tuple(vs...);
}

template<typename... Ts>
KOKKOS_INLINE_FUNCTION
constexpr auto dyablo_tuple_tie( Ts&... vs )
{
    return std::tie(vs...);
}

template<size_t I, typename Tuple>
KOKKOS_INLINE_FUNCTION
constexpr decltype(auto) dyablo_tuple_get(Tuple& t) 
{ 
    return std::get<I>(t); 
}

} // namespace dyablo

#else

#include <utility>
#include <cstdlib>
#include <tuple>
#include <Kokkos_Core.hpp>

namespace dyablo {

template<typename... Ts> 
struct dyablo_tuple;

template<> 
struct dyablo_tuple<> 
{};

template<typename T0, typename... Ts>
struct dyablo_tuple<T0, Ts...> : dyablo_tuple<Ts...>
{
    T0 value;
    
    KOKKOS_FORCEINLINE_FUNCTION 
    constexpr dyablo_tuple(T0 v, Ts... rest)
    : dyablo_tuple<Ts...>(rest...), value(v) 
    {}

    KOKKOS_FORCEINLINE_FUNCTION 
    constexpr dyablo_tuple(const dyablo_tuple& o)
    : dyablo_tuple<Ts...>(o), value(o.value)
    {}

    template< typename T2_0, typename... T2_s >
    KOKKOS_FORCEINLINE_FUNCTION
    constexpr dyablo_tuple& operator=(const dyablo_tuple<T2_0, T2_s...>& o)
    {
        this->value = o.value;
        if constexpr (sizeof...(T2_s) > 0)
        {
            dyablo_tuple<Ts...>& this_base = *this;
            const dyablo_tuple<T2_s...>& o_base = o;
            this_base = o_base;
        }    
        return *this;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr dyablo_tuple& operator=(const dyablo_tuple& o)
    {
        return operator=<T0, Ts...>(o);
    }
};

} // namespace dyablo

/// Define necessary stuff for c++17 structured bindings
namespace std {
template<typename... Ts>
struct tuple_size<dyablo::dyablo_tuple<Ts...>> 
: std::integral_constant<size_t, sizeof...(Ts)> 
{};

template<size_t I, typename... Ts>
struct tuple_element<I, dyablo::dyablo_tuple<Ts...>> 
{ 
    using type = std::tuple_element_t<I, std::tuple<Ts...>>; 
};
} // namespace std

namespace dyablo {

template<typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION
constexpr dyablo_tuple<Ts...> make_dyablo_tuple( Ts... vs )
{
    return dyablo_tuple<Ts...>(vs...);
}

template<typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION
constexpr dyablo_tuple<Ts&...> dyablo_tuple_tie( Ts&... vs )
{
    return dyablo_tuple<Ts&...>(vs...);
}

template<size_t I, typename T0, typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION
constexpr std::tuple_element_t<I, dyablo_tuple<T0, Ts...>>& dyablo_tuple_get(dyablo_tuple<T0, Ts...>& t) 
{ 
    if constexpr( I == 0 )
        return t.value;
    else
        return dyablo_tuple_get<I-1>( static_cast<dyablo_tuple<Ts...>&>(t) );
}

template<size_t I, typename T0, typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION
constexpr const std::tuple_element_t<I, dyablo_tuple<T0, Ts...>>&  dyablo_tuple_get(const dyablo_tuple<T0, Ts...>& t) 
{ 
    if constexpr( I == 0 )
        return t.value;
    else
        return dyablo_tuple_get<I-1>( static_cast<const dyablo_tuple<Ts...>&>(t) );
}

/// Define dyablo::get for c++17 structured bindings
template<size_t I, typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION
constexpr decltype(auto) get(dyablo_tuple<Ts...>& t) { return dyablo_tuple_get<I>(t); }

template<size_t I, typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION
constexpr decltype(auto) get(const dyablo_tuple<Ts...>& t) { return dyablo_tuple_get<I>(t); }

} // namespace dyablo

#endif