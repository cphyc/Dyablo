/**
 * Define arithmetic operations on States. States are structures containing 
 * variables for a cell (or a particle) needed inside kernels. 
 * 
 * A state `State_t` contains N variables of type `real_t`.
 * To enable arithmetic operators +,-,*,/,+=,-=,*=,/= as well as `state_get<I>(state)` and  `state_foreach_var()` :
 * * DECLARE_STATE_TYPE( State_t, N ) must be called to declare State_t as a state
 * * An index for every member variable must be set with DECLARE_STATE_GET( State_t, <index>, <member> )
 *      Every index from 0 to N-1 must be set
 * See State_hydro.h for an example
 **/

#pragma once

#include <Kokkos_Core.hpp>
#include <tuple>

#include "real_type.h"
#include "utils/misc/dyablo_tuple.h"

namespace dyablo {

/// By default T is not a State
template<typename T>
struct State_traits
{
    static constexpr bool is_state = false;
};

#define DECLARE_STATE_TYPE_ARRAY_AUX(constness, State, N_VARS, N_FIELDS ) \
template<> \
struct State_traits<constness State> \
{ \
    static constexpr bool is_state = true; \
    static constexpr int nvars = N_VARS; \
    static constexpr int nfields = N_FIELDS; \
}; \

/// Type-trait for States without arrays
#define DECLARE_STATE_TYPE( State, N_VARS) \
DECLARE_STATE_TYPE_ARRAY_AUX(, State, N_VARS, N_VARS ) \
DECLARE_STATE_TYPE_ARRAY_AUX(const, State, N_VARS, N_VARS ) \

/// Type-trait for States with arrays
/// N_VARS is the total number of vars (summing array lengths)
/// N_FIELDS are the number of fields in struct : (1 array = 1 field)
#define DECLARE_STATE_TYPE_ARRAY( State, N_VARS, N_FIELDS ) \
DECLARE_STATE_TYPE_ARRAY_AUX(, State, N_VARS, N_FIELDS ) \
DECLARE_STATE_TYPE_ARRAY_AUX(const, State, N_VARS, N_FIELDS ) \

#define DECLARE_STATE_GET( State, I, var ) /*empty*/

template< typename T >
concept State = State_traits<T>::is_state;

namespace {

/***
 * Use structured bindings (auto& [e0,e1,...] = s;) to transform State into tuple
 * Note : this doesn't support > 15 fields because structured bindings needs to statically list the variables
 ***/
template< State State_t >
KOKKOS_INLINE_FUNCTION
auto as_dyablo_tuple( State_t& s )
{
    constexpr int N = State_traits<State_t>::nfields;

    if      constexpr (N == 1) { auto& [e0] = s; return dyablo_tuple_tie( e0 ); }
    #define DEFINE_AS_TUPLE(n, ...) else if constexpr ( N == (n) ) \
    {\
        auto& [__VA_ARGS__] = s; \
        return dyablo_tuple_tie(__VA_ARGS__); \
    }
    DEFINE_AS_TUPLE(2,e0,e1)
    DEFINE_AS_TUPLE(3,e0,e1,e2)
    DEFINE_AS_TUPLE(4,e0,e1,e2,e3)
    DEFINE_AS_TUPLE(5,e0,e1,e2,e3,e4)
    DEFINE_AS_TUPLE(6,e0,e1,e2,e3,e4,e5)
    DEFINE_AS_TUPLE(7,e0,e1,e2,e3,e4,e5,e6)
    DEFINE_AS_TUPLE(8,e0,e1,e2,e3,e4,e5,e6,e7)
    DEFINE_AS_TUPLE(9,e0,e1,e2,e3,e4,e5,e6,e7,e8)
    DEFINE_AS_TUPLE(10,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9)
    DEFINE_AS_TUPLE(11,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10)
    DEFINE_AS_TUPLE(12,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11)
    DEFINE_AS_TUPLE(13,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12)
    DEFINE_AS_TUPLE(14,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13)
    DEFINE_AS_TUPLE(15,e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14)

    static_assert( N <= 15, "States with more than 15 fields are not supported (1 real_t or array = 1 field)" );
}

template<typename F, size_t I, typename... Tuple_t>
KOKKOS_INLINE_FUNCTION
constexpr void state_foreach_at(const F& f, Tuple_t... t)
{
    constexpr bool is_array = ( std::is_bounded_array_v<std::remove_reference_t<decltype(dyablo_tuple_get<I>(t))>> || ... );
    static_assert( ( (is_array == std::is_bounded_array_v<std::remove_reference_t<decltype(dyablo_tuple_get<I>(t))>>) && ... ),
                   "state_foreach State mismatch : all States must have arrays and real_t at the same place" );
    if constexpr (is_array) {
        constexpr size_t array_len = std::min({ std::extent_v<std::remove_reference_t<decltype(dyablo_tuple_get<I>(t))>>... });
        for (size_t i=0; i<array_len; i++) f( dyablo_tuple_get<I>(t)[i]... );
    } else {
        f( dyablo_tuple_get<I>(t)... );
    }
}

template<typename F, typename... Tuple_t, size_t... Is>
KOKKOS_INLINE_FUNCTION
constexpr void state_foreach_impl(const F& f, std::index_sequence<Is...>, Tuple_t... t)
{
    ( state_foreach_at<F, Is, Tuple_t...>(f, t...), ... );
}
} // namespace

/**
 * Iterate over each member variable for a set of states
 * @tparam I start index (mainly here for metaprogramming purpose)
 * @param states... states to read or modify. They can be const or not.
 *                  Mixing state types is not advised
 * @param f function to apply to each field in states of type
 *          f : (real_t(&), real_t(&), ...) -> void
 *          one real_t for each const State& in `states...`
 *          one real& for each State& in `states`
 *          e.g. state_foreach_var( [](real_t&, real_t, real_t){...}, State&, const State&, const State& );
 **/
template< typename F, State... State_t >
KOKKOS_INLINE_FUNCTION
void state_foreach_var( const F& f, State_t&... states )
{
    using State_t0 = std::tuple_element_t<0, std::tuple<State_t...>>;
    constexpr int N = State_traits<State_t0>::nfields;
    static_assert( ( (N == State_traits<State_t>::nfields) && ... ), 
                   "state_foreach State mismatch : States are not the same size" );
    state_foreach_impl(f, std::make_index_sequence<N>{}, as_dyablo_tuple(states)...);
}

//################
// Arithmetic operators on states
//################

// See https://github.com/llvm/llvm-project/issues/49197
// Fixed in https://github.com/llvm/llvm-project/pull/131777
// operator*(real_t, State) with concept was not filtered out when operator is called with an enum instead of real_t 
// This uses enable_if on top of concept to ensure sfinae actually triggers
#define CLANG_ISSUE_49197_WORKAROUND , std::enable_if_t<State<State_t>, int> = 0

// Operator +
template< State State_t >
KOKKOS_INLINE_FUNCTION
State_t operator+(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l+r;}, res, lhs, rhs );
    return res;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND >
KOKKOS_INLINE_FUNCTION
State_t operator+(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l+rhs;}, res, lhs );    
    return res;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND >
KOKKOS_INLINE_FUNCTION
State_t operator+(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs+r;}, res, rhs );    
    return res;
}

template< State State_t, std::enable_if_t<State<State_t>, int> = 0 >
KOKKOS_INLINE_FUNCTION
State_t& operator+=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l+=r;}, lhs, rhs );
    return lhs;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND >
KOKKOS_INLINE_FUNCTION
State_t& operator+=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l+=rhs;}, lhs );
    return lhs;
}

// Operator -
template< State State_t >
KOKKOS_INLINE_FUNCTION
State_t operator-(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l-r;}, res, lhs, rhs );
    return res;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND > 
KOKKOS_INLINE_FUNCTION
State_t operator-(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l-rhs;}, res, lhs );    
    return res;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND > 
KOKKOS_INLINE_FUNCTION
State_t operator-(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs-r;}, res, rhs );    
    return res;
}

template< State State_t > 
KOKKOS_INLINE_FUNCTION
State_t& operator-=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l-=r;}, lhs, rhs );
    return lhs;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND >
KOKKOS_INLINE_FUNCTION
State_t& operator-=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l-=rhs;}, lhs );
    return lhs;
}

// Operator *
template< State State_t > 
KOKKOS_INLINE_FUNCTION
State_t operator*(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l*r;}, res, lhs, rhs );
    return res;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND > 
KOKKOS_INLINE_FUNCTION
State_t operator*(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l*rhs;}, res, lhs );    
    return res;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND > 
KOKKOS_INLINE_FUNCTION
State_t operator*(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs*r;}, res, rhs );    
    return res;
}

template< State State_t > 
KOKKOS_INLINE_FUNCTION
State_t& operator*=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l*=r;}, lhs, rhs );
    return lhs;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND > 
KOKKOS_INLINE_FUNCTION
State_t& operator*=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l*=rhs;}, lhs );
    return lhs;
}

// Operator /
template< State State_t >
KOKKOS_INLINE_FUNCTION
State_t operator/(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l/r;}, res, lhs, rhs );
    return res;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND >
KOKKOS_INLINE_FUNCTION
State_t operator/(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l/rhs;}, res, lhs );    
    return res;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND > 
KOKKOS_INLINE_FUNCTION
State_t operator/(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs/r;}, res, rhs );    
    return res;
}

template< State State_t >
KOKKOS_INLINE_FUNCTION
State_t& operator/=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l/=r;}, lhs, rhs );
    return lhs;
}

template< State State_t CLANG_ISSUE_49197_WORKAROUND > 
KOKKOS_INLINE_FUNCTION
State_t& operator/=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l/=rhs;}, lhs );
    return lhs;
}


}