#pragma once

namespace dyablo{

template< typename LegacyState_t >
class FiniteVolumePolicy_Slope_minmod
{
public:
  using PrimState = typename LegacyState_t::PrimState;
  using ConsState = typename LegacyState_t::ConsState;

  FiniteVolumePolicy_Slope_minmod( ConfigMap& configMap )
  {}

  static std::string name() {return "minmod";}

  KOKKOS_INLINE_FUNCTION
  PrimState compute_slope( PrimState qL, PrimState qC, PrimState qR, real_t dL, real_t dR) const
  {
    auto dqp = (qR - qC) / dR;
    auto dqm = (qC - qL) / dL;

    PrimState slope{};
    state_foreach_var([](real_t& res, real_t dvp, real_t dvm) {
      if (dvp * dvm <= 0.0)
        res = 0.0;
      else
        res = fabs(dvp) > fabs(dvm) ? dvm : dvp;
    }, slope, dqp, dqm);

    return slope;
  }

};

template< typename LegacyState_t >
class FiniteVolumePolicy_Slope_superbee
{
public:
  using PrimState = typename LegacyState_t::PrimState;
  using ConsState = typename LegacyState_t::ConsState;

  FiniteVolumePolicy_Slope_superbee( ConfigMap& configMap )
  {}

  static std::string name() {return "superbee";}

  KOKKOS_INLINE_FUNCTION
  PrimState compute_slope( PrimState qL, PrimState qC, PrimState qR, real_t dL, real_t dR) const
  {
    auto dqp = (qR - qC) / dR;
    auto dqm = (qC - qL) / dL;

    PrimState slope{};
    state_foreach_var([](real_t& res, real_t dvp, real_t dvm) {
      if( dvp * dvm <= 0 )
        res = 0.0;
      else
        res = (fabs(2.0 * dvp) < fabs(dvm) ? 2.0 * dvp : dvm);
    }, slope, dqp, dqm);

    return slope;
  }
};


template< typename F, typename Tuple_t, size_t... Is >
KOKKOS_INLINE_FUNCTION 
bool tuple_foreach_until( F f, const Tuple_t& t, std::index_sequence<Is...>)
{
  return ( f(Is, std::get<Is>(t)) || ... );
}

template< typename... Ts >
size_t tuple_find_name( std::string name, const std::tuple<Ts...>& t)
{
  size_t res;
  auto find_name = [&](size_t i, const auto& ti)
  {
    if( ti.name() == name ) 
      res = i;
    return ti.name() == name;
  };

  bool found = tuple_foreach_until( find_name, t, std::index_sequence_for<Ts...>{} );
  DYABLO_ASSERT_HOST_RELEASE( found, "Could not find slope limiter '" << name << "'" );

  return res;
}

template< typename F, typename... Ts >
KOKKOS_INLINE_FUNCTION 
auto tuple_apply_nth( int n, F f, const std::tuple<Ts...>& t)
{
  using ret_type = std::invoke_result_t< F, std::tuple_element_t<0, std::tuple<Ts...> > >;
  static_assert( ( std::is_same_v< ret_type, std::invoke_result_t< F, Ts > > && ... ), "Return type mismatch for F" );

  ret_type res;

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

template< typename LegacyState_t, typename... Ts >
class FiniteVolumePolicy_Slope_dynamic_impl
{
  std::tuple<Ts...> slopes;
  size_t selected_slope;
public:
  using PrimState = typename LegacyState_t::PrimState;
  using ConsState = typename LegacyState_t::ConsState;

  FiniteVolumePolicy_Slope_dynamic_impl( ConfigMap& configMap )
  : slopes(Ts(configMap)...)
  {
    std::string slope_limiter = configMap.getValue<std::string>("hydro", "slope_limiter", "minmod");

    this->selected_slope = tuple_find_name( slope_limiter, slopes );
  }

public:
  KOKKOS_INLINE_FUNCTION
  PrimState compute_slope( PrimState qL, PrimState qC, PrimState qR, real_t dL, real_t dR) const
  {
    return tuple_apply_nth( selected_slope, [&](const auto& s){return s.compute_slope( qL, qC, qR, dL, dR );}, slopes );
  }
};

template<typename LegacyState_t>
using FiniteVolumePolicy_Slope_dynamic = FiniteVolumePolicy_Slope_dynamic_impl< LegacyState_t,
  FiniteVolumePolicy_Slope_minmod<LegacyState_t>,
  FiniteVolumePolicy_Slope_superbee<LegacyState_t>
>;

} //namespace dyablo