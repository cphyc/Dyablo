#pragma once

#include "user_data/UserData.h"

namespace dyablo {

template< typename Impl >
class AnalyticalFormula
{
public:
    Impl impl;

    AnalyticalFormula(ConfigMap& configMap)
     : impl(configMap)
    {}
    
    using State = typename Impl::State;

    std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo() const
    {
        return impl.getFieldsInfo();
    }

    KOKKOS_INLINE_FUNCTION
    void setState( const UserData::FieldAccessor& Uout, const ForeachCell::CellIndex& iCell, const State& u ) const
    {
        impl.setState(Uout, iCell, u);
    }

    /// the final value hydro state for the cell at position {x,y,z} with size {dx,dy,dz}    
    KOKKOS_INLINE_FUNCTION
    State value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    {
        return impl.value(x,y,z, dx,dy,dz);
    }
};

} // namespace dyablo