#pragma once

#include "Units.h"

namespace dyablo {

namespace Units{

struct UnitDims
{
    int Time, Length, Mass, Current, Temp, mol, LuminousIntensity;
    template< int... Dims >
    static UnitDims from_unit( const Unit<Dims...>& u)
    {
        return UnitDims{Dims...};
    }    
};

real_t parse_unit( const std::string& str, const UnitDims& expected_dims );

template<typename Unit_t> 
Unit_t parse_unit( const std::string& str )
{
    using traits = Units::Unit_traits<Unit_t>;
    static_assert( traits::is_unit, "parse_unit : expected type is not a unit" );

    real_t SI_value = parse_unit( str, UnitDims::from_unit( Unit_t(1) ) );   

    return Unit_t( SI_value );
}

} // namespace Units
} // namespace dyablo