#include "Units.h"

#include "utils/misc/Dyablo_assert.h"

namespace dyablo{
namespace Units{

namespace {
auto& code_units_impl()
{
    struct code_units_container{
        UnitSystem code_units;
        bool initialized;
    };
    static code_units_container res {};
    return res;
}
} // namespace

void code_units_init( const UnitSystem& code_units )
{
    code_units_impl().initialized = true;
    code_units_impl().code_units = code_units;
}

const UnitSystem& code_units()
{
    DYABLO_ASSERT_HOST_RELEASE( code_units_impl().initialized, "Code units are used but not initialized yet!" );
    return code_units_impl().code_units;
}

} // namespace Units
} // namespace dyablo