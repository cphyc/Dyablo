#pragma once

#include "user_data/UserData.h" // IWYU pragma: export
#include "particles/ForeachParticle.h"

namespace dyablo {
namespace UserData_Impl{

struct UserData_ParticleAccessor_AttributeInfo
{
    std::string name; /// Name as in VarIndex.h
    VarIndex id; /// id to use to access with at()
};

class UserData_ParticleAccessor
{
public:
    using AttributeInfo = UserData_ParticleAccessor_AttributeInfo;

    UserData_ParticleAccessor() = default;
    UserData_ParticleAccessor(const UserData_ParticleAccessor& ) = default;
    UserData_ParticleAccessor(UserData_ParticleAccessor& ) = default;
    UserData_ParticleAccessor& operator=(const UserData_ParticleAccessor& ) = default;
    UserData_ParticleAccessor& operator=(UserData_ParticleAccessor& ) = default;

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return ivar_to_arrayindex.size();
    }

    UserData_ParticleAccessor(const UserData_Particles_Pdata& user_data, const std::string& array_name, const std::vector<AttributeInfo>& attr_info);
    
    KOKKOS_INLINE_FUNCTION
    real_t& at( const ParticleArray::ParticleIndex& iPart, const VarIndex& varindex ) const
    {
        return particles.at_ivar( iPart, var_to_arrayindex(varindex));
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at_ivar( const ForeachParticle::ParticleIndex& iPart, int ivar ) const
    {
        return particles.at_ivar( iPart, ivar_to_arrayindex(ivar));
    }

    KOKKOS_INLINE_FUNCTION
    ParticleArray getShape() const
    {
        return particles;
    }

private:
    Kokkos::View<int*> var_to_arrayindex; // Index conversion for at()
    Kokkos::View<int*> ivar_to_arrayindex; // Index conversion for at_ivar()

protected:
    ParticleData particles;
};

} //namespace UserData_Impl
} //namespace dyablo