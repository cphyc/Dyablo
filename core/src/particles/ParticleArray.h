#pragma once

#include <Kokkos_Core.hpp>

#include "real_type.h"
#include "enums.h"
#include "VarIndex.h"

namespace dyablo {

class ForeachParticle;

class ParticleArray
{
    //friend ForeachParticle;
    //friend UserData;
public:
    using ParticleIndex = uint32_t;
    ParticleArray() = default;
    ParticleArray( const ParticleArray& pa ) = default;
    ParticleArray( ParticleArray&& pa ) = default;
    ParticleArray& operator=( const ParticleArray& pa ) = default;
    ParticleArray& operator=( ParticleArray&& pa ) = default;    

    ParticleArray( const std::string& name, uint32_t count )
    : particle_position( name, count, 3 )
    {}

    KOKKOS_INLINE_FUNCTION
    uint32_t getNumParticles() const 
    {
        return particle_position.extent(0);
    }

    KOKKOS_INLINE_FUNCTION
    real_t& pos( const ParticleIndex& iPart, ComponentIndex3D iDir ) const
    {
        return particle_position(iPart, iDir);
    }

//protected:
    Kokkos::View< real_t**, Kokkos::LayoutLeft > particle_position;
};

class ParticleData : public ParticleArray
{
    //friend ForeachParticle;
    //friend UserData;
public:
    ParticleData() = default;
    ParticleData( const ParticleData& pa ) = default;
    ParticleData( ParticleData&& pa ) = default;
    ParticleData& operator=( const ParticleData& pa ) = default;
    ParticleData& operator=( ParticleData&& pa ) = default;  

    ParticleData( const ParticleArray& pa, uint32_t nbAttributes )
    : ParticleArray( pa ),
      particle_data( this->particle_position.label()+"_data", pa.getNumParticles(), nbAttributes )
    {}

    ParticleData( const std::string& name, uint32_t nbParticles, uint32_t nbAttributes )
    : ParticleArray( name, nbParticles ),
      particle_data( this->particle_position.label()+"_data", nbParticles, nbAttributes )
    {}

    KOKKOS_INLINE_FUNCTION
    int nbAttributes() const
    {
        return particle_data.extent(1);
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at( const ParticleIndex& iPart, VarIndex field ) const
    {
        return at_ivar( iPart, field );
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at_ivar( const ParticleIndex& iPart, int ivar ) const
    {
        return particle_data( iPart, ivar );
    }
//private:
    Kokkos::View< real_t**, Kokkos::LayoutLeft > particle_data;
};

} // namespace dyablo