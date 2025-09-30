#pragma once

#include <map>
#include <vector>

#include "utils/config/ConfigMap.h"
#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

namespace dyablo {

class UserData_particles
{
public:
    /***
     * Nested class to access values in fields
     * UserData_particles should not be used directly in Kokkos kernels,
     * create a new FieldAccessor with get*Accessor() to use in kernels
     ***/
    class ParticleAccessor; 
    struct ParticleAccessor_AttributeInfo;

    using ParticleArray_t = ParticleArray;
    using ParticleAttribute_t = ParticleData;

    class ParticleContainer
    {
        friend UserData_particles::ParticleAccessor;

    public:
        ParticleContainer( const ParticleContainer& ) = default;
        ParticleContainer( ParticleContainer&& ) = default;
        ParticleContainer( const std::string& name, const ForeachParticle& foreach_particle, uint32_t num_particles )
          : name(name), foreach_particle(foreach_particle), particles(name, num_particles, FieldManager(0))
        {}
        int nbAttributes() const
        {
            return attribute_index.size();
        }
        void new_ParticleAttribute( const std::string& attribute_name )
        {
            int nb_new_attributes = 1;
            int needed_attr_count = nbAttributes() + nb_new_attributes;
            this->max_particle_count = std::max( this->max_particle_count, needed_attr_count );
            int allocated_attr_count = nbAttributes();
            if( needed_attr_count > allocated_attr_count )
            {
                ParticleData particles_new( particles, FieldManager(needed_attr_count) );
                if( allocated_attr_count != 0 )
                {
                    Kokkos::deep_copy( 
                        Kokkos::subview( particles_new.particle_data, Kokkos::ALL(), std::pair(0,allocated_attr_count) ),
                        particles.particle_data
                    );
                }
                this->particles = particles_new;
            }

            for( const std::string& name : {attribute_name} )
            {
                DYABLO_ASSERT_HOST_RELEASE( !has_ParticleAttribute(name), "new_ParticleAttribute() - attribute already exist : " << name );
            
                auto first_free = [&]() -> int
                {
                    for(int i=0; i<particles.nbfields(); i++)
                    {
                        bool free = true;
                        for( auto& p : attribute_index )
                        {
                            if(p.second == i)
                                free = false;
                        }
                        if( free ) return i;
                    }
                    DYABLO_ASSERT_HOST_RELEASE(false, "new_ParticleAttribute internal error : not enough fields allocated");
                    return -1;
                };

                int index = first_free();
                attribute_index[name] = index;
                
                const auto& particles = this->particles;

                foreach_particle.foreach_particle( "zero_attribute", particles,
                    KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
                {
                    particles.at_ivar(iPart, index) = 0;
                });
            }
            
        }
        bool has_ParticleAttribute( const std::string& name ) const
        {
            return attribute_index.end() != attribute_index.find(name); 
        }
        std::set<std::string> getEnabledParticleAttributes() const
        {
            std::set<std::string> res;
            for( const auto& p : attribute_index )
            {
                res.insert( p.first );
            }
            return res;
        }
        ParticleArray_t getParticleArray() const
        {
            return particles;
        }
        ParticleAttribute_t getParticleAttribute( const std::string& attribute_name ) const
        {
            ParticleAttribute_t res( particles, FieldManager(1) );

            int index = attribute_index.at(attribute_name);

            const auto& particles = this->particles;
            foreach_particle.foreach_particle( "copy_attr", particles,
                KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
            {
                res.at_ivar(iPart, 0) = particles.at_ivar( iPart, index );
            });

            return res;
        }
        void move_ParticleAttribute( const std::string& attr_dest, const std::string& attr_src )
        {
            DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleAttribute(attr_src), "move_ParticleAttribute() - source attribute doesn't exist : " << attr_src);

            attribute_index[ attr_dest ] = attribute_index.at(attr_src);
            attribute_index.erase( attr_src );
        }
        void delete_ParticleAttribute( const std::string& attribute_name)
        {
            attribute_index.erase( attribute_name );
        }

        void resize( uint32_t new_num_particles )
        {
            uint32_t old_size = particles.getNumParticles();
            if (new_num_particles == old_size) return;

            size_t Nprops = particles.particle_data.extent(1);
            Kokkos::resize(particles.particle_data, new_num_particles, Nprops);
            size_t Ndim = particles.particle_position.extent(1);
            Kokkos::resize(particles.particle_position, new_num_particles, Ndim);
        }

        void distributeParticles()
        {
            ViewCommunicator part_comm = foreach_particle.get_distribute_communicator( particles );
            uint32_t nbParticles_new = part_comm.getNumGhosts();

            ParticleAttribute_t particles_new( this->name, nbParticles_new, FieldManager(particles.nbfields()) );

            part_comm.exchange_ghosts<0>( particles.particle_data, particles_new.particle_data );
            part_comm.exchange_ghosts<0>( particles.particle_position, particles_new.particle_position );

            this->particles = particles_new;
        }

    private:
        std::string name;
        ForeachParticle foreach_particle;
        ParticleAttribute_t particles;
        std::map<std::string, int> attribute_index;
        int max_particle_count = 0;
    };
    

public:
    UserData_particles( const UserData_particles& ) = default;
    UserData_particles( UserData_particles&& ) = default;

    UserData_particles( ConfigMap& configMap, ForeachCell& foreach_cell )
    :   foreach_particle( foreach_cell.get_amr_mesh(), configMap )
    {}    

    /// Create a new particle array
    void new_ParticleArray( const std::string& name, uint32_t num_particles )
    {
        DYABLO_ASSERT_HOST_RELEASE( !this->has_ParticleArray(name), "UserData_particles::new_ParticleArray() - particle array already exists : " << name );
        particle_containers.emplace( name, ParticleContainer(name, foreach_particle, num_particles) );
    }

    void delete_ParticleArray( const std::string& name )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleArray(name), "UserData_particles::delete_ParticleArray() - particle array does not exist : " << name );
        particle_containers.erase( name );
    }

    void merge_particles_if( const std::string& id_dest, const std::string& id_to_merge, const std::string& mask_field );

private:
    ParticleContainer& getParticleContainer( const std::string& array_name )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleArray(array_name), "Particle array does not exist : " << array_name );
        return particle_containers.at(array_name);
    }

    const ParticleContainer& getParticleContainer( const std::string& array_name ) const
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleArray(array_name), "Particle array does not exist : " << array_name );
        return particle_containers.at(array_name);
    }

public:
    /// Create a new attribute for particle array `array_name` (must exist)
    void new_ParticleAttribute( const std::string& array_name, const std::string& attribute_name )
    {
        ParticleContainer& array = getParticleContainer( array_name );
        DYABLO_ASSERT_HOST_RELEASE( !array.has_ParticleAttribute(attribute_name), "UserData_particles::new_ParticleAttribute() - particle attribute already exists : " << attribute_name );
        array.new_ParticleAttribute( attribute_name );
    }

    /// Check if UserData_particles contains a ParticleArray with this name
    bool has_ParticleArray(const std::string& name) const
    {
        return particle_containers.end() != particle_containers.find(name);
    }

    // Check is ParticleArray `array_name` (must exist) has an attribute wiuth this name
    bool has_ParticleAttribute(const std::string& array_name, const std::string& attribute_name ) const
    {
        return getParticleContainer( array_name ).has_ParticleAttribute( attribute_name );
    }

    /// Get identifiers for all enabled particle arrays
    std::set<std::string> getEnabledParticleArrays() const
    {
        std::set<std::string> res;
        for( const auto& p : particle_containers )
        {
            res.insert( p.first );
        }
        return res;
    } 

    /// Get identifiers for all enabled attribudes of particle array `array_name`
    std::set<std::string> getEnabledParticleAttributes( const std::string& array_name ) const
    {
        return getParticleContainer(array_name).getEnabledParticleAttributes();
    } 

    // Get particle array associated with name
    ParticleArray_t getParticleArray( const std::string& array_name ) const
    {        
        return getParticleContainer(array_name).getParticleArray();;
    }
    
    // Get particle attribute with name `attribute_name` from particle array `array_name`
    ParticleAttribute_t getParticleAttribute( const std::string& array_name, const std::string& attribute_name ) const
    {
        return getParticleContainer(array_name).getParticleAttribute(attribute_name);
    }

    /***
     * @brief Change name of particle attribute from `array_name`/`attr_src` to `array_name`/`attr_dest`
     * NOTE : order of parameters is dest, src like in Kokkos deep_copy
     * WARNING : Invalidates all accessors containing source attribute 
     ***/
    void move_ParticleAttribute( const std::string& array_name, const std::string& attr_dest, const std::string& attr_src )
    {
        getParticleContainer(array_name).move_ParticleAttribute(attr_dest, attr_src);   
    }

    /***
     * @brief Delete attribte `array_name`/`attribute_name` from user data
     * WARNING : Invalidates all accessors containing this attribute
     ***/
    void delete_ParticleAttribute(const std::string& array_name, const std::string& attribute_name)
    {
        getParticleContainer(array_name).delete_ParticleAttribute(attribute_name);
    }

    /***
     * @brief create a ParticleAccessor to access attributes listed in `attribute_info` from particle array `array_name`
     * NOTE : Accessors may be invalidated by some methods from UserData_particles (e.g. deleting or moving an attribute or an array)
     * Do not keep invalidated accessors since live accessors may prevent Kokkos::View deallocation in when deleting or moving user data
     ***/
    ParticleAccessor getParticleAccessor( const std::string& array_name, const std::vector<ParticleAccessor_AttributeInfo>& attribute_info ) const;

    /***
     * @brief Distribute position array and attributes for particle array `array_name`
     * WARNING : Invalidates all accessors containing this particle array
     ***/
    void distributeParticles( const std::string& array_name )
    {
        getParticleContainer(array_name).distributeParticles();
    }
    void distributeAllParticles()
    {
        for(auto& pair : particle_containers)
            pair.second.distributeParticles();
    }

private:
    ForeachParticle foreach_particle;
    std::map<std::string, ParticleContainer> particle_containers;
};

struct UserData_particles::ParticleAccessor_AttributeInfo
{
    std::string name; /// Name as in VarIndex.h
    VarIndex id; /// id to use to access with at()
};

class UserData_particles::ParticleAccessor
{
public:
    using AttributeInfo = ParticleAccessor_AttributeInfo;

    ParticleAccessor() = default;
    ParticleAccessor(const ParticleAccessor& ) = default;
    ParticleAccessor(ParticleAccessor& ) = default;
    ParticleAccessor& operator=(const ParticleAccessor& ) = default;
    ParticleAccessor& operator=(ParticleAccessor& ) = default;

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return ivar_to_arrayindex.size();
    }

    ParticleAccessor(const UserData_particles& user_data, const std::string& array_name, const std::vector<AttributeInfo>& attr_info)
        : ParticleAccessor( user_data.getParticleContainer( array_name ), attr_info )
    {}
    
    ParticleAccessor( const ParticleContainer& particles, const std::vector<AttributeInfo>& attr_info )
     : particles(particles.particles)
    {
        DYABLO_ASSERT_HOST_RELEASE( attr_info.size() > 0, "fields_info cannot be empty" );

        auto unknown_attr_error = [&](std::string attr_name)
        {
            std::stringstream s;
            s << "Could not find particle attribute '" << attr_name << "' in ParticleArray" << std::endl;
            s << "Available attributes are :" << std::endl;
            for( const std::string& particle_attr : particles.getEnabledParticleAttributes() )
            {
                s << " - '" << particle_attr << "'" << std::endl;
            }
            return s.str();
        };

        int max_varindex = 0;
        for( const AttributeInfo& info : attr_info )
        {
            max_varindex = std::max( max_varindex, info.id );
        }

        this->var_to_arrayindex = Kokkos::View<int*>( "varindex_to_viewindex", max_varindex+1 );
        this->ivar_to_arrayindex = Kokkos::View<int*>( "ivar_to_viewindex", attr_info.size() );
        auto var_to_arrayindex_host = Kokkos::create_mirror_view( var_to_arrayindex );
        this->ivar_to_arrayindex_host = Kokkos::create_mirror_view( ivar_to_arrayindex );
        for(size_t i=0; i<var_to_arrayindex_host.size(); i++)
            var_to_arrayindex_host(i) = -1;

        int i=0; 
        for( const AttributeInfo& info : attr_info )
        {
            DYABLO_ASSERT_HOST_RELEASE( particles.has_ParticleAttribute(info.name),
                                        unknown_attr_error(info.name));
            int index = particles.attribute_index.at(info.name);
            var_to_arrayindex_host(info.id) = index;
            ivar_to_arrayindex_host(i) = index;
            i++;
        }
        Kokkos::deep_copy( var_to_arrayindex, var_to_arrayindex_host );
        Kokkos::deep_copy( ivar_to_arrayindex, ivar_to_arrayindex_host );
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at( const ForeachParticle::ParticleIndex& iPart, const VarIndex& varindex ) const
    {
        return particles.at_ivar( iPart, get_index_from_varindex(varindex));
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at_ivar( const ForeachParticle::ParticleIndex& iPart, int ivar ) const
    {
        return particles.at_ivar( iPart, get_index_from_ivar_device(ivar));
    }

    KOKKOS_INLINE_FUNCTION
    ParticleArray_t getShape() const
    {
        return particles;
    }

private:
    Kokkos::View<int*> var_to_arrayindex; // Index conversion for at()
    Kokkos::View<int*> ivar_to_arrayindex; // Index conversion for at_ivar()
    Kokkos::View<int*>::HostMirror ivar_to_arrayindex_host;

    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex(VarIndex var) const
    {
        return var_to_arrayindex(var);
    }

    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device(int ivar) const
    {
        return ivar_to_arrayindex(ivar);
    }
protected:
    ParticleData particles;
};

inline UserData_particles::ParticleAccessor UserData_particles::getParticleAccessor( const std::string& array_name, const std::vector<ParticleAccessor_AttributeInfo>& attribute_info ) const
{
    return ParticleAccessor( *this, array_name, attribute_info );
}

inline void UserData_particles::merge_particles_if( const std::string& id_dest, const std::string& id_to_merge, const std::string& mask_field )
{
    DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleArray(id_dest), "merge_particles_if error : destination array '"<<id_dest<<"' does not exist" );
    DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleArray(id_to_merge), "merge_particles_if error : array to merge '"<<id_to_merge<<"' does not exist" );
    DYABLO_ASSERT_HOST_RELEASE( this->has_ParticleAttribute(id_to_merge, mask_field), "merge_particles_if error : mask field '"<<id_to_merge<<"/"<<mask_field<<"' does not exist" );

    std::vector<ParticleAccessor::AttributeInfo> all_attr;
    VarIndex Imask = -1;
    {
        for( const std::string& attr : this->getEnabledParticleAttributes( id_dest ) )
        {
            VarIndex ivar = all_attr.size();
            all_attr.push_back({attr, ivar});
            if( attr == mask_field )
            {
                Imask = ivar;
            }
        }
    }
    int nb_attr = all_attr.size();

    int n_to_merge = 0;
    {
        auto Pto_merge = this->getParticleArray(id_to_merge);
        auto Pmask = this->getParticleAccessor(id_to_merge, {{mask_field, 0}});   
        foreach_particle.reduce_particle("count_to_merge", Pto_merge,
            KOKKOS_LAMBDA(const ForeachParticle::ParticleIndex& iPart, int& count)
        {
            if( Pmask.at(iPart, 0) != 0.0 )
                count++;
        }, n_to_merge);
    }

    auto& Pout_container = this->getParticleContainer( id_dest );
    uint32_t Pout_initial_size = Pout_container.getParticleArray().getNumParticles();
    Pout_container.resize( Pout_initial_size + n_to_merge );
    
    {
        auto Pin = this->getParticleArray(id_to_merge);
        auto Pin_data = this->getParticleAccessor( id_to_merge, all_attr );

        auto Pout = this->getParticleArray(id_dest);
        auto Pout_data = this->getParticleAccessor( id_dest, all_attr );
    
        Kokkos::View<uint32_t> count("count");
        foreach_particle.foreach_particle( "merge_copy", Pin,
            KOKKOS_LAMBDA(const ForeachParticle::ParticleIndex& iPart_in)
        {   
            if( Pin_data.at_ivar(iPart_in, Imask) != 0 )
            {         
                ForeachParticle::ParticleIndex iPart_out = Pout_initial_size + Kokkos::atomic_fetch_add(&count(), 1);

                Pout.pos( iPart_out, IX ) = Pin.pos( iPart_in, IX );
                Pout.pos( iPart_out, IY ) = Pin.pos( iPart_in, IY );
                Pout.pos( iPart_out, IZ ) = Pin.pos( iPart_in, IZ );

                for( int i=0; i<nb_attr; i++ )
                {
                    Pout_data.at_ivar(iPart_out, i) = Pin_data.at_ivar(iPart_in, i);
                }
            }
        });
    }        
    
    this->delete_ParticleArray(id_to_merge);
}


}// namespace dyablo