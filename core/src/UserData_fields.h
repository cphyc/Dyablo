#pragma once

#include <map>
#include <vector>

#include "utils/config/ConfigMap.h"
#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

namespace dyablo {

class UserData_fields
{
public:
    /***
     * Nested class to access values in fields
     * UserData_fields should not be used directly in Kokkos kernels,
     * create a new FieldAccessor with get*Accessor() to use in kernels
     ***/
    class FieldAccessor; 
    struct FieldAccessor_FieldInfo;

public: // FIXME: When cuda is enabled, needs to be put public for some reason...
    struct field_index_t
    {
        int index;
    };

public:
    using FieldView_t = ForeachCell::CellArray_global_ghosted;

    UserData_fields( const UserData_fields& ) = default;
    UserData_fields( UserData_fields&& ) = default;

    UserData_fields( ConfigMap& configMap, ForeachCell& foreach_cell )
    :   foreach_cell(foreach_cell)
    {}

    /***
     * @brief Return a CellArray_global_ghosted::Shape_t instance 
     * with the same size as all fields in current UserData_fields
     * UserData_fields must have at least on active field
     ***/
    const FieldView_t::Shape_t getShape() const
    {
        DYABLO_ASSERT_HOST_RELEASE( field_index.size() > 0, "Cannot getShape() of an empty UserData_fields" );
        return fields.getShape();
    }

    void extend_fields( )
    {
        int nbOcts = this->foreach_cell.get_amr_mesh().getNumOctants();
        int nbGhosts = this->foreach_cell.get_amr_mesh().getNumGhosts();
        extend_fields_aux(foreach_cell, fields, nbOcts, nbGhosts, max_field_count);
    }

    static void extend_fields_aux( ForeachCell& foreach_cell, FieldView_t& fields, int nbOcts, int nbGhosts, int max_field_count )
    {
        int allocated_field_count = fields.nbfields();
        auto fields_new = foreach_cell.allocate_ghosted_array( "UserData_fields", FieldManager(max_field_count), nbOcts, nbGhosts );
        if( allocated_field_count != 0 )
        {
            Kokkos::deep_copy( 
                Kokkos::subview(fields_new.U, Kokkos::ALL(), std::pair(0,allocated_field_count), Kokkos::ALL() ),
                fields.U
            );
            Kokkos::deep_copy( 
                Kokkos::subview(fields_new.Ughost, Kokkos::ALL(), std::pair(0,allocated_field_count), Kokkos::ALL() ),
                fields.Ughost
            );
        }
        fields = fields_new;
    }
    
    static void new_fields_aux( const std::set<std::string>& names,
                                size_t nbOcts,
                                size_t nbGhosts,
                                int& max_field_count,
                                std::map<std::string, field_index_t>& field_index,
                                FieldView_t& fields,
                                ForeachCell& foreach_cell)
    {
        if( field_index.size() != 0 )
        {
            DYABLO_ASSERT_HOST_RELEASE( fields.U.extent(2) == nbOcts, "UserData_fields internal error : mismatch between allocated size and octant count" );
            DYABLO_ASSERT_HOST_RELEASE( fields.Ughost.extent(2) == nbGhosts, "UserData_fields internal error : mismatch between allocated size and ghost octant count" );
        }
        
        int needed_field_count = field_index.size() + names.size();
        max_field_count = std::max( max_field_count, needed_field_count );
        int allocated_field_count = fields.nbfields();
        if( needed_field_count > allocated_field_count )
        {   // Not enough fields : resize to add fields
            std::cout << "Reallocate : add fields " << allocated_field_count << " -> " << max_field_count << std::endl;
            extend_fields_aux(foreach_cell, fields, nbOcts, nbGhosts, max_field_count);
        }

        for( const std::string& name : names )
        {
            if( 1 == field_index.count(name) )
                throw std::runtime_error(std::string("UserData_fields::new_fields() - field already exists : ") + name);
            /// Find first free ivar in `fields` view
            auto first_free = [&]() -> int
            {
                for(int i=0; i<fields.nbfields(); i++)
                {
                    bool free = true;
                    for( auto& p : field_index )
                    {
                        if(p.second.index == i)
                            free = false;
                    }
                    if( free ) return i;
                }
                DYABLO_ASSERT_HOST_RELEASE(false, "UserData_fields internal error : not enough fields allocated");
                return -1;
            };
            
            int index = first_free();
            field_index[name].index = index;
            const auto& U = fields.U;
            Kokkos::parallel_for( "zero_new_field", U.extent(0)*U.extent(2),
                KOKKOS_LAMBDA( uint32_t i )
            {
                uint32_t iCell = i%U.extent(0);
                uint32_t iOct  = i/U.extent(0);
                U(iCell, index, iOct) = 0;
            });
            const auto& Ughost = fields.Ughost;
            Kokkos::parallel_for( "zero_new_field_ghost", Ughost.extent(0)*Ughost.extent(2),
                KOKKOS_LAMBDA( uint32_t i )
            {
                uint32_t iCell = i%Ughost.extent(0);
                uint32_t iOct  = i/Ughost.extent(0);
                Ughost(iCell, index, iOct) = 0;
            });
        }
    }

    /**
     * Add new fields with unique identifiers 
     * names should not be already present
     **/
    void new_fields( const std::set<std::string>& names)
    {
        size_t nbOcts = this->foreach_cell.get_amr_mesh().getNumOctants();
        size_t nbGhosts = this->foreach_cell.get_amr_mesh().getNumGhosts();
        int& max_field_count = this->max_field_count;
        std::map<std::string, field_index_t>& field_index = this->field_index;
        FieldView_t& fields = this->fields;

        new_fields_aux(names,nbOcts,nbGhosts,max_field_count,field_index,fields,this->foreach_cell);
    }

    void new_intermediate_fields( const std::set<std::string>& names)
    {
        size_t nbOcts = this->foreach_cell.get_amr_mesh().getLightOctree().getNumIntermediateOctants();
        size_t nbGhosts = this->foreach_cell.get_amr_mesh().getLightOctree().getNumIntermediateGhosts();
        int& max_field_count = this->max_field_count_intermediate;
        std::map<std::string, field_index_t>& field_index = this->field_index_intermediate;
        FieldView_t& fields = this->fields_intermediate;

        new_fields_aux(names,nbOcts,nbGhosts,max_field_count,field_index,fields,this->foreach_cell);
    }

    /// Check if field exists
    bool has_field(const std::string& name) const
    {
        return field_index.end() != field_index.find(name);
    }

    std::set<std::string> getEnabledFields() const
    {
        std::set<std::string> res;
        for( const auto& p : field_index )
        {
            res.insert( p.first );
        }
        return res;
    }   

    // Get View associated with field name
    const FieldView_t getField(const std::string& name) const
    {
        if( !this->has_field(name)  )
            throw std::runtime_error(std::string("UserData_fields::getField() - field doesn't exist : ") + name);
        
        int index = field_index.at(name).index;

        auto field = foreach_cell.allocate_ghosted_array( std::string("field_")+name, FieldManager(1) );
        Kokkos::deep_copy( 
            field.U,
            Kokkos::subview(fields.U, Kokkos::ALL(), std::pair(index, index+1) , Kokkos::ALL() )
        );

        return field;
    }

    /// Change field name from src to dest. If dest already exist it is replaced
    void move_field( const std::string& dest, const std::string& src )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_field(src), "UserData_fields::move_field() - field doesn't exist : " << src);

        field_index[ dest ] = field_index.at( src );
        field_index.erase( src );
    }

    void delete_field( const std::string& name )
    {
        field_index.erase( name );
    }

    void delete_intermediate_field( const std::string& name )
    {
        field_index_intermediate.erase( name );
    }

    /// Get the number of active fields in UserData_fields
    int nbFields() const
    {
        return field_index.size();
    }


    void exchange_loadbalance( const ViewCommunicator& ghost_comm );

    FieldAccessor getAccessor( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const;
    
    FieldAccessor getAccessor_intermediate( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const;

    FieldAccessor backup_and_realloc();

private:
    ForeachCell& foreach_cell;
    FieldView_t fields, fields_intermediate;
    std::map<std::string, field_index_t> field_index, field_index_intermediate;
    int max_field_count = 0, max_field_count_intermediate = 0;
};

struct UserData_fields::FieldAccessor_FieldInfo
{
    std::string name; /// Name as in VarIndex.h
    VarIndex id; /// id to use to access with at()
};

class GhostCommunicator_full_blocks;

class UserData_fields::FieldAccessor
{
friend GhostCommunicator_full_blocks;
friend UserData_fields;
public:
    static constexpr int MAX_FIELD_COUNT = id2index_t::MAX_INDEX_COUNT;
    using FieldInfo = FieldAccessor_FieldInfo;

    FieldAccessor() = default;
    FieldAccessor(const FieldAccessor& ) = default;
    FieldAccessor(FieldAccessor& ) = default;
    FieldAccessor& operator=(const FieldAccessor& ) = default;
    FieldAccessor& operator=(FieldAccessor& ) = default;

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return ivar_to_arrayindex.size();
    }

    FieldAccessor(const UserData_fields& user_data, const std::vector<FieldInfo>& fields_info)
    : FieldAccessor( user_data.fields,  user_data.field_index, fields_info)
    {}        

    FieldAccessor(const UserData_fields::FieldView_t& fields,
                  const std::map<std::string, field_index_t>& field_index, 
                  const std::vector<FieldInfo>& fields_info)
    : fields(fields)
    {
        DYABLO_ASSERT_HOST_RELEASE( fields_info.size() > 0, "fields_info cannot be empty" );

        auto unknown_field_error = [&](std::string field_name)
        {
            std::stringstream s;
            s << "Could not find field '" << field_name << "' in UserData" << std::endl;
            s << "Available fields are :" << std::endl;
            for( auto& p : field_index )
            {
                s << " - '" << p.first << "'" << std::endl;
            }
            return s.str();
        };

        int max_varindex = 0;
        for( const FieldInfo& info : fields_info )
        {
            max_varindex = std::max( max_varindex, info.id );
        }

        this->var_to_arrayindex = Kokkos::View<int*>( "varindex_to_viewindex", max_varindex+1 );
        this->ivar_to_arrayindex = Kokkos::View<int*>( "ivar_to_viewindex", fields_info.size() );
        auto var_to_arrayindex_host = Kokkos::create_mirror_view( var_to_arrayindex );
        this->ivar_to_arrayindex_host = Kokkos::create_mirror_view( ivar_to_arrayindex );
        for(size_t i=0; i<var_to_arrayindex_host.size(); i++)
            var_to_arrayindex_host(i) = -1;

        int i=0; 
        for( const FieldInfo& info : fields_info )
        {
            DYABLO_ASSERT_HOST_RELEASE( 1 == field_index.count(info.name), 
                                        unknown_field_error(info.name) );
            int index = field_index.at(info.name).index;
            var_to_arrayindex_host(info.id) = index;
            ivar_to_arrayindex_host(i) = index;
            i++;
        }
        Kokkos::deep_copy( var_to_arrayindex, var_to_arrayindex_host );
        Kokkos::deep_copy( ivar_to_arrayindex, ivar_to_arrayindex_host );
    }
    KOKKOS_INLINE_FUNCTION
    real_t& at( const ForeachCell::CellIndex& iCell, const VarIndex& varindex ) const
    {
        return fields.at_ivar( iCell, get_index_from_varindex(varindex) );
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at_ivar( const ForeachCell::CellIndex& iCell, int ivar ) const
    {
        return fields.at_ivar( iCell, get_index_from_ivar_device(ivar) );
    }

    KOKKOS_INLINE_FUNCTION
    FieldView_t::Shape_t getShape() const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG(nbFields() > 0, "Cannot getShape() of an empty UserData_fields" );
        return fields.getShape();
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
protected:
    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device(int ivar) const
    {
        return ivar_to_arrayindex(ivar);
    }
    int get_index_from_ivar_host(int ivar) const
    {
        return ivar_to_arrayindex_host(ivar);
    }
    FieldView_t fields;
};

inline void UserData_fields::exchange_loadbalance( const ViewCommunicator& ghost_comm )
{
    {  
        UserData_fields::FieldAccessor fields_old = this->backup_and_realloc();
        int nb_fields = this->nbFields();
        auto old_fields_View = Kokkos::subview( fields_old.fields.U, 
                                                Kokkos::ALL(),
                                                std::make_pair(0, nb_fields),
                                                Kokkos::ALL()  );
        auto new_fields_View = Kokkos::subview(this->fields.U, 
                                                Kokkos::ALL(),
                                                std::make_pair(0, nb_fields),
                                                Kokkos::ALL()  );
        
        ghost_comm.exchange_ghosts<2>(old_fields_View, new_fields_View );
    }// This block is important to deallocate fields_old before extend_fields()

    this->extend_fields();
}

inline UserData_fields::FieldAccessor UserData_fields::getAccessor( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const
{
    return FieldAccessor(*this, fields_info);
}

inline UserData_fields::FieldAccessor UserData_fields::getAccessor_intermediate( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const
{
    return FieldAccessor(this->fields_intermediate, this->field_index_intermediate, fields_info);
}

inline UserData_fields::FieldAccessor UserData_fields::backup_and_realloc()
{
    std::vector<FieldAccessor::FieldInfo> all_fields;
    int i=0;
    for( const std::string& field : this->getEnabledFields() )
        all_fields.push_back({field, i++});
    FieldAccessor fields_old = this->getAccessor( all_fields );

    this->fields = foreach_cell.allocate_ghosted_array( "UserData_fields", FieldManager(this->field_index.size()) );
    // Reorder fields to reduce fragmentation
    std::map<std::string, field_index_t> field_index_new;
    {
        int new_index = 0;
        for( const auto& [field_name, old_index] : this->field_index )
        {
            field_index_new[field_name].index = new_index;
            new_index++;
        }
    }
    this->field_index = field_index_new;

    return fields_old;
}


}// namespace dyablo