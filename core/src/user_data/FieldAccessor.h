#pragma once

#include <type_traits>

#include "user_data/UserData.h" // IWYU pragma: export
#include "utils/misc/dyablo_tuple.h"

namespace dyablo {

class GhostCommunicator_full_blocks;

namespace UserData_Impl{

struct UserData_FieldAccessor_FieldInfo
{
  std::string name; /// Name as in VarIndex.h
  VarIndex id; /// id to use to access with at()
};

void FieldAccessor_init(const UserData_Fields_Pdata& user_data, const std::vector<UserData_FieldAccessor_FieldInfo>& fields_info,
                        int max_field_count, bool has_intermediates,
                        UserData::FieldView_t& fields,
                        UserData::FieldView_t& fields_intermediates
                      );

void FieldAccessor_FieldManager_init_static( const UserData_Fields_Pdata& user_data, const std::vector<UserData_FieldAccessor_FieldInfo>& fields_info,
                        int max_field_count, bool has_intermediates,
                        int& _nbFields,
                        int* var_to_arrayindex,
                        int* ivar_to_arrayindex
                      );

template< int _MAX_FIELD_COUNT >
class FieldAccessor_FieldManager
{
    template< int > friend class FieldAccessor_FieldManager; // For conversions from/to FieldAccessor_FieldManager<DYNAMIC_FIELD_COUNT>
private:
    static constexpr int MAX_FIELD_COUNT = _MAX_FIELD_COUNT;
    using FieldInfo = UserData_FieldAccessor_FieldInfo;

    int _nbFields;
    Kokkos::Array<int, MAX_FIELD_COUNT> var_to_arrayindex; // Index conversion for at()
    Kokkos::Array<int, MAX_FIELD_COUNT> ivar_to_arrayindex; // Index conversion for at_ivar()

public:
    FieldAccessor_FieldManager() = default;
    FieldAccessor_FieldManager(const FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager(FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager& operator=(const FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager& operator=(FieldAccessor_FieldManager& ) = default;

    FieldAccessor_FieldManager(const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info, bool has_intermediates)
    {
        FieldAccessor_FieldManager_init_static( user_data, fields_info, 
                                                MAX_FIELD_COUNT, has_intermediates,
                                                this->_nbFields,
                                                this->var_to_arrayindex.data(),
                                                this->ivar_to_arrayindex.data() );
    }

    /**
     * Conversion from a FieldAccessor_FieldManager with a dynamic number of fields
     * Checks (DYABLO_ASSERT_HOST_RELEASE) that the number of fields and every VarIndex fit in MAX_FIELD_COUNT
     * Defined after FieldAccessor_FieldManager<DYNAMIC_FIELD_COUNT>
     **/
    explicit FieldAccessor_FieldManager(const FieldAccessor_FieldManager<DYNAMIC_FIELD_COUNT>& o);

    /**
     * FieldManager for all the fields of a CellArray : VarIndex i and ivar i are field i of the array
     * Checks (DYABLO_ASSERT_HOST_RELEASE) that nbFields fits in MAX_FIELD_COUNT
     **/
    static FieldAccessor_FieldManager identity( int nbFields )
    {
        DYABLO_ASSERT_HOST_RELEASE( nbFields <= MAX_FIELD_COUNT,
            "Too many fields to convert to FieldAccessor : " << nbFields << " fields, MAX_FIELD_COUNT is " << MAX_FIELD_COUNT );
        FieldAccessor_FieldManager res;
        res._nbFields = nbFields;
        for( int i=0; i<MAX_FIELD_COUNT; i++ )
        {
            res.ivar_to_arrayindex[i] = ( i < nbFields ) ? i : 0;
            res.var_to_arrayindex[i] = ( i < nbFields ) ? i : -1;
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return _nbFields;
    }

    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex(VarIndex var) const
    {
        return var_to_arrayindex[var];
    }

    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device(int ivar) const
    {
        return ivar_to_arrayindex[ivar];
    }

    int get_index_from_ivar_host(int ivar) const
    {
        return ivar_to_arrayindex[ivar];
    }
};

template<>
class FieldAccessor_FieldManager<DYNAMIC_FIELD_COUNT>
{
    template< int > friend class FieldAccessor_FieldManager; // For conversions from/to FieldAccessor_FieldManager<MAX_FIELD_COUNT>
private:
    static constexpr int MAX_FIELD_COUNT = 3;
    using FieldInfo = UserData_FieldAccessor_FieldInfo;

    Kokkos::View<int*> var_to_arrayindex; // Index conversion for at()
    Kokkos::View<int*> ivar_to_arrayindex_device; // Index conversion for at_ivar()
    Kokkos::View<int*>::host_mirror_type ivar_to_arrayindex_host;

public:
    FieldAccessor_FieldManager() = default;
    FieldAccessor_FieldManager(const FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager(FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager& operator=(const FieldAccessor_FieldManager& ) = default;
    FieldAccessor_FieldManager& operator=(FieldAccessor_FieldManager& ) = default;

    FieldAccessor_FieldManager(const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info, bool has_intermediates);

    /// Conversion from a FieldAccessor_FieldManager with a static number of fields
    template< int MAX_FIELD_COUNT_o >
    requires ( MAX_FIELD_COUNT_o >= 0 )
    FieldAccessor_FieldManager(const FieldAccessor_FieldManager<MAX_FIELD_COUNT_o>& o)
      : var_to_arrayindex("var_to_arrayindex", MAX_FIELD_COUNT_o),
        ivar_to_arrayindex_device("ivar_to_arrayindex", o.nbFields()),
        ivar_to_arrayindex_host( Kokkos::create_mirror_view(ivar_to_arrayindex_device) )
    {
        auto var_to_arrayindex_host = Kokkos::create_mirror_view(var_to_arrayindex);
        for( int i=0; i<MAX_FIELD_COUNT_o; i++ )
            var_to_arrayindex_host(i) = o.var_to_arrayindex[i];
        for( int i=0; i<o.nbFields(); i++ )
            ivar_to_arrayindex_host(i) = o.ivar_to_arrayindex[i];
        Kokkos::deep_copy( var_to_arrayindex, var_to_arrayindex_host );
        Kokkos::deep_copy( ivar_to_arrayindex_device, ivar_to_arrayindex_host );
    }

    /// FieldManager for all the fields of a CellArray : VarIndex i and ivar i are field i of the array
    static FieldAccessor_FieldManager identity( int nbFields )
    {
        FieldAccessor_FieldManager res;
        res.var_to_arrayindex = Kokkos::View<int*>("var_to_arrayindex", nbFields);
        res.ivar_to_arrayindex_device = Kokkos::View<int*>("ivar_to_arrayindex", nbFields);
        res.ivar_to_arrayindex_host = Kokkos::create_mirror_view(res.ivar_to_arrayindex_device);
        for( int i=0; i<nbFields; i++ )
            res.ivar_to_arrayindex_host(i) = i;
        Kokkos::deep_copy( res.ivar_to_arrayindex_device, res.ivar_to_arrayindex_host );
        Kokkos::deep_copy( res.var_to_arrayindex, res.ivar_to_arrayindex_host );
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return ivar_to_arrayindex_device.size();
    }

    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex(VarIndex var) const
    {
        return var_to_arrayindex(var);
    }

    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device(int ivar) const
    {
        return ivar_to_arrayindex_device(ivar);
    }

    int get_index_from_ivar_host(int ivar) const
    {
        return ivar_to_arrayindex_host(ivar);
    }
};

template< int _MAX_FIELD_COUNT >
FieldAccessor_FieldManager<_MAX_FIELD_COUNT>::FieldAccessor_FieldManager(const FieldAccessor_FieldManager<DYNAMIC_FIELD_COUNT>& o)
  : _nbFields( o.nbFields() )
{
    DYABLO_ASSERT_HOST_RELEASE( _nbFields <= MAX_FIELD_COUNT,
        "Too many fields to convert to FieldAccessor : " << _nbFields << " fields, MAX_FIELD_COUNT is " << MAX_FIELD_COUNT );

    for( int i=0; i<MAX_FIELD_COUNT; i++ )
    {
        ivar_to_arrayindex[i] = 0;
        var_to_arrayindex[i] = -1;
    }

    for( int i=0; i<_nbFields; i++ )
        ivar_to_arrayindex[i] = o.ivar_to_arrayindex_host(i);

    auto var_to_arrayindex_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), o.var_to_arrayindex );
    for( int var=0; var<(int)var_to_arrayindex_host.size(); var++ )
    {
        if( var < MAX_FIELD_COUNT )
            var_to_arrayindex[var] = var_to_arrayindex_host(var);
        else
            DYABLO_ASSERT_HOST_RELEASE( var_to_arrayindex_host(var) == -1,
                "VarIndex " << var << " is used and cannot be converted to FieldAccessor : VarIndex must be in [0," << MAX_FIELD_COUNT << "[" );
    }
}

template< bool has_intermediates, int _MAX_FIELD_COUNT >
class UserData_FieldAccessor_impl
{
friend GhostCommunicator_full_blocks;
friend UserData_Fields_Pdata;
template< bool, int > friend class UserData_FieldAccessor_impl; // For conversions between static and dynamic number of fields
public:
    static constexpr int MAX_FIELD_COUNT = _MAX_FIELD_COUNT;
    using FieldInfo = UserData_FieldAccessor_FieldInfo;
    using FieldView_t = UserData::FieldView_t;
    using FieldManager = FieldAccessor_FieldManager<MAX_FIELD_COUNT>;

    UserData_FieldAccessor_impl() = default;
    UserData_FieldAccessor_impl(const UserData_FieldAccessor_impl& ) = default;
    UserData_FieldAccessor_impl(UserData_FieldAccessor_impl& ) = default;
    UserData_FieldAccessor_impl& operator=(const UserData_FieldAccessor_impl& ) = default;
    UserData_FieldAccessor_impl& operator=(UserData_FieldAccessor_impl& ) = default;

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return fm.nbFields();
    }

    UserData_FieldAccessor_impl(const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info)
        : fm( user_data, fields_info, false )
    {
        if constexpr (has_intermediates)
            fm_intermediates = FieldManager( user_data, fields_info, true );

        UserData_Impl::FieldAccessor_init(
            user_data, fields_info,
            MAX_FIELD_COUNT, has_intermediates,
            this->fields, this->fields_intermediates
        );
    }

    /**
     * Conversion from an accessor with a static number of fields to an accessor with a dynamic
     * number of fields (MAX_FIELD_COUNT == DYNAMIC_FIELD_COUNT)
     * Copies field indices to device : avoid in performance-critical loops
     **/
    template< int MAX_FIELD_COUNT_o >
    requires ( MAX_FIELD_COUNT == DYNAMIC_FIELD_COUNT && MAX_FIELD_COUNT_o >= 0 )
    UserData_FieldAccessor_impl(const UserData_FieldAccessor_impl<has_intermediates, MAX_FIELD_COUNT_o>& o)
        : fm( o.fm ),
          fields( o.fields ),
          fields_intermediates( o.fields_intermediates )
    {
        if constexpr (has_intermediates)
            fm_intermediates = FieldManager( o.fm_intermediates );
    }

    /**
     * Conversion from an accessor with a dynamic number of fields (MAX_FIELD_COUNT_o == DYNAMIC_FIELD_COUNT) to an accessor
     * with a static number of fields
     * Undefined behavior if the number of fields or the biggest ivar is too big (usually DYABLO_ASSERT)
     **/
    template< int MAX_FIELD_COUNT_o >
    requires ( MAX_FIELD_COUNT >= 0 && MAX_FIELD_COUNT_o == DYNAMIC_FIELD_COUNT )
    explicit UserData_FieldAccessor_impl(const UserData_FieldAccessor_impl<has_intermediates, MAX_FIELD_COUNT_o>& o)
        : fm( o.fm ),
          fields( o.fields ),
          fields_intermediates( o.fields_intermediates )
    {
        if constexpr (has_intermediates)
            fm_intermediates = FieldManager( o.fm_intermediates );
    }

    /**
     * Accessor for all the fields of a CellArray : VarIndex i and ivar i are field i of the array
     * With a static number of fields, checks (DYABLO_ASSERT_HOST_RELEASE) that the fields fit in MAX_FIELD_COUNT
     **/
    explicit UserData_FieldAccessor_impl(const FieldView_t& U)
    requires ( !has_intermediates )
        : fm( FieldManager::identity( U.nbFields() ) ),
          fields( U )
    {}

    template< typename ... VarIndex_s >
    KOKKOS_INLINE_FUNCTION
    decltype(auto) at( const ForeachCell::CellIndex& iCell, VarIndex varindex_0, VarIndex_s... varindex_s) const
    {
        bool is_intermediate = has_intermediates && iCell.iOct().isIntermediate;

        real_t* origin = this->at( iCell );
        auto value = [&]( VarIndex varindex ) -> real_t&
        {
            auto offset = is_intermediate ?
                  fields_intermediates.get_offset_ivar( get_index_from_varindex_intermediates(varindex) )
                : fields.get_offset_ivar( get_index_from_varindex(varindex) ) ;

            return origin[offset];
        };

        if constexpr ( sizeof...(VarIndex_s) == 0 )
            return (value(varindex_0)); // Parenthesis are important here to keep real_t& reference
        else
            return dyablo_tuple_tie( value(varindex_0), value(varindex_s)... );
    }

    template< typename ... Int_t >
    KOKKOS_INLINE_FUNCTION
    decltype(auto) at_ivar( const ForeachCell::CellIndex& iCell, int ivar0, Int_t... ivars) const
    {
        bool is_intermediate = has_intermediates && iCell.iOct().isIntermediate;

        real_t* origin = this->at( iCell );
        auto value = [&]( int ivar ) -> real_t&
        {
            auto offset = is_intermediate ?
                  fields_intermediates.get_offset_ivar( get_index_from_ivar_device_intermediates(ivar) )
                : fields.get_offset_ivar( get_index_from_ivar_device(ivar) ) ;

            return origin[offset];
        };

        if constexpr ( sizeof...(Int_t) == 0 )
            return (value(ivar0));  // Parenthesis are important here to keep real_t& reference
        else
            return dyablo_tuple_tie( value(ivar0), value(ivars)... );
    }
    

    KOKKOS_INLINE_FUNCTION
    FieldView_t::Shape_t getShape() const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG(nbFields() > 0, "Cannot getShape() of an empty UserData_fields" );
        auto iter_space = fields.getShape();
        auto iter_space_inter = fields_intermediates.getShape();
        return FieldView_t::Shape_t{
            .bx = iter_space.bx,
            .by = iter_space.by,
            .bz = iter_space.bz,
            .nbFields = (uint32_t)nbFields(),
            .nbOcts = iter_space.nbOcts,
            .nbGhosts = iter_space.nbGhosts,
            .nbIntermediateOcts = iter_space_inter.nbOcts,
            .nbIntermediateGhosts = iter_space_inter.nbGhosts,
        };
    }

private:    
    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex(VarIndex var) const
    {
        return fm.get_index_from_varindex(var);
    }

    KOKKOS_INLINE_FUNCTION
    int get_index_from_varindex_intermediates(VarIndex var) const
    {
        return fm_intermediates.get_index_from_varindex(var);
    }
protected:
    FieldAccessor_FieldManager<_MAX_FIELD_COUNT> fm;
    FieldAccessor_FieldManager<_MAX_FIELD_COUNT> fm_intermediates;

    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device(int ivar) const
    {
        return fm.get_index_from_ivar_device(ivar);
    }
    int get_index_from_ivar_host(int ivar) const
    {
        return fm.get_index_from_ivar_host(ivar);
    }
    FieldView_t fields;

    KOKKOS_INLINE_FUNCTION
    int get_index_from_ivar_device_intermediates(int ivar) const
    {
        return fm_intermediates.get_index_from_ivar_device(ivar);
    }
    int get_index_from_ivar_host_intermediates(int ivar) const
    {
        return fm_intermediates.get_index_from_ivar_host(ivar);
    }
    FieldView_t fields_intermediates;

    KOKKOS_INLINE_FUNCTION
    real_t* at( const ForeachCell::CellIndex& iCell ) const
    {
        if constexpr ( has_intermediates )
        {
            if( iCell.iOct().isIntermediate )
            {
                // Remove isIntermediate flag from iCell
                auto iOct_intermediate = iCell.iOct();
                iOct_intermediate.isIntermediate=false;
                ForeachCell::CellIndex iCell_intermediate = iCell;
                iCell_intermediate.set_iOct(iOct_intermediate);
                return fields_intermediates.at( iCell_intermediate );
            }
        }
        DYABLO_ASSERT_KOKKOS_DEBUG( !iCell.iOct().isIntermediate, "Accessing intermediates in array without intermediates" );
        return fields.at( iCell );
    }
};

/// Accessor with a dynamic number of fields : UserData::FieldAccessor_impl<DYNAMIC_FIELD_COUNT> or UserData::FieldAccessor_fulltree_impl<DYNAMIC_FIELD_COUNT>
template< bool has_intermediates >
using FieldAccessor_dynamic = UserData_FieldAccessor_impl<has_intermediates, DYNAMIC_FIELD_COUNT>;

template< typename Array_t, bool has_intermediates >
inline constexpr bool converts_to_FieldAccessor_dynamic = std::is_constructible_v<FieldAccessor_dynamic<has_intermediates>, const Array_t&>;

} //namespace UserData_Impl

/**
 * Arrays that can be converted (explicitly or not) to an accessor with a dynamic number of fields :
 * UserData::FieldAccessor_impl<DYNAMIC_FIELD_COUNT> or UserData::FieldAccessor_fulltree_impl<DYNAMIC_FIELD_COUNT>
 * (e.g. FieldAccessor_impl<N>, FieldAccessor_fulltree_impl<N>, CellArray_global_ghosted)
 **/
template< typename Array_t >
concept FieldAccessor_dynamic_convertible = UserData_Impl::converts_to_FieldAccessor_dynamic<Array_t, false>
                                         || UserData_Impl::converts_to_FieldAccessor_dynamic<Array_t, true>;

/// Accessor with a dynamic number of fields to convert Array_t to (with intermediates when possible)
template< FieldAccessor_dynamic_convertible Array_t >
using FieldAccessor_dynamic_t = UserData_Impl::FieldAccessor_dynamic< UserData_Impl::converts_to_FieldAccessor_dynamic<Array_t, true> >;

} //namespace dyablo