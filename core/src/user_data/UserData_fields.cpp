#include "FieldAccessor.h"

#include <map>
#include <vector>

#include "utils/config/ConfigMap.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo {

namespace UserData_Impl{

struct UserData_Fields_Pdata
{
  struct field_index_t
  {
      int index;
  };

  template<typename T>
  struct TypedFields
  {
    using View_t = ForeachCell::CellArray_global_ghosted_t<T>;
    View_t fields;
    std::map<std::string, field_index_t> field_index;
    int max_field_count = 0;
    View_t fields_intermediate;
    std::map<std::string, field_index_t> field_index_intermediate;
    int max_field_count_intermediate = 0;
  };
public:
    using FieldView_t = ForeachCell::CellArray_global_ghosted;

    UserData_Fields_Pdata( const UserData_Fields_Pdata& ) = default;
    UserData_Fields_Pdata( UserData_Fields_Pdata&& ) = default;

    UserData_Fields_Pdata( ConfigMap& configMap, ForeachCell& foreach_cell )
    :   foreach_cell(foreach_cell)
    {}

    /***
     * @brief Return a CellArray_global_ghosted::Shape_t instance 
     * with the same size as all fields in current UserData_fields
     * UserData_fields must have at least one active field
     * WARNING : resulting shape doesn't account for intermediates
     ***/
    template<typename T = real_t>
    const typename ForeachCell::CellArray_global_ghosted_t<T>::Shape_t getShape() const
    {
        const auto& typed = typed_fields<T>();
        DYABLO_ASSERT_HOST_RELEASE( typed.field_index.size() > 0, "Cannot getShape() of an empty UserData_fields" );
        return typed.fields.getShape();
    }

    template<typename T = real_t>
    void extend_fields( )
    {
        int nbOcts = this->foreach_cell.get_amr_mesh().getNumOctants();
        int nbGhosts = this->foreach_cell.get_amr_mesh().getNumGhosts();
        auto& typed = typed_fields<T>();
        extend_fields_aux(foreach_cell, typed.fields, nbOcts, nbGhosts, typed.max_field_count);
    }

    template<typename View_t>
    static void initialise_new_aux(View_t& fields, const int index)
    {
        const auto& U = fields._U;
        const uint32_t extent_0 = U.extent(0);
        const uint32_t extent_2 = U.extent(2);

        Kokkos::parallel_for( "zero_new_field", Kokkos::RangePolicy<>(0, extent_0*extent_2),
            KOKKOS_LAMBDA( const uint32_t i )
        {
            const uint32_t iCell = i%extent_0;
            const uint32_t iOct  = i/extent_0;
            U(iCell, index, iOct) = 0;
        });
    }

    template<typename View_t>
    static void extend_fields_aux( ForeachCell& foreach_cell, View_t& fields, uint32_t nbOcts, uint32_t nbGhosts, uint32_t max_field_count )
    {
        int allocated_field_count = fields.nbfields();
        FieldView_t::Shape_t shape{
          .bx = foreach_cell.blockSize()[IX],
          .by = foreach_cell.blockSize()[IY],
          .bz = foreach_cell.blockSize()[IZ],
          .nbFields = max_field_count,
          .nbOcts = nbOcts,
          .nbGhosts = nbGhosts,
        };
        View_t fields_new( "UserData_fields", shape );
        
        if( allocated_field_count != 0 )
        {
            Kokkos::deep_copy( 
                Kokkos::subview(fields_new._U, Kokkos::ALL(), std::pair(0,allocated_field_count), Kokkos::ALL() ),
                fields._U
            );
        }
        fields = fields_new;
    }

    template<typename View_t>
    static void new_fields_aux( const std::set<std::string>& names,
                                const size_t nbOcts,
                                const size_t nbGhosts,
                                int& max_field_count,
                                std::map<std::string, field_index_t>& field_index,
                                View_t& fields,
                                ForeachCell& foreach_cell)
    {
        if( field_index.size() != 0 )
        {
            DYABLO_ASSERT_HOST_RELEASE( fields._U.extent(2) == nbOcts+nbGhosts, "UserData_fields internal error : mismatch between allocated size and octant count" );
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
            initialise_new_aux(fields, index);
        }
    }

    /**
     * Add new fields with unique identifiers 
     * names should not be already present
     **/
    template<typename T = real_t>
    void new_fields( const std::set<std::string>& names)
    {
        static_assert(std::is_same_v<T, real_t> || std::is_same_v<T, float> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>,
                      "UserData fields support real_t, float, int32_t, and int64_t");
        auto& typed = typed_fields<T>();
        const size_t nbOcts = foreach_cell.get_amr_mesh().getNumOctants();
        const size_t nbGhosts = foreach_cell.get_amr_mesh().getNumGhosts();
        new_fields_aux(names, nbOcts, nbGhosts, typed.max_field_count,
                       typed.field_index, typed.fields, foreach_cell);
    }

    template<typename T = real_t>
    void new_intermediate_fields( const std::set<std::string>& names)
    {
        size_t nbOcts = this->foreach_cell.get_amr_mesh().getNumIntermediates();
        size_t nbGhosts = this->foreach_cell.get_amr_mesh().getNumIntermediateGhosts();
        auto& typed = typed_fields<T>();

        new_fields_aux(names, nbOcts, nbGhosts, typed.max_field_count_intermediate,
                       typed.field_index_intermediate, typed.fields_intermediate,
                       this->foreach_cell);
    }

    /// Check if field exists
    template<typename T = real_t>
    bool has_field(const std::string& name) const
    {
        const auto& typed = typed_fields<T>();
        return typed.field_index.end() != typed.field_index.find(name);
    }

    /// Check if field exists
    template<typename T = real_t>
    bool has_intermediate_field(const std::string& name) const
    {
        const auto& typed = typed_fields<T>();
        return typed.field_index_intermediate.end() != typed.field_index_intermediate.find(name);
    }

    template<typename T = real_t>
    std::set<std::string> getEnabledFields() const
    {
        std::set<std::string> res;
        for( const auto& p : typed_fields<T>().field_index )
        {
            res.insert( p.first );
        }
        return res;
    } 
    
    // Get View associated with field name
    template<typename T = real_t>
    const ForeachCell::CellArray_global_t<T> getFieldCopy(const std::string& name) const
    {
        const auto& typed = typed_fields<T>();
        if( !this->has_field<T>(name)  )
            throw std::runtime_error(std::string("UserData_fields::getFieldCopy() - field doesn't exist : ") + name);
        
        int index = typed.field_index.at(name).index;

        using GlobalView_t = ForeachCell::CellArray_global_t<T>;
        typename GlobalView_t::Shape_t shape{typed.fields.getShape().bx, typed.fields.getShape().by,
                                             typed.fields.getShape().bz, 1, typed.fields.getShape().nbOcts};
        GlobalView_t res( name+"_copy", shape);

        Kokkos::deep_copy( 
            res.subview_U(),
            Kokkos::subview(typed.fields.subview_U(), Kokkos::ALL(), std::pair(index, index+1) , Kokkos::ALL() )
        );

        return res;
    }

    /// Change field name from src to dest. If dest already exist it is replaced
    template<typename T = real_t>
    void move_field( const std::string& dest, const std::string& src )
    {
        auto& typed = typed_fields<T>();
        DYABLO_ASSERT_HOST_RELEASE( this->has_field<T>(src), "UserData_fields::move_field() - field doesn't exist : " << src);

        typed.field_index[ dest ] = typed.field_index.at( src );
        typed.field_index.erase( src );
    }

    template<typename T = real_t>
    void delete_field( const std::string& name )
    {
        typed_fields<T>().field_index.erase( name );
    }

    template<typename T = real_t>
    void clear_intermediates()
    {
        auto& typed = typed_fields<T>();
        typed.fields_intermediate = typename TypedFields<T>::View_t();
        typed.field_index_intermediate.clear();
    }

    /// Get the number of active fields in UserData_fields
    template<typename T = real_t>
    int nbFields() const
    {
        return typed_fields<T>().field_index.size();
    }

    template<typename T = real_t>
    int nbFields_intermediates()
    {
      return typed_fields<T>().field_index_intermediate.size();
    }

    void exchange_loadbalance( const ViewCommunicator& ghost_comm )
    {
      auto& typed = typed_fields<real_t>();
      DYABLO_ASSERT_HOST_RELEASE( 0 == nbFields_intermediates<real_t>(), "UserData::exchange_loadbalance : Keeping intermediates between interations is not supported yet" );
      {  
        UserData::FieldAccessor fields_old = this->backup_and_realloc();
        int nb_fields = this->nbFields<real_t>();
        auto old_fields_View = Kokkos::subview( fields_old.fields.subview_U(), 
                                                Kokkos::ALL(),
                                                std::make_pair(0, nb_fields),
                                                Kokkos::ALL()  );
        auto new_fields_View = Kokkos::subview(typed.fields.subview_U(), 
                                                Kokkos::ALL(),
                                                std::make_pair(0, nb_fields),
                                                Kokkos::ALL()  );
        
        ghost_comm.exchange_ghosts<2>(old_fields_View, new_fields_View );
      }// This block is important to deallocate fields_old before extend_fields()

      this->extend_fields<real_t>();
    }

    UserData::FieldAccessor backup_and_realloc()
    {
      DYABLO_ASSERT_HOST_RELEASE( 0 == nbFields_intermediates<real_t>(), "UserData::backup_and_realloc : Keeping intermediates between interations is not supported yet" );

      using FieldAccessor = UserData::FieldAccessor;
      std::vector<typename FieldAccessor::FieldInfo> all_fields;
      int i=0;
      for( const std::string& field : this->getEnabledFields<real_t>() )
        all_fields.push_back({field, i++});
      FieldAccessor fields_old( *this, all_fields );

      auto& typed = typed_fields<real_t>();
      typed.fields = foreach_cell.allocate_ghosted_array( "UserData_fields", typed.field_index.size() );
      // Reorder fields to reduce fragmentation
      std::map<std::string, field_index_t> field_index_new;
      {
        int new_index = 0;
        for( const auto& [field_name, old_index] : typed.field_index )
        {
          field_index_new[field_name].index = new_index;
          new_index++;
        }
      }
      typed.field_index = field_index_new;

      return fields_old;
    }

//private:
    ForeachCell& foreach_cell;
    TypedFields<real_t> fields_real;
    TypedFields<float> fields_float;
    TypedFields<int32_t> fields_int32;
    TypedFields<int64_t> fields_int64;

    template<typename T>
    TypedFields<T>& typed_fields()
    {
      if constexpr (std::is_same_v<T, real_t>) return fields_real;
      else if constexpr (std::is_same_v<T, float>) return fields_float;
      else if constexpr (std::is_same_v<T, int32_t>) return fields_int32;
      else return fields_int64;
    }

    template<typename T>
    const TypedFields<T>& typed_fields() const
    {
      if constexpr (std::is_same_v<T, real_t>) return fields_real;
      else if constexpr (std::is_same_v<T, float>) return fields_float;
      else if constexpr (std::is_same_v<T, int32_t>) return fields_int32;
      else return fields_int64;
    }
};

} //namespace UserData_Impl

namespace {

using Pdata = UserData_Impl::UserData_Fields_Pdata;
using FieldView_t = UserData::FieldView_t;

}

UserData::Fields::Fields(ConfigMap& configMap, ForeachCell& foreach_cell)
  : pdata( std::make_unique<Pdata>(configMap, foreach_cell) )
{}

UserData::Fields::~Fields()
{}

const FieldView_t::Shape_t UserData::getShape() const
{
  return this->fields.pdata->getShape();
}

template<typename T>
void UserData::new_fields( const std::set<std::string>& names)
{
  this->fields.pdata->new_fields<T>(names);
}

template<typename T>
void UserData::new_intermediate_fields( const std::set<std::string>& names)
{
  this->fields.pdata->new_intermediate_fields<T>(names);
}

template<typename T>
bool UserData::has_field(const std::string& name) const
{
  return this->fields.pdata->has_field<T>(name);
}

template<typename T>
std::set<std::string> UserData::getEnabledFields() const
{
  return this->fields.pdata->getEnabledFields<T>();
}

template<typename T>
const ForeachCell::CellArray_global_t<T> UserData::getFieldCopy(const std::string& name) const
{
  return this->fields.pdata->getFieldCopy<T>(name);
}

template<typename T>
void UserData::move_field( const std::string& dest, const std::string& src )
{
  this->fields.pdata->move_field<T>(dest, src);
}

template<typename T>
void UserData::delete_field( const std::string& name )
{
  this->fields.pdata->delete_field<T>(name);
}

template<typename T>
void UserData::clear_intermediates()
{
  this->fields.pdata->clear_intermediates<T>();
}

void UserData::exchange_loadbalance( const ViewCommunicator& ghost_comm )
{
  this->fields.pdata->exchange_loadbalance(ghost_comm);
}

template<typename T>
int UserData::nbFields() const
{
  return this->fields.pdata->nbFields<T>();
}

UserData::FieldAccessor UserData::backup_and_realloc()
{
  return this->fields.pdata->backup_and_realloc();
}

void UserData::extend_fields()
{
  this->fields.pdata->extend_fields();
}

template<typename T>
[[deprecated]] UserData::FieldAccessor_fulltree_t<T> UserData::getAccessor_intermediates( const std::vector<UserData::FieldAccessor_FieldInfo>& fields_info ) const
{
  return getAccessor_fulltree<T>(fields_info);
}

namespace UserData_Impl {

using FieldInfo = UserData_FieldAccessor_FieldInfo;

template<typename T>
void FieldAccessor_init( const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info,
                        int max_field_count, bool has_intermediates,
                        ForeachCell::CellArray_global_ghosted_t<T>& fields,
                        ForeachCell::CellArray_global_ghosted_t<T>& fields_intermediates
                      )
{
  if constexpr (std::is_same_v<T, real_t>)
  {
    fields = user_data.typed_fields<T>().fields;
    if( has_intermediates )
      fields_intermediates = user_data.typed_fields<T>().fields_intermediate;
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    fields = user_data.typed_fields<T>().fields;
    if( has_intermediates ) fields_intermediates = user_data.typed_fields<T>().fields_intermediate;
  }
  else if constexpr (std::is_same_v<T, int32_t>)
  {
    fields = user_data.typed_fields<T>().fields;
    if( has_intermediates ) fields_intermediates = user_data.typed_fields<T>().fields_intermediate;
  }
  else
  {
    fields = user_data.typed_fields<T>().fields;
    if( has_intermediates ) fields_intermediates = user_data.typed_fields<T>().fields_intermediate;
  }
}


template<typename T>
void FieldAccessor_FieldManager_init_static( const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info,
                        int max_field_count, bool has_intermediates,
                        int& _nbFields,
                        int* var_to_arrayindex,
                        int* ivar_to_arrayindex
                      )
{
    DYABLO_ASSERT_HOST_RELEASE( fields_info.size() > 0, "fields_info cannot be empty" );

    auto unknown_field_error = [&](std::string field_name)
    {
        std::stringstream s;
        s << "Could not find field '" << field_name << "' in UserData" << std::endl;
        s << "Available fields are :" << std::endl;
        for( auto& p : user_data.typed_fields<T>().field_index )
        {
            s << " - '" << p.first << "'" << std::endl;
        }
        return s.str();
    };

    _nbFields = fields_info.size();

    DYABLO_ASSERT_HOST_RELEASE( _nbFields <= max_field_count, "Too many fields for FieldAccessor, use getAccessor<_MAX_FIELD_COUNT>() to increase number of fields." );
    for( [[maybe_unused]] const FieldInfo& info : fields_info )
    {
      DYABLO_ASSERT_HOST_RELEASE( info.id >= 0 && info.id < max_field_count, "VarIndex must be included in [0,MAX_FIELD_COUNT[. "
                                                                             "VarIndex is " << info.id << ", MAX_FIELD_COUNT is " << max_field_count  );
    }

    for(int i=0; i<max_field_count; i++)
    {
      ivar_to_arrayindex[i] = 0;
      var_to_arrayindex[i] = -1;
    }

    int i=0; 
    for( const FieldInfo& info : fields_info )
    {
        DYABLO_ASSERT_HOST_RELEASE( has_intermediates ? user_data.has_intermediate_field<T>(info.name) : user_data.has_field<T>(info.name), 
                                    unknown_field_error(info.name) );
        int index = has_intermediates ? 
                        user_data.typed_fields<T>().field_index_intermediate.at(info.name).index
                      : user_data.typed_fields<T>().field_index.at(info.name).index;
        var_to_arrayindex[info.id] = index;
        ivar_to_arrayindex[i] = index;
        i++;
    }
}

template void FieldAccessor_init<real_t>( const UserData_Fields_Pdata&, const std::vector<FieldInfo>&,
                                          int, bool,
                                          ForeachCell::CellArray_global_ghosted_t<real_t>&,
                                          ForeachCell::CellArray_global_ghosted_t<real_t>& );
template void FieldAccessor_init<float>( const UserData_Fields_Pdata&, const std::vector<FieldInfo>&,
                                         int, bool,
                                         ForeachCell::CellArray_global_ghosted_t<float>&,
                                         ForeachCell::CellArray_global_ghosted_t<float>& );
template void FieldAccessor_init<int32_t>( const UserData_Fields_Pdata&, const std::vector<FieldInfo>&,
                                           int, bool,
                                           ForeachCell::CellArray_global_ghosted_t<int32_t>&,
                                           ForeachCell::CellArray_global_ghosted_t<int32_t>& );
template void FieldAccessor_init<int64_t>( const UserData_Fields_Pdata&, const std::vector<FieldInfo>&,
                                           int, bool,
                                           ForeachCell::CellArray_global_ghosted_t<int64_t>&,
                                           ForeachCell::CellArray_global_ghosted_t<int64_t>& );

template<typename T>
FieldAccessor_FieldManager<-1, T>::FieldAccessor_FieldManager( const UserData_Fields_Pdata& user_data, const std::vector<FieldInfo>& fields_info, bool has_intermediates )
  : var_to_arrayindex("var_to_arrayindex", fields_info.size()),
    ivar_to_arrayindex_device("ivar_to_arrayindex", fields_info.size()),
    ivar_to_arrayindex_host( Kokkos::create_mirror_view(ivar_to_arrayindex_device) )
{
  auto var_to_arrayindex_host = Kokkos::create_mirror_view(var_to_arrayindex);

  int dummy;
  FieldAccessor_FieldManager_init_static<T>( user_data, fields_info,
    fields_info.size(), has_intermediates,
    dummy, ivar_to_arrayindex_host.data(), var_to_arrayindex_host.data() );

  Kokkos::deep_copy( var_to_arrayindex, var_to_arrayindex_host );
  Kokkos::deep_copy( ivar_to_arrayindex_device, ivar_to_arrayindex_host );
}


} // namespace UserData_Impl
} // namespace dyablo

template void dyablo::UserData::new_fields<real_t>( const std::set<std::string>& );
template void dyablo::UserData::new_fields<float>( const std::set<std::string>& );
template void dyablo::UserData::new_fields<int32_t>( const std::set<std::string>& );
template void dyablo::UserData::new_fields<int64_t>( const std::set<std::string>& );
template void dyablo::UserData::new_intermediate_fields<real_t>( const std::set<std::string>& );
template void dyablo::UserData::new_intermediate_fields<float>( const std::set<std::string>& );
template void dyablo::UserData::new_intermediate_fields<int32_t>( const std::set<std::string>& );
template void dyablo::UserData::new_intermediate_fields<int64_t>( const std::set<std::string>& );
template bool dyablo::UserData::has_field<real_t>( const std::string& ) const;
template bool dyablo::UserData::has_field<float>( const std::string& ) const;
template bool dyablo::UserData::has_field<int32_t>( const std::string& ) const;
template bool dyablo::UserData::has_field<int64_t>( const std::string& ) const;
template std::set<std::string> dyablo::UserData::getEnabledFields<real_t>() const;
template std::set<std::string> dyablo::UserData::getEnabledFields<float>() const;
template std::set<std::string> dyablo::UserData::getEnabledFields<int32_t>() const;
template std::set<std::string> dyablo::UserData::getEnabledFields<int64_t>() const;
template void dyablo::UserData::move_field<real_t>( const std::string&, const std::string& );
template void dyablo::UserData::move_field<float>( const std::string&, const std::string& );
template void dyablo::UserData::move_field<int32_t>( const std::string&, const std::string& );
template void dyablo::UserData::move_field<int64_t>( const std::string&, const std::string& );
template void dyablo::UserData::delete_field<real_t>( const std::string& );
template void dyablo::UserData::delete_field<float>( const std::string& );
template void dyablo::UserData::delete_field<int32_t>( const std::string& );
template void dyablo::UserData::delete_field<int64_t>( const std::string& );
template void dyablo::UserData::clear_intermediates<real_t>();
template void dyablo::UserData::clear_intermediates<float>();
template void dyablo::UserData::clear_intermediates<int32_t>();
template void dyablo::UserData::clear_intermediates<int64_t>();
template int dyablo::UserData::nbFields<real_t>() const;
template int dyablo::UserData::nbFields<float>() const;
template int dyablo::UserData::nbFields<int32_t>() const;
template int dyablo::UserData::nbFields<int64_t>() const;
