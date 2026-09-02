#pragma once

#include "InitialConditions_base.h"

#include "user_data/FieldAccessor.h"

namespace dyablo{

class InitialConditions_uniform : public InitialConditions
{
protected:
    ForeachCell& foreach_cell;
    std::vector<std::string> fields;
    std::vector<real_t> values;

    /// Simple Constructor to build deriverd InitialConditions with static values with static fields and values
    InitialConditions_uniform(ForeachCell& foreach_cell)
    : foreach_cell(foreach_cell)
    {}

public:
    InitialConditions_uniform(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
        :  InitialConditions_uniform(foreach_cell)
    {
        this->fields = configMap.getValue<std::vector<std::string>>( "InitialConditions_uniform", "fields" );
        this->values = configMap.getValue<std::vector<real_t>>( "InitialConditions_uniform", "values" );
    }


    void init( UserData& U )
    {
        DYABLO_ASSERT_HOST_RELEASE( values.size() >= fields.size(), "InitialConditions_uniform : too many fields, not enough values" );

        std::vector<UserData::FieldAccessor::FieldInfo> fields_info;
        std::set<std::string> new_fields;
        for( const std::string& field : fields )
        {
            if( U.has_field(field) )
            {
                U.delete_field(field);
                std::cout << "WARNING : field " <<  field << " exists but will be overwritten by InitialConditions_uniform" << std::endl;
            }
            new_fields.insert(field); 
            VarIndex ivar = fields_info.size();
            fields_info.push_back({field,ivar});
        }
        U.new_fields( new_fields ); 

        auto Uout = U.getAccessor( fields_info );
        
        Kokkos::View<real_t*> values_view("InitialConditions_uniform::values", values.size());
        {
            Kokkos::View<real_t*> values_view_cpu( values.data(), values.size() );
            Kokkos::deep_copy(values_view, values_view_cpu);
        }

        foreach_cell.foreach_cell( "InitialConditions_uniform::fill", U.getShape(),
            KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
        {
            for( size_t i=0; i<values_view.size(); i++ )
            {
                Uout.at(iCell, i) = values_view(i);
            };
        });
    }
};

} // namespace dyablo
