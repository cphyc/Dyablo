#pragma once

#include "InitialConditions_base.h"

namespace dyablo{

class InitialConditions_uniform : public InitialConditions
{
protected:
    ForeachCell& foreach_cell;
    std::vector<std::string> fields;
    std::vector<real_t> values;
    std::vector<bool> conservative;

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
        if (configMap.hasValue("InitialConditions_uniform", "conservative")) {
            this->conservative = configMap.getValue<std::vector<bool>>( "InitialConditions_uniform", "conservative", {});
            DYABLO_ASSERT_HOST_RELEASE( this->conservative.size() == this->fields.size(),
                "InitialConditions_uniform : size of conservative vector (" << this->conservative.size() << ") does not match size of fields vector (" << this->fields.size() << ")" );
        }
        else
            this->conservative = std::vector<bool>( this->fields.size(), true );
    }


    void init( UserData& U )
    {
        DYABLO_ASSERT_HOST_RELEASE( values.size() >= fields.size(), "InitialConditions_uniform : too many fields, not enough values" );

        std::vector<UserData::FieldAccessor::FieldInfo> fields_info;
        fields_info.push_back( {"rho", (VarIndex)0} ); // Assume rho is field 0
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

        Kokkos::View<uint8_t*> conservative_view("InitialConditions_uniform::conservative", conservative.size());
        {
            // First convert to uint8_t to avoid issues with bool specialization in Kokkos
            std::vector<uint8_t> conservative_uint(conservative.begin(), conservative.end());
            Kokkos::View<uint8_t*> conservative_view_cpu( conservative_uint.data(), conservative_uint.size() );
            Kokkos::deep_copy(conservative_view, conservative_view_cpu);
        }

        foreach_cell.foreach_cell( "InitialConditions_uniform::fill", U.getShape(),
            KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
        {
            for( size_t i=0; i<values_view.size(); i++ )
            {
                if (conservative_view(i))
                    Uout.at(iCell, i+1) = values_view(i);
                else
                    Uout.at(iCell, i+1) = values_view(i) * Uout.at(iCell, 0);
            };
        });
    }
};

} // namespace dyablo
