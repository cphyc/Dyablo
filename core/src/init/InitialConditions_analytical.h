#pragma once

#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "foreach_cell/ForeachCell.h"
#include "mpi/GhostCommunicator.h"
#include "refine_condition/RefineCondition.h"

namespace dyablo{


/** 
 * Helper to implement InitialConditions with analytical formula
 * User implements the AnalyticalFormula class that implements the AnalyticalFormula_base interface
 * 
 * And the init() method :
 * 1) initializes the AMR tree by calling successively need_refine for each levels between level_min and level_max
 * 2) Fills the final AMR mesh using value()
 **/
template< typename AnalyticalFormula_impl >
class InitialConditions_analytical : public InitialConditions
{ 
    AnalyticalFormula<AnalyticalFormula_impl> analytical_formula;
    std::unique_ptr<RefineCondition> refine_condition; 

    struct Data{
        ForeachCell& foreach_cell;
        int level_min, level_max;
    } data;
public:
  InitialConditions_analytical(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : analytical_formula(configMap),
    refine_condition( RefineConditionFactory::make_instance( 
        configMap.getValue<std::string>("amr", "markers_kernel", "RefineCondition_second_derivative_error"),
        configMap,
        foreach_cell,
        timers
      ) ),
    data({
      foreach_cell,
      foreach_cell.get_amr_mesh().get_level_min(),
      foreach_cell.get_amr_mesh().get_level_max()
    })
  {}

  void init( UserData& U )
  {
    ForeachCell& foreach_cell = data.foreach_cell;
    AMRmesh& pmesh     = foreach_cell.get_amr_mesh();

    uint32_t level_min = data.level_min;
    uint32_t level_max = data.level_max;

    auto& analytical_formula = this->analytical_formula;

    auto fields_info = analytical_formula.getFieldsInfo();
    {
        std::set<std::string> new_fields;
        for( const auto& fi : fields_info)
            if( !U.has_field(fi.name) ) 
                new_fields.insert(fi.name);
        U.new_fields( new_fields );
    }

    // Reallocate and fill U
    auto fill_U = [&]()
    {
        U.backup_and_realloc();

        ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
        const UserData::FieldAccessor Uout = U.getAccessor( fields_info );

        foreach_cell.foreach_cell( "InitialConditions_analytical::fill_U", Uout.getShape(),
                    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell_U )
        {
            ForeachCell::CellMetaData::pos_t c = cellmetadata.getCellCenter(iCell_U);
            ForeachCell::CellMetaData::pos_t s = cellmetadata.getCellSize(iCell_U);

            auto val = analytical_formula.value( c[IX], c[IY], c[IZ], s[IX], s[IY], s[IZ] );
            analytical_formula.setState( Uout, iCell_U, val );
        });
        
        int ghost_count = std::min( {U.getShape().bx, U.getShape().by, (uint32_t)4} );
        GhostCommunicator ghost_comm(pmesh, U.getShape(), ghost_count);
        ghost_comm.exchange_ghosts( Uout );
    };

    if( this->refine_condition )
    {
      RefineCondition& refine_condition = *(this->refine_condition);
      ScalarSimulationData scalar_data; // Empty scalardata
      // Refine until level_max using a RefineConditions plugin
      for (uint32_t level=level_min; level<level_max; ++level)
      {
          // Init fields for RefineConditions
          fill_U();

          refine_condition.mark_cells(U, scalar_data);
          
          // Refine the mesh according to markers
          pmesh.adapt();
          // Load balance at each level to avoid excessive inbalance
          pmesh.loadBalance();
      }
    }
    pmesh.loadBalance();

    // Reallocate and fill U fields
    fill_U();
    
  }  
}; 



} // namespace dyablo