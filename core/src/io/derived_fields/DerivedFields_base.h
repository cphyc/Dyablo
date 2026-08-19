#pragma once 

#include "kokkos_shared.h"
#include "amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "user_data/UserData.h"
#include "utils/io/HDF5ViewWriter.h"

namespace dyablo{

/**
 * @brief Base class for derived fields
 * 
 * Derived fields are sub-plugins for IOManagers to output additional fields derived from
 * fields needed for the simulation. They are meant to be used only for output and are not used
 * in the rest of the simulation. 
 **/
class DerivedFields{
public:
  using OutputArray = ForeachCell::CellArray_global;
  // DerivedFields(
  //     ConfigMap& configMap,
  //     ForeachCell& foreach_cell,  
  //     Timers& timers);
  /** 
   * @brief Returns the name of all the new derived fields
   */
  virtual std::vector<std::string> get_fields_names() const = 0;
  
  /**
   * @brief Compute derived fields
   * 
   * Compute the derived field(s) using data for U and store them in out.
   * 
   * @param U input UserData containing simulation fields
   * @param out new derived fields are stored in out after execution.
   * 
   * Note : out must be allocated to the right size before calling compute_derived_fields()
   */
  virtual void compute_derived_fields( OutputArray &out, const UserData &U ) const = 0;
  virtual ~DerivedFields(){}
};

using DerivedFieldsFactory = RegisteringFactory<DerivedFields, 
  ConfigMap&,
  ForeachCell&, 
  Timers&>;


} // namespace dyablo