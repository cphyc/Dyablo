#pragma once

#include "amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "user_data/UserData.h"
#include "mpi/GhostCommunicator_full_blocks.h"

namespace dyablo {

/**
 * Base class for MapUserData plugin
 * Contains the remap() method to fill new mesh after AMR refinement/corasening 
 **/
class MapUserData{
public: 
  // MapUserData(
  //     ConfigMap& configMap,
  //     ForeachCell& foreach_cell,
  //     Timers& timers );
  virtual ~MapUserData(){}

  /**
   * Save a snapshot of the current AMR mesh in ForeachCell
   * This should be called before calling AMRmesh::refine()
   **/
  virtual void save_old_mesh( UserData& user_data ) = 0;
  
  /**
   * Fill new array Uout from Uin
   * @param Uin A cell array on the old amr mesh (mesh at last save_old_mesh() call) to copy data from
   * @param Uout An array to store new cells for the new mesh to write new mesh data to
   **/
  virtual void remap( UserData& user_data ) = 0;
};

class CellIndexRemapper;

/**
 * Implementation of MapUserData using mean of smaller cells when coarseneing
 **/
class MapUserData_base : public MapUserData{
public: 
  MapUserData_base(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : foreach_cell(foreach_cell)
  {}
  
  ~MapUserData_base(){}

  void save_old_mesh(UserData& user_data) override;

  void remap( UserData& user_data ) override;
    
  virtual void remap_aux( const UserData::FieldAccessor_t<real_t>&, const UserData::FieldAccessor_t<real_t>&, const CellIndexRemapper& ) = 0;
  virtual void remap_aux( const UserData::FieldAccessor_t<float>&, const UserData::FieldAccessor_t<float>&, const CellIndexRemapper& ) = 0;
  virtual void remap_aux( const UserData::FieldAccessor_t<int32_t>&, const UserData::FieldAccessor_t<int32_t>&, const CellIndexRemapper& ) = 0;
  virtual void remap_aux( const UserData::FieldAccessor_t<int64_t>&, const UserData::FieldAccessor_t<int64_t>&, const CellIndexRemapper& ) = 0;

protected:
  ForeachCell& foreach_cell;
  LightOctree lmesh_old;
  std::unique_ptr<GhostCommunicator_full_blocks> ghost_comm_full;
};

using MapUserDataFactory = RegisteringFactory< MapUserData, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 