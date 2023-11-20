#include "io/IOManager_base.h"

namespace dyablo { 


class IOManager_diagnostics : public IOManager{
public: 
  IOManager_diagnostics(
    ConfigMap& configMap,
    ForeachCell& foreach_cell,
    Timers& timers )
  : foreach_cell(foreach_cell)
  {}

  void save_snapshot( const UserData& U_, ScalarSimulationData& scalar_data )
  {
    using CellIndex = ForeachCell::CellIndex;

    int iter = scalar_data.get<int>("iter");

    enum VarIndex_diag { ID, IE };
    auto cells = foreach_cell.getCellMetaData();
    UserData::FieldAccessor Uin = U_.getAccessor( {{"rho", ID}, {"e_tot", IE}} );

    real_t total_mass_local = 0;
    real_t total_energy_local = 0;
    foreach_cell.reduce_cell( "diagnostics", U_.getShape(),
      KOKKOS_LAMBDA( const CellIndex& iCell, real_t& mass, real_t& energy )
    { 
      auto size = cells.getCellSize(iCell);
      real_t V = size[IX]*size[IY]*size[IZ];      
      mass += Uin.at( iCell, ID ) * V;
      energy += Uin.at(iCell, IE)*V;
      
    }, total_mass_local, total_energy_local);

    MpiComm mpi_comm = foreach_cell.get_amr_mesh().getMpiComm();

    real_t total_mass = 0;
    real_t total_energy = 0;

    mpi_comm.MPI_Allreduce(&total_mass_local, &total_mass, 1, MpiComm::MPI_Op_t::SUM);
    mpi_comm.MPI_Allreduce(&total_energy_local, &total_energy, 1, MpiComm::MPI_Op_t::SUM);

    // Only print on rank 0
    if( mpi_comm.MPI_Comm_rank() == 0 )
    {
      if( !initialized )
      {
        this->initialized = true;

        this->mass_initial = total_mass;
        this->mass_last = total_mass;

        this->energy_initial = total_energy;
        this->energy_last = total_energy;

        std::cout << "Variable, iter : value, diff (abs), error (rel), cumulative diff (abs)" << std::endl;
      }

      real_t mass_error_last = total_mass - mass_last;
      real_t mass_error_cumulative = total_mass - mass_initial;
      std::cout << "Mass  , " << iter << " : " << total_mass << ", " <<  mass_error_last << ", " << mass_error_last/mass_last << ", " << mass_error_cumulative << std::endl;
      this->mass_last = total_mass;

      real_t energy_error_last = total_energy - energy_last;
      real_t energy_error_cumulative = total_energy - energy_initial;
      std::cout << "Energy, " << iter << " : " << total_energy << ", " <<  energy_error_last << ", " << energy_error_last/energy_last << ", " << energy_error_cumulative << std::endl;
      this->energy_last = total_energy;
    }
  }
  
private:
  ForeachCell& foreach_cell;
  bool initialized = false;
  real_t mass_initial, mass_last;
  real_t energy_initial, energy_last;

};

}// namespace dyablo


FACTORY_REGISTER( dyablo::IOManagerFactory, dyablo::IOManager_diagnostics, "IOManager_diagnostics" );

