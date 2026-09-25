/**
 * \file test_UserData_typed.cpp
 * Tests for UserData fields stored as float next to real_t (double) fields
 *
 * Every cell gets a value encoded from its position so that local cells, ghosts,
 * load-balanced, remapped and restarted cells can all be checked exactly, on any number of ranks.
 */

#include "gtest/gtest.h"

#include <map>
#include <set>
#include <string>
#include <type_traits>

#include "amr/AMRmesh.h"
#include "amr/MapUserData.h"
#include "init/InitialConditions.h"
#include "io/IOManager.h"
#include "mpi/GhostCommunicator.h"
#include "mpi/GhostCommunicator_full_blocks.h"
#include "user_data/UserData.h"
#include "user_data/FieldAccessor.h"
#include "DyabloSession.hpp"
#include "utils/mpi/MpiBufferPool.h"

namespace dyablo {
namespace {

using CellIndex = ForeachCell::CellIndex;

constexpr int level_min = 3;
constexpr int level_max = 5;
constexpr uint32_t bsize = 4;
// Cell centers are at half-integer multiples of the finest cell size, so 2*pos/dx_min is an integer
constexpr real_t two_inv_dx = 2 * (1 << level_max) * bsize; // 256 -> 8 bits per dimension

/// Exactly representable in float and double, so both types hold the same value
template< typename T >
KOKKOS_INLINE_FUNCTION
T encode( real_t x, real_t y, real_t z )
{
  int64_t ix = (int64_t)(x*two_inv_dx + 0.5);
  int64_t iy = (int64_t)(y*two_inv_dx + 0.5);
  int64_t iz = (int64_t)(z*two_inv_dx + 0.5);
  int64_t code = 1 + ix + (iy << 8) + (iz << 16); // never 0, < 2^24
  return (T)code * (T)0.25;
}

const char* mesh_config = R"ini(
[mesh]
ndim=3
[amr]
bx=4
by=4
bz=4
level_min=3
level_max=5
)ini";

/// Mesh with a local refinement at the end of rank 0 so that load balancing moves octants
std::shared_ptr<AMRmesh> make_mesh()
{
  auto amr_mesh = std::make_shared<AMRmesh>( 3, std::array<bool,3>{false,false,false}, level_min, level_max );
  for( int i=0; i<2; i++ )
  {
    if( amr_mesh->getMpiComm().MPI_Comm_rank() == 0 )
      amr_mesh->setMarker( amr_mesh->getNumOctants()-1, 1 );
    amr_mesh->adapt();
  }
  return amr_mesh;
}

int global_sum( int v )
{
  int res = 0;
  GlobalMpiSession::get_comm_world().MPI_Allreduce( &v, &res, 1, MpiComm::MPI_Op_t::SUM );
  return res;
}

template< typename T >
void init_field( ForeachCell& foreach_cell, UserData& U, const std::string& name )
{
  auto acc = U.getAccessor<T>( {{name, 0}} );
  auto cells = foreach_cell.getCellMetaData();
  foreach_cell.foreach_cell( "init_typed", acc.getShape(),
    KOKKOS_LAMBDA( const CellIndex& iCell )
  {
    auto c = cells.getCellCenter( iCell );
    acc.at_ivar( iCell, 0 ) = encode<T>( c[IX], c[IY], c[IZ] );
  });
}

template< typename T >
int count_local_errors( ForeachCell& foreach_cell, const UserData& U, const std::string& name )
{
  auto acc = U.getAccessor<T>( {{name, 0}} );
  auto cells = foreach_cell.getCellMetaData();
  int errors = 0;
  foreach_cell.reduce_cell( "check_typed", acc.getShape(),
    KOKKOS_LAMBDA( const CellIndex& iCell, int& errors )
  {
    auto c = cells.getCellCenter( iCell );
    if( acc.at_ivar( iCell, 0 ) != encode<T>( c[IX], c[IY], c[IZ] ) )
      errors++;
  }, errors);
  return errors;
}

/// Number of cells where a float copy differs from `encode`
int count_copy_errors( ForeachCell& foreach_cell, const ForeachCell::CellArray_global_t<float>& copy )
{
  auto cells = foreach_cell.getCellMetaData();
  int errors = 0;
  foreach_cell.reduce_cell( "check_copy", copy.getShape(),
    KOKKOS_LAMBDA( const CellIndex& iCell, int& errors )
  {
    auto c = cells.getCellCenter( iCell );
    if( copy.at_ivar( iCell, 0 ) != encode<float>( c[IX], c[IY], c[IZ] ) )
      errors++;
  }, errors);
  return errors;
}

/// Number of cells where the float field differs from the real_t field beyond float rounding
int count_mismatches( ForeachCell& foreach_cell, const ForeachCell::CellArray_global_t<real_t>& rho,
                      const ForeachCell::CellArray_global_t<float>& f32 )
{
  int errors = 0;
  foreach_cell.reduce_cell( "compare_remap", rho.getShape(),
    KOKKOS_LAMBDA( const CellIndex& iCell, int& errors )
  {
    real_t r = rho.at_ivar( iCell, 0 );
    real_t f = f32.at_ivar( iCell, 0 );
    if( Kokkos::abs( f - r ) > 1e-6 * Kokkos::abs( r ) )
      errors++;
  }, errors);
  return errors;
}

/// With DYABLO_USE_DOUBLE=OFF, real_t fields and float fields share the same store
constexpr bool separate_stores = !std::is_same_v<real_t, float>;

/// Setup shared by most tests : one real_t field "rho" and one float field "f32"
struct TestSetup
{
  ConfigMap configMap;
  std::shared_ptr<AMRmesh> amr_mesh;
  ForeachCell foreach_cell;
  UserData U;

  TestSetup( const std::string& config = mesh_config )
  : configMap(config),
    amr_mesh(make_mesh()),
    foreach_cell(*amr_mesh, configMap),
    U(configMap, foreach_cell)
  {
    U.new_fields( {"rho"} );
    U.new_fields<float>( {"f32"} );
    init_field<real_t>( foreach_cell, U, "rho" );
    init_field<float>( foreach_cell, U, "f32" );
  }
};

} // namespace

// ============================================================================
// Create and access fields that are not real_t
// ============================================================================
TEST( Test_UserData_typed, api )
{
  TestSetup s;
  UserData& U = s.U;

  EXPECT_TRUE( U.has_field("rho") );
  EXPECT_TRUE( U.has_field<float>("f32") );
  if constexpr ( separate_stores )
  {
    EXPECT_FALSE( U.has_field("f32") );
    EXPECT_EQ( U.nbFields(), 1 );
    EXPECT_EQ( U.nbFields<float>(), 1 );
    EXPECT_EQ( U.getEnabledFields<float>(), (std::set<std::string>{"f32"}) );
  }

  auto acc_f = U.getAccessor<float>( {{"f32", 0}} );
  static_assert( std::is_same_v< decltype(acc_f.at_ivar(std::declval<CellIndex>(), 0)), float& > );
  static_assert( std::is_same_v< decltype(acc_f.at(std::declval<CellIndex>(), VarIndex(0))), float& > );

  EXPECT_EQ( 0, count_local_errors<real_t>( s.foreach_cell, U, "rho" ) );
  EXPECT_EQ( 0, count_local_errors<float>( s.foreach_cell, U, "f32" ) );

  // Runtime-sized accessor (MAX_FIELD_COUNT = -1)
  {
    auto acc_dyn = U.getAccessor<float, -1>( {{"f32", 0}} );
    EXPECT_EQ( acc_dyn.nbFields(), 1 );
  }

  // getFieldCopy<float> is readable in kernels
  {
    auto copy = U.getFieldCopy<float>( "f32" );
    static_assert( std::is_same_v< std::decay_t<decltype(copy)>::View_t::value_type, float > );
    EXPECT_EQ( copy.getShape().nbOcts, s.amr_mesh->getNumOctants() );
    EXPECT_EQ( count_copy_errors( s.foreach_cell, copy ), 0 );
  }

  // move / delete
  U.new_fields<float>( {"f32b"} );
  U.move_field<float>( "f32c", "f32" );
  EXPECT_TRUE( U.has_field<float>("f32c") );
  EXPECT_FALSE( U.has_field<float>("f32") );
  EXPECT_EQ( 0, count_local_errors<float>( s.foreach_cell, U, "f32c" ) );
  U.delete_field<float>( "f32b" );
  EXPECT_EQ( U.nbFields<float>(), separate_stores ? 1 : 2 );

  // Names are unique across types
  EXPECT_ANY_THROW( U.new_fields<float>( {"rho"} ) );
  EXPECT_ANY_THROW( U.new_fields( {"f32c"} ) );
  if constexpr ( separate_stores ) // Otherwise this is a rename within one type, which replaces dest
  {
    EXPECT_ANY_THROW( U.move_field<float>( "rho", "f32c" ) );
    EXPECT_ANY_THROW( U.move_field( "f32c", "rho" ) );
  }
  EXPECT_TRUE( U.has_field("rho") );
  EXPECT_TRUE( U.has_field<float>("f32c") );
}

// ============================================================================
// Ghost exchange, both communicators, each type
// ============================================================================
template< typename T >
class Test_UserData_typed_ghosts : public testing::Test {};
using FieldTypes = testing::Types<double, float>;
TYPED_TEST_SUITE( Test_UserData_typed_ghosts, FieldTypes );

template< typename T, typename GhostComm_t >
void run_ghost_test()
{
  ConfigMap configMap( mesh_config );
  auto amr_mesh = make_mesh();
  ForeachCell foreach_cell( *amr_mesh, configMap );
  UserData U( configMap, foreach_cell );
  U.new_fields<T>( {"f"} ); // zero initialized, ghosts included
  init_field<T>( foreach_cell, U, "f" );

  auto acc = U.getAccessor<T>( {{"f", 0}} );
  GhostComm_t ghost_comm( *amr_mesh, acc.getShape(), 2 );
  ghost_comm.exchange_ghosts( acc );

  auto shape = acc.getShape();
  uint32_t bx = shape.bx, by = shape.by, bz = shape.bz, nbCells = bx*by*bz;
  auto cells = foreach_cell.getCellMetaData();
  int filled = 0, errors = 0;
  Kokkos::parallel_reduce( "check_ghosts", shape.nbGhosts*nbCells,
    KOKKOS_LAMBDA( uint32_t i, int& filled, int& errors )
  {
    uint32_t iOct = i/nbCells, c = i%nbCells;
    CellIndex iCell{ {iOct, true, false}, c%bx, (c/bx)%by, c/(bx*by), bx, by, bz };
    T v = acc.at_ivar( iCell, 0 );
    if( v != T(0) ) // cells not exchanged by partial_blocks stay 0
    {
      filled++;
      auto p = cells.getCellCenter( iCell );
      if( v != encode<T>( p[IX], p[IY], p[IZ] ) )
        errors++;
    }
  }, filled, errors);

  int filled_g = global_sum(filled), errors_g = global_sum(errors);
  std::cout << "[typed] " << GhostComm_t::name() << " sizeof(T)=" << sizeof(T)
            << " ghost cells filled=" << filled_g << " wrong=" << errors_g << std::endl;
  if( GlobalMpiSession::get_comm_world().MPI_Comm_size() > 1 )
  {
    EXPECT_GT( filled_g, 0 );
  }
  EXPECT_EQ( errors_g, 0 );
}

TYPED_TEST( Test_UserData_typed_ghosts, partial_blocks )
{
  run_ghost_test< TypeParam, GhostCommunicator_impl<GhostCommunicator_partial_blocks> >();
}

TYPED_TEST( Test_UserData_typed_ghosts, full_blocks )
{
  run_ghost_test< TypeParam, GhostCommunicator_impl<GhostCommunicator_full_blocks> >();
}

// ============================================================================
// Load balancing
// ============================================================================
TEST( Test_UserData_typed, loadbalance )
{
  TestSetup s;
  uint32_t nbOcts_before = s.amr_mesh->getNumOctants();
  s.amr_mesh->loadBalance_userdata( 4, s.U );
  uint32_t nbOcts = s.amr_mesh->getNumOctants();
  std::cout << "[typed] loadbalance nbOcts " << nbOcts_before << " -> " << nbOcts << std::endl;

  EXPECT_EQ( s.U.getFieldCopy<real_t>("rho").getShape().nbOcts, nbOcts );
  EXPECT_EQ( s.U.getFieldCopy<float>("f32").getShape().nbOcts, nbOcts );
  EXPECT_EQ( 0, global_sum( count_local_errors<real_t>( s.foreach_cell, s.U, "rho" ) ) );
  EXPECT_EQ( 0, global_sum( count_local_errors<float>( s.foreach_cell, s.U, "f32" ) ) );
}

// Temporary field pattern : create, delete, change the mesh, create again
TEST( Test_UserData_typed, delete_then_loadbalance )
{
  TestSetup s;
  s.U.new_fields<float>( {"tmp"} );
  s.U.delete_field<float>( "tmp" );
  s.U.delete_field<float>( "f32" ); // no float field left

  s.amr_mesh->loadBalance_userdata( 4, s.U );

  s.U.new_fields<float>( {"tmp"} );
  auto shape = s.U.getAccessor<float>( {{"tmp", 0}} ).getShape();
  EXPECT_EQ( shape.nbOcts, s.amr_mesh->getNumOctants() );
  EXPECT_EQ( shape.nbGhosts, s.amr_mesh->getNumGhosts() );
}

// ============================================================================
// AMR remap : float fields follow the same mapping as real_t fields
// ============================================================================
class Test_UserData_typed_remap : public testing::TestWithParam<std::string> {};

TEST_P( Test_UserData_typed_remap, float_matches_real )
{
  TestSetup s;
  s.U.new_fields<float>( {"tmp"} );
  s.U.delete_field<float>( "tmp" );

  Timers timers;
  auto mapUserData = MapUserDataFactory::make_instance( GetParam(), s.configMap, s.foreach_cell, timers );
  mapUserData->save_old_mesh( s.U );
  // Coarsen everything that can be, refine one octant
  uint32_t nbOcts_old = s.amr_mesh->getNumOctants();
  for( uint32_t iOct=0; iOct<nbOcts_old; iOct++ )
    s.amr_mesh->setMarker( iOct, -1 );
  s.amr_mesh->setMarker( nbOcts_old/2, 1 );
  s.amr_mesh->adapt();
  mapUserData->remap( s.U );

  uint32_t nbOcts = s.amr_mesh->getNumOctants();
  std::cout << "[typed] " << GetParam() << " nbOcts " << nbOcts_old << " -> " << nbOcts << std::endl;
  auto rho = s.U.getFieldCopy<real_t>("rho");
  auto f32 = s.U.getFieldCopy<float>("f32");
  ASSERT_EQ( rho.getShape().nbOcts, nbOcts );
  ASSERT_EQ( f32.getShape().nbOcts, nbOcts );

  // float accumulates coarsening means in float : allow float rounding
  EXPECT_EQ( 0, global_sum( count_mismatches( s.foreach_cell, rho, f32 ) ) );

  // Next allocations follow the new mesh
  EXPECT_NO_THROW( s.U.new_fields<float>( {"g32"} ) );
  EXPECT_EQ( s.U.getAccessor<float>( {{"g32", 0}} ).getShape().nbOcts, nbOcts );
}

INSTANTIATE_TEST_SUITE_P( Test_UserData_typed_remap, Test_UserData_typed_remap,
  testing::Values( "MapUserData_mean", "MapUserData_linear" ) );

// ============================================================================
// IO : snapshot, checkpoint and restart
// ============================================================================
TEST( Test_UserData_typed, io_restart )
{
  int mpi_size = GlobalMpiSession::get_comm_world().MPI_Comm_size();
  std::string prefix = "typed_" + std::to_string(mpi_size);
  {
    TestSetup s( std::string(mesh_config) + R"ini(
[output]
hdf5_enabled=true
write_variables=rho,f32
outputDir=./typed_io
outputPrefix=)ini" + prefix + "\n" );

    Timers timers;
    ScalarSimulationData scalar_data;
    scalar_data.set<int>( "iter", 0 );
    scalar_data.set<real_t>( "time", 0.0 );
    for( std::string id : {"IOManager_hdf5", "IOManager_checkpoint"} )
    {
      auto io = IOManagerFactory::make_instance( id, s.configMap, s.foreach_cell, timers );
      io->save_snapshot( s.U, scalar_data );
    }
  }

  ConfigMap configMap = ConfigMap::broadcast_parameters( "./typed_io/restart_" + prefix + "_0.ini" );
  AMRmesh amr_mesh( 3, std::array<bool,3>{false,false,false}, level_min, level_max );
  ForeachCell foreach_cell( amr_mesh, configMap );
  UserData U( configMap, foreach_cell );
  Timers timers;
  auto initial_conditions = InitialConditionsFactory::make_instance( "restart", configMap, foreach_cell, timers );
  initial_conditions->init( U );

  EXPECT_TRUE( U.has_field<real_t>("rho") );
  EXPECT_TRUE( U.has_field<float>("f32") );
  if constexpr ( separate_stores )
  {
    EXPECT_FALSE( U.has_field<real_t>("f32") );
  }
  EXPECT_EQ( 0, global_sum( count_local_errors<real_t>( foreach_cell, U, "rho" ) ) );
  EXPECT_EQ( 0, global_sum( count_local_errors<float>( foreach_cell, U, "f32" ) ) );
}

// ============================================================================
// Memory footprint : float vs double fields
// ============================================================================
namespace {
std::map<std::string, uint64_t> g_alloc_bytes;
void record_alloc( const Kokkos_Profiling_SpaceHandle, const char* label, const void*, const uint64_t size )
{
  g_alloc_bytes[label] += size;
}

template< typename T >
std::map<std::string, uint64_t> measure_fields( AMRmesh& amr_mesh, ConfigMap& configMap, int nb_fields, int nb_fields_real, bool exchange )
{
  ForeachCell foreach_cell( amr_mesh, configMap );
  DyabloSession::get_MpiBufferPool().clear();
  g_alloc_bytes.clear();
  {
    UserData U( configMap, foreach_cell );
    std::set<std::string> names, names_real;
    std::vector<UserData::FieldAccessor_FieldInfo> info;
    for( int i=0; i<nb_fields; i++ )
    {
      names.insert( "v"+std::to_string(i) );
      info.push_back( {"v"+std::to_string(i), (VarIndex)i} );
    }
    for( int i=0; i<nb_fields_real; i++ )
      names_real.insert( "r"+std::to_string(i) );
    if( nb_fields_real > 0 )
      U.new_fields( names_real );
    U.new_fields<T>( names );
    if( exchange )
    {
      auto acc = U.getAccessor<T>( info );
      GhostCommunicator ghost_comm( amr_mesh, acc.getShape(), 2 );
      ghost_comm.exchange_ghosts( acc );
    }
  }
  DyabloSession::get_MpiBufferPool().clear();
  return g_alloc_bytes;
}
} // namespace

TEST( Test_UserData_typed, memory_footprint )
{
  ConfigMap configMap( mesh_config );
  auto amr_mesh = make_mesh();
  constexpr int nf = 8;
  uint64_t nbCells = bsize*bsize*bsize;
  uint64_t nbSlots = amr_mesh->getNumOctants() + amr_mesh->getNumGhosts();

  Kokkos::Tools::Experimental::set_allocate_data_callback( record_alloc );
  auto m64 = measure_fields<double>( *amr_mesh, configMap, nf, 0, false );
  auto m32 = measure_fields<float>( *amr_mesh, configMap, nf, 0, false );
  auto mix = measure_fields<float>( *amr_mesh, configMap, nf/2, nf/2, false );
  auto x64 = measure_fields<double>( *amr_mesh, configMap, nf, 0, true );
  auto x32 = measure_fields<float>( *amr_mesh, configMap, nf, 0, true );
  Kokkos::Tools::Experimental::set_allocate_data_callback( nullptr );

  const std::string L = "UserData_fields";
  std::cout << "[typed] field storage bytes : double=" << m64[L] << " float=" << m32[L]
            << " (4 double + 4 float)=" << mix[L] << std::endl;
  EXPECT_EQ( m64[L], nbCells*nbSlots*nf*sizeof(double) );
  EXPECT_EQ( 2*m32[L], m64[L] );
  EXPECT_EQ( 4*mix[L], 3*m64[L] );

  const std::string B = "MPI Buffer";
  std::cout << "[typed] ghost exchange MPI buffer bytes : double=" << x64[B] << " float=" << x32[B] << std::endl;
  if( GlobalMpiSession::get_comm_world().MPI_Comm_size() > 1 && x64[B] > 0 )
  {
    EXPECT_NEAR( (double)x32[B]/x64[B], 0.5, 0.01 ) << "ghost exchange buffers are not sized by field type";
  }
}

} // namespace dyablo
