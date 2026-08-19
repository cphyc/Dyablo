#pragma once 

#include <iostream>
#include <fstream>

#include <Kokkos_Core.hpp>

#include "amr/AMRmesh.h"
#include "amr/LightOctree.h"

namespace dyablo{
namespace debug{

inline void output_vtk( const std::string& name, AMRmesh& mesh )
{
  auto print_array = []( std::ostream& out, const auto& v )
  {
    auto v_host = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy( v_host, v );

    for( size_t i=0; i<v_host.size(); i++ )
      out << v_host(i) << " ";
    out << std::endl;
  };

  int mpi_rank = mesh.getMpiComm().MPI_Comm_rank();
  std::string filename = name+"_"+std::to_string(mpi_rank)+".vtu";

  std::cout << "DEBUG OUPUT MESH " << filename << std::endl;
 
  uint32_t nbOcts = mesh.getNumOctants();
  uint32_t nbGhosts = mesh.getNumGhosts();
  uint32_t nofCubes = nbOcts + nbGhosts;

  Kokkos::View<int*> cells_is_ghost("cells_is_ghost", nofCubes);
  Kokkos::View<int*> cells_iOct("cells_iOct", nofCubes);

  Kokkos::View<double*> nodes_Coordinates("nodes_Coordinates", nofCubes*8*3);
  Kokkos::View<uint32_t*> cells_Connectivity("cells_Connectivity", nofCubes*8);
  Kokkos::View<uint32_t*> cells_offsets("cells_offsets", nofCubes);
  Kokkos::View<int*> cells_types("cells_types", nofCubes);

  const LightOctree& lmesh = mesh.getLightOctree();

  Kokkos::parallel_for( "init_vtk_arrays", nofCubes,
    KOKKOS_LAMBDA( uint32_t i )
  {
    bool isGhost = i>=nbOcts;
    LightOctree::OctantIndex iOct { i - isGhost*nbOcts, isGhost };

    cells_iOct(i) = i;
    cells_is_ghost(i) = (int)isGhost;
    cells_types(i) = 11;

    auto pos = lmesh.getCorner( iOct );
    auto size = lmesh.getSize( iOct );
    real_t px = pos[0];
    real_t py = pos[1];
    real_t pz = pos[2];
    real_t size_x = size[0];
    real_t size_y = size[1];
    real_t size_z = size[2];

    for( int16_t dz=0; dz<2; dz++ )
    for( int16_t dy=0; dy<2; dy++ )
    for( int16_t dx=0; dx<2; dx++ )
    {
      int di = dx + 2*dy + 4*dz;
      nodes_Coordinates(3*(8*i+di) + 0) = px + size_x * dx;
      nodes_Coordinates(3*(8*i+di) + 1) = py + size_y * dy;
      nodes_Coordinates(3*(8*i+di) + 2) = pz + size_z * dz;
      cells_Connectivity(8*i+di) = 8*i+di;
    }

    cells_offsets(i) = 8*i+8;
  });

  std::ofstream out( filename );
  out << "<?xml version=\"1.0\"?>"                                                      << std::endl;
  out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
  out << "  <UnstructuredGrid>"                                                         << std::endl;
  out << "    <Piece NumberOfCells=\"" << nofCubes << "\" NumberOfPoints=\"" << nofCubes*8 << "\">" << std::endl;
  out << "      <Points>"                                                               << std::endl;
  out << "        <DataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
  print_array(out, nodes_Coordinates);
  out << "        </DataArray>"                                                         << std::endl;
  out << "      </Points>"                                                              << std::endl;
  out << "      <CellData>"                                                             << std::endl;
  out << "        <DataArray type=\"Int32\" Name=\"is_ghost\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_is_ghost);
  out << "        </DataArray>"                                                         << std::endl;
  out << "        <DataArray type=\"Int32\" Name=\"iOct\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_iOct);
  out << "        </DataArray>"                                                         << std::endl;
  out << "      </CellData>"                                                            << std::endl;
  out << "      <Cells>"                                                                << std::endl;
  out << "        <DataArray type=\"UInt32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_Connectivity);
  out << "        </DataArray>"                                                         << std::endl;
  out << "        <DataArray type=\"UInt32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_offsets);
  out << "        </DataArray>"                                                         << std::endl;
  out << "        <DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_types);
  out << "        </DataArray>"                                                         << std::endl;
  out << "      </Cells>"                                                               << std::endl;
  out << "    </Piece>"                                                                 << std::endl;
  out << "  </UnstructuredGrid>"                                                        << std::endl;
  out << "</VTKFile>"                                                                   << std::endl;

}

}
} //namespace dyablo