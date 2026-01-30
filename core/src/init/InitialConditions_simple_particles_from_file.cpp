#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

namespace dyablo{

class InitialConditions_simple_particles_from_file : public InitialConditions{
    //ForeachCell& foreach_cell;
    ForeachParticle foreach_particle;

    std::vector<std::string> file_names;
    std::vector<std::string> array_names;
    std::vector<std::string> attribute_names;
    std::vector<int> npart;
    std::vector<Kokkos::View<double**, Kokkos::LayoutLeft>> attribute_values;
    std::vector<Kokkos::View<double**, Kokkos::LayoutLeft>> pos_values;
    real_t xmin, xmax, ymin, ymax, zmin, zmax;
public:
  InitialConditions_simple_particles_from_file(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  : //foreach_cell(foreach_cell),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    file_names(configMap.getValue<std::vector<std::string>>("simple_particles", "file_names")),
    array_names(configMap.getValue<std::vector<std::string>>("simple_particles", "array_names")),
    attribute_names(configMap.getValue<std::vector<std::string>>("simple_particles", "attribute_names")),
    xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ),
    xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
    ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ),
    ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
    zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ),
    zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) )
  {
    // Few sanity checks
    // -- file_names and array_names must have the same size
    DYABLO_ASSERT_HOST_RELEASE(
        file_names.size() == array_names.size(),
        "'file_names' and 'array_names' must have the same size, "
        "but got " << file_names.size() << " and " << array_names.size()
    );
    // -- attribute_names should contain px, py, pz at least
    DYABLO_ASSERT_HOST_RELEASE(
        attribute_names.size() >= 4,
        "'attribute_names' must contain at least 4 entries (px, py, pz, mass), "
        "but got only " << attribute_names.size()
    );
    for (const auto& required_attr : {"px", "py", "pz", "mass"}) {
      DYABLO_ASSERT_HOST_RELEASE(
          std::find(attribute_names.begin(), attribute_names.end(), required_attr) != attribute_names.end(),
          "'attribute_names' must contain '" << required_attr << "'"
      );
    }

    // Setting up particles
    int rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();
    if (rank != 0) return; // Only rank 0 reads files

    for( const auto& fname : file_names ) {
      int count = 0;
      {
        std::ifstream file( fname );
        DYABLO_ASSERT_HOST_RELEASE( file.is_open(),
          "Could not open particle file: " << fname
        );

        // Count number of lines (particles) in the file
        std::string line;
        while( std::getline(file, line) )
        {
          count++;
        }
        npart.push_back(count);
        file.close();
      }

      std::cout << "InitialConditions_simple_particles_from_file: "
                << "File '" << fname << "' contains " << count << " particles." << std::endl;

      // Allocate Host view of shape ( total_npart, n_attributes )
      Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>
          attribute_values_host("attributes_host", count, attribute_names.size() - 3),
          pos_values_host("pos_values_host", count, 3);

      // Reset file to beginning
      {
        std::ifstream file( fname );
        DYABLO_ASSERT_HOST_RELEASE( file.is_open(),
          "Could not open particle file: " << fname
        );
        count = 0;
        std::string line;
        while( std::getline(file, line) )
        {
          std::istringstream iss(line);
          size_t attr_index = 0;
          for( size_t a=0; a<attribute_names.size(); a++ )
          {
            double val;
            iss >> val; // Read attribute value
            if      (attribute_names[a] == "px")
              pos_values_host(count, 0) = val;
            else if (attribute_names[a] == "py")
              pos_values_host(count, 1) = val;
            else if (attribute_names[a] == "pz")
              pos_values_host(count, 2) = val;
            else {
              // Other attributes
              attribute_values_host(count, attr_index) = val;
              attr_index++;
            }
          }
          count++;
        }

        DYABLO_ASSERT_HOST_RELEASE(
          count == npart.back(),
          "Mismatch in particle count when reading file: " << fname
        );
        file.close();
      }

      // Copy to device
      Kokkos::View<double**, Kokkos::LayoutLeft>
          attribute_values_device("attributes_device", attribute_values_host.extent(0), attribute_values_host.extent(1)),
          pos_values_device("pos_values_device", pos_values_host.extent(0), pos_values_host.extent(1));

      Kokkos::deep_copy(attribute_values_device, attribute_values_host);
      Kokkos::deep_copy(pos_values_device, pos_values_host);

      // Store in class
      attribute_values.push_back( attribute_values_device );
      pos_values.push_back( pos_values_device );
    }
  }

  void init( UserData& U )
  {
    // Setting up particles
    int rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();

    for (size_t file_idx = 0; file_idx < file_names.size(); file_idx++) {
      int npart_local = (rank==0) ? npart[file_idx] : 0;

      U.new_ParticleArray("new_particles", npart_local);

      for( std::string& attr : attribute_names )
      {
        if (attr == "px" || attr == "py" || attr == "pz") {
          continue; // Position attributes are handled separately
        }
        U.new_ParticleAttribute("new_particles", attr);
      }

      if (rank == 0) {
        const ForeachParticle::ParticleArray& P = U.getParticleArray("new_particles");
        std::vector<UserData::ParticleAccessor::AttributeInfo> attr_info;
        size_t index = 0;
        for( size_t i=0; i<attribute_names.size(); i++ )
        {
          if (attribute_names[i] == "px" || attribute_names[i] == "py" || attribute_names[i] == "pz") {
            continue; // Skip position attributes
          }
          attr_info.push_back( {attribute_names[i], (VarIndex)(index++)} );
        }
        const UserData::ParticleAccessor Pdata = U.getParticleAccessor("new_particles", attr_info);

        const auto& attribute_values = this->attribute_values[file_idx];
        const auto& pos_values = this->pos_values[file_idx];

        std::cout << "Filling particle array 'new_particles' with "
                  << P.getNumParticles() << " particles. attribute_values.size = " << attribute_values.extent(0) << " × " << attribute_values.extent(1) << std::endl;

        size_t nbAttr = attribute_values.extent(1);
        real_t xmin = this->xmin,
               xmax = this->xmax,
               ymin = this->ymin,
               ymax = this->ymax,
               zmin = this->zmin,
               zmax = this->zmax;

        foreach_particle.foreach_particle("InitialConditions_simple_particles", P,
          KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart)
        {
          if (pos_values(iPart, 0) < xmin || pos_values(iPart, 0) > xmax ||
              pos_values(iPart, 1) < ymin || pos_values(iPart, 1) > ymax ||
              pos_values(iPart, 2) < zmin || pos_values(iPart, 2) > zmax ) {
            // Particle outside domain : just ignore
            return;
          }
          P.pos(iPart, IX) = pos_values(iPart, 0);
          P.pos(iPart, IY) = pos_values(iPart, 1);
          P.pos(iPart, IZ) = pos_values(iPart, 2);

          for( size_t ivar = 0; ivar<nbAttr; ivar++ ) {
            real_t tmp = attribute_values(iPart, ivar);
            Pdata.at_ivar(iPart, ivar) = tmp;
          }
        });
      }

      // Store array under the requested name
      const std::string& array_name = array_names[file_idx];

      // Create destination if needed
      if ( !U.has_ParticleArray(array_name) )
        U.new_ParticleArray( array_name, 0 );

      for( std::string& attr : attribute_names )
      {
        if (attr == "px" || attr == "py" || attr == "pz")
          continue;

        if ( !U.has_ParticleAttribute(array_name, attr) )
          U.new_ParticleAttribute( array_name, attr);
      }

      // Keep those particles that have mass>0
      U.merge_particles_if( array_name, "new_particles", "mass" );

      // Distribute particles
      U.distributeParticles(array_name);
    }

  }
};

} // namespace dyablo


FACTORY_REGISTER(dyablo::InitialConditionsFactory,
                 dyablo::InitialConditions_simple_particles_from_file,
                 "simple_particles_from_file");
