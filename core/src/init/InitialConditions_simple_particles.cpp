#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

namespace dyablo{

class InitialConditions_simple_particles : public InitialConditions{ 
    //ForeachCell& foreach_cell;
    ForeachParticle foreach_particle;
    real_t gamma0;
    int npart;
    Kokkos::View<double*> px, py, pz, vx, vy, vz, mass;
    std::vector<std::string> extra_attributes;
    std::vector<Kokkos::View<double*>> extra_attr_arrays;
public:
  InitialConditions_simple_particles(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : //foreach_cell(foreach_cell),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    gamma0(configMap.getValue<real_t>("hydro", "gamma0", 1.4)),
    npart(configMap.getValue<int>("simple_particles", "npart", 1)),
    px( "px", npart ), py( "py", npart ), pz( "pz", npart ),
    vx( "vx", npart ), vy( "vy", npart ), vz( "vz", npart ),
    mass("mass", npart ),
    extra_attributes(configMap.getValue<std::vector<std::string>>("simple_particles", "attributes", {}))
  {    
    auto parse_array = [&](const Kokkos::View<double*>& a, const std::string& var)
    {
      std::vector<double> values = configMap.getValue< std::vector<double> >("simple_particles", var, {});
      int nb_values = std::max((int)values.size(),npart); // Select at most npart values from .ini

      // Create unmanaged view to copy vector
      using UnmanagedHostView = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
      UnmanagedHostView values_host( values.data(), nb_values ); 
      
      auto values_device = Kokkos::subview( a, std::make_pair(0,nb_values) );

      Kokkos::deep_copy(values_device, values_host);
    };

    parse_array(px, "px");
    parse_array(py, "py");
    parse_array(pz, "pz");
    parse_array(vx, "vx");
    parse_array(vy, "vy");
    parse_array(vz, "vz");
    parse_array(mass, "mass");
    for (auto& f: extra_attributes) {
      Kokkos::View<double*> arr("extra_attribute_" + f, npart);
      parse_array(arr, f);
      extra_attr_arrays.push_back(arr);
    }

  }

  void init( UserData& U )
  {
    // Setting up particles
    int rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();

    int npart_local = (rank==0) ? npart : 0;

    U.new_ParticleArray("particles", npart_local);
    U.new_ParticleAttribute("particles", "vx");
    U.new_ParticleAttribute("particles", "vy");
    U.new_ParticleAttribute("particles", "vz");
    U.new_ParticleAttribute("particles", "mass");
    for (auto& f: extra_attributes) {
      U.new_ParticleAttribute("particles", f);
    }

    if (rank == 0) { 

      const ForeachParticle::ParticleArray& P = U.getParticleArray("particles"); 

      enum VarIndex_particle{
        IVX, IVY, IVZ, IM
      };

      const auto Pdata = U.getParticleAccessor("particles", {
        {"vx", IVX},
        {"vy", IVY},
        {"vz", IVZ},
        {"mass", IM},
      });
      std::vector<dyablo::UserData_particles::ParticleAccessor_AttributeInfo> extra_attr_info = {};
      int Nextra = extra_attributes.size();
      for (auto &f: extra_attributes) {
        int idx = extra_attr_info.size();
        extra_attr_info.push_back({f, idx});
      }
      const auto Pdata_extra = U.getParticleAccessor("particles", extra_attr_info);

      const auto& px = this->px;
      const auto& py = this->py;
      const auto& pz = this->pz;
      const auto& vx = this->vx;
      const auto& vy = this->vy;
      const auto& vz = this->vz;
      const auto& mass = this->mass;
      const auto& extra_attr_arrays = this->extra_attr_arrays;

      foreach_particle.foreach_particle("InitialConditions_simple_particles", P,
        KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart) {      
          P.pos(iPart, IX) = px(iPart);
          P.pos(iPart, IY) = py(iPart);
          P.pos(iPart, IZ) = pz(iPart);

          Pdata.at(iPart, IVX) = vx(iPart);
          Pdata.at(iPart, IVY) = vy(iPart);
          Pdata.at(iPart, IVZ) = vz(iPart);
          Pdata.at(iPart, IM)  = mass(iPart);
        });

      // Set extra attributes
      for (auto i = 0; i < Nextra; i++) {
        auto arr = extra_attr_arrays[i];
        foreach_particle.foreach_particle("InitialConditions_simple_particles_extra", P,
          KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart) {
            Pdata_extra.at(iPart, i) = arr(iPart);
          });
        }
    }

    U.distributeParticles("particles");
  }  
}; 

} // namespace dyablo


FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_simple_particles, 
                 "simple_particles");

