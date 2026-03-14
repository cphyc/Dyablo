#include "ParticleUpdate_base.h"
#include "utils/units/Units.h"
#include "ForeachParticle.h"
#include "states/State_hydro.h"
#include "mpi/GhostCommunicator_partial_blocks.h"
#include "foreach_cell/ForeachCell_utils.h"

#include <Kokkos_Core.hpp>
#include <cmath>

namespace {
  using namespace dyablo;

  /**
   * @brief Compute the momentum injection and energy contribution
   * from a supernova.
   *
   * See Kimm & Cen 2014, Geen et al. 2015 and Kimm et al. 2017 for details.
   *
   * @param dx: cell size [code units]
   * @param rS: Stromgren sphere radius [code units]
   * @param ESN: supernova energy [erg]
   * @param Zp: metallicity [solar units]
   * @param beta_SN: fraction of SN mass redistributed in neighbour cells
   * @param Mej: ejected mass from the SN [Msun]
   * @param rho: density in the cell containing the SN [mp/cm³]
   * @param rho_host: mass in the cell containing the SN [mp/cm³]
   * @param V_host: volume of the cell containing the SN [cm³]
   *
   * @return the momentum to inject [Msun * km/s], to be multiplied by dΩ/4π
   */
  KOKKOS_INLINE_FUNCTION
  real_t compute_dp(const real_t dx, const real_t rS, const real_t ESN, const real_t Zp, const real_t beta_SN, const real_t Mej, const real_t rho, const real_t rho_host, const real_t V_host) {

    real_t M_host = rho_host * V_host;
    constexpr real_t XH = Units::XH().convert_to(Units::one());

    real_t E51 = ESN / 1e51;
    real_t nH = rho * XH;
    real_t nH_m217 = pow(nH, -2.0/17.0);
    real_t nH_m417 = pow(nH, -4.0/17.0);

    real_t E51_1617 = pow(E51, 16.0/17.0);
    real_t E51_m217 = pow(E51, -2.0/17.0);

    real_t Zp_m014 = pow(Zp, -0.14);
    real_t Zp_m028 = pow(Zp, -0.28);

    // Snowplow phase momentum

    // Eq. (14) from Kimm et al. 2017
    real_t pSN_snow = 3e5 * nH_m217 * E51_1617 * Zp_m014;
    // Eq. (15) from Kimm et al. 2017
    real_t pSN_PH = 4.2e5 * E51_1617 * Zp_m014;

    // Injected momentum at the snowplow phase, accounting for
    // boost from photo-heating (Geen et al. 2015)
    real_t pSN;
    {
      // Eq. (16) from Kimm et al. 2017
      real_t exp_factor = exp(-dx / rS);
      pSN = pSN_snow * exp_factor + pSN_PH * (1.0 - exp_factor);
    }

    // Eq. (A5) from Kimm et Cen 2014, threshold mass ratio
    real_t chi_tr = 87.54 * nH_m417 * E51_m217 * Zp_m028;

    // Eq. (17, 18, 19) from Kimm et al. 2017
    // Note: we're assuming the local cell is ⅛ of the volume of the cell hosting the SN
    real_t dMej = (1 - beta_SN) * Mej / 48.0;
    real_t dMswept = rho * V_host / 8.0 + (1 - beta_SN) * M_host / 48.0 + dMej;
    real_t chi = dMswept / dMej;

    if (chi < chi_tr) {
      // Momentum from the adiabatic phase
      // Eq. (21) from Kimm et al. 2017
      real_t f_ESN = 1 - (chi - 1) / (3 * (chi_tr - 1));
      pSN = SQRT(2 * chi * Mej * f_ESN * ESN);
    }
    return pSN;
  }

};

namespace dyablo {

class ParticleUpdate_momentum_feedback : public ParticleUpdate {
public:
  using pos_t = Kokkos::Array<real_t, 3>;

  using CellIndex = ForeachCell::CellIndex;

  // Number of neighboring cells to deposit mass/momentum/energy
  static constexpr int nSNnei = 48;
  // Number of cells corresponding to the central cell to deposit mass
  static constexpr int nSNcen = 4;

  ParticleUpdate_momentum_feedback(
    ConfigMap& configMap,
    ForeachCell& foreach_cell,
    Timers& timers)
  : foreach_cell    ( foreach_cell ),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    timers          ( timers ),
    eta_SNII        ( configMap.getValue<real_t>("star_feedback", "eta_SNII", 0.1) ),
    yield_SNII      ( configMap.getValue<real_t>("star_feedback", "yield_SNII", 0.1) ),
    E_SNII_physical ( configMap.getValue_in_code_unit<Units::Energy>("star_feedback", "E_SNII", "1e51 erg") ),
    M_SNII_physical ( configMap.getValue_in_code_unit<Units::Mass>  ("star_feedback", "M_SNII", "10 Msun") ),
    t_SNII_physical ( configMap.getValue_in_code_unit<Units::Time>  ("star_feedback", "t_SNII", "10 Myr") ),
    cosmology       ( configMap.getValue<bool>("cosmology", "active", false) ),
    array_names     ( configMap.getValue<std::vector<std::string>>("particles", "ParticleUpdate_momentum_feedback_array_names") ),
    A_SN            ( configMap.getValue<real_t>("star_feedback", "A_SN", 3e5) ),
    expN_SN         ( configMap.getValue<real_t>("star_feedback", "expN_SN", -2.0/17.0) ),
    expE_SN         ( configMap.getValue<real_t>("star_feedback", "expE_SN", 16.0/17.0) ),
    expZ_SN         ( configMap.getValue<real_t>("star_feedback", "expZ_SN", -0.14) ),
    f_LOAD          ( static_cast<real_t>(nSNnei) / static_cast<real_t>(nSNcen + nSNnei) ),
    f_LOAD_CEN      ( configMap.getValue<real_t>("star_feedback", "f_LOAD_CEN", 0.0) ),
    f_CANCEL        ( configMap.getValue<real_t>("star_feedback", "f_CANCEL", 0.9387) ),
    f_ESN           ( configMap.getValue<real_t>("star_feedback", "f_ESN", 0.676) )
  {
    initialize_neighbor_positions();
    DYABLO_ASSERT_HOST_RELEASE(
      configMap.getValue<int>("mesh", "ndim", 3) == 3,
      "ParticleUpdate_momentum_feedback only works in 3D"
    );
  }

  ~ParticleUpdate_momentum_feedback() {}

  void initialize_neighbor_positions() {
    // Initialize neighbor positions (normalized to dx=1)
    // from -0.75 to 0.75, excluding edges and center
    int ind = 0;

    for (int k = 0; k < 4; k++) {
      for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
          bool ok = true;
          // Exclude edges
          if ((i == 0 || i == 3) && (j == 0 || j == 3) && (k == 0 || k == 3)) ok = false;
          // Exclude center
          if ((i == 1 || i == 2) && (j == 1 || j == 2) && (k == 1 || k == 2)) ok = false;

          if (ok) {
            real_t x = static_cast<real_t>(i) + 0.5 - 2.0;
            real_t y = static_cast<real_t>(j) + 0.5 - 2.0;
            real_t z = static_cast<real_t>(k) + 0.5 - 2.0;
            real_t rr = std::sqrt(x*x + y*y + z*z);

            xSNnei[ind][IX] = x / 2.0;
            xSNnei[ind][IY] = y / 2.0;
            xSNnei[ind][IZ] = z / 2.0;
            vSNnei[ind][IX] = x / rr;
            vSNnei[ind][IY] = y / rr;
            vSNnei[ind][IZ] = z / rr;
            ind++;
          }
        }
      }
    }
  }

  void update(UserData& U, ScalarSimulationData& scalar_data) {
    for (const std::string& array_name : array_names) {
      if (!U.has_ParticleArray(array_name)) {
        throw std::runtime_error("ParticleUpdate_momentum_feedback: Particle array '" + array_name + "' does not exist in UserData.");
      }

      timers.get("ParticleUpdate_momentum_feedback").start();

      U.new_fields({"rho_star", "rho_vx_star", "rho_vy_star", "rho_vz_star"});
      this->update_aux(U, scalar_data, array_name);

      U.delete_field("rho_star");
      U.delete_field("rho_vx_star");
      U.delete_field("rho_vy_star");
      U.delete_field("rho_vz_star");

      timers.get("ParticleUpdate_momentum_feedback").stop();
    }
  }


  void update_aux(UserData& U, ScalarSimulationData& scalar_data, const std::string& array_name)
  {
    const real_t t = cosmology ? scalar_data.get<real_t>("time_physical") : scalar_data.get<real_t>("time");
    const real_t dt = scalar_data.get<real_t>("dt");

    enum VarIndex {
      IRho, IE_tot, IRho_vx, IRho_vy, IRho_vz, IRho_Z, IRho_star, IRho_vx_star, IRho_vy_star, IRho_vz_star
    };
    enum VarIndex_particle {
      IMASS, IVX, IVY, IVZ, IBIRTH, IMETAL
    };

    std::vector<UserData_fields::FieldAccessor_FieldInfo> Uin_infos = {
        {"rho", IRho},
        {"e_tot", IE_tot},
        {"rho_vx", IRho_vx},
        {"rho_vy", IRho_vy},
        {"rho_vz", IRho_vz},
      };

    std::vector<UserData_particles::ParticleAccessor_AttributeInfo>
      pinfos = {{"mass", IMASS}, {"vx", IVX}, {"vy", IVY}, {"vz", IVZ}, {"birth_time", IBIRTH}};

    bool has_metallicity = U.has_field("metallicity");
    if (has_metallicity) {
      Uin_infos.push_back( {"metallicity", IRho_Z} );
      pinfos.push_back( {"metallicity", IMETAL} );
    }

    // Get accessors
    auto Ppos = U.getParticleArray( array_name );
    auto Pdata = U.getParticleAccessor( array_name, pinfos );
    auto Uin = U.getAccessor( Uin_infos );
    auto Ustar = U.getAccessor( {{"rho_star", IRho_star},
                                 {"rho_vx_star", IRho_vx_star},
                                 {"rho_vy_star", IRho_vy_star},
                                 {"rho_vz_star", IRho_vz_star}} );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    real_t aexp = scalar_data.get<real_t>("aexp");

    // Gather SN feedback parameters
    const real_t eta_SNII = this->eta_SNII;
    const real_t yield_SNII = this->yield_SNII;
    const real_t E_SNII = Units::physical_to_supercomoving<Units::Energy>(E_SNII_physical, aexp);
    const real_t M_SNII = Units::physical_to_supercomoving<Units::Mass>(M_SNII_physical, aexp);
    const real_t e_SN = E_SNII / M_SNII; // Specific energy
    const real_t t_SNII_physical = this->t_SNII_physical;
    const real_t dt_physical = Units::supercomoving_to_physical<Units::Time>(dt, aexp);

    // Momentum feedback parameters
    const real_t f_LOAD = this->f_LOAD;
    const real_t f_LOAD_CEN = this->f_LOAD_CEN;
    const real_t f_CANCEL = this->f_CANCEL;
    const real_t f_ESN = this->f_ESN;
    const real_t A_SN = this->A_SN;
    const real_t expN_SN = this->expN_SN;
    const real_t expE_SN = this->expE_SN;
    const real_t expZ_SN = this->expZ_SN;

    // Copy neighbor positions to device-accessible arrays
    Kokkos::View<pos_t[nSNnei], Kokkos::LayoutRight> xSNnei_dev("xSNnei_dev", nSNnei);
    Kokkos::View<pos_t[nSNnei], Kokkos::LayoutRight> vSNnei_dev("vSNnei_dev", nSNnei);
    auto xSNnei_host = Kokkos::create_mirror_view(xSNnei_dev);
    auto vSNnei_host = Kokkos::create_mirror_view(vSNnei_dev);
    for (int i = 0; i < nSNnei; i++) {
      xSNnei_host(i) = this->xSNnei[i];
      vSNnei_host(i) = this->vSNnei[i];
    }
    Kokkos::deep_copy(xSNnei_dev, xSNnei_host);
    Kokkos::deep_copy(vSNnei_dev, vSNnei_host);

    auto mp_per_cc = Units::PROTON_MASS() / Units::cm3();
    auto code_density  = Units::code_units().getUnit<Units::Density>();

    real_t XH = Units::XH().convert_to(Units::one());

    uint SN_count = 0;
    uint SN_count_global = 0;

    // //----------------------------------------------------------
    // // FIRST PASS: count mass and momentum of SN II explosions
    // foreach_particle.reduce_particle( "particles_update_momentum_feedback_count", Ppos,
    //   KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart, uint& SN_count )
    // {
    //   // Age of the particle
    //   real_t age_physical = t - Pdata.at(iPart, IBIRTH);

    //   // If the SN will explode in this time step
    //   if ((age_physical < t_SNII_physical) && ((age_physical + dt_physical) > t_SNII_physical)) {
    //     SN_count++;
    //     pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
    //     pos_t part_vel = {Pdata.at(iPart, IVX), Pdata.at(iPart, IVY), Pdata.at(iPart, IVZ)};

    //     ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );

    //     pos_t cell_size = cells.getCellSize( iCell );
    //     real_t cell_volume = cell_size[IX] * cell_size[IY] * cell_size[IZ];
    //     Kokkos::atomic_add(&Ustar.at(iCell, IRho_star), Pdata.at(iPart, IMASS) / cell_volume);
    //     Kokkos::atomic_add(&Ustar.at(iCell, IRho_vx_star), Pdata.at(iPart, IMASS) * part_vel[IX] / cell_volume);
    //     Kokkos::atomic_add(&Ustar.at(iCell, IRho_vy_star), Pdata.at(iPart, IMASS) * part_vel[IY] / cell_volume);
    //     Kokkos::atomic_add(&Ustar.at(iCell, IRho_vz_star), Pdata.at(iPart, IMASS) * part_vel[IZ] / cell_volume);
    //   }
    // }, SN_count );

    // // Count the number of SN explosions in the whole box
    // MPI_Allreduce( &SN_count, &SN_count_global, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD );

    // if (SN_count_global == 0) {
    //   return;
    // }

    // // Communicate ghost zones
    // {
    //   int ghost_count = 1;
    //   GhostCommunicator_partial_blocks ghost_comm (
    //     foreach_cell.get_amr_mesh(),
    //     Ustar.getShape(),
    //     ghost_count );
    //   ghost_comm.exchange_ghosts( Ustar );
    // }

    // //----------------------------------------------------------
    // // SECOND PASS: apply mass and momentum feedback following
    // // Kimm & Cen 2014, Geen et al. 2016, Kimm et al. 2017
    // foreach_cell.foreach_cell( "particles_update_momentum_feedback_apply",
    //   Uin.getShape(),
    //   KOKKOS_LAMBDA( const CellIndex& iCell ) {
    //     pos_t cell_size = cells.getCellSize( iCell );

    //     // Get injection cell properties
    //     real_t rho = Uin.at(iCell, IRho);
    //     real_t rho_vx = Uin.at(iCell, IRho_vx);
    //     real_t rho_vy = Uin.at(iCell, IRho_vy);
    //     real_t rho_vz = Uin.at(iCell, IRho_vz);

    //     real_t drho = 0, drho_vx = 0, drho_vy = 0, drho_vz = 0, drho_Z = 0;

    //     // Helper function doing the SN injection from iCell_SN into iCell
    //     auto& do_SN_injection = [&] (const CellIndex& iCell_SN, int level_diff) {
    //       pos_t cell_size_host = cells.getCellSize( iCell_SN );

    //       // Get SN host cell properties
    //       real_t rho_host = Uin.at(iCell_SN, IRho);
    //       real_t rho_vx_host = Uin.at(iCell_SN, IRho_vx);
    //       real_t rho_vy_host = Uin.at(iCell_SN, IRho_vy);
    //       real_t rho_vz_host = Uin.at(iCell_SN, IRho_vz);

    //       // Get SN mass and momentum
    //       real_t V_host = cell_size_host[IX] * cell_size_host[IY] * cell_size_host[IZ];
    //       real_t MSN = eta_SNII * Ustar.at(iCell_SN, IRho_star) * V_host;
    //       real_t py_SN = eta_SNII * Ustar.at(iCell_SN, IRho_vy_star) * V_host;
    //       real_t pz_SN = eta_SNII * Ustar.at(iCell_SN, IRho_vz_star) * V_host;
    //       real_t px_SN = eta_SNII * Ustar.at(iCell_SN, IRho_vx_star) * V_host;

    //       // Compute dp to inject
    //       real_t dp = 0;

    //       drho += 0;
    //       drho_vx += 0;
    //       drho_vy += 0;
    //       drho_vz += 0;
    //       drho_Z += 0;
    //     };


    //     CellIndex::offset_t offset;
    //     int8_t nSN = 0;

    //     for (int8_t i = -1; i <= 1; ++i)
    //     for (int8_t j = -1; j <= 1; ++j)
    //     for (int8_t k = -1; k <= 1; ++k) {
    //       // Skip corners
    //       if (((i == -1) || (i == 1)) && ((j == -1) || (j == 1)) && ((k == -1) || (k == 1)))
    //         continue;

    //       // Get neighbours
    //       offset = {i, j, k};
    //       CellIndex iCell_neighbor = iCell.getNeighbor_ghost(offset, Uin);

    //       int level_diff = iCell_neighbor.level_diff();

    //       if (iCell_neighbor.is_boundary()) {
    //         continue; // Skip boundaries
    //       }

    //       if (level_diff == -1) { // neighbour is smaller
    //         // Note: this will iterate over
    //         //      - 4 smaller neighbors for neighbor that share a face,
    //         //      - 2 for those that share an edge,
    //         //      - 1 for those that share a corner [but we've skipped this case]
    //         foreach_smaller_neighbor<3, true>( iCell_neighbor, offset, Uin.getShape(),
    //           [&]( const CellIndex& iCell_smaller ) {
    //             if (Uin.at(iCell_smaller, IRho_star) > 0) {
    //               do_SN_injection(iCell_smaller, iCell, level_diff);
    //             }
    //           });
    //       }
    //       else if (level_diff >= 0) { // neighbour is same size or bigger
    //         real_t rho_star = Ustar.at(iCell_neighbor, IRho_star);
    //         if (rho_star > 0) do_SN_injection(iCell_neighbor, iCell, level_diff);
    //       }
    //     } // end for neighbours

    //     // Update cell quantities
    //     Uin.at(iCell, IRho)     += drho;
    //     Uin.at(iCell, IRho_vx)  += drho_vx;
    //     Uin.at(iCell, IRho_vy)  += drho_vy;
    //     Uin.at(iCell, IRho_vz)  += drho_vz;

    //     if (has_metallicity) {
    //       Uin.at(iCell, IRho_Z)   += drho_Z;
    //     }

    // });

    foreach_particle.reduce_particle( "particles_update_momentum_feedback", Ppos,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart, uint& SN_count )
    {
      // Age of the particle
      real_t age_physical = t - Pdata.at(iPart, IBIRTH);

      // If the SN will explode in this time step
      if ((age_physical < t_SNII_physical) && ((age_physical + dt_physical) > t_SNII_physical)) {
        pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
        pos_t part_vel = {Pdata.at(iPart, IVX), Pdata.at(iPart, IVY), Pdata.at(iPart, IVZ)};

        ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );

        pos_t cell_size = cells.getCellSize( iCell );
        real_t dx_loc = cell_size[IX]; // Assuming uniform cell size
        DYABLO_ASSERT_KOKKOS_DEBUG( cell_size[IX] == cell_size[IY] && cell_size[IX] == cell_size[IZ],
          "ParticleUpdate_momentum_feedback: Non-cubic cells are not supported." );
        real_t cell_volume = cell_size[IX] * cell_size[IY] * cell_size[IZ];

        // Compute supernova properties
        real_t Mstar = Pdata.at(iPart, IMASS);
        real_t mejecta = Mstar * eta_SNII;
        real_t num_sn = mejecta / M_SNII;

        real_t up = part_vel[IX];
        real_t vp = part_vel[IY];
        real_t wp = part_vel[IZ];

        real_t rho_loss = mejecta / cell_volume;
        real_t ekloss = rho_loss * 0.5 * (up*up + vp*vp + wp*wp);

        real_t zloss = yield_SNII;
        if (has_metallicity) {
          zloss = yield_SNII + (1.0 - yield_SNII) * Pdata.at(iPart, IMETAL);
        }
        real_t dzloss = rho_loss * zloss;

        // NOTE: there is a race condition here, since a particle exploding
        //       in a nearby cell may update this very cell ^o^.
        // TODO: fix it!
        // Get central cell properties
        real_t rho = Uin.at(iCell, IRho);
        real_t rho_vx = Uin.at(iCell, IRho_vx);
        real_t rho_vy = Uin.at(iCell, IRho_vy);
        real_t rho_vz = Uin.at(iCell, IRho_vz);
        real_t u = rho_vx / rho;
        real_t v = rho_vy / rho;
        real_t w = rho_vz / rho;
        real_t e = Uin.at(iCell, IE_tot);
        real_t ekk = 0.5 * rho * (u*u + v*v + w*w);
        real_t eth = e - ekk;

        real_t Z = 0.02; // Default solar metallicity
        if (has_metallicity) {
          Z = Uin.at(iCell, IRho_Z) / rho;
        }

        // Update central cell -> only dump 1-f_LOAD of the ejecta, the rest will be loaded onto neighbors
        Kokkos::atomic_add(&Uin.at(iCell, IRho),    (1 - f_LOAD) * rho_loss    - rho*        f_LOAD_CEN);
        Kokkos::atomic_add(&Uin.at(iCell, IRho_vx), (1 - f_LOAD) * rho_loss*up - rho*u     * f_LOAD_CEN);
        Kokkos::atomic_add(&Uin.at(iCell, IRho_vy), (1 - f_LOAD) * rho_loss*vp - rho*v     * f_LOAD_CEN);
        Kokkos::atomic_add(&Uin.at(iCell, IRho_vz), (1 - f_LOAD) * rho_loss*wp - rho*w     * f_LOAD_CEN);
        Kokkos::atomic_add(&Uin.at(iCell, IE_tot),  (1 - f_LOAD) * ekloss      - (ekk+eth) * f_LOAD_CEN);

        if (has_metallicity) {
          Kokkos::atomic_add(&Uin.at(iCell, IRho_Z), (1 - f_LOAD) * dzloss - rho*Z*f_LOAD_CEN);
        }

        // Update neighboring cells
        real_t rho_ejecta = rho_loss * f_LOAD / static_cast<real_t>(nSNnei);
        real_t rho_load = rho_ejecta + rho * f_LOAD_CEN / static_cast<real_t>(nSNnei);

        // Loop over solid angles (neighboring cells)
        for (int j = 0; j < nSNnei; j++) {
          // Get neighboring cell position
          pos_t xnei = {
            part_pos[IX] + xSNnei_dev(j)[IX] * dx_loc,
            part_pos[IY] + xSNnei_dev(j)[IY] * dx_loc,
            part_pos[IZ] + xSNnei_dev(j)[IZ] * dx_loc
          };

          ForeachCell::CellIndex iCellNei = cells.getCellFromPos( xnei );
          // DYABLO_ASSERT_KOKKOS_DEBUG(
          //   !((iCellNei.i == iCell.i) && (iCellNei.j == iCell.j) && (iCellNei.k == iCell.k) && (iCellNei.iOct.iOct == iCell.iOct.iOct)),
          //   "ParticleUpdate_momentum_feedback: Neighboring cell is the same as the central cell: " <<
          //   "{i, j, k} = {" << iCellNei.i << ", " << iCellNei.j << ", " << iCellNei.k << "} | iOct = " << iCellNei.iOct.iOct <<
          //   " | j = " << j << " | xSNnei(j) = {" << xSNnei_dev(j)[IX] << ", " << xSNnei_dev(j)[IY] << ", " << xSNnei_dev(j)[IZ] << "} × " << dx_loc );

          pos_t cell_size_nei = cells.getCellSize( iCellNei );
          real_t relative_volume = (cell_size_nei[IX] * cell_size_nei[IY] * cell_size_nei[IZ]) / cell_volume;

          // Get neighboring cell properties
          real_t rho_nei = Uin.at(iCellNei, IRho);
          real_t Z_nei = 0.02; // Default solar metallicity
          if (has_metallicity) {
            Z_nei = Uin.at(iCellNei, IRho_Z) / rho_nei;
          }

          // Compute actual mass ratio
          real_t f_w_cell = (rho_load + rho_nei / 8.0) / rho_ejecta - 1.0;

          // Compute critical mass ratio
          real_t nH_nei;
          {
            auto rho_physical = Units::supercomoving_to_physical<Units::Density>(rho_nei, aexp) * code_density;
            nH_nei = (rho_physical * XH).convert_to(mp_per_cc);
          }
          real_t f_w_crit;
          {
            real_t Zdepen = pow(FMAX(0.01, Z_nei / 0.02), expZ_SN * 2.0);
            f_w_crit = pow(A_SN / 1e4, 2.0) / (f_ESN * M_SNII) *
                       pow(num_sn, (expE_SN - 1.0) * 2.0) *
                       pow(nH_nei, expN_SN * 2.0) * Zdepen - 1.0;
            f_w_crit = FMAX(0.0, f_w_crit);
          }

          // Compute SN terminal momentum
          real_t vload_rad = SQRT(2.0 * f_ESN * e_SN * (1.0 + f_w_crit)) /
                            f_CANCEL / (1.0 + f_w_cell) / f_LOAD;

          // Determine in which phase is the blast wave
          real_t vload;
          if (f_w_cell >= f_w_crit) {
            // Radiative phase
            vload = vload_rad;
          } else {
            // Adiabatic phase
            real_t f_esn2 = 1.0 - (1.0 - f_ESN) * f_w_cell / f_w_crit;
            vload = SQRT(2.0 * f_esn2 * e_SN / (1.0 + f_w_cell)) / f_LOAD;
          }
          if (vload > vload_rad) vload = vload_rad;

          // Add mass, momentum and energy from central cell loading
          real_t inv_nSNnei_vol_nei = 1.0 / (static_cast<real_t>(nSNnei) * relative_volume);

          Kokkos::atomic_add(&Uin.at(iCellNei, IRho),     (rho_loss *      f_LOAD + rho *         f_LOAD_CEN) * inv_nSNnei_vol_nei);
          Kokkos::atomic_add(&Uin.at(iCellNei, IRho_vx),  (rho_loss * up * f_LOAD + rho * u *     f_LOAD_CEN) * inv_nSNnei_vol_nei);
          Kokkos::atomic_add(&Uin.at(iCellNei, IRho_vy),  (rho_loss * vp * f_LOAD + rho * v *     f_LOAD_CEN) * inv_nSNnei_vol_nei);
          Kokkos::atomic_add(&Uin.at(iCellNei, IRho_vz),  (rho_loss * wp * f_LOAD + rho * w *     f_LOAD_CEN) * inv_nSNnei_vol_nei);
          Kokkos::atomic_add(&Uin.at(iCellNei, IE_tot),   (ekloss *        f_LOAD + (ekk + eth) * f_LOAD_CEN) * inv_nSNnei_vol_nei);

          if (has_metallicity) {
            Kokkos::atomic_add(&Uin.at(iCellNei, IRho_Z),
              (dzloss * f_LOAD + rho * Z * f_LOAD_CEN) * inv_nSNnei_vol_nei);
          }

          // Compute radial momentum and kinetic energy
          real_t p_solid = (1.0 + f_w_cell) * rho_ejecta * vload;
          real_t ek_solid = p_solid * (vload * f_LOAD) / 2.0;

          // Important: this is *not* momentum conserving since
          // what's injected on the left may not exactly match
          // what's injected on the right!
          // Add momentum and energy from the cold shell
          Kokkos::atomic_add(&Uin.at(iCellNei, IRho_vx), p_solid * vSNnei_dev(j)[IX] / relative_volume);
          Kokkos::atomic_add(&Uin.at(iCellNei, IRho_vy), p_solid * vSNnei_dev(j)[IY] / relative_volume);
          Kokkos::atomic_add(&Uin.at(iCellNei, IRho_vz), p_solid * vSNnei_dev(j)[IZ] / relative_volume);
          Kokkos::atomic_add(&Uin.at(iCellNei, IE_tot), ek_solid / relative_volume);
        }

        // Update particle properties
        Pdata.at(iPart, IMASS) -= mejecta;

        SN_count++;
      }
    }, SN_count);

    if (SN_count > 0)
      std::cout << "[ParticleUpdate_momentum_feedback] Number of SNII explosions in array '" << array_name << "': " << SN_count << std::endl;
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;

  real_t eta_SNII;
  real_t yield_SNII;
  real_t E_SNII_physical;
  real_t M_SNII_physical;
  real_t t_SNII_physical;

  bool cosmology;

  std::vector<std::string> array_names;

  // Momentum feedback specific parameters
  real_t A_SN;
  real_t expN_SN;
  real_t expE_SN;
  real_t expZ_SN;
  real_t f_LOAD;
  real_t f_LOAD_CEN;
  real_t f_CANCEL;
  real_t f_ESN;

  // Neighbor positions and velocities
  pos_t xSNnei[nSNnei];
  pos_t vSNnei[nSNnei];
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory,
                  dyablo::ParticleUpdate_momentum_feedback,
                  "ParticleUpdate_momentum_feedback")
