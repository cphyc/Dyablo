#include "SourceUpdate_base.h"

#include "ionization/Ionization_utils.h"

namespace dyablo{

namespace{

enum VarIndex_Chem{ Irho,Ie_tot, Irho_vx,Irho_vy,Irho_vz, Ie_rad, Ifx_rad,Ify_rad,Ifz_rad, Ixe,Izr, Itemp };

KOKKOS_INLINE_FUNCTION
void apply_rad_chem( const ForeachCell::CellIndex& iCell_Uin,
                    const UserData::FieldAccessor& Uout,
                    int ndim, real_t xpos, real_t ypos, real_t zpos, real_t gamma0, real_t rho_crit,
                    real_t dt, real_t dx, real_t ctilde, real_t aexp, real_t tstar, real_t rhostar, 
                    real_t vstar, real_t size, real_t Ndot, RadType mode,
                    bool apply_cooling, bool dynamic, real_t sigma_n_c, real_t sigma_e_c, real_t typical_energy)
{
  
  real_t dtSI = dt*tstar*(aexp*aexp); // Timestep in s
  real_t dxSI = dx*aexp; // dx in m
  real_t nstar = rhostar/Units::PROTON_MASS;

   // Local Gaz density
  real_t rho = Uout.at(iCell_Uin, VarIndex_Chem::Irho);
  real_t rhoSI = rho*rhostar/(aexp*aexp*aexp); // Physical gas density in kg/m3. We expect to have rhoSI = 1e3*mass_proton 
  real_t nHSI = rhoSI/Units::PROTON_MASS; // Physical atom number density in atoms/m3

  // Local Photon number Density
  real_t Norg = Uout.at(iCell_Uin, VarIndex_Chem::Ie_rad);
  real_t NSI = Norg*nstar/(aexp*aexp*aexp);

  // Local Flux. 
  // In principle we need a full conversion to physical quantites but since F is not used it's not necessary
  real_t FXSI = Uout.at(iCell_Uin, VarIndex_Chem::Ifx_rad);
  real_t FYSI = Uout.at(iCell_Uin, VarIndex_Chem::Ify_rad);
  real_t FZSI = Uout.at(iCell_Uin, VarIndex_Chem::Ifz_rad);

  // Local ionisation fraction
  real_t x = Uout.at(iCell_Uin, VarIndex_Chem::Ixe);

  // Local reionisation time
  real_t zreold = Uout.at(iCell_Uin, VarIndex_Chem::Izr);

  // Derive temperature
  real_t e_tot = Uout.at(iCell_Uin, VarIndex_Chem::Ie_tot);
  real_t rho_u = Uout.at(iCell_Uin, VarIndex_Chem::Irho_vx);
  real_t rho_v = Uout.at(iCell_Uin, VarIndex_Chem::Irho_vy);
  real_t rho_w = Uout.at(iCell_Uin, VarIndex_Chem::Irho_vz);

  real_t pstar = rhostar * vstar * vstar;
  real_t e_cin = 0.5 * (rho_u*rho_u + rho_v*rho_v + rho_w*rho_w)/rho;
  real_t pressure = (e_tot - e_cin)*(gamma0-1.0);
  real_t pressure_SI = pressure*pstar/(aexp*aexp*aexp*aexp*aexp);

  real_t temp_SI = Uout.at(iCell_Uin, VarIndex_Chem::Itemp);
  // In case of dynamic mode, we need to recompute the temperature related to the e_tot and rho
  if(dynamic)
    temp_SI = pressure_SI /( (gamma0 - 1.0) * 1.5 * nHSI*(1+x) * Units::KBOLTZ);

  real_t temp_new_SI = temp_SI;
  real_t e_tot_new = e_tot;

  // Source of photons
  real_t deltaN = 0.0;
  real_t x1 = 0.0, y1 = 0.0, z1 = 0.0;  // Position of the source
  real_t r1 = (xpos-x1)*(xpos-x1)+(ypos-y1)*(ypos-y1)+(zpos-z1)*(zpos-z1);
  
  // Increase the photons number density due to sources when criterion is met
  if ( (mode==REGULAR && rho>rho_crit) || (mode==STROMGREN && r1<size*size && (xpos-x1)>0 && (ypos-y1)>0 && (zpos-z1)>0) ){ 
    deltaN = dtSI*(Ndot/(dxSI*dxSI*dxSI)); 
    NSI = NSI+deltaN;
  }

  // Absorption polynomial coefficients
  real_t nh_square_dt = nHSI*nHSI*dtSI;
  real_t alphab = get_alpha_b(temp_SI);
  real_t alpha = get_alpha_a(temp_SI);
  real_t beta = get_beta(temp_SI);
  
  real_t m = (alphab + beta)*nh_square_dt;
  real_t n = nHSI - (alpha + beta)*nHSI/sigma_n_c - alphab*nh_square_dt - 2.0*beta*nh_square_dt;
  real_t p = -nHSI*(1+x) - NSI - 1./(sigma_n_c*dtSI) + beta*nHSI/sigma_n_c + beta*nh_square_dt;
  real_t q = NSI + nHSI*x + x/(sigma_n_c*dtSI);

  // Update ionisation fraction
  real_t xnew = solve_raphson_newton(x, m, n, q, p);
  // xnew= x - (m*x*x*x +n*x*x +p*x +q)/(3*m*x*x +2*n*x +p);

  // Update NSI (equation 5 from  Aubert & Teyssier 2008)
  real_t NSI_new = NSI - alphab*nHSI*nHSI*xnew*xnew*dtSI - nHSI*(xnew-x);
  NSI_new = (NSI_new/nstar)*(aexp*aexp*aexp); // switch back to code units

  // Update fluxes
  real_t fact = 1.0 + sigma_n_c*nHSI*dtSI*(1-xnew); //fact is dimensionless
  FXSI = FXSI/fact;
  FYSI = FYSI/fact;
  FZSI = FZSI/fact;

  real_t F = sqrt(FXSI*FXSI + FYSI*FYSI + FZSI*FZSI);
  real_t Fred = F/(ctilde*NSI_new);

  if(Fred > 1.0){
    FXSI=FXSI/F*ctilde*NSI_new;
    FYSI=FYSI/F*ctilde*NSI_new;
    FZSI=FZSI/F*ctilde*NSI_new;
  }

  if(apply_cooling){
    // Compute cooling and heating effects and derive new temperature
    const real_t c_rate = cooling_rate_density(temp_SI, nHSI, xnew);
    const real_t h_rate = heating_rate(nHSI, xnew, NSI, sigma_n_c, sigma_e_c, typical_energy);  // here we use the non-updated NSI value
    const real_t coef = 2. * (h_rate - c_rate) * dtSI / (3.0 * nHSI * (1.0 + xnew) * Units::KBOLTZ);
    temp_new_SI = FMAX((coef + temp_SI) / (1.0 + xnew - x), 10.0);

    // Update e_tot value
    real_t pressure_new_SI = (gamma0 - 1.0) * 1.5 * nHSI*(1+xnew) * Units::KBOLTZ * temp_new_SI;
    real_t pressure_new = pressure_new_SI / pstar * (aexp*aexp*aexp*aexp*aexp);
    e_tot_new = e_cin + pressure_new/(gamma0-1.0);
  }

  // Update zreion
  real_t zrenew = zreold;
  if( xnew>0.5 && abs(zreold)<=1e-6 ){
      zrenew = 1.0/aexp-1.0;
  }

  // Store results
  Uout.at(iCell_Uin, VarIndex_Chem::Ie_rad) = NSI_new; // there should be a NSMIN like PMIN
  Uout.at(iCell_Uin, VarIndex_Chem::Ifx_rad) = FXSI;
  Uout.at(iCell_Uin, VarIndex_Chem::Ify_rad) = FYSI;
  Uout.at(iCell_Uin, VarIndex_Chem::Ifz_rad) = FZSI;
  Uout.at(iCell_Uin, VarIndex_Chem::Ixe) = xnew;
  Uout.at(iCell_Uin, VarIndex_Chem::Izr) = zrenew;
  Uout.at(iCell_Uin, VarIndex_Chem::Itemp) = temp_new_SI;

 if(dynamic)
    Uout.at( iCell_Uin, VarIndex_Chem::Ie_tot) = e_tot_new;
}
}

/**
 * @brief Ionization 'Chem' source term
 */
class SourceUpdate_Ionization_Chem : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;

  real_t gamma0;
  real_t rho_crit;
  real_t ndot;
  real_t sigma_n_c;
  real_t sigma_e_c;
  real_t typical_energy;
  real_t ctilde_a0;
  
  RadType mode;

  bool apply_cooling;
  bool dynamic;

  // TODO use units for this
  real_t dx;
  real_t tstar;
  real_t rhostar;
  real_t vstar;  
public:
  SourceUpdate_Ionization_Chem(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  : foreach_cell(foreach_cell),
    timers(timers),

    gamma0(configMap.getValue<real_t>("hydro", "gamma0", 1.666)),
    rho_crit(configMap.getValue<real_t>("ionization", "rho_crit", 0.3)),
    ndot(configMap.getValue<real_t>("ionization", "ndot", 1e56)),
    sigma_n_c(configMap.getValue<real_t>( "ionization", "sigma_n_c" )),
    sigma_e_c(configMap.getValue<real_t>( "ionization", "sigma_e_c" )),
    typical_energy(configMap.getValue<real_t>( "ionization", "typical_energy" )),
    ctilde_a0(configMap.getValue<real_t>( "cosmology", "ctilde" ) / configMap.getValue<real_t>( "cosmology", "astart" )),
    mode(configMap.getValue<RadType>("ionization", "mode", REGULAR)),
    apply_cooling(configMap.getValue<bool>("ionization", "apply_cooling", true)),
    dynamic(configMap.getValue<bool>("ionization", "dynamic", false)),

    dx(configMap.getValue<real_t>( "cosmology", "dx") ),
    tstar(configMap.getValue<real_t>( "cosmology", "tstar") ),    
    rhostar(configMap.getValue<real_t>( "cosmology", "rhostar") ),
    vstar(configMap.getValue<real_t>( "cosmology", "vstar") )
  {}

  void update( UserData &U,
               ScalarSimulationData& scalar_data)
  {
    uint32_t ndim = foreach_cell.getDim();

    ForeachCell& foreach_cell = this->foreach_cell;

    timers.get("SourceUpdate_Ionization_Chem::update").start();

    enum VarIndex {IDR,IUR,IVR,IWR};

    UserData::FieldAccessor Uout = U.getAccessor( 
      {
        {"rho_next",    Irho    }, 
        {"e_tot_next",  Ie_tot  }, 
        {"rho_vx_next", Irho_vx },
        {"rho_vy_next", Irho_vy },
        {"rho_vz_next", Irho_vz }, 
        {"e_rad_next",  Ie_rad  }, 
        {"fx_rad_next", Ifx_rad },
        {"fy_rad_next", Ify_rad },
        {"fz_rad_next", Ifz_rad }, 
        {"xe_next",     Ixe     },
        {"zr_next",     Izr     }, 
        {"temp_next",   Itemp   }
      });

    real_t dt = scalar_data.get<real_t>("dt");
    real_t aexp = scalar_data.get<real_t>("aexp");

    real_t gamma0 = this->gamma0;
    real_t rho_crit = this->rho_crit;
    real_t ndot = this->ndot;
    real_t sigma_n_c = this->sigma_n_c;
    real_t sigma_e_c = this->sigma_e_c;
    real_t typical_energy = this->typical_energy;
    real_t ctilde = this->ctilde_a0 * aexp;
    
    RadType mode = this->mode;

    bool apply_cooling = this->apply_cooling;
    bool dynamic = this->dynamic;

    // TODO use units for this
    real_t dx = this->dx;
    real_t tstar = this->tstar;
    real_t rhostar = this->rhostar;
    real_t vstar = this->vstar;    

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    foreach_cell.foreach_cell( "SourceUpdate_Ionization_Chem", Uout.getShape(), 
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell_Uout) 
    {
      auto pos = cells.getCellCenter(iCell_Uout);
      auto size = cells.getCellSize(iCell_Uout);
      DYABLO_ASSERT_KOKKOS_DEBUG( size[IX] == SIZE[IY] && size[IX] == SIZE[IZ], "Only square cells supported" );

      apply_rad_chem( iCell_Uout, Uout, ndim,
                      pos[IX], pos[IY], pos[IZ],
                      gamma0, rho_crit, dt, dx, ctilde, aexp, 
                      tstar, rhostar, vstar, size[IX], ndot, 
                      mode, apply_cooling, dynamic, sigma_n_c, sigma_e_c, typical_energy);
    });

    timers.get("SourceUpdate_Ionization_Chem").stop();
  }
};


} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory, 
                  dyablo::SourceUpdate_Ionization_Chem, 
                  "SourceUpdate_Ionization_Chem" );