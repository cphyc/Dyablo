#pragma once

#include "utils/units/Units.h"

namespace dyablo {

namespace{

    // Collisional ionisation rate in cm^{3}.s^{-1} (Maselli et al. 2003)
    KOKKOS_INLINE_FUNCTION
    real_t gamma_h0(const real_t temp)
    {
        real_t res = 5.85e-11;
        res *= sqrt(temp);
        res *= 1.0 / (1.0 + sqrt(temp / 1.e5));
        res *= exp(-157809.1 / temp);

        // Convert to m^{3}.s^{-1}
        res *= 1e-6;
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    real_t get_alpha_a(const real_t temp)
    {
        const real_t lambda = 2.0 * 157807.0 / temp;
        real_t alpha_ah = 1.269e-13;
        alpha_ah *= pow(lambda, 1.503);
        alpha_ah /= pow(1.0 + pow(lambda / 0.522, 0.47), 1.923);

        // Convert to m^{3}.s^{-1}
        alpha_ah *= 1e-6;

        return alpha_ah;
    }

    KOKKOS_INLINE_FUNCTION
    real_t get_alpha_b(const real_t temp)
    {
        const real_t lambda = 2.0 * 157807.0 / temp;

        real_t alpha_bh = 2.753e-14;
        alpha_bh *= pow(lambda, 1.5);
        alpha_bh /= pow(1.0 + pow(lambda / 2.74, 0.407), 2.242);

        // Convert to m^{3}.s^{-1}
        alpha_bh *= 1e-6;

        return alpha_bh;
    }

    KOKKOS_INLINE_FUNCTION
    real_t get_beta(const real_t temp)
    {
        const real_t lambda = 2.0 * 157807.0 / temp;
 
        real_t beta_h = 21.11 * pow(temp, -3.0 / 2.0) * exp(-lambda / 2.0) * pow(lambda, -1.089);
        beta_h /= pow(1.0 + pow(lambda / 0.354, 0.874), 1.01);

        // Convert to m^{3}.s^{-1}
        beta_h *= 1e-6;

        return beta_h;
    }

    // Case A - Recombination rate in cm^{3}.s^{-1} (Hui & Gnedin 1997)
    KOKKOS_INLINE_FUNCTION
    real_t alpha_ah(const real_t temp)
    {
        const real_t lambda = 2.0 * 157807.0 / temp;
        real_t res = 1.269e-13;
        res *= pow(lambda, 1.503);
        res /= pow(1.0 + pow(lambda / 0.522, 0.47), 1.923);

        // Convert to m^{3}.s^{-1}
        res *= 1e-6;
        return res;
    }

    // Case A - HII recombination cooling rate in erg.cm^{3}.s^{-1} (Hui & Gnedin 1997)
    KOKKOS_INLINE_FUNCTION
    real_t recombination_cooling_rate_ah(const real_t temp)
    {
        const real_t lambda = 2.0 * 157807.0 / temp;
        real_t res = 1.778e-29 * pow(lambda, 1.965);
        res /= pow(1.0 + pow(lambda / 0.541, 0.502), 2.697);

        // Convert to J.m^{3}.s^{-1}
        res *= 1e-13;
        return res;
    }

    // Case B - Recombination rate from in cm^{3}.s^{-1} (Hui & Gnedin 1997)
    KOKKOS_INLINE_FUNCTION
    real_t alpha_bh(const real_t temp)
    {
        const real_t lambda = 2.0 * 157807.0 / temp;
        real_t res = 2.753e-14;

        res *= lambda * lambda / sqrt(lambda);

        res /= pow(1.0 + pow(lambda / 2.74, 0.407), 2.242);

        // Convert to m^{3}.s^{-1}
        res *= 1e-6;
        return res;
    }

    // HI collisional ionisation coefficient in cm^{3}.s^{-1}.K^{3/2} (Hui & Gnedin
    // 1997)
    KOKKOS_INLINE_FUNCTION
    real_t beta_h(const real_t temp)
    {
        const real_t lambda = 2.0 * 157807.0 / temp;
        real_t res = 21.11 * sqrt(temp) / (temp * temp) * exp(-0.5 * lambda) * pow(lambda, -1.089);

        res /= pow(1.0 + pow(lambda / 0.354, 0.874), 1.01);

        // Convert to m^{3}.s^{-1}
        res *= 1e-6;
        return res;
    }

    // Collisional ionisation cooling in erg.cm^{3}.s^{-1} (Maselli et al. 2003)
    KOKKOS_INLINE_FUNCTION
    real_t ksi_h0(const real_t temp)
    {
        const real_t t0 = sqrt(temp);
        const real_t t1 = 0.0031622776601683;
        real_t res = 1.27e-21 * t0 / (1.0 + t0 * t1);

        res *= exp(-157809.1 / temp);

        // Convert to J.m^{3}.s^{-1}
        res *= 1e-13;
        return res;
    }

    // Recombination cooling for H0 in erg.cm^{3}.s^{-1} (Maselli et al. 2003)
    // Case A or B or total ?
    KOKKOS_INLINE_FUNCTION
    real_t eta_h0(const real_t temp)
    {
        real_t res = 8.7e-27 * sqrt(temp) * pow(temp / 1.e3, -0.2) / (1.0 + pow(temp / 1.e6, 0.7));

        // Convert to J.m^{3}.s^{-1}
        res *= 1e-13;
        return res;
    }

    // Collisional exciation cooling for H0 in erg.cm^{3}.s^{-1} (Maselli et al. 2003)
    KOKKOS_INLINE_FUNCTION
    real_t psi_h0(const real_t temp)
    {
        const real_t t0 = 0.0031622776601683;
        real_t res = 7.5e-19 / (1.0 + t0 * sqrt(temp));

        res *= exp(-118348.0 / temp);

        // Convert to J.m^{3}.s^{-1}
        res *= 1e-13;
        return res;
    }

    // Bremsstrahlung cooling in erg.cm^{3}.s^{-1} (Maselli et al. 2003) WARNING: we
    // took the densities out of the formula so one needs to multiply the result by
    // rho_electrons^2 (in case of pure Hydrogen chemistry)
    KOKKOS_INLINE_FUNCTION
    real_t beta_bremsstrahlung(const real_t temp)
    {
        // Convert to J.m^{3}.s^{-1} using * 1e-13
        return 1.42e-40 * sqrt(temp);
    }

    // Total cooling rate (sum of terms below) in erg.cm^{3}.s^{-1}
    KOKKOS_INLINE_FUNCTION
    real_t cooling_rate(const real_t temp, const real_t x)
    {
        return (beta_bremsstrahlung(temp) + eta_h0(temp)) * x * x +
            (psi_h0(temp) + ksi_h0(temp)) * (1.0 - x)*(1.0 - x);
    }

    KOKKOS_INLINE_FUNCTION
    real_t cooling_rate_density(const real_t temp, const real_t nH, const real_t x_n)
    {
        const real_t f1 = (nH * x_n) * (nH * x_n);
        const real_t f2 = (nH * x_n) * nH - f1;

        return (beta_bremsstrahlung(temp) + eta_h0(temp)) * f1 + (psi_h0(temp) + ksi_h0(temp)) * f2;
    }

    KOKKOS_INLINE_FUNCTION
    real_t heating_rate(const real_t nH, const real_t x_n, const real_t N, const real_t sigma_n, const real_t sigma_e, const real_t typical_energy)
    {
        const real_t e = (typical_energy*sigma_e - 13.6*sigma_n) * 1.60218e-19;

        // Short time step
        return nH * (1.0 - x_n) * N * e;
    }

    KOKKOS_INLINE_FUNCTION
    real_t solve_raphson_newton(const real_t x, const real_t m, const real_t n, const real_t q, const real_t p){

        real_t xnew = 0.0;
        int nmax = 500;
        real_t error = 1e3;

        real_t xold = x;
        int i = 0;
        while((error>1e-6) && (i<nmax)){
            real_t a = m*xold*xold*xold + n*xold*xold + p*xold + q;
            real_t b = 3*m*xold*xold + 2*n*xold + p;
            xnew = xold - a/b;
            if(isnan(xnew)){
                xnew = xold;
                printf("nan\n");
            }
            error = abs((xnew-xold)/xold);
            xold =  xnew;
            i = i+1;
        }

        xnew = FMIN(1.0-1e-6,xnew);
        xnew = FMAX(1e-6,xnew);

        return xnew;
    }


    void spectrum_BBody(real_t* energy, real_t* spec, real_t temp, size_t size){
        real_t Estar = temp * Units::KBOLTZ / Units::EV;
        real_t x = 0.0;
        for(size_t i=0; i<size; i++){
            x = energy[i] / Estar;
            spec[i] = x*x*x / (exp(x) - 1 + 1e-9); // adimensionalized spectrum
        }
    }


    void sigma_HI(real_t* egy, real_t* sig, size_t size){

        // Table E1 & E2 Rosdahl+
        real_t EION = 13.6; // eV
        real_t E0 = 0.4298; // eV
        real_t S0 = 5.475e-14; // cm2
        real_t P = 2.963;
        real_t ya = 32.88;
        real_t yw = 0.0;
        real_t y0 = 0.0; 
        real_t y1 = 0.0;

        real_t x = 0;
        real_t y = 0;

        // Formula E1 from Rosdahl +
        for(size_t i=0; i<size; i++){
            x = egy[i]/E0-y0;
            y = sqrt(x*x + y1*y1);
            sig[i] = S0*((x-1.0)*(x-1.0)+yw*yw)* pow(y,(0.5*P-5.5)) / pow((1.0 + sqrt(y/ya)),P);

            // Cross section is zero below ionisation energy
            if(x <= EION)
                sig[i] = 0;
       
            // Switch from cm2 to m2
            sig[i] = sig[i]*1e-4;
        } 

    }


    // Trapezoidal rule for integration
    real_t trapezoid(real_t* y, real_t* x, size_t size){
        real_t sum = 0.0;
        for(size_t i=0; i<size-1; i++)
            sum = sum + 0.5*(y[i]+y[i+1])*(x[i+1]-x[i]);
        return sum;
    }


    auto computeSigma(real_t temp){

        struct S {
            real_t sn;
            real_t se;
            real_t etyp;
        } s;

        // Max energy for computations (eV)
        real_t egymax = 500.0;

        // Sampling points number for energy
        size_t negy = 163840;

        // An energy array in eV
        real_t egy[negy];
        real_t step = (egymax-13.6)/(negy-1.0);
        for(size_t i=0; i<negy; i++)
            egy[i] = 13.6 + i*step;

        // Ionization cross section as a function of energy
        real_t sig[negy];
        sigma_HI(egy, sig, negy); 

        real_t spec[negy];  
        spectrum_BBody(egy, spec, temp, negy);

        real_t sig_spec[negy];
        real_t spec_norm[negy];
        real_t sig_spec_norm[negy];

        for(size_t i=0; i<negy; i++){
            sig_spec[i] = sig[i]*spec[i];
            spec_norm[i] = spec[i]/(egy[i] + 1e-9);
            sig_spec_norm[i] = spec_norm[i]*sig[i];
        }

        // Number averaged cross section
        s.sn = trapezoid(sig_spec_norm, egy, negy) / trapezoid(spec_norm, egy, negy);

        // Energy averaged cross section
        s.se = trapezoid(sig_spec, egy, negy) / trapezoid(spec, egy, negy);

        // Typical energy
        s.etyp = trapezoid(spec, egy, negy) / trapezoid(spec_norm, egy, negy);

        printf("T=%f sigma_n=%e sigma_e=%e typical_energy=%f\n", temp, s.sn, s.se, s.etyp);

        return s;
    }
}
}