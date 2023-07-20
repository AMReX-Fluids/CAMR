#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <AMReX_FArrayBox.H>
#include "CAMR_Constants.H"
#include "IndexDefines.H"
#include "CAMR.H"
#include "EOS.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_reset_internal_e(
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real> const& S,
  const int l_allow_small_energy,
  const int l_allow_negative_energy,
  const int l_dual_energy_update_E_from_e,
  const amrex::Real l_dual_energy_eta2,
  const amrex::Real small_t,
  const int /*verbose*/)
{
    //First, check if the internal energy variable is
    //smaller than the internal energy computed via
    //a call to the EOS using the small temperature.
    //If so, reset it using the current temperature,
    //assuming it is at least as large as small_temp.
    //Note that allow_small_energy .eq. 0 overrides
    //allow_negative_energy .eq. 0 since a negative
    //energy is of course smaller than the smallest
    //allowed energy.

    const amrex::Real rho    =  S(i, j, k, URHO);
#ifdef AMREX_USE_EB
    if (rho > 0.)
#endif
    {
    const amrex::Real rhoInv = 1.0 / rho;
    AMREX_D_TERM( const amrex::Real Up = S(i, j, k, UMX) * rhoInv;,
                  const amrex::Real Vp = S(i, j, k, UMY) * rhoInv;,
                  const amrex::Real Wp = S(i, j, k, UMZ) * rhoInv;);
#if (AMREX_SPACEDIM == 2)
    const amrex::Real ke = 0.5 * (Up * Up + Vp * Vp);
#elif (AMREX_SPACEDIM == 3)
    const amrex::Real ke = 0.5 * (Up * Up + Vp * Vp + Wp * Wp);
#endif
    const amrex::Real eden = S(i, j, k, UEDEN) * rhoInv;

    amrex::Real massfrac[NUM_SPECIES];
    for (int sp = 0; sp < NUM_SPECIES; sp++) {
        massfrac[sp] = S(i, j, k, sp + UFS) * rhoInv;
    }

    amrex::Real e_out = 0.;
    amrex::Real small_e = 0.;
    const amrex::Real small_T = small_t;

    if (l_allow_small_energy == 0) {

        EOS::RTY2E(rho,small_T,massfrac,small_e);

        // E < small_e
        if (eden < small_e) {
            if (S(i, j, k, UEINT) * rhoInv < small_e) {
              const amrex::Real T = std::max(S(i,j,k,UTEMP), small_T);
              EOS::RTY2E(rho,T,massfrac,e_out);
              S(i, j, k, UEINT) = rho * e_out;
            }
            S(i, j, k, UEDEN) = S(i, j, k, UEINT) + rho * ke;

        // E >= small_e
        } else {

            const amrex::Real rho_eint = S(i, j, k, UEDEN) - S(i, j, k, URHO) * ke;

            // e/E > eta2
            if (rho_eint > 0. && rho_eint / S(i,j,k,UEDEN) > l_dual_energy_eta2) {
                S(i, j, k, UEINT) = rho_eint;
            }

            // e  < small_e
            if (S(i, j, k, UEINT) * rhoInv < small_e) {
                const amrex::Real T = std::max(S(i,j,k,UTEMP), small_T);
                EOS::RTY2E(rho,T,massfrac,e_out);
                if (l_dual_energy_update_E_from_e == 1) {
                  S(i, j, k, UEDEN) += (rho * e_out - S(i, j, k, UEINT));
                }
                S(i, j, k, UEINT) = S(i, j, k, URHO) * e_out;
            }
        }

    } else if (l_allow_negative_energy == 0) {

        if (S(i, j, k, UEDEN) < 0.) {
            if (S(i, j, k, UEINT) < 0.) {
                EOS::RTY2E(rho,small_T,massfrac,e_out);
                S(i, j, k, UEINT) = rho * e_out;
            }
            S(i, j, k, UEDEN) = S(i, j, k, UEINT) + rho * ke;

        } else {

            const amrex::Real rho_eint = S(i, j, k, UEDEN) - rho * ke;
            if (rho_eint > 0. && rho_eint / S(i,j,k,UEDEN) > l_dual_energy_eta2) {
                S(i, j, k, UEINT) = rho_eint;

            } else if (S(i,j,k,UEINT) > 0 && l_dual_energy_update_E_from_e == 1) {
                S(i, j, k, UEDEN) = S(i,j,k,UEINT) + rho * ke;

            } else if (S(i,j,k,UEINT) <= 0.) {
                EOS::RTY2E(rho,small_T,massfrac,e_out);
                S(i, j, k, UEINT) = rho * e_out;

                if (l_dual_energy_update_E_from_e == 1) {
                    S(i, j, k, UEDEN) += (rho * e_out - S(i,j,k,UEINT));
                }
            }
        }

     // If (l_allow_negative_energy == 1) and (l_allow_small_energy == 1)
     // then just reset (rho e) from (rho E)
     } else {
         S(i, j, k, UEINT) = S(i, j, k, UEDEN) - rho * ke;
     }

    } // rho > 0
}

#endif
