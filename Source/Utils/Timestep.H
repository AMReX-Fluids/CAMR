#ifndef _TIMESTEP_H_
#define _TIMESTEP_H_

#include <AMReX_FArrayBox.H>
#include "IndexDefines.H"
#include "CAMR_Constants.H"

#ifdef AMREX_USE_EB
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBCellFlag.H>
#endif

// EstDt routines

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
CAMR_estdt_hydro(
  const amrex::Box& bx,
  const amrex::Array4<const amrex::Real>& u,
#ifdef AMREX_USE_EB
  const amrex::Array4<const amrex::EBCellFlag>& flags,
#endif
  AMREX_D_DECL(
    const amrex::Real& dx,
    const amrex::Real& dy,
    const amrex::Real& dz)) noexcept
{
  amrex::Real dt = std::numeric_limits<amrex::Real>::max();

  amrex::Loop(bx, [=, &dt](int i, int j, int k) {
#ifdef AMREX_USE_EB
    if (!flags(i, j, k).isCovered()) {
#endif
      const amrex::Real rho = u(i, j, k, URHO);
      const amrex::Real rhoInv = 1.0 / rho;
      amrex::Real massfrac[NUM_SPECIES];
      amrex::Real eint = u(i, j, k, UEINT)/rho;
      amrex::Real c, gam, pres;
      for (int n = 0; n < NUM_SPECIES; ++n) {
        massfrac[n] = u(i, j, k, UFS + n) * rhoInv;
      }
      EOS::REY2P(rho,eint,massfrac,pres);
      EOS::REY2Gam(rho,eint,massfrac,gam);
      c = std::sqrt(gam*pres/rho);
      AMREX_D_TERM(const amrex::Real ux = u(i, j, k, UMX) * rhoInv;
                   const amrex::Real dt1 = dx / (c + std::abs(ux));
                   dt = std::min(dt, dt1);
                   , const amrex::Real uy = u(i, j, k, UMY) * rhoInv;
                   const amrex::Real dt2 = dy / (c + std::abs(uy));
                   dt = std::min(dt, dt2);
                   , const amrex::Real uz = u(i, j, k, UMZ) * rhoInv;
                   const amrex::Real dt3 = dz / (c + std::abs(uz));
                   dt = std::min(dt, dt3););
#ifdef AMREX_USE_EB
    }
#endif
  });
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dt > 0.0, "ERROR: dt needs to be positive.");
  return dt;
}
#endif
