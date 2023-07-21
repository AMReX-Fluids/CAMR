#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>

#include "IndexDefines.H"
#include "prob_parm.H"
#include "CAMR_params.H"
#include "EOS.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_initdata( int i, int j, int k,
               amrex::Array4<amrex::Real> const& state,
               amrex::GeometryData const& geomdata,
               ProbParmDevice const& prob_parm)
{
    // Geometry
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real*      dx = geomdata.CellSize();

    amrex::Real x = prob_lo[0] + (i+amrex::Real(0.5))*dx[0];
    amrex::Real Pt, rhot, uxt;
    if (x < amrex::Real(0.5)) {
        Pt   = prob_parm.p_l;
        rhot = prob_parm.rho_l;
        uxt  = prob_parm.u_l;
    } else {
        Pt   = prob_parm.p_r;
        rhot = prob_parm.rho_r;
        uxt  = prob_parm.u_r;
    }
    state(i,j,k,URHO ) = rhot;
    state(i,j,k,UMX  ) = rhot*uxt;
    state(i,j,k,UMY  ) = amrex::Real(0.0);
#if (AMREX_SPACEDIM == 3)
    state(i,j,k,UMZ  ) = amrex::Real(0.0);
#endif
    amrex::Real et = Pt/(EOS::gamma-amrex::Real(1.0));
    state(i,j,k,UEINT) = et;
    state(i,j,k,UEDEN) = et + amrex::Real(0.5)*rhot*uxt*uxt;
    if (i == 10 and j == 6) amrex::Print() <<" UEDEN " << state(i,j,k,UEDEN) << std::endl;
    state(i,j,k,UTEMP) = amrex::Real(0.0);

    state(i,j,k,UFS  ) = amrex::Real(1.0) * state(i,j,k,URHO);
    state(i,j,k,UFS+1) = amrex::Real(0.0) * state(i,j,k,URHO);
    state(i,j,k,UFS+2) = amrex::Real(0.0) * state(i,j,k,URHO);

    if (i == -5 and j == -5) amrex::Print() <<" EINT IN PROB " << state(i,j,k,UEINT) << std::endl;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real* /*x[AMREX_SPACEDIM]*/,
  const amrex::Real s_int[NVAR],
  amrex::Real s_ext[NVAR],
  const int /*idir*/,
  const int /*sgn*/,
  const amrex::Real /*time*/,
  amrex::GeometryData const& /*geomdata*/,
  ProbParmDevice const& /*prob_parm*/)
{
  for (int n = 0; n < NVAR; n++) {
    s_ext[n] = s_int[n];
  }
}
#endif
