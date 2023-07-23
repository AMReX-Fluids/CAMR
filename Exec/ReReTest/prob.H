#ifndef PROB_H
#define PROB_H

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
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real*      dx = geomdata.CellSize();

    const amrex::Real half_point = (prob_hi[0] - prob_lo[0]) * 0.5;

    amrex::Real x = prob_lo[0] + (i+amrex::Real(0.5))*dx[0];
    amrex::Real Pt, rhot, uxt;

    if (x < half_point - 1.0) { // shift one coarse cell to left of c/f interface
        Pt   = prob_parm.p_l;
        rhot = prob_parm.rho_l;
        uxt  = prob_parm.u_l;
        state(i,j,k,UFS  ) = amrex::Real(1.0) * rhot;
        state(i,j,k,UFS+1) = amrex::Real(0.0) * rhot;
        state(i,j,k,UFS+2) = amrex::Real(0.0) * rhot;
    } else {
        Pt   = prob_parm.p_r;
        rhot = prob_parm.rho_r;
        uxt  = prob_parm.u_r;
        state(i,j,k,UFS  ) = amrex::Real(0.0) * rhot;
        state(i,j,k,UFS+1) = amrex::Real(1.0) * rhot;
        state(i,j,k,UFS+2) = amrex::Real(0.0) * rhot;
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
    state(i,j,k,UTEMP) = amrex::Real(0.0);
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
