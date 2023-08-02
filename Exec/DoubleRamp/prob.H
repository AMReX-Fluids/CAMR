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
    const amrex::Real*      dx = geomdata.CellSize();

    amrex::Real Pt=0;
    amrex::Real rhot=0;
    amrex::Real  uxt=0;
    amrex::Real idot,jdot;

    int n_pts = 3;
#if (AMREX_SPACEDIM == 3)
    for(int inside_i = 0; inside_i <= n_pts+1; inside_i++){
        for(int inside_j = 0; inside_j <= n_pts+1; inside_j++){
        idot = (prob_lo[0] + (i+amrex::Real(inside_i/(n_pts+1)))*dx[0]);
            for(int inside_k = 0;inside_k <= n_pts+1;inside_k++){
        jdot = (prob_lo[1] + (j+amrex::Real(inside_j/(n_pts+1)))*dx[1]);
        if( idot  < (amrex::Real(-sqrt(1.0/3.0))*jdot)){
                    Pt  += prob_parm.p_l/((n_pts+2)*(n_pts+2)*(n_pts+2));
                    rhot += prob_parm.rho_l/((n_pts+2)*(n_pts+2)*(n_pts+2));
                    uxt  += prob_parm.u_l/((n_pts+2)*(n_pts+2)*(n_pts+2));
                }else{
                    Pt  += prob_parm.p_r/((n_pts+2)*(n_pts+2)*(n_pts+2));
                    rhot += prob_parm.rho_r/((n_pts+2)*(n_pts+2)*(n_pts+2));
                    uxt  += prob_parm.u_r/((n_pts+2)*(n_pts+2)*(n_pts+2));
                 }
            }
        }
    }

#endif

#if (AMREX_SPACEDIM == 2)
    for(int inside_i = 0; inside_i <= n_pts+1; inside_i++){
        idot = (prob_lo[0] + (i+amrex::Real(inside_i/(n_pts+1)))*dx[0]);
        for(int inside_j = 0; inside_j <= n_pts+1; inside_j++){
            jdot = (prob_lo[1] + (j+amrex::Real(inside_j/(n_pts+1)))*dx[1]);
            if(idot < (amrex::Real(-sqrt(1.0/3.0))*jdot)){
            Pt  += prob_parm.p_l/((n_pts+2)*(n_pts+2));
                rhot += prob_parm.rho_l/((n_pts+2)*(n_pts+2));
                uxt  += prob_parm.u_l/((n_pts+2)*(n_pts+2));
            }else{
                   Pt  += prob_parm.p_r/((n_pts+2)*(n_pts+2));
                rhot += prob_parm.rho_r/((n_pts+2)*(n_pts+2));
                uxt  += prob_parm.u_r/((n_pts+2)*(n_pts+2));
             }
    }
    }

#endif

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

    state(i,j,k,UFS  ) = amrex::Real(1.0) * state(i,j,k,URHO);
    state(i,j,k,UFS+1) = amrex::Real(0.0) * state(i,j,k,URHO);
    state(i,j,k,UFS+2) = amrex::Real(0.0) * state(i,j,k,URHO);
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
