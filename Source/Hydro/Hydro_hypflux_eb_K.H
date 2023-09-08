#ifndef HYDRO_HYPFLUX_K_H
#define HYDRO_HYPFLUX_K_H

#include "IndexDefines.H"
#include <AMReX_FArrayBox.H>
#include <Hydro_riemann.H>
#include <cmath>

#ifdef AMREX_USE_EB
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
compute_hyp_wallflux (int i, int j, int k,
                      amrex::Array4<amrex::Real const> const& q,
                      amrex::Array4<amrex::Real const> const& qa,
                      const amrex::Real axm, const amrex::Real axp,
                      const amrex::Real aym, const amrex::Real ayp,
#if (AMREX_SPACEDIM == 3)
                      const amrex::Real azm, const amrex::Real azp,
#endif
                      const amrex::Real small,
                      const amrex::Real small_dens,
                      const amrex::Real small_pres,
                      amrex::GpuArray<amrex::Real,NVAR>& fluxw,
                      amrex::Real smallu) noexcept
{
    using amrex::Real;

    Real u = q(i,j,k,QU);
    Real v = q(i,j,k,QV);
#if (AMREX_SPACEDIM == 2)
    Real apnorm = std::sqrt( (axm-axp)*(axm-axp) + (aym-ayp)*(aym-ayp) );
    Real apnorminv = 1./apnorm;
    Real un = u*(axm-axp)*apnorminv + v*(aym-ayp)*apnorminv;
#else
    Real w = q(i,j,k,QW);
    Real apnorm = std::sqrt( (axm-axp)*(axm-axp) + (aym-ayp)*(aym-ayp) + (azm-azp)*(azm-azp) );
    Real apnorminv = 1./apnorm;
    Real un = u*(axm-axp)*apnorminv + v*(aym-ayp)*apnorminv + w*(azm-azp)*apnorminv;
#endif

    Real AMREX_D_DECL(momfluxn, momfluxt1, momfluxt2);

    Real dummy_gu, dummy_gv, dummy_gp, dummy_gg;
#if (AMREX_SPACEDIM == 3)
    Real dummy_gv2;
#endif

    Real rho = q(i,j,k,QRHO);
    Real p   = q(i,j,k,QPRES);
    Real qrei = q(i,j,k,QREINT);
    Real gamc = qa(i,j,k,QGAMC);
    Real cav = qa(i,j,k,QC);
    Real smallc = qa(i,j,k,QCSML);
    Real ustar;

    riemannus(
       rho,  AMREX_D_DECL(un, 0., 0.), p, qrei,
       rho, AMREX_D_DECL(-un, 0., 0.), p, qrei,
       cav, gamc, gamc,
       small, smallc, small_dens, small_pres, ustar,
       fluxw[URHO],  AMREX_D_DECL(momfluxn, momfluxt1, momfluxt2), fluxw[UEDEN],
       fluxw[UEINT], AMREX_D_DECL(dummy_gu, dummy_gv, dummy_gv2), dummy_gp, dummy_gg,
       smallu);

    AMREX_D_TERM(fluxw[UMX] =  (axm-axp) * momfluxn;,
                 fluxw[UMY] =  (aym-ayp) * momfluxn;,
                 fluxw[UMZ] =  (azm-azp) * momfluxn;);
}
#endif
#endif
