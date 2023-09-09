#ifndef HYDRO_CTOPRIM_H
#define HYDRO_CTOPRIM_H

#include <AMReX_FArrayBox.H>
#include "IndexDefines.H"
#include "EOS.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
hydro_ctoprim(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& u,
  amrex::Array4<      amrex::Real> const& q,
  amrex::Array4<      amrex::Real> const& qa,
  PassMap const& pmap,
  amrex::Real l_small_num,
  amrex::Real l_dual_energy_eta1,
  amrex::Real l_allow_negative_energy)
{
    const amrex::Real rho = u(i, j, k, URHO);

    if (rho > 0.) {
        const amrex::Real rhoinv = 1.0 / rho;
        AMREX_D_TERM(const amrex::Real vx = u(i, j, k, UMX) * rhoinv;,
                     const amrex::Real vy = u(i, j, k, UMY) * rhoinv;,
                     const amrex::Real vz = u(i, j, k, UMZ) * rhoinv;);
#if (AMREX_SPACEDIM == 2)
        const amrex::Real kineng = 0.5 * rho * (vx * vx + vy * vy);
#else
        const amrex::Real kineng = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
#endif

        q(i, j, k, QRHO) = rho;
        AMREX_D_TERM(q(i, j, k, QU) = vx;,
                     q(i, j, k, QV) = vy;,
                     q(i, j, k, QW) = vz;);

        for (int ipassive = 0; ipassive < NPASSIVE; ++ipassive) {
           const int n = pmap.upassMap[ipassive];
           const int nq = pmap.qpassMap[ipassive];
           q(i, j, k, nq) = u(i, j, k, n) * rhoinv;
         }

         // Get the internal energy, which we'll use for
         // determining the pressure.  We use a dual energy
         // formalism. If (E - K) < eta1 and eta1 is suitably
         // small, then we risk serious numerical truncation error
         // in the internal energy.  Therefore we'll use the result
         // of the separately updated internal energy equation.
         // Otherwise, we'll set e = E - K.

        if (!l_allow_negative_energy) {
            AMREX_ALWAYS_ASSERT(u(i,j,k,UEDEN) > 0.);
        }

        if ( (u(i,j,k,UEDEN) - kineng) / u(i,j,k,UEDEN) > l_dual_energy_eta1) {
           q(i,j,k,QREINT) = (u(i,j,k,UEDEN) - kineng) * rhoinv;
        } else {
           q(i,j,k,QREINT) = u(i,j,k,UEINT) * rhoinv;
        }

        const amrex::Real e = q(i,j,k,QREINT);

        if (!l_allow_negative_energy) {
            AMREX_ALWAYS_ASSERT(u(i,j,k,UEINT) > 0.);
        }

        amrex::Real T = u(i, j, k, UTEMP);
        amrex::Real massfrac[NUM_SPECIES];
        for (int sp = 0; sp < NUM_SPECIES; ++sp) {
          q(i, j, k, sp + QFS) = u(i,j,k,UFS+sp) * rhoinv;
          massfrac[sp] = q(i, j, k, sp + QFS);
        }

        amrex::Real dpdr_e, dpde, gam1, cs,  p;

        // Are all these EOS calls needed? Seems fairly convoluted.
        EOS::REY2T  (rho, e, massfrac, T);
        EOS::REY2P  (rho, e, massfrac, p);
        EOS::REY2Gam(rho, e, massfrac, gam1);

        EOS::REY2dpde  (rho, e, massfrac, dpde);
        EOS::REY2dpdr_e(rho, e, massfrac, dpdr_e);

        q(i, j, k, QTEMP)  = T;
        q(i, j, k, QREINT) = e * rho;
        q(i, j, k, QPRES)  = p;
        q(i, j, k, QGAME)  = p / (e * rho) + 1.0;

        // Auxilary Fab
        qa(i, j, k, QDPDR) = dpdr_e;
        qa(i, j, k, QDPDE) = dpde;
        qa(i, j, k, QGAMC) = gam1;
        cs =  std::sqrt(gam1*p/rho);
        qa(i, j, k, QC) = cs;
        qa(i, j, k, QCSML) = std::max(l_small_num, l_small_num * cs);
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
hydro_srctoprim(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& q,
  amrex::Array4<const amrex::Real> const& qa,
  amrex::Array4<const amrex::Real> const& src,
  amrex::Array4<amrex::Real> const& srcq,
  PassMap const& pmap)
{
  // Local loading Vars ==> Load once from global memory use multiple times
  const amrex::Real srcrho = src(i, j, k, URHO);

  if (q(i,j,k,QRHO) > 0.) {

      const amrex::Real rhoinv = 1.0 / q(i, j, k, QRHO);
      AMREX_D_TERM(const amrex::Real vx = q(i, j, k, QU);,
                   const amrex::Real vy = q(i, j, k, QV);,
                   const amrex::Real vz = q(i, j, k, QW););

      AMREX_D_TERM(const amrex::Real smx = src(i, j, k, UMX);,
                   const amrex::Real smy = src(i, j, k, UMY);,
                   const amrex::Real smz = src(i, j, k, UMZ););

      // SrcQ Calc
      srcq(i, j, k, QRHO) = srcrho;
      AMREX_D_TERM(srcq(i, j, k, QU) = (smx - vx * srcrho) * rhoinv;,
                   srcq(i, j, k, QV) = (smy - vy * srcrho) * rhoinv;,
                   srcq(i, j, k, QW) = (smz - vz * srcrho) * rhoinv;);

      srcq(i, j, k, QREINT) = src(i, j, k, UEINT);

      for (int ip = 0; ip < NPASSIVE; ++ip) {
        const int n = pmap.upassMap[ip];
        const int nq = pmap.qpassMap[ip];
        srcq(i, j, k, nq) = (src(i, j, k, n) - q(i, j, k, nq) * srcrho) * rhoinv;
      }

      srcq(i, j, k, QPRES) =
        qa(i, j, k, QDPDE) *
          (srcq(i, j, k, QREINT) - q(i, j, k, QREINT) * srcrho * rhoinv) * rhoinv +
        qa(i, j, k, QDPDR) * srcrho;

    } else {
      srcq(i, j, k, QRHO  ) = amrex::Real(0.);
      AMREX_D_TERM(srcq(i, j, k, QU    ) = amrex::Real(0.);,
                   srcq(i, j, k, QV    ) = amrex::Real(0.);,
                   srcq(i, j, k, QW    ) = amrex::Real(0.););
      srcq(i, j, k, QPRES) = amrex::Real(0.);
      srcq(i, j, k, QREINT) = amrex::Real(0.);
    }
}
#endif
