#ifndef FLATTEN_H
#define FLATTEN_H

#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>

#include "IndexDefines.H"

#ifdef AMREX_USE_EB
#include "AMReX_MultiFabUtil.H"
#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>
#include "eb_utils.H"
#endif

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
flatten (
  const int i,
  const int j,
  const int k,
  const int dir,
  amrex::Array4<const amrex::Real> const& q)
{
  amrex::ignore_unused(k);
  const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
  const amrex::IntVect dvec(amrex::IntVect::TheDimensionVector(dir));
  const amrex::IntVect ivm3(iv - 3 * dvec);
  const amrex::IntVect ivm2(iv - 2 * dvec);
  const amrex::IntVect ivm(iv - dvec);
  const amrex::IntVect ivp(iv + dvec);
  const amrex::IntVect ivp2(iv + 2 * dvec);
  const amrex::IntVect ivp3(iv + 3 * dvec);

#if (AMREX_SPACEDIM == 2)
  if ((!q.contains(ivp3[0], ivp3[1], k))  ||
      (!q.contains(ivm3[0], ivm3[1], k))) {
    return 1.0;
  }
#elif (AMREX_SPACEDIM == 3)
  if ( !q.contains(ivp3[0], ivp3[1], ivp3[2]) ||
       !q.contains(ivm3[0], ivm3[1], ivm3[2]) ) {
    return 1.0;
  }
#endif

  const int n = QPRES;
  // Parameters from uflatten
  const amrex::Real small_pres = 1.e-200;
  const amrex::Real shktst = 0.33e0;
  const amrex::Real zcut1 = 0.75e0;
  const amrex::Real zcut2 = 0.85e0;
  const amrex::Real dzcut = 1.0 / (zcut2 - zcut1);

  // amrex::Real dlft, drgt, dcen, dfm, dfp, dlim, dsgn, dtemp, drgtm1, dlftp1,
  // slop; const amrex::Real qc = q(i, j, k, n);
  const amrex::Real qm = q(ivm, n);
  const amrex::Real qp = q(ivp, n);
  // assumes QU,QV,QW ordering
  const amrex::Real qmun = q(ivm, QU + dir);
  const amrex::Real qpun = q(ivp, QU + dir);
  const amrex::Real qmm = q(ivm2, n);
  const amrex::Real qpp = q(ivp2, n);

  amrex::Real dp = qp - qm;

  const int ishft = (dp > 0.0) ? 1 : -1;

  amrex::Real denom = std::max(small_pres, std::abs(qpp - qmm));
  amrex::Real zeta = std::abs(dp) / denom;
  const amrex::Real z = std::min(1.0, std::max(0.0, dzcut * (zeta - zcut1)));

  amrex::Real tst = (qmun - qpun >= 0.0) ? 1.0 : 0.0;

  amrex::Real tmp = std::min(qp, qm);

  const amrex::Real chi = (std::abs(dp) / tmp) > shktst ? tst : 0.0;

  const amrex::Real qpshft = q(iv + (1 - ishft) * dvec, n);
  const amrex::Real qmshft = q(iv - (1 + ishft) * dvec, n);
  const amrex::Real qpunshft = q(iv + (1 - ishft) * dvec, QU + dir);
  const amrex::Real qmunshft = q(iv - (1 + ishft) * dvec, QU + dir);
  const amrex::Real qppshft = q(iv + (2 - ishft) * dvec, n);
  const amrex::Real qmmshft = q(iv - (2 + ishft) * dvec, n);

  dp = qpshft - qmshft;

  denom = std::max(small_pres, std::abs(qppshft - qmmshft));
  zeta  = std::abs(dp) / denom;
  const amrex::Real z2 = std::min(1.0, std::max(0.0, dzcut * (zeta - zcut1)));

  tst = (qmunshft - qpunshft >= 0.0) ? 1.0 : 0.0;

  tmp = std::min(qpshft, qmshft);

  const amrex::Real chi2 = (std::abs(dp) / tmp) > shktst ? tst : 0.0;

  return 1.0 - std::max(chi2 * z2, chi * z);
}

#ifdef AMREX_USE_EB
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real flatten_eb(const int i,
                       const int j,
                       const int k,
                       const int dir,
                       amrex::Array4<amrex::EBCellFlag const> const& flags,
                       amrex::Array4<const amrex::Real> const &q) {

  const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
  const amrex::IntVect dvec(amrex::IntVect::TheDimensionVector(dir));
  const amrex::IntVect ng(3 * dvec);
  if(is_cut_neighborhood(iv, flags, ng)){
    return 1.0;
  }
  return flatten(i, j, k, dir, q);
}
#endif
#endif
