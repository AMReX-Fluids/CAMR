#ifndef PLM_H
#define PLM_H

#include <cmath>

#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_Box.H>

#include "IndexDefines.H"

// This header file contains functions and declarations for the 3D hydro portion
// on the GPU. As per the convention of AMReX, inlined functions are
// defined here, whereas non-inline functions are declared here.

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
plm_slope (
  const int i,
  const int j,
  const int k,
  const int n,
  const int dir,
  amrex::Array4<const amrex::Real> const& q,
  amrex::Real flat,
  const int order)
{
  if (order == 1) {
      return 0.;

  } else {
      amrex::ignore_unused(k);
      const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
      const amrex::IntVect dvec(amrex::IntVect::TheDimensionVector(dir));
      const amrex::IntVect ivm2(iv - 2 * dvec);
      const amrex::IntVect ivm(iv - dvec);
      const amrex::IntVect ivp(iv + dvec);
      const amrex::IntVect ivp2(iv + 2 * dvec);

      amrex::Real dlft, drgt, dcen, dfm = 0.0, dfp = 0.0, dlim, dsgn, dtemp;
      const amrex::Real qc = q(iv, n);
      const amrex::Real qm2 = q(ivm2, n);
      const amrex::Real qm = q(ivm, n);
      const amrex::Real qp = q(ivp, n);
      const amrex::Real qp2 = q(ivp2, n);

      if (order == 4) {
        dlft = qm - qm2;
        drgt = qc - qm;
        dcen = 0.5 * (dlft + drgt);
        dsgn = amrex::Math::copysign(1.0, dcen);
        dlim = (dlft * drgt >= 0.0)
                   ? 2.0 * std::min(std::abs(dlft), std::abs(drgt))
                   : 0.0;
        dfm = dsgn * std::min(dlim, std::abs(dcen));

        dlft = qp - qc;
        drgt = (order == 4) ? qp2 - qp : 0.0;
        dcen = 0.5 * (dlft + drgt);
        dsgn = amrex::Math::copysign(1.0, dcen);
        dlim = (dlft * drgt >= 0.0)
                   ? 2.0 * std::min(std::abs(dlft), std::abs(drgt))
                   : 0.0;
        dfp = dsgn * std::min(dlim, std::abs(dcen));
      }

      dlft = qc - qm;
      drgt = qp - qc;
      dcen = 0.5 * (dlft + drgt);
      dsgn = amrex::Math::copysign(1.0, dcen);
      dlim = (dlft * drgt >= 0.0)
               ? 2.0 * std::min(std::abs(dlft), std::abs(drgt)) : 0.0;

      dtemp = 4.0 / 3.0 * dcen - 1.0 / 6.0 * (dfp + dfm);

      return flat * dsgn * std::min(dlim, std::abs(dtemp));
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
plm_pslope (
  const int i,
  const int j,
  const int k,
  const int n,
  const int dir,
  amrex::Array4<const amrex::Real> const& q,
  const amrex::Real dx,
  amrex::Array4<const amrex::Real> const& srcQ,
  amrex::Real flat,
  const int order)
{
  if (order == 1) {
      return 0.;

  } else {

      amrex::ignore_unused(k);
      const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
      const amrex::IntVect dvec(amrex::IntVect::TheDimensionVector(dir));
      const amrex::IntVect ivm2(iv - 2 * dvec);
      const amrex::IntVect ivm(iv - dvec);
      const amrex::IntVect ivp(iv + dvec);
      const amrex::IntVect ivp2(iv + 2 * dvec);

      amrex::Real dlft, drgt, dcen, dfm = 0.0, dfp = 0.0, dlim, dsgn, dtemp;
      const amrex::Real qc  = q(iv  , n);
      const amrex::Real qm2 = q(ivm2, n);
      const amrex::Real qm  = q(ivm , n);
      const amrex::Real qp  = q(ivp , n);
      const amrex::Real qp2 = q(ivp2, n);

      int QIU;
      if (dir == 0) QIU = QU;
      if (dir == 1) QIU = QV;
#if (AMREX_SPACEDIM == 3)
      if (dir == 2) QIU = QW;
#endif

      if (order == 4) {
        dlft = qm - qm2 -
               0.25 * dx * (q(ivm, QRHO) + q(ivm2, QRHO)) *
                   (srcQ(ivm, QIU) + srcQ(ivm2, QIU));
        drgt = qc - qm -
               0.25 * dx * (q(ivm, QRHO) + q(iv, QRHO)) *
                   (srcQ(ivm, QIU) + srcQ(iv, QIU));
        dcen = 0.5 * (dlft + drgt);
        dsgn = amrex::Math::copysign(1.0, dcen);
        dlim = (dlft * drgt >= 0.0)
                   ? 2.0 * std::min(std::abs(dlft), std::abs(drgt))
                   : 0.0;
        dfm = dsgn * std::min<amrex::Real>(dlim, std::abs(dcen));

        dlft = qp - qc -
               0.25 * dx * (q(ivp, QRHO) + q(iv, QRHO)) *
                   (srcQ(ivp, QIU) + srcQ(iv, QIU));
        drgt = qp2 - qp -
               0.25 * dx * (q(ivp, QRHO) + q(ivp2, QRHO)) *
                   (srcQ(ivp, QIU) + srcQ(ivp2, QIU));
        dcen = 0.5 * (dlft + drgt);
        dsgn = amrex::Math::copysign(1.0, dcen);
        dlim = (dlft * drgt >= 0.0)
                   ? 2.0 * std::min(std::abs(dlft), std::abs(drgt))
                   : 0.0;
        dfp = dsgn * std::min(dlim, std::abs(dcen));
      }

      dlft = qc - qm - 0.25 * dx * (q(iv,QRHO)+q(ivm,QRHO)) * (srcQ(iv,QIU)+srcQ(ivm,QIU));
      drgt = qp - qc - 0.25 * dx * (q(iv,QRHO)+q(ivp,QRHO)) * (srcQ(iv,QIU)+srcQ(ivp,QIU));
      dcen = 0.5 * (dlft + drgt);
      dsgn = amrex::Math::copysign(1.0, dcen);
      dlim = (dlft * drgt >= 0.0)
               ? 2.0 * std::min(std::abs(dlft), std::abs(drgt)) : 0.0;

      dtemp = 4.0 / 3.0 * dcen - 1.0 / 6.0 * (dfp + dfm);

      amrex::Real dp = flat * dsgn * std::min<amrex::Real>(dlim, std::abs(dtemp));

      dp += q(iv,QRHO) * srcQ(iv,QIU) * dx;
      return dp;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
hydro_plm_d (
  const int i,
  const int j,
  const int k,
  const int dir,
  amrex::Array4<amrex::Real> const& qm,
  amrex::Array4<amrex::Real> const& qp,
  amrex::Real const slope[QVAR],
  amrex::Array4<const amrex::Real> const& q,
  amrex::Real const cc,
  amrex::Real const dx,
  amrex::Real const dt,
  amrex::Real const small_dens,
  amrex::Real const small_pres,
  PassMap const& pmap)
{
  amrex::ignore_unused(k);
  const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
  const amrex::IntVect ivp(iv + amrex::IntVect::TheDimensionVector(dir));
  const amrex::GpuArray<const int, 3> bdim{{dir == 0, dir == 1, dir == 2}};
  const amrex::GpuArray<const int, 3> l_idx{
    {bdim[0] * 0 + bdim[1] * 1 + bdim[2] * 2,
     bdim[0] * 1 + bdim[1] * 0 + bdim[2] * 0,
     bdim[0] * 2 + bdim[1] * 2 + bdim[2] * 1}};

  const amrex::Real dtdx = dt / dx;
  const amrex::Real cs = cc * cc;
  const amrex::Real rho = q(iv, QRHO);
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vel{
    AMREX_D_DECL(q(iv, QU), q(iv, QV), q(iv, QW))};
  const amrex::Real p = q(iv, QPRES);
  const amrex::Real rhoe = q(iv, QREINT);
  const amrex::Real enth = ((rhoe + p) / rho) / cs;
  const amrex::Real drho = slope[QRHO];
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
     dvel{AMREX_D_DECL(slope[QU], slope[QV], slope[QW])};
  const amrex::Real dp = slope[QPRES];
  const amrex::Real drhoe = slope[QREINT];
  const amrex::Real alpham = 0.5 * (dp / (rho * cc) - dvel[dir]) * rho / cc;
  const amrex::Real alphap = 0.5 * (dp / (rho * cc) + dvel[dir]) * rho / cc;
  const amrex::Real alpha0r = drho - dp / cs;
  const amrex::Real alpha0e = drhoe - dp * enth;
  AMREX_D_TERM(, const amrex::Real alpha0v = dvel[l_idx[1]];
               , const amrex::Real alpha0w = dvel[l_idx[2]];)
  const amrex::GpuArray<amrex::Real, 3> wv = {
    vel[dir] - cc, vel[dir], vel[dir] + cc};

  //  Construct the right state on the i-1/2 interface
  {
  const amrex::Real rho_ref =
    rho - 0.5 * (1.0 + dtdx * std::min(wv[0], 0.0)) * drho;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vel_ref{
    AMREX_D_DECL(0.0, 0.0, 0.0)};
  for (int dim = 0; dim < AMREX_SPACEDIM; dim++) {
    vel_ref[dim] =
      vel[dim] -
      0.5 * (1.0 + dtdx * std::min(wv[0], 0.0)) * dvel[dim];
  }
  const amrex::Real p_ref =
    p - 0.5 * (1.0 + dtdx * std::min(wv[0], 0.0)) * dp;
  const amrex::Real rhoe_ref =
    rhoe - 0.5 * (1.0 + dtdx * std::min(wv[0], 0.0)) * drhoe;

  const amrex::Real apright = 0.25 * dtdx * (wv[0] - wv[2]) *
                              (1.0 - amrex::Math::copysign(1.0, wv[2])) *
                              alphap;
  const amrex::Real amright = 0.0;

  const amrex::Real azrright = 0.25 * dtdx * (wv[0] - wv[1]) *
                               (1.0 - amrex::Math::copysign(1.0, wv[1])) *
                               alpha0r;
  const amrex::Real azeright = 0.25 * dtdx * (wv[0] - wv[1]) *
                               (1.0 - amrex::Math::copysign(1.0, wv[1])) *
                               alpha0e;
  AMREX_D_TERM(, const amrex::Real azv1rght =
                   0.25 * dtdx * (wv[0] - wv[1]) *
                   (1.0 - amrex::Math::copysign(1.0, wv[1])) * alpha0v;
               , const amrex::Real azw1rght =
                   0.25 * dtdx * (wv[0] - wv[1]) *
                   (1.0 - amrex::Math::copysign(1.0, wv[1])) * alpha0w;)

  qp(iv, QRHO) = rho_ref + apright + amright + azrright;
  qp(iv, QRHO) = std::max( qp(iv, QRHO), small_dens);
  AMREX_D_TERM(qp(iv, QU + l_idx[0]) =
                 vel_ref[l_idx[0]] + (apright - amright) * cc / rho;
               qp(iv, QU + l_idx[1]) = 0.; qp(iv, QU + l_idx[2]) = 0.;
               , qp(iv, QU + l_idx[1]) = vel_ref[l_idx[1]] + azv1rght;
               , qp(iv, QU + l_idx[2]) = vel_ref[l_idx[2]] + azw1rght;);
  qp(iv, QPRES) = p_ref + (apright + amright) * cs;
  qp(iv, QPRES) = std::max( qp(iv, QPRES), small_pres);
  qp(iv, QREINT) = rhoe_ref + (apright + amright) * enth * cs + azeright;
  }

  //  Construct the left state on the i+1/2 interface
  {
  const amrex::Real rho_ref =
    rho + 0.5 * (1.0 - dtdx * std::max(wv[2], 0.0)) * drho;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vel_ref{
    AMREX_D_DECL(0.0, 0.0, 0.0)};
  for (int dim = 0; dim < AMREX_SPACEDIM; dim++) {
    vel_ref[dim] =
      vel[dim] +
      0.5 * (1.0 - dtdx * std::max(wv[2], 0.0)) * dvel[dim];
  }
  const amrex::Real p_ref = p + 0.5 * (1.0 - dtdx * std::max(wv[2], 0.0)) * dp;

  const amrex::Real rhoe_ref =
    rhoe + 0.5 * (1.0 - dtdx * std::max(wv[2], 0.0)) * drhoe;

  const amrex::Real apleft = 0.0;
  const amrex::Real amleft = 0.25 * dtdx * (wv[2] - wv[0]) *
                             (1.0 + amrex::Math::copysign(1.0, wv[0])) * alpham;

  const amrex::Real azrleft = 0.25 * dtdx * (wv[2] - wv[1]) *
                              (1.0 + amrex::Math::copysign(1.0, wv[1])) *
                              alpha0r;
  const amrex::Real azeleft = 0.25 * dtdx * (wv[2] - wv[1]) *
                              (1.0 + amrex::Math::copysign(1.0, wv[1])) *
                              alpha0e;
  AMREX_D_TERM(, const amrex::Real azv1left =
                   0.25 * dtdx * (wv[2] - wv[1]) *
                   (1.0 + amrex::Math::copysign(1.0, wv[1])) * alpha0v;
               , const amrex::Real azw1left =
                   0.25 * dtdx * (wv[2] - wv[1]) *
                   (1.0 + amrex::Math::copysign(1.0, wv[1])) * alpha0w;)
  qm(ivp, QRHO) = rho_ref + apleft + amleft + azrleft;
  qm(ivp, QRHO) = std::max( qm(ivp, QRHO), small_dens);
  AMREX_D_TERM(qm(ivp, QU + l_idx[0]) =
                 vel_ref[l_idx[0]] + (apleft - amleft) * cc / rho;
               qm(ivp, QU + l_idx[1]) = 0.; qm(ivp, QU + l_idx[2]) = 0.;
               , qm(ivp, QU + l_idx[1]) = vel_ref[l_idx[1]] + azv1left;
               , qm(ivp, QU + l_idx[2]) = vel_ref[l_idx[2]] + azw1left;);
  qm(ivp, QPRES) = p_ref + (apleft + amleft) * cs;
  qm(ivp, QPRES) = std::max( qm(ivp, QPRES), small_pres);
  qm(ivp, QREINT) = rhoe_ref + (apleft + amleft) * enth * cs + azeleft;
  }

  // Upwind the passive variables
  for (int ipassive = 0; ipassive < NPASSIVE; ++ipassive) {
    const int n = pmap.qpassMap[ipassive];
    const amrex::Real vel_ad = q(iv, QU + l_idx[0]);
    // Right state
    {
      const amrex::Real spzero = (vel_ad > 0) ? -1.0 : vel_ad * dtdx;
      const amrex::Real acmprght = 0.5 * (-1.0 - spzero) * slope[n];
      qp(iv, n) = q(iv, n) + acmprght;
    }
    {
      const amrex::Real spzero = vel_ad >= 0 ? vel_ad * dtdx : 1.0;
      const amrex::Real acmpleft = 0.5 * (1.0 - spzero) * slope[n];
      qm(ivp, n) = q(iv, n) + acmpleft;
    }
  }
}
#endif
