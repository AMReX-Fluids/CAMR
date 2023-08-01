#if (AMREX_SPACEDIM == 3)

#ifndef GODUNOV_UTILS_H
#define GODUNOV_UTILS_H

#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>

#include "IndexDefines.H"

#ifdef AMREX_USE_EB
#include "AMReX_MultiFabUtil.H"
#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>
#include "eb_utils.H"
#endif

// Transverse Correction for Predicted dir-states, using other_dir-Flux
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_transdo (
  const int i,
  const int j,
  const int k,
  const int dir,
  const int other_dir,
  amrex::Array4<amrex::Real> const& qm,
  amrex::Array4<amrex::Real> const& qp,
  amrex::Array4<const amrex::Real> const& qym,
  amrex::Array4<const amrex::Real> const& qyp,
  amrex::Array4<const amrex::Real> const& flxx,
  amrex::Array4<const amrex::Real> const& qa,
  amrex::Array4<const amrex::Real> const& qint,
  const amrex::Real cdtdx,
  PassMap const& pmap,
  int transverse_reset_density,
  amrex::Real small_pres)
{
  amrex::ignore_unused(k);
  const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
  const amrex::IntVect ivp0(iv + amrex::IntVect::TheDimensionVector(other_dir));
  const amrex::IntVect ivp1(iv + amrex::IntVect::TheDimensionVector(dir));
  const int qvidx = (other_dir == 0) ? GDU : ((other_dir == 1) ? GDV : GDW);

  const amrex::Real flxrho = cdtdx * (flxx(ivp0, URHO) - flxx(iv, URHO));
  const amrex::Real flxu = cdtdx * (flxx(ivp0, UMX) - flxx(iv, UMX));
  const amrex::Real flxv = cdtdx * (flxx(ivp0, UMY) - flxx(iv, UMY));
  const amrex::Real flxw = cdtdx * (flxx(ivp0, UMZ) - flxx(iv, UMZ));
  const amrex::Real flxe = cdtdx * (flxx(ivp0, UEDEN) - flxx(iv, UEDEN));
  const amrex::Real c = qa(iv, QGAMC);

  // Update passive variables
  for (int ipass = 0; ipass < NPASSIVE; ++ipass) {
    const int n = pmap.upassMap[ipass];
    const int nqp = pmap.qpassMap[ipass];
    const amrex::Real compn = cdtdx * (flxx(ivp0, n) - flxx(iv, n));
    amrex::Real rr = qyp(iv, QRHO);
    amrex::Real rrnew = rr - flxrho;
    amrex::Real compo = rr * qyp(iv, nqp) - compn;
    qp(iv, nqp) = compo / rrnew;

    rr = qym(ivp1, QRHO);
    rrnew = rr - flxrho;
    compo = rr * qym(ivp1, nqp) - compn;
    qm(ivp1, nqp) = compo / rrnew;
  }
  // Update hydro vars

  const amrex::Real pggp = qint(ivp0, GDPRES);
  const amrex::Real pggm = qint(iv, GDPRES);
  const amrex::Real ugp = qint(ivp0, qvidx);
  const amrex::Real ugm = qint(iv, qvidx);

  const amrex::Real dAup = pggp * ugp - pggm * ugm;
  const amrex::Real pav = 0.5 * (pggp + pggm);
  const amrex::Real dAu = ugp - ugm;

  // QP

  // Convert to conservative
  const amrex::Real rrr = qyp(iv, QRHO);
  amrex::Real rur = qyp(iv, QU);
  amrex::Real rvr = qyp(iv, QV);
  amrex::Real rwr = qyp(iv, QW);
  const amrex::Real ekinr = 0.5 * rrr * (rur * rur + rvr * rvr + rwr * rwr);
  rur *= rrr;
  rvr *= rrr;
  rwr *= rrr;

  const amrex::Real rer = qyp(iv, QREINT) + ekinr;
  // Add transverse predictor
  amrex::Real rrnewr = rrr - flxrho;
  amrex::Real runewr = rur - flxu;
  amrex::Real rvnewr = rvr - flxv;
  amrex::Real rwnewr = rwr - flxw;
  amrex::Real renewr = rer - flxe;

  bool reset_state = false;

  if (transverse_reset_density == 1 && rrnewr < 0.) {
     rrnewr = rrr;
     runewr = rur;
     rvnewr = rvr;
     rwnewr = rwr;
     renewr = rer;
     reset_state = true;
  }

  // Convert back to primitive
  qp(iv, QRHO) = rrnewr;
  qp(iv, QU) = runewr / rrnewr;
  qp(iv, QV) = rvnewr / rrnewr;
  qp(iv, QW) = rwnewr / rrnewr;

  const amrex::Real rhoekinr = 0.5 * (runewr * runewr + rvnewr * rvnewr + rwnewr * rwnewr) / rrnewr;

  qp(iv, QREINT) = renewr - rhoekinr;

  amrex::Real pnewr;
  if(!reset_state){
     pnewr = qyp(iv, QPRES) - cdtdx * (dAup + pav * dAu * (c - 1.0));
     qp(iv, QPRES) = std::max(pnewr, small_pres);
  } else {
     qp(iv, QPRES) = qyp(iv, QPRES);
  }
  // QM

  // Conversion to Conservative
  const amrex::Real rrl = qym(ivp1, QRHO);
  amrex::Real rul = qym(ivp1, QU);
  amrex::Real rvl = qym(ivp1, QV);
  amrex::Real rwl = qym(ivp1, QW);
  const amrex::Real ekinl = 0.5 * rrl * (rul * rul + rvl * rvl + rwl * rwl);
  rul *= rrl;
  rvl *= rrl;
  rwl *= rrl;
  const amrex::Real rel = qym(ivp1, QREINT) + ekinl;

  // Transverse fluxes
  amrex::Real rrnewl = rrl - flxrho;
  amrex::Real runewl = rul - flxu;
  amrex::Real rvnewl = rvl - flxv;
  amrex::Real rwnewl = rwl - flxw;
  amrex::Real renewl = rel - flxe;

  reset_state = false;
  if (transverse_reset_density == 1 && rrnewl < 0.) {
     rrnewl = rrl;
     runewl = rul;
     rvnewl = rvl;
     rwnewl = rwl;
     renewl = rel;
     reset_state = true;
  }

  qm(ivp1, QRHO) = rrnewl;
  qm(ivp1, QU) = runewl / rrnewl;
  qm(ivp1, QV) = rvnewl / rrnewl;
  qm(ivp1, QW) = rwnewl / rrnewl;
  const amrex::Real rhoekinl =
    0.5 * (runewl * runewl + rvnewl * rvnewl + rwnewl * rwnewl) / rrnewl;

  qm(ivp1, QREINT) = renewl - rhoekinl;

  amrex::Real pnewl;
  if(!reset_state){
    pnewl = qym(ivp1, QPRES) - cdtdx * (dAup + pav * dAu * (c - 1.0));
    qm(ivp1, QPRES) = std::max(pnewl, small_pres);
  } else {
    qm(ivp1, QPRES) = qym(ivp1, QPRES);
  }
}

// dir corrected from other two dirs
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_transdd (
  const int i,
  const int j,
  const int k,
  const int dir,
  amrex::Array4<amrex::Real> const& qm,
  amrex::Array4<amrex::Real> const& qp,
  amrex::Array4<const amrex::Real> const& qzm,
  amrex::Array4<const amrex::Real> const& qzp,
  amrex::Array4<const amrex::Real> const& flxx,
  amrex::Array4<const amrex::Real> const& flxy,
  amrex::Array4<const amrex::Real> const& qx,
  amrex::Array4<const amrex::Real> const& qy,
  amrex::Array4<const amrex::Real> const& qa,
  amrex::Array4<const amrex::Real> const& srcq,
  const amrex::Real hdt,
  const amrex::Real cdtdx0,
  const amrex::Real cdtdx1,
  PassMap const& pmap,
  int transverse_reset_density,
  amrex::Real small_pres)
{
  amrex::ignore_unused(k);
  const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
  const int qvidx0 = (dir == 0) ? GDV : GDU;
  const int qvidx1 = (dir == 2) ? GDV : GDW;
  const amrex::GpuArray<const int, 3> bdim{{dir == 0, dir == 1, dir == 2}};
  const amrex::GpuArray<const int, 3> l_idx{
    {bdim[0] * 0 + bdim[1] * 1 + bdim[2] * 2,
     bdim[0] * 1 + bdim[1] * 0 + bdim[2] * 0,
     bdim[0] * 2 + bdim[1] * 2 + bdim[2] * 1}};
  const amrex::IntVect ivp0(iv + amrex::IntVect::TheDimensionVector(l_idx[1]));
  const amrex::IntVect ivp1(iv + amrex::IntVect::TheDimensionVector(l_idx[2]));
  const amrex::IntVect ivp2(iv + amrex::IntVect::TheDimensionVector(dir));

  const amrex::Real flxrho = cdtdx0 * (flxx(ivp0, URHO) - flxx(iv, URHO)) +
                             cdtdx1 * (flxy(ivp1, URHO) - flxy(iv, URHO));
  const amrex::Real flxu = cdtdx0 * (flxx(ivp0, UMX) - flxx(iv, UMX)) +
                           cdtdx1 * (flxy(ivp1, UMX) - flxy(iv, UMX));
  const amrex::Real flxv = cdtdx0 * (flxx(ivp0, UMY) - flxx(iv, UMY)) +
                           cdtdx1 * (flxy(ivp1, UMY) - flxy(iv, UMY));
  const amrex::Real flxw = cdtdx0 * (flxx(ivp0, UMZ) - flxx(iv, UMZ)) +
                           cdtdx1 * (flxy(ivp1, UMZ) - flxy(iv, UMZ));
  const amrex::Real flxe = cdtdx0 * (flxx(ivp0, UEDEN) - flxx(iv, UEDEN)) +
                           cdtdx1 * (flxy(ivp1, UEDEN) - flxy(iv, UEDEN));
  const amrex::Real c = qa(iv, QGAMC);
  const amrex::Real srcrho = srcq(iv, QRHO);
  const amrex::Real srcu = srcq(iv, QU);
  const amrex::Real srcv = srcq(iv, QV);
  const amrex::Real srcw = srcq(iv, QW);
  const amrex::Real srce = srcq(iv, QREINT);
  const amrex::Real srcp = srcq(iv, QPRES);

  // Update passive variables
  const amrex::Real rrr = qzp(iv, QRHO);
  const amrex::Real rrl = qzm(ivp2, QRHO);
  amrex::Real rrnewl = rrl - flxrho;
  amrex::Real rrnewr = rrr - flxrho;

  for (int ipass = 0; ipass < NPASSIVE; ++ipass) {
    const int n = pmap.upassMap[ipass];
    const int nqp = pmap.qpassMap[ipass];
    const amrex::Real srcpass = srcq(iv, nqp);
    const amrex::Real compn = cdtdx0 * (flxx(ivp0, n) - flxx(iv, n)) +
                              cdtdx1 * (flxy(ivp1, n) - flxy(iv, n));
    // qp
    amrex::Real compo = rrr * qzp(iv, nqp) - compn;
    qp(iv, nqp) = compo / rrnewr + hdt * srcpass;

    // qm
    compo = rrl * qzm(ivp2, nqp) - compn;
    qm(ivp2, nqp) = compo / rrnewl + hdt * srcpass;
  }

  // Update hydro vars
  const amrex::Real pggpx = qx(ivp0, GDPRES);
  const amrex::Real pggmx = qx(iv, GDPRES);
  const amrex::Real ugpx = qx(ivp0, qvidx0);
  const amrex::Real ugmx = qx(iv, qvidx0);

  const amrex::Real dAupx = pggpx * ugpx - pggmx * ugmx;
  const amrex::Real pavx = 0.5 * (pggpx + pggmx);
  const amrex::Real dAux = ugpx - ugmx;

  const amrex::Real pggpy = qy(ivp1, GDPRES);
  const amrex::Real pggmy = qy(iv, GDPRES);
  const amrex::Real ugpy = qy(ivp1, qvidx1);
  const amrex::Real ugmy = qy(iv, qvidx1);

  const amrex::Real dAupy = pggpy * ugpy - pggmy * ugmy;
  const amrex::Real pavy = 0.5 * (pggpy + pggmy);
  const amrex::Real dAuy = ugpy - ugmy;
  const amrex::Real pxnew = cdtdx0 * (dAupx + pavx * dAux * (c - 1.0));
  const amrex::Real pynew = cdtdx1 * (dAupy + pavy * dAuy * (c - 1.0));

  // qp state
  const amrex::Real rur = rrr * qzp(iv, QU);
  const amrex::Real rvr = rrr * qzp(iv, QV);
  const amrex::Real rwr = rrr * qzp(iv, QW);
  const amrex::Real ekinr = 0.5 * (rur * rur + rvr * rvr + rwr * rwr) / rrr;
  const amrex::Real rer = qzp(iv, QREINT) + ekinr;

  amrex::Real runewr = rur - flxu;
  amrex::Real rvnewr = rvr - flxv;
  amrex::Real rwnewr = rwr - flxw;
  amrex::Real renewr = rer - flxe;

  bool reset_state = false;

  if (transverse_reset_density == 1 && rrnewr < 0.) {
     rrnewr = rrr;
     runewr = rur;
     rvnewr = rvr;
     rwnewr = rwr;
     renewr = rer;
     reset_state = true;
  }

  qp(iv, QRHO) = rrnewr + hdt * srcrho;
  qp(iv, QU) = runewr / rrnewr + hdt * srcu;
  qp(iv, QV) = rvnewr / rrnewr + hdt * srcv;
  qp(iv, QW) = rwnewr / rrnewr + hdt * srcw;

  const amrex::Real rhoekinr =
    0.5 * (runewr * runewr + rvnewr * rvnewr + rwnewr * rwnewr) / rrnewr;
  qp(iv, QREINT) = renewr - rhoekinr + hdt * srce;

  if (!reset_state){
   qp(iv, QPRES) =  qzp(iv, QPRES) - pxnew - pynew + hdt * srcp;
  } else {
    qp(iv, QPRES) = qzp(iv, QPRES) + hdt * srcp;
  }
   qp(iv, QPRES) = std::max( qp(iv,QPRES), small_pres);

  // qm state
  const amrex::Real rul = rrl * qzm(ivp2, QU);
  const amrex::Real rvl = rrl * qzm(ivp2, QV);
  const amrex::Real rwl = rrl * qzm(ivp2, QW);
  const amrex::Real ekinl = 0.5 * (rul * rul + rvl * rvl + rwl * rwl) / rrl;
  const amrex::Real rel = qzm(ivp2, QREINT) + ekinl;

  amrex::Real runewl = rul - flxu;
  amrex::Real rvnewl = rvl - flxv;
  amrex::Real rwnewl = rwl - flxw;
  amrex::Real renewl = rel - flxe;

  reset_state = false;
  if (transverse_reset_density == 1 && rrnewl < 0.) {
     rrnewl = rrl;
     runewl = rul;
     rvnewl = rvl;
     rwnewl = rwl;
     renewl = rel;
     reset_state = true;
  }

  qm(ivp2, QRHO) = rrnewl + hdt * srcrho;
  qm(ivp2, QU) = runewl / rrnewl + hdt * srcu;
  qm(ivp2, QV) = rvnewl / rrnewl + hdt * srcv;
  qm(ivp2, QW) = rwnewl / rrnewl + hdt * srcw;

  const amrex::Real rhoekinl =
    0.5 * (runewl * runewl + rvnewl * rvnewl + rwnewl * rwnewl) / rrnewl;
  qm(ivp2, QREINT) = renewl - rhoekinl + hdt * srce;

  if(!reset_state){
    qm(ivp2, QPRES) = qzm(ivp2, QPRES) - pxnew - pynew + hdt * srcp;
  } else {
    qm(ivp2, QPRES) = qzm(ivp2, QPRES) + hdt * srcp;
  }
  qm(ivp2, QPRES) = std::max(qm(ivp2,QPRES), small_pres);
}
#endif
#endif
