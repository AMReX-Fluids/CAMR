#if (AMREX_SPACEDIM == 2)

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

// 2D version of transdd and transdo
// Corrects the left and right dir-states with the other-Flux
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_transd (
  const int i,
  const int j,
  const int k,
  const int dir,
  amrex::Array4<amrex::Real> const& qm,
  amrex::Array4<amrex::Real> const& qp,
  amrex::Array4<const amrex::Real> const& qym,
  amrex::Array4<const amrex::Real> const& qyp,
  amrex::Array4<const amrex::Real> const& flxx,
  amrex::Array4<const amrex::Real> const& srcQ,
  amrex::Array4<const amrex::Real> const& qa,
  amrex::Array4<const amrex::Real> const& qint,
  const amrex::Real hdt,
  const amrex::Real cdtdx,
  PassMap const& pmap,
  int transverse_reset_density,
  amrex::Real small_pres
#ifdef AMREX_USE_EB
  , amrex::Array4<const amrex::Real> const& area = {}
#endif
)
{
  amrex::ignore_unused(k);
  const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
  const amrex::IntVect ivp0(
    iv + amrex::IntVect::TheDimensionVector(dir == 0 ? 1 : 0));
  const amrex::IntVect ivp1(
    iv + amrex::IntVect::TheDimensionVector(dir == 0 ? 0 : 1));
  const int qvidx = (dir == 0) ? GDV : GDU;

  const bool no_cut_face =
#ifdef AMREX_USE_EB
    (area(ivp0) > 0.0) && (area(iv) > 0.0);
#else
  true;
#endif
  const amrex::Real flxrho = no_cut_face ? cdtdx * (flxx(ivp0, URHO) - flxx(iv, URHO)) : 0.0;
  const amrex::Real flxu   = no_cut_face ? cdtdx * (flxx(ivp0, UMX) - flxx(iv, UMX)) : 0.0;
  const amrex::Real flxv   = no_cut_face ? cdtdx * (flxx(ivp0, UMY) - flxx(iv, UMY)) : 0.0;
  const amrex::Real flxe   = no_cut_face ? cdtdx * (flxx(ivp0, UEDEN) - flxx(iv, UEDEN)) : 0.0;
  const amrex::Real srcr   = srcQ(iv, QRHO);
  const amrex::Real srce   = srcQ(iv, QREINT);
  const amrex::Real srcp   = srcQ(iv, QPRES);
  const amrex::Real    c   =   qa(iv, QGAMC);

  amrex::Real dAup = 0.0, pav = 0.0, dAu = 0.0;
  if (no_cut_face) {
    const amrex::Real pggp = qint(ivp0, GDPRES);
    const amrex::Real pggm = qint(iv, GDPRES);
    const amrex::Real ugp = qint(ivp0, qvidx);
    const amrex::Real ugm = qint(iv, qvidx);

    dAup = pggp * ugp - pggm * ugm;
    pav = 0.5 * (pggp + pggm);
    dAu = ugp - ugm;
  }

  // ****************************************************************************
  // Update passive variables
  // ****************************************************************************
  for (int ipass = 0; ipass < NPASSIVE; ++ipass) {
    const int n = pmap.upassMap[ipass];
    const int nqp = pmap.qpassMap[ipass];
    const amrex::Real srcpass = srcQ(iv, nqp);
    const amrex::Real compn = no_cut_face ? cdtdx * (flxx(ivp0, n) - flxx(iv, n)) : 0.0;
    amrex::Real rr = qyp(iv, QRHO);
    amrex::Real rrnew = rr - flxrho;
    amrex::Real compo = rr * qyp(iv, nqp) - compn;
    qp(iv, nqp) = compo / rrnew + hdt * srcpass;

    rr = qym(ivp1, QRHO);
    rrnew = rr - flxrho;
    compo = rr * qym(ivp1, nqp) - compn;
    qm(ivp1, nqp) = compo / rrnew + hdt * srcpass;
  }

  // ****************************************************************************
  // QP
  // ****************************************************************************

  // Convert to conservative
  const amrex::Real rrr = qyp(iv, QRHO);
  amrex::Real rur = qyp(iv, QU);
  amrex::Real rvr = qyp(iv, QV);
  const amrex::Real ekinr = 0.5 * rrr * (rur * rur + rvr * rvr);
  rur *= rrr;
  rvr *= rrr;

  const amrex::Real rer = qyp(iv, QREINT) + ekinr;
  // Add transverse predictor
  amrex::Real rrnewr = rrr - flxrho;
  amrex::Real runewr = rur - flxu;
  amrex::Real rvnewr = rvr - flxv;
  amrex::Real renewr = rer - flxe;

  bool reset_state = false;
  if (transverse_reset_density == 1 && rrnewr < 0.) {
     rrnewr = rrr;
     runewr = rur;
     rvnewr = rvr;
     renewr = rer;
     reset_state = true;
  }

  // Convert back to primitive
  qp(iv, QRHO) = rrnewr + hdt * srcr;
  qp(iv, QU) = runewr / rrnewr + hdt * srcQ(iv, QU);
  qp(iv, QV) = rvnewr / rrnewr + hdt * srcQ(iv, QV);

  const amrex::Real rhoekinr =
    0.5 * (runewr * runewr + rvnewr * rvnewr) / rrnewr;
  qp(iv, QREINT) = renewr - rhoekinr + hdt * srce;

  if(!reset_state){
    amrex::Real pnewr = qyp(iv, QPRES) - cdtdx * (dAup + pav * dAu * (c - 1.));
    pnewr += hdt * srcp;
    qp(iv, QPRES) = pnewr;
  } else {
    qp(iv, QPRES) = qyp(iv,QPRES);
  }

  qp(iv,QPRES) = std::max( qp(iv,QPRES), small_pres);

  // ****************************************************************************
  // QM
  // ****************************************************************************

  // Conversion to Conservative
  const amrex::Real rrl = qym(ivp1, QRHO);
  amrex::Real rul = qym(ivp1, QU);
  amrex::Real rvl = qym(ivp1, QV);
  const amrex::Real ekinl = 0.5 * rrl * (rul * rul + rvl * rvl);
  rul *= rrl;
  rvl *= rrl;
  const amrex::Real rel = qym(ivp1, QREINT) + ekinl;

  // Transverse fluxes
  amrex::Real rrnewl = rrl - flxrho;
  amrex::Real runewl = rul - flxu;
  amrex::Real rvnewl = rvl - flxv;
  amrex::Real renewl = rel - flxe;

  reset_state=false;
  if (transverse_reset_density == 1 && rrnewl < 0.) {
     rrnewl = rrl;
     runewl = rul;
     rvnewl = rvl;
     renewl = rel;
     reset_state = true;
  }

  qm(ivp1, QRHO) = rrnewl + hdt * srcr;
  qm(ivp1, QU) = runewl / rrnewl + hdt * srcQ(iv, QU);
  qm(ivp1, QV) = rvnewl / rrnewl + hdt * srcQ(iv, QV);

  const amrex::Real rhoekinl =
    0.5 * (runewl * runewl + rvnewl * rvnewl) / rrnewl;
  qm(ivp1, QREINT) = renewl - rhoekinl + hdt * srce;

  if(!reset_state){
    amrex::Real pnewl = qym(ivp1, QPRES) - cdtdx * (dAup + pav * dAu * (c - 1.));
    qm(ivp1, QPRES) = pnewl + hdt * srcp;
  } else {
    qm(ivp1, QPRES) = qym(ivp1, QPRES)  + hdt * srcp;
  }

  qm(ivp1, QPRES) = std::max( qm(ivp1, QPRES), small_pres);
}
#endif
#endif
