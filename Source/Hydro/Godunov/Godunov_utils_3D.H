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
hydro_transdo (
  const int i,
  const int j,
  const int k,
  const int dir,
  const int other_dir,
  amrex::Array4<amrex::Real> const& qm,
  amrex::Array4<amrex::Real> const& qp,
  amrex::Array4<const amrex::Real> const& qnormm,
  amrex::Array4<const amrex::Real> const& qnormp,
  amrex::Array4<const amrex::Real> const& flxx,
  amrex::Array4<const amrex::Real> const& qa,
  amrex::Array4<const amrex::Real> const& qint,
  const amrex::Real cdtdx,
  PassMap const& pmap,
  int transverse_reset_density,
  amrex::Real small_pres
#ifdef AMREX_USE_EB
  , amrex::Array4<const amrex::Real> const&  norm_area = {}
  , amrex::Array4<const amrex::Real> const& trans_area = {}
#endif
)
{
  amrex::ignore_unused(k);
  const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
  const amrex::IntVect ivpn(iv + amrex::IntVect::TheDimensionVector(dir));
  const amrex::IntVect ivpt(iv + amrex::IntVect::TheDimensionVector(other_dir));
  const int qvidx = (other_dir == 0) ? GDU : ((other_dir == 1) ? GDV : GDW);

  // ivpn is an offset by 1 in the     normal direction
  // ivpt is an offset by 1 in the transverse direction
  // If (dir == 0 and other_dir == 1),  ivpt = (i,j+1,k) and ivpn = (i+1,j,k) and qvidx = GDV
  // If (dir == 0 and other_dir == 2),  ivpt = (i,j,k+1) and ivpn = (i+1,j,k) and qvidx = GDW
  // If (dir == 1 and other_dir == 0),  ivpt = (i+1,j,k) and ivpn = (i,j+1,k) and qvidx = GDU
  // If (dir == 1 and other_dir == 2),  ivpt = (i,j,k+1) and ivpn = (i,j+1,k) and qvidx = GDW
  // If (dir == 2 and other_dir == 0),  ivpt = (i+1,j,k) and ivpn = (i,j,k+1) and qvidx = GDU
  // If (dir == 2 and other_dir == 1),  ivpt = (i,j+1,k) and ivpn = (i,j,k+1) and qvidx = GDV

  bool no_cov_face = true;
  bool lo_face_not_covered = true;
  bool hi_face_not_covered = true;

#ifdef AMREX_USE_EB
   if (trans_area) {
       no_cov_face = (trans_area(ivpt) > 0.0) && (trans_area(iv) > 0.0);
   }
   if (norm_area) {
       lo_face_not_covered = (norm_area(iv  ) > amrex::Real(0.0));
       hi_face_not_covered = (norm_area(ivpn) > amrex::Real(0.0));
   }
#endif

  const amrex::Real flxrho = no_cov_face ? cdtdx * (flxx(ivpt, URHO ) - flxx(iv, URHO )) : amrex::Real(0.0);
  const amrex::Real flxu   = no_cov_face ? cdtdx * (flxx(ivpt, UMX  ) - flxx(iv, UMX  )) : amrex::Real(0.0);
  const amrex::Real flxv   = no_cov_face ? cdtdx * (flxx(ivpt, UMY  ) - flxx(iv, UMY  )) : amrex::Real(0.0);
  const amrex::Real flxw   = no_cov_face ? cdtdx * (flxx(ivpt, UMZ  ) - flxx(iv, UMZ  )) : amrex::Real(0.0);
  const amrex::Real flxe   = no_cov_face ? cdtdx * (flxx(ivpt, UEDEN) - flxx(iv, UEDEN)) : amrex::Real(0.0);
  const amrex::Real      c = qa(iv, QGAMC);

  // Update passive variables
  for (int ipass = 0; ipass < NPASSIVE; ++ipass) {
      const int n = pmap.upassMap[ipass];
      const int nqp = pmap.qpassMap[ipass];
      const amrex::Real compn = no_cov_face ? cdtdx * (flxx(ivpt, n) - flxx(iv, n)) : amrex::Real(0.0);

      amrex::Real rr, rrnew, compo;

      if (lo_face_not_covered)
      {
          rr = qnormp(iv, QRHO);
          rrnew = rr - flxrho;
          compo = rr * qnormp(iv, nqp) - compn;
          qp(iv, nqp) = compo / rrnew;
      } else {
          qp(iv, nqp) = qnormp(iv, nqp);
      }

      if (hi_face_not_covered)
      {
          rr = qnormm(ivpn, QRHO);
          rrnew = rr - flxrho;
          compo = rr * qnormm(ivpn, nqp) - compn;
          qm(ivpn, nqp) = compo / rrnew;
      } else {
          qm(ivpn, nqp) = qnormm(ivpn, nqp);
      }
  }

  // Update hydro vars

  amrex::Real dAup = 0.0, pav = 0.0, dAu = 0.0;
  if (no_cov_face) {
      const amrex::Real pggp = qint(ivpt, GDPRES);
      const amrex::Real pggm = qint(iv, GDPRES);
      const amrex::Real ugp = qint(ivpt, qvidx);
      const amrex::Real ugm = qint(iv, qvidx);

      dAup = pggp * ugp - pggm * ugm;
      pav = 0.5 * (pggp + pggm);
      dAu = ugp - ugm;
  }

  // QP

  if (lo_face_not_covered)
  {
      // Convert to conservative
      const amrex::Real rrr = qnormp(iv, QRHO);
      amrex::Real rur = qnormp(iv, QU);
      amrex::Real rvr = qnormp(iv, QV);
      amrex::Real rwr = qnormp(iv, QW);
      const amrex::Real ekinr = 0.5 * rrr * (rur * rur + rvr * rvr + rwr * rwr);
      rur *= rrr;
      rvr *= rrr;
      rwr *= rrr;

      const amrex::Real rer = qnormp(iv, QREINT) + ekinr;
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
         pnewr = qnormp(iv, QPRES) - cdtdx * (dAup + pav * dAu * (c - 1.0));
         qp(iv, QPRES) = std::max(pnewr, small_pres);
      } else {
         qp(iv, QPRES) = qnormp(iv, QPRES);
      }
  } else {
      qp(iv,QRHO  ) = qnormp(iv,QRHO  );
      qp(iv,QU    ) = qnormp(iv,QU    );
      qp(iv,QV    ) = qnormp(iv,QV    );
      qp(iv,QREINT) = qnormp(iv,QREINT);
      qp(iv,QPRES ) = qnormp(iv,QPRES );
  }

  // QM

  if (hi_face_not_covered)
  {
      // Conversion to Conservative
      const amrex::Real rrl = qnormm(ivpn, QRHO);
      amrex::Real rul = qnormm(ivpn, QU);
      amrex::Real rvl = qnormm(ivpn, QV);
      amrex::Real rwl = qnormm(ivpn, QW);
      const amrex::Real ekinl = 0.5 * rrl * (rul * rul + rvl * rvl + rwl * rwl);
      rul *= rrl;
      rvl *= rrl;
      rwl *= rrl;
      const amrex::Real rel = qnormm(ivpn, QREINT) + ekinl;

      // Transverse fluxes
      amrex::Real rrnewl = rrl - flxrho;
      amrex::Real runewl = rul - flxu;
      amrex::Real rvnewl = rvl - flxv;
      amrex::Real rwnewl = rwl - flxw;
      amrex::Real renewl = rel - flxe;

      bool reset_state = false;
      if (transverse_reset_density == 1 && rrnewl < 0.) {
         rrnewl = rrl;
         runewl = rul;
         rvnewl = rvl;
         rwnewl = rwl;
         renewl = rel;
         reset_state = true;
      }

      qm(ivpn, QRHO) = rrnewl;
      qm(ivpn, QU) = runewl / rrnewl;
      qm(ivpn, QV) = rvnewl / rrnewl;
      qm(ivpn, QW) = rwnewl / rrnewl;
      const amrex::Real rhoekinl =
        0.5 * (runewl * runewl + rvnewl * rvnewl + rwnewl * rwnewl) / rrnewl;

      qm(ivpn, QREINT) = renewl - rhoekinl;

      amrex::Real pnewl;
      if(!reset_state){
        pnewl = qnormm(ivpn, QPRES) - cdtdx * (dAup + pav * dAu * (c - 1.0));
        qm(ivpn, QPRES) = std::max(pnewl, small_pres);
      } else {
        qm(ivpn, QPRES) = qnormm(ivpn, QPRES);
      }

  } else {
      qm(ivpn,QRHO  ) = qnormm(ivpn,QRHO  );
      qm(ivpn,QU    ) = qnormm(ivpn,QU    );
      qm(ivpn,QV    ) = qnormm(ivpn,QV    );
      qm(ivpn,QW    ) = qnormm(ivpn,QW    );
      qm(ivpn,QREINT) = qnormm(ivpn,QREINT);
      qm(ivpn,QPRES ) = qnormm(ivpn,QPRES );
  }
}

// dir corrected from other two dirs
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
hydro_transdd (
  const int i,
  const int j,
  const int k,
  const int dir,
  amrex::Array4<amrex::Real> const& qm,
  amrex::Array4<amrex::Real> const& qp,
  amrex::Array4<const amrex::Real> const& qnormm,
  amrex::Array4<const amrex::Real> const& qnormp,
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
  amrex::Real small_pres
#ifdef AMREX_USE_EB
  , amrex::Array4<const amrex::Real> const&  norm_area = {}
  , amrex::Array4<const amrex::Real> const& trans0_area = {}
  , amrex::Array4<const amrex::Real> const& trans1_area = {}
#endif
)
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
  const amrex::IntVect ivpt0(iv + amrex::IntVect::TheDimensionVector(l_idx[1]));
  const amrex::IntVect ivpt1(iv + amrex::IntVect::TheDimensionVector(l_idx[2]));
  const amrex::IntVect ivpn(iv + amrex::IntVect::TheDimensionVector(dir));

  // ivpn is an offset by 1 in the normal direction
  // ivpt0 is an offset by 1 in the first transverse direction
  // ivpt1 is an offset by 1 in the second transverse direction
  // If (dir == 0),  ivpt0 = (i,j+1,k) and ivpt1 = (i,j,k+1) and ivpn = (i+1,j,k) and qvidx0 = GDV and qvidx1 = GVW
  // If (dir == 1),  ivpt0 = (i+1,j,k) and ivpt1 = (i,j,k+1) and ivpn = (i,j+1,k) and qvidx0 = GDU and qvidx1 = GVW
  // If (dir == 2),  ivpt0 = (i+1,j,k) and ivpt1 = (i,j+1,k) and ivpn = (i,j,k+1) and qvidx0 = GDU and qvidx1 = GVV

  bool no_cov_face = true;
  bool lo_face_not_covered = true;
  bool hi_face_not_covered = true;
#ifdef AMREX_USE_EB
    if (trans0_area && trans1_area) {
        no_cov_face = (trans0_area(ivpt0) > 0.0) && (trans1_area(ivpt1) > 0.0)
                   && (trans0_area(iv   ) > 0.0) && (trans1_area(iv   ) > 0.0);
    }
    if (norm_area) {
        lo_face_not_covered = (norm_area(iv  ) > amrex::Real(0.0));
        hi_face_not_covered = (norm_area(ivpn) > amrex::Real(0.0));
    }
#endif

  const amrex::Real flxrho = no_cov_face ? cdtdx0 * (flxx(ivpt0, URHO) - flxx(iv, URHO)) +
                             cdtdx1 * (flxy(ivpt1, URHO) - flxy(iv, URHO)) : amrex::Real(0.0);
  const amrex::Real flxu = no_cov_face ? cdtdx0 * (flxx(ivpt0, UMX) - flxx(iv, UMX)) +
                           cdtdx1 * (flxy(ivpt1, UMX) - flxy(iv, UMX)) : amrex::Real(0.0);
  const amrex::Real flxv = no_cov_face ? cdtdx0 * (flxx(ivpt0, UMY) - flxx(iv, UMY)) +
                           cdtdx1 * (flxy(ivpt1, UMY) - flxy(iv, UMY)) : amrex::Real(0.0);
  const amrex::Real flxw = no_cov_face ? cdtdx0 * (flxx(ivpt0, UMZ) - flxx(iv, UMZ)) +
                           cdtdx1 * (flxy(ivpt1, UMZ) - flxy(iv, UMZ)) : amrex::Real(0.0);
  const amrex::Real flxe = no_cov_face ? cdtdx0 * (flxx(ivpt0, UEDEN) - flxx(iv, UEDEN)) +
                           cdtdx1 * (flxy(ivpt1, UEDEN) - flxy(iv, UEDEN)) : amrex::Real(0.0);
  const amrex::Real c = qa(iv, QGAMC);
  const amrex::Real srcrho = srcq(iv, QRHO);
  const amrex::Real srcu = srcq(iv, QU);
  const amrex::Real srcv = srcq(iv, QV);
  const amrex::Real srcw = srcq(iv, QW);
  const amrex::Real srce = srcq(iv, QREINT);
  const amrex::Real srcp = srcq(iv, QPRES);

  // Update passive variables
  const amrex::Real rrr = qnormp(iv, QRHO);
  const amrex::Real rrl = qnormm(ivpn, QRHO);
  amrex::Real rrnewl = rrl - flxrho;
  amrex::Real rrnewr = rrr - flxrho;

  for (int ipass = 0; ipass < NPASSIVE; ++ipass) {
    const int n = pmap.upassMap[ipass];
    const int nqp = pmap.qpassMap[ipass];
    const amrex::Real srcpass = srcq(iv, nqp);
    const amrex::Real compn = no_cov_face ?
      cdtdx0 * (flxx(ivpt0, n) - flxx(iv, n)) +
      cdtdx1 * (flxy(ivpt1, n) - flxy(iv, n)) : amrex::Real(0.0);

    amrex::Real compo;

    // qp
    if (lo_face_not_covered)
    {
      compo = rrr * qnormp(iv, nqp) - compn;
      qp(iv, nqp) = compo / rrnewr + hdt * srcpass;
    } else {
      qp(iv, nqp) = qnormp(iv, nqp);
    }

    // qm

    if (hi_face_not_covered)
    {
      compo = rrl * qnormm(ivpn, nqp) - compn;
      qm(ivpn, nqp) = compo / rrnewl + hdt * srcpass;
    } else {
      qm(ivpn, nqp) = qnormm(ivpn, nqp);
    }
  }

  // Update hydro vars
  amrex::Real dAupx = 0.0, pavx = 0.0, dAux = 0.0;
  amrex::Real dAupy = 0.0, pavy = 0.0, dAuy = 0.0;
  if (no_cov_face) {
    const amrex::Real pggpx = qx(ivpt0, GDPRES);
    const amrex::Real pggmx = qx(iv, GDPRES);
    const amrex::Real ugpx = qx(ivpt0, qvidx0);
    const amrex::Real ugmx = qx(iv, qvidx0);

    dAupx = pggpx * ugpx - pggmx * ugmx;
    pavx = 0.5 * (pggpx + pggmx);
    dAux = ugpx - ugmx;

    const amrex::Real pggpy = qy(ivpt1, GDPRES);
    const amrex::Real pggmy = qy(iv, GDPRES);
    const amrex::Real ugpy = qy(ivpt1, qvidx1);
    const amrex::Real ugmy = qy(iv, qvidx1);

    dAupy = pggpy * ugpy - pggmy * ugmy;
    pavy = 0.5 * (pggpy + pggmy);
    dAuy = ugpy - ugmy;
  }
  const amrex::Real pxnew = cdtdx0 * (dAupx + pavx * dAux * (c - 1.0));
  const amrex::Real pynew = cdtdx1 * (dAupy + pavy * dAuy * (c - 1.0));

  // qp state
  if (lo_face_not_covered)
  {
    const amrex::Real rur = rrr * qnormp(iv, QU);
    const amrex::Real rvr = rrr * qnormp(iv, QV);
    const amrex::Real rwr = rrr * qnormp(iv, QW);
    const amrex::Real ekinr = 0.5 * (rur * rur + rvr * rvr + rwr * rwr) / rrr;
    const amrex::Real rer = qnormp(iv, QREINT) + ekinr;

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
      qp(iv, QPRES) =  qnormp(iv, QPRES) - pxnew - pynew + hdt * srcp;
    } else {
      qp(iv, QPRES) = qnormp(iv, QPRES) + hdt * srcp;
    }
    qp(iv, QPRES) = std::max( qp(iv,QPRES), small_pres);
  } else {
      qp(iv,QRHO  ) = qnormp(iv,QRHO  );
      qp(iv,QU    ) = qnormp(iv,QU    );
      qp(iv,QV    ) = qnormp(iv,QV    );
      qp(iv,QREINT) = qnormp(iv,QREINT);
      qp(iv,QPRES ) = qnormp(iv,QPRES );
  }

  // qm state
  if (hi_face_not_covered)
  {
    const amrex::Real rul = rrl * qnormm(ivpn, QU);
    const amrex::Real rvl = rrl * qnormm(ivpn, QV);
    const amrex::Real rwl = rrl * qnormm(ivpn, QW);
    const amrex::Real ekinl = 0.5 * (rul * rul + rvl * rvl + rwl * rwl) / rrl;
    const amrex::Real rel = qnormm(ivpn, QREINT) + ekinl;

    amrex::Real runewl = rul - flxu;
    amrex::Real rvnewl = rvl - flxv;
    amrex::Real rwnewl = rwl - flxw;
    amrex::Real renewl = rel - flxe;

    bool reset_state = false;
    if (transverse_reset_density == 1 && rrnewl < 0.) {
      rrnewl = rrl;
      runewl = rul;
      rvnewl = rvl;
      rwnewl = rwl;
      renewl = rel;
      reset_state = true;
    }

    qm(ivpn, QRHO) = rrnewl + hdt * srcrho;
    qm(ivpn, QU) = runewl / rrnewl + hdt * srcu;
    qm(ivpn, QV) = rvnewl / rrnewl + hdt * srcv;
    qm(ivpn, QW) = rwnewl / rrnewl + hdt * srcw;

    const amrex::Real rhoekinl =
      0.5 * (runewl * runewl + rvnewl * rvnewl + rwnewl * rwnewl) / rrnewl;
    qm(ivpn, QREINT) = renewl - rhoekinl + hdt * srce;

    if(!reset_state){
      qm(ivpn, QPRES) = qnormm(ivpn, QPRES) - pxnew - pynew + hdt * srcp;
    } else {
      qm(ivpn, QPRES) = qnormm(ivpn, QPRES) + hdt * srcp;
    }
    qm(ivpn, QPRES) = std::max(qm(ivpn,QPRES), small_pres);

  } else {
      qm(ivpn,QRHO  ) = qnormm(ivpn,QRHO  );
      qm(ivpn,QU    ) = qnormm(ivpn,QU    );
      qm(ivpn,QV    ) = qnormm(ivpn,QV    );
      qm(ivpn,QW    ) = qnormm(ivpn,QW    );
      qm(ivpn,QREINT) = qnormm(ivpn,QREINT);
      qm(ivpn,QPRES ) = qnormm(ivpn,QPRES );
  }
}
#endif
#endif
