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
  amrex::Array4<const amrex::Real> const& qnormm,
  amrex::Array4<const amrex::Real> const& qnormp,
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
  , amrex::Array4<const amrex::Real> const&  norm_area = {}
  , amrex::Array4<const amrex::Real> const& trans_area = {}
#endif
)
{
  amrex::ignore_unused(k);

  const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
  const amrex::IntVect ivpt(iv + amrex::IntVect::TheDimensionVector(dir == 0 ? 1 : 0));
  const amrex::IntVect ivpn(iv + amrex::IntVect::TheDimensionVector(dir == 0 ? 0 : 1));
  const int qvidx = (dir == 0) ? GDV : GDU;

  // ivpt is an offset by 1 in the transverse direction
  // ivpn is an offset by 1 in the     normal direction
  // If (dir == 0),  ivpt = (i,j+1) and ivpn = (i+1,j) and qvidx = GDV
  // If (dir == 1),  ivpt = (i+1,j) and ivpn = (i,j+1) and qvidx = GDU

  const bool no_cov_face =
#ifdef AMREX_USE_EB
    (trans_area(ivpt) > 0.0) && (trans_area(iv) > 0.0);
#else
  true;
#endif

  const amrex::Real flxrho = no_cov_face ? cdtdx * (flxx(ivpt, URHO ) - flxx(iv, URHO )) : 0.0;
  const amrex::Real flxu   = no_cov_face ? cdtdx * (flxx(ivpt, UMX  ) - flxx(iv, UMX  )) : 0.0;
  const amrex::Real flxv   = no_cov_face ? cdtdx * (flxx(ivpt, UMY  ) - flxx(iv, UMY  )) : 0.0;
  const amrex::Real flxe   = no_cov_face ? cdtdx * (flxx(ivpt, UEDEN) - flxx(iv, UEDEN)) : 0.0;
  const amrex::Real srcr   = srcQ(iv, QRHO);
  const amrex::Real srce   = srcQ(iv, QREINT);
  const amrex::Real srcp   = srcQ(iv, QPRES);
  const amrex::Real    c   =   qa(iv, QGAMC);

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

  // ****************************************************************************
  // Update passive variables
  // ****************************************************************************
  for (int ipass = 0; ipass < NPASSIVE; ++ipass) {
    const int n = pmap.upassMap[ipass];
    const int nqp = pmap.qpassMap[ipass];
    const amrex::Real srcpass = srcQ(iv, nqp);
    const amrex::Real compn = no_cov_face ? cdtdx * (flxx(ivpt, n) - flxx(iv, n)) : 0.0;

    if (norm_area(iv) > amrex::Real(0.0)) {
        amrex::Real rr = qnormp(iv, QRHO);
        amrex::Real rrnew = rr - flxrho;
        amrex::Real compo = rr * qnormp(iv, nqp) - compn;
        qp(iv, nqp) = compo / rrnew + hdt * srcpass;
    } else {
        qp(iv, nqp) = qnormp(iv, nqp);
    }

    if (norm_area(ivpn) > amrex::Real(0.0)) {
        amrex::Real rr = qnormm(ivpn, QRHO);
        amrex::Real rrnew = rr - flxrho;
        amrex::Real compo = rr * qnormm(ivpn, nqp) - compn;
        qm(ivpn, nqp) = compo / rrnew + hdt * srcpass;
    } else {
        qm(iv, nqp) = qnormm(iv, nqp);
    }
  }

  bool reset_state;

  // ****************************************************************************
  // QP
  // ****************************************************************************

  if (norm_area(iv) > amrex::Real(0.0))
  {
      // Convert to conservative
      const amrex::Real rrr = qnormp(iv, QRHO);
      amrex::Real rur = qnormp(iv, QU);
      amrex::Real rvr = qnormp(iv, QV);
      const amrex::Real ekinr = 0.5 * rrr * (rur * rur + rvr * rvr);
      rur *= rrr;
      rvr *= rrr;

      const amrex::Real rer = qnormp(iv, QREINT) + ekinr;
      // Add transverse predictor
      amrex::Real rrnewr = rrr - flxrho;
      amrex::Real runewr = rur - flxu;
      amrex::Real rvnewr = rvr - flxv;
      amrex::Real renewr = rer - flxe;

      reset_state = false;
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
          amrex::Real pnewr = qnormp(iv, QPRES) - cdtdx * (dAup + pav * dAu * (c - 1.));
          pnewr += hdt * srcp;
          qp(iv, QPRES) = pnewr;
      } else {
          qp(iv, QPRES) = qnormp(iv,QPRES);
      }

      qp(iv,QPRES) = std::max( qp(iv,QPRES), small_pres);

  } else {

      qp(iv,QRHO  ) = qnormp(iv,QRHO  );
      qp(iv,QU    ) = qnormp(iv,QU    );
      qp(iv,QV    ) = qnormp(iv,QV    );
      qp(iv,QREINT) = qnormp(iv,QREINT);
      qp(iv,QPRES ) = qnormp(iv,QPRES );
  }

  // ****************************************************************************
  // QM
  // ****************************************************************************

  if (norm_area(ivpn) > amrex::Real(0.0))
  {
      // Conversion to Conservative
      const amrex::Real rrl = qnormm(ivpn, QRHO);
      amrex::Real rul = qnormm(ivpn, QU);
      amrex::Real rvl = qnormm(ivpn, QV);
      const amrex::Real ekinl = 0.5 * rrl * (rul * rul + rvl * rvl);
      rul *= rrl;
      rvl *= rrl;
      const amrex::Real rel = qnormm(ivpn, QREINT) + ekinl;

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

      qm(ivpn, QRHO) = rrnewl + hdt * srcr;
      qm(ivpn, QU) = runewl / rrnewl + hdt * srcQ(iv, QU);
      qm(ivpn, QV) = rvnewl / rrnewl + hdt * srcQ(iv, QV);

      const amrex::Real rhoekinl = 0.5 * (runewl * runewl + rvnewl * rvnewl) / rrnewl;
      qm(ivpn, QREINT) = renewl - rhoekinl + hdt * srce;

      if(!reset_state){
          amrex::Real pnewl = qnormm(ivpn, QPRES) - cdtdx * (dAup + pav * dAu * (c - 1.));
          qm(ivpn, QPRES) = pnewl + hdt * srcp;
      } else {
          qm(ivpn, QPRES) = qnormm(ivpn, QPRES)  + hdt * srcp;
      }

      qm(ivpn, QPRES) = std::max( qm(ivpn, QPRES), small_pres);

  } else {

      qm(ivpn,QRHO  ) = qnormm(ivpn,QRHO  );
      qm(ivpn,QU    ) = qnormm(ivpn,QU    );
      qm(ivpn,QV    ) = qnormm(ivpn,QV    );
      qm(ivpn,QREINT) = qnormm(ivpn,QREINT);
      qm(ivpn,QPRES ) = qnormm(ivpn,QPRES );
  }
}
#endif
#endif
