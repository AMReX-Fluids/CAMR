#ifndef _CMPFLXH_
#define _CMPFLXH_

#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>

#include "IndexDefines.H"
#include "Hydro_riemann.H"

#if (AMREX_SPACEDIM == 2)

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_cmpflx(
  const int i,
  const int j,
  const int k,
  const int bclo,
  const int bchi,
  const int domlo,
  const int domhi,
  amrex::Array4<amrex::Real> const& ql,
  amrex::Array4<amrex::Real> const& qr,
  amrex::Array4<amrex::Real> const& flx,
  amrex::Array4<      amrex::Real> const& q,
  amrex::Array4<const amrex::Real> const& qa,
  const int dir,
  PassMap const& pmap,
  const amrex::Real small,
  const amrex::Real small_dens,
  const amrex::Real small_pres)
{
    amrex::ignore_unused(k);
    amrex::Real cav = 0.0, ustar;
    amrex::Real gamcl, gamcr, smallc = 0.0;
    amrex::Real ul, ur, vl, vr;
    int idx;
    int IU = 0, IV = 0;
    int GU = 0, GV = 0;
    int f_idx[2];
    if (dir == 0) {
      IU = QU;
      IV = QV;
      GU = GDU;
      GV = GDV;
      cav = 0.5 * (qa(i, j, k, QC) + qa(i - 1, j, k, QC));
      smallc = std::max (qa(i, j, k, QCSML) , qa(i - 1, j, k, QCSML));
      gamcl = qa(i-1,j,k,QGAMC);
      gamcr = qa(i  ,j,k,QGAMC);
      f_idx[0] = UMX;
      f_idx[1] = UMY;
    } else if (dir == 1) {
      IU = QV;
      IV = QU;
      GU = GDV;
      GV = GDU;
      cav = 0.5 * (qa(i, j, k, QC) + qa(i, j - 1, k, QC));
      smallc = std::max (qa(i, j, k, QCSML) , qa(i, j - 1, k, QCSML));
      gamcl = qa(i,j-1,k,QGAMC);
      gamcr = qa(i,j  ,k,QGAMC);
      f_idx[0] = UMY;
      f_idx[1] = UMX;
    }
    else{
      amrex::Abort();
    }

    ul = ql(i, j, k, IU);
    vl = ql(i, j, k, IV);

    ur = qr(i, j, k, IU);
    vr = qr(i, j, k, IV);

    idx = (dir == 0) ? i : j;

    if ( idx == domlo)
    {
        if (bclo == Symmetry || bclo == SlipWall || bclo == NoSlipWall)
        {
            ql(i,j,k,QRHO)   = qr(i,j,k,QRHO);
            ql(i,j,k,QPRES)  = qr(i,j,k,QPRES);
            ql(i,j,k,QREINT) = qr(i,j,k,QREINT);
            gamcl = gamcr;
            ul  = -ur;
            vl  =  vr; // Note this is ok for NoSlip because there will be no
                       // advection velocity through the wall
        } else if (bclo == Outflow) {
            ul = ur;
            vl = vr;
            gamcl = gamcr;
            ql(i,j,k,QRHO)   = qr(i,j,k,QRHO);
            ql(i,j,k,QPRES)  = qr(i,j,k,QPRES);
            ql(i,j,k,QREINT) = qr(i,j,k,QREINT);
        }
    }
    if ( idx == domhi+1)
    {
        if (bchi == Symmetry || bchi == SlipWall || bchi == NoSlipWall)
        {
            qr(i,j,k,QRHO)   = ql(i,j,k,QRHO);
            qr(i,j,k,QPRES)  = ql(i,j,k,QPRES);
            qr(i,j,k,QREINT) = ql(i,j,k,QREINT);
            gamcr = gamcl;
            ur    = -ul;
            vr    =  vl; // Note this is ok for NoSlip because there will be no
                         // advection velocity through the wall
        } else if (bchi == Outflow) {
            qr(i,j,k,QRHO)   = ql(i,j,k,QRHO);
            qr(i,j,k,QPRES)  = ql(i,j,k,QPRES);
            qr(i,j,k,QREINT) = ql(i,j,k,QREINT);
            gamcr = gamcl;
            ur    = ul;
            vr    = vl;
        }
    }

    riemannus(
       ql(i, j, k, QRHO), ul, vl, ql(i, j, k, QPRES), ql(i,j,k,QREINT),
       qr(i, j, k, QRHO), ur, vr, qr(i, j, k, QPRES), qr(i,j,k,QREINT),
       cav, gamcl, gamcr,
       small, smallc, small_dens, small_pres, ustar,
       flx(i, j, k, URHO),
       AMREX_D_DECL(flx(i, j, k, f_idx[0]),
                    flx(i, j, k, f_idx[1]),
                    flx(i, j, k, f_idx[2])),
       flx(i, j, k, UEDEN), flx(i, j, k, UEINT),
       q(i, j, k, GU), q(i, j, k, GV),
#if (AMREX_SPACEDIM == 3)
       q(i, j, k, GV2),
#endif
       q(i, j, k, GDPRES), q(i, j, k, GDGAME));

    amrex::Real flxrho = flx(i, j, k, URHO);
    for (int ipass = 0; ipass < NPASSIVE; ++ipass) {
      int n = pmap.upassMap[ipass];
      int nqp = pmap.qpassMap[ipass];
      if (ustar > 0.0) {
        flx(i, j, k, n) = flxrho * ql(i, j, k, nqp);
      } else if (ustar < 0.0) {
        flx(i, j, k, n) = flxrho * qr(i, j, k, nqp);
      } else {
        flx(i, j, k, n) = flxrho * 0.50 * (ql(i, j, k, nqp) + qr(i, j, k, nqp));
      }
  }
}

#elif (AMREX_SPACEDIM == 3)

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_cmpflx(
  const int i,
  const int j,
  const int k,
  const int bclo,
  const int bchi,
  const int domlo,
  const int domhi,
  amrex::Array4<amrex::Real> const& ql,
  amrex::Array4<amrex::Real> const& qr,
  amrex::Array4<amrex::Real> const& flx,
  amrex::Array4<      amrex::Real> const& q,
  amrex::Array4<const amrex::Real> const& qa,
  const int dir,
  PassMap const& pmap,
  const amrex::Real small,
  const amrex::Real small_dens,
  const amrex::Real small_pres)
{
    amrex::ignore_unused(k);
    amrex::Real cav, ustar;
    amrex::Real gamcl, gamcr, smallc;
    amrex::Real ul, ur, vl, vr, v2l, v2r;
    int idx;
    int IU, IV, IV2;
    int GU, GV, GV2;
    int f_idx[3];
    if (dir == 0) {
      IU = QU;
      IV = QV;
      IV2 = QW;
      GU = GDU;
      GV = GDV;
      GV2 = GDW;
      cav = 0.5 * (qa(i, j, k, QC) + qa(i - 1, j, k, QC));
      smallc = std::max (qa(i, j, k, QCSML) , qa(i - 1, j, k, QCSML));
      gamcl = qa(i-1,j,k,QGAMC);
      gamcr = qa(i  ,j,k,QGAMC);
      f_idx[0] = UMX;
      f_idx[1] = UMY;
      f_idx[2] = UMZ;
    } else if (dir == 1) {
      IU = QV;
      IV = QU;
      IV2 = QW;
      GU = GDV;
      GV = GDU;
      GV2 = GDW;
      cav = 0.5 * (qa(i, j, k, QC) + qa(i, j - 1, k, QC));
      smallc = std::max (qa(i, j, k, QCSML) , qa(i, j - 1, k, QCSML));
      gamcl = qa(i,j-1,k,QGAMC);
      gamcr = qa(i,j  ,k,QGAMC);
      f_idx[0] = UMY;
      f_idx[1] = UMX;
      f_idx[2] = UMZ;
    } else {
      IU = QW;
      IV = QU;
      IV2 = QV;
      GU = GDW;
      GV = GDU;
      GV2 = GDV;
      cav = 0.5 * (qa(i, j, k, QC) + qa(i, j, k - 1, QC));
      smallc = std::max (qa(i, j, k, QCSML) , qa(i, j, k - 1, QCSML));
      gamcl = qa(i,j,k-1,QGAMC);
      gamcr = qa(i,j,k  ,QGAMC);
      f_idx[0] = UMZ;
      f_idx[1] = UMX;
      f_idx[2] = UMY;
    }

    ul = ql(i, j, k, IU);
    vl = ql(i, j, k, IV);
    v2l = ql(i, j, k, IV2);

    ur = qr(i, j, k, IU);
    vr = qr(i, j, k, IV);
    v2r = qr(i, j, k, IV2);

    if (dir == 2) {
      idx = k;
    } else {
      idx = (dir == 0) ? i : j;
    }

    if ( idx == domlo)
    {
        if (bclo == Symmetry || bclo == SlipWall || bclo == NoSlipWall)
        {
            ql(i,j,k,QRHO)   = qr(i,j,k,QRHO);
            ql(i,j,k,QPRES)  = qr(i,j,k,QPRES);
            ql(i,j,k,QREINT) = qr(i,j,k,QREINT);
            gamcl = gamcr;
            ul  = -ur;
            vl  =  vr;
            v2l = v2r;
        } else if (bclo == Outflow) {
            ul = ur;
            vl = vr;
            v2l = v2r;
            gamcl = gamcr;
            ql(i,j,k,QRHO)   = qr(i,j,k,QRHO);
            ql(i,j,k,QPRES)  = qr(i,j,k,QPRES);
            ql(i,j,k,QREINT) = qr(i,j,k,QREINT);
        }
    }
    if ( idx == domhi+1)
    {
        if (bchi == Symmetry || bchi == SlipWall || bchi == NoSlipWall)
        {
            qr(i,j,k,QRHO)   = ql(i,j,k,QRHO);
            qr(i,j,k,QPRES)  = ql(i,j,k,QPRES);
            qr(i,j,k,QREINT) = ql(i,j,k,QREINT);
            gamcr = gamcl;
            ur    = -ul;
            vr    =  vl;
            v2r   = v2l;
        } else if (bchi == Outflow) {
            qr(i,j,k,QRHO)   = ql(i,j,k,QRHO);
            qr(i,j,k,QPRES)  = ql(i,j,k,QPRES);
            qr(i,j,k,QREINT) = ql(i,j,k,QREINT);
            gamcr = gamcl;
            ur    = ul;
            vr    = vl;
            v2r   = v2l;
        }
    }

    riemannus(
       ql(i, j, k, QRHO), ul, vl, v2l, ql(i, j, k, QPRES), ql(i,j,k,QREINT),
       qr(i, j, k, QRHO), ur, vr, v2r, qr(i, j, k, QPRES), qr(i,j,k,QREINT),
       cav, gamcl, gamcr,
       small, smallc, small_dens, small_pres, ustar,
       flx(i, j, k, URHO),
       AMREX_D_DECL(flx(i, j, k, f_idx[0]),
                    flx(i, j, k, f_idx[1]),
                    flx(i, j, k, f_idx[2])),
       flx(i, j, k, UEDEN), flx(i, j, k, UEINT),
       q(i, j, k, GU), q(i, j, k, GV), q(i, j, k, GV2),
       q(i, j, k, GDPRES), q(i, j, k, GDGAME));

    amrex::Real flxrho = flx(i, j, k, URHO);
    for (int ipass = 0; ipass < NPASSIVE; ++ipass) {
      int n = pmap.upassMap[ipass];
      int nqp = pmap.qpassMap[ipass];
      if (ustar > 0.0) {
        flx(i, j, k, n) = flxrho * ql(i, j, k, nqp);
      } else if (ustar < 0.0) {
        flx(i, j, k, n) = flxrho * qr(i, j, k, nqp);
      } else {
        flx(i, j, k, n) = flxrho * 0.50 * (ql(i, j, k, nqp) + qr(i, j, k, nqp));
      }
  }
}
#endif
#endif
