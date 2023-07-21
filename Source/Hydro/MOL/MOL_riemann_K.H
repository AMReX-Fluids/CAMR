#ifndef MOL_RIEMANN_K_H
#define MOL_RIEMANN_K_H

#include "IndexDefines.H"
#include "Hydro_cmpflx.H"
#include "Hydro_riemann.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
mol_riemann_x (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fx,
               amrex::Array4<amrex::Real const> const& dq,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& qa,
               amrex::Array4<amrex::Real      > const& q1,
               amrex::Array4<amrex::Real      > const& ql,
               amrex::Array4<amrex::Real      > const& qr,
               amrex::Real small,
               amrex::Real small_dens, amrex::Real small_pres,
               const int bclo,
               const int bchi,
               const int domlo,
               const int domhi,
               PassMap const& pmap) noexcept
{
    using amrex::Real;

    Real cspeed = qa(i-1,j,k,QC);
    Real enth = (q(i-1,j,k,QREINT)+ q(i-1,j,k,QPRES))/ amrex::max(q(i-1,j,k,QRHO),small_dens);

#if 0
    Real rl = q(i-1,j,k,QRHO) + Real(0.5) * ( (dq(i-1,j,k,0)+dq(i-1,j,k,2))/cspeed + dq(i-1,j,k,1));
    rl = amrex::max(rl, small_dens);
    ql(i,j,k,QRHO) = rl;
#endif

    Real pl = q(i-1,j,k,QPRES) + Real(0.5) *  (dq(i-1,j,k,0)+dq(i-1,j,k,2))*cspeed;
    pl = amrex::max(pl, small_pres);
    ql(i,j,k,QPRES) = pl;

    ql(i,j,k,QU) = q(i-1,j,k,QU) + Real(0.5) * ( (dq(i-1,j,k,2)-dq(i-1,j,k,0))/q(i-1,j,k,QRHO));
    ql(i,j,k,QV) = q(i-1,j,k,QV) + Real(0.5) * dq(i-1,j,k,3);
#if (AMREX_SPACEDIM == 3)
    ql(i,j,k,QW) = q(i-1,j,k,QW) + Real(0.5) * dq(i-1,j,k,4);
#endif
#if (NUM_ADV > 0)
        for (int n=0; n < NUM_ADV; n++)
        {
           ql(i,j,k,QFA+n) = q(i-1,j,k,QFA+n) + Real(0.5) * dq(i-1,j,k,QFA+n);
        }
#endif
#if 0
    Real sum = 0;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real massfrac = q(i-1,j,k,QFS+n)+ 0.5*dq(i-1,j,k,QFS+n);
        massfrac = amrex::max(0., amrex::min(1.,massfrac));
        sum += massfrac;
        ql(i,j,k,QFS+n) = massfrac;
    }
    for (int n=0; n < NUM_SPECIES ; n++)
    {
        ql(i,j,k,QFS+n) = ql(i,j,k,QFS+n)/sum;
    }

#else
    Real sum = 0.;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real specden = q(i-1,j,k,QRHO)*q(i-1,j,k,QFS+n)+ 0.5*(dq(i-1,j,k,QFS+n)
            +q(i-1,j,k,QFS+n)*(dq(i-1,j,k,0)+dq(i-1,j,k,2))/cspeed);
        sum += specden;
        ql(i,j,k,QFS+n) = specden;
    }

    if (sum < small_dens) {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            ql(i,j,k,QFS+n) = q(i,j,k,QFS+n);
        }
    } else {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            ql(i,j,k,QFS+n) = ql(i,j,k,QFS+n)/sum;
        }
    }
    ql(i,j,k,QRHO) = amrex::max(sum, small_dens);

#endif

    ql(i,j,k,QREINT) = q(i-1,j,k,QREINT)+0.5*(dq(i-1,j,k,QREINT)+enth*(dq(i-1,j,k,0)+dq(i-1,j,k,2))/cspeed);

    cspeed = qa(i,j,k,QC);
    enth = (q(i,j,k,QREINT)+ q(i,j,k,QPRES))/ amrex::max(q(i,j,k,QRHO),small_dens);

    Real pr = q(i,j,k,QPRES) - Real(0.5) * (dq(i,j,k,0)+dq(i,j,k,2))*cspeed;
    pr = amrex::max(pr, small_pres);
    qr(i,j,k,QPRES) = pr;

#if 0
    Real rr = q(i,j,k,QRHO) - Real(0.5) * ( (dq(i,j,k,0)+dq(i,j,k,2))/cspeed + dq(i,j,k,1));
    rr = amrex::max(rr, small_dens);
    qr(i,j,k,QRHO) = rr;
#endif

    qr(i,j,k,QU) = q(i,j,k,QU) - Real(0.5) * ( (dq(i,j,k,2)-dq(i,j,k,0))/q(i,j,k,QRHO));
    qr(i,j,k,QV) = q(i,j,k,QV) - Real(0.5) * dq(i,j,k,3);
#if (AMREX_SPACEDIM == 3)
    qr(i,j,k,QW) = q(i,j,k,QW) - Real(0.5) * dq(i,j,k,4);
#endif
#if (NUM_ADV > 0)
        for (int n=0; n < NUM_ADV; n++)
        {
           qr(i,j,k,QFA+n) = q(i,j,k,QFA+n) - Real(0.5) * dq(i,j,k,QFA+n);
        }
#endif
#if 0
    sum = 0.;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real massfrac = q(i,j,k,QFS+n)- 0.5*dq(i,j,k,QFS+n);
        massfrac = amrex::max(0., amrex::min(1.,massfrac));
        sum += massfrac;
        qr(i,j,k,QFS+n) = massfrac;
    }
    for (int n=0; n < NUM_SPECIES ; n++)
    {
        qr(i,j,k,QFS+n) = qr(i,j,k,QFS+n)/sum;
    }
#else
    sum = 0.;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real specden = q(i,j,k,QRHO)*q(i,j,k,QFS+n) - 0.5*(dq(i,j,k,QFS+n)
                      +q(i,j,k,QFS+n)*(dq(i,j,k,0)+dq(i,j,k,2))/cspeed);
        sum += specden;
        qr(i,j,k,QFS+n) = specden;
    }

    if (sum < small_dens) {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            qr(i,j,k,QFS+n) = q(i,j,k,QFS+n);
        }
    } else {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            qr(i,j,k,QFS+n) = qr(i,j,k,QFS+n)/sum;
        }
    }
    qr(i,j,k,QRHO) = amrex::max(sum, small_dens);

#endif

    qr(i,j,k,QREINT) = q(i,j,k,QREINT) - 0.5*(dq(i,j,k,QREINT)+enth*(dq(i,j,k,0)+dq(i,j,k,2))/cspeed);

    int dir = 0;
    CAMR_cmpflx(i,j,k,bclo,bchi,domlo,domhi,ql,qr,fx,q1,qa,dir,pmap,small,small_dens,small_pres);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
mol_riemann_y (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fy,
               amrex::Array4<amrex::Real const> const& dq,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& qa,
               amrex::Array4<amrex::Real      > const& q2,
               amrex::Array4<amrex::Real      > const& ql,
               amrex::Array4<amrex::Real      > const& qr,
               amrex::Real small,
               amrex::Real small_dens, amrex::Real small_pres,
               const int bclo,
               const int bchi,
               const int domlo,
               const int domhi,
               PassMap const& pmap) noexcept
{
    using amrex::Real;

    Real cspeed = qa(i,j-1,k,QC);
    Real enth = (q(i,j-1,k,QREINT)+ q(i,j-1,k,QPRES))/ amrex::max(q(i,j-1,k,QRHO),small_dens);

#if 0
    Real rl = q(i,j-1,k,QRHO) + Real(0.5) * ( (dq(i,j-1,k,0)+dq(i,j-1,k,2))/cspeed + dq(i,j-1,k,1));
    rl = amrex::max(rl, small_dens);
    ql(i,j,k,QRHO) = rl;
#endif

    Real pl = q(i,j-1,k,QPRES) + Real(0.5) *  (dq(i,j-1,k,0)+dq(i,j-1,k,2))*cspeed;
    pl = amrex::max(pl, small_pres);
    ql(i,j,k,QPRES) = pl;

    ql(i,j,k,QV) = q(i,j-1,k,QV) + Real(0.5) * ( (dq(i,j-1,k,2)-dq(i,j-1,k,0))/q(i,j-1,k,QRHO));
    ql(i,j,k,QU) = q(i,j-1,k,QU) + Real(0.5) * dq(i,j-1,k,3);
#if (AMREX_SPACEDIM == 3)
    ql(i,j,k,QW) = q(i,j-1,k,QW) + Real(0.5) * dq(i,j-1,k,4);
#endif
#if (NUM_ADV > 0)
        for (int n=0; n < NUM_ADV; n++)
        {
           ql(i,j,k,QFA+n) = q(i,j-1,k,QFA+n) + Real(0.5) * dq(i,j-1,k,QFA+n);
        }
#endif
#if 0
    Real sum = 0.;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real massfrac = q(i,j-1,k,QFS+n)+ 0.5*dq(i,j-1,k,QFS+n);
        massfrac = amrex::max(0., amrex::min(1.,massfrac));
        sum += massfrac;
        ql(i,j,k,QFS+n) = massfrac;
    }
    for (int n=0; n < NUM_SPECIES ; n++)
    {
        ql(i,j,k,QFS+n) = ql(i,j,k,QFS+n)/sum;
    }
#else
    Real sum = 0.;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real specden = q(i,j-1,k,QRHO)*q(i,j-1,k,QFS+n) + 0.5*(dq(i,j-1,k,QFS+n)
                      +q(i,j-1,k,QFS+n)*(dq(i,j-1,k,0)+dq(i,j-1,k,2))/cspeed);
        sum += specden;
        ql(i,j,k,QFS+n) = specden;
    }

    if (sum < small_dens) {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            ql(i,j,k,QFS+n) = q(i,j,k,QFS+n);
        }
    } else {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            ql(i,j,k,QFS+n) = ql(i,j,k,QFS+n)/sum;
        }
    }
    ql(i,j,k,QRHO) = amrex::max(sum, small_dens);

#endif

    ql(i,j,k,QREINT) = q(i,j-1,k,QREINT)+0.5*(dq(i,j-1,k,QREINT)+enth*(dq(i,j-1,k,0)+dq(i,j-1,k,2))/cspeed);

    cspeed = qa(i,j,k,QC);
    enth = (q(i,j,k,QREINT)+ q(i,j,k,QPRES))/ amrex::max(q(i,j,k,QRHO),small_dens);

#if 0
    Real rr = q(i,j,k,QRHO) - Real(0.5) * ( (dq(i,j,k,0)+dq(i,j,k,2))/cspeed + dq(i,j,k,1));
    rr = amrex::max(rr, small_dens);
    qr(i,j,k,QRHO) = rr;
#endif

    Real pr = q(i,j,k,QPRES) - Real(0.5) * (dq(i,j,k,0)+dq(i,j,k,2))*cspeed;
    pr = amrex::max(pr, small_pres);
    qr(i,j,k,QPRES) = pr;

    qr(i,j,k,QV) = q(i,j,k,QV) - Real(0.5) * ( (dq(i,j,k,2)-dq(i,j,k,0))/q(i,j,k,QRHO));
    qr(i,j,k,QU) = q(i,j,k,QU) - Real(0.5) * dq(i,j,k,3);
#if (AMREX_SPACEDIM == 3)
    qr(i,j,k,QW) = q(i,j,k,QW) - Real(0.5) * dq(i,j,k,4);
#endif
#if (NUM_ADV > 0)
        for (int n=0; n < NUM_ADV; n++)
        {
           qr(i,j,k,QFA+n) = q(i,j,k,QFA+n) - Real(0.5) * dq(i,j,k,QFA+n);
        }
#endif
#if 0
    sum = 0.;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real massfrac = q(i,j,k,QFS+n) - 0.5*dq(i,j,k,QFS+n);
        massfrac = amrex::max(0., amrex::min(1.,massfrac));
        sum += massfrac;
        qr(i,j,k,QFS+n) = massfrac;
    }
    for (int n=0; n < NUM_SPECIES ; n++)
    {
        qr(i,j,k,QFS+n) = qr(i,j,k,QFS+n)/sum;
    }
#else
    sum = 0.;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real specden = q(i,j,k,QRHO)*q(i,j,k,QFS+n)- 0.5*(dq(i,j,k,QFS+n)
            +q(i,j,k,QFS+n)*(dq(i,j,k,0)+dq(i,j,k,2))/cspeed);
        sum += specden;
        qr(i,j,k,QFS+n) = specden;
    }

    if (sum < small_dens) {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            qr(i,j,k,QFS+n) = q(i,j,k,QFS+n);
        }
    } else {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            qr(i,j,k,QFS+n) = qr(i,j,k,QFS+n)/sum;
        }
    }
    qr(i,j,k,QRHO) = amrex::max(sum, small_dens);
#endif

    qr(i,j,k,QREINT) = q(i,j,k,QREINT) - 0.5*(dq(i,j,k,QREINT)+enth*(dq(i,j,k,0)+dq(i,j,k,2))/cspeed);

    int dir = 1;
    CAMR_cmpflx(i,j,k,bclo,bchi,domlo,domhi,ql,qr,fy,q2,qa,dir,pmap,small,small_dens,small_pres);
}

#if (AMREX_SPACEDIM == 3)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
mol_riemann_z (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fz,
               amrex::Array4<amrex::Real const> const& dq,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& qa,
               amrex::Array4<amrex::Real      > const& q3,
               amrex::Array4<amrex::Real      > const& ql,
               amrex::Array4<amrex::Real      > const& qr,
               amrex::Real small,
               amrex::Real small_dens, amrex::Real small_pres,
               const int bclo,
               const int bchi,
               const int domlo,
               const int domhi,
               PassMap const& pmap) noexcept
{
    using amrex::Real;

    Real cspeed = qa(i,j,k-1,QC);
    Real enth = (q(i,j,k-1,QREINT)+ q(i,j,k-1,QPRES))/ amrex::max(q(i,j,k-1,QRHO),small_dens);

#if 0
   Real rl = q(i,j,k-1,QRHO) + Real(0.5) * ( (dq(i,j,k-1,0)+dq(i,j,k-1,2))/cspeed + dq(i,j,k-1,1));
   rl = amrex::max(rl, small_dens);
   ql(i,j,k,QRHO) = rl;
#endif

    Real pl = q(i,j,k-1,QPRES) + Real(0.5) *  (dq(i,j,k-1,0)+dq(i,j,k-1,2))*cspeed;
    pl = amrex::max(pl, small_pres);
    ql(i,j,k,QPRES) = pl;

    ql(i,j,k,QW) = q(i,j,k-1,QW) + Real(0.5) * ( (dq(i,j,k-1,2)-dq(i,j,k-1,0))/q(i,j,k-1,QRHO));
    ql(i,j,k,QU) = q(i,j,k-1,QU) + Real(0.5) * dq(i,j,k-1,3);
#if (AMREX_SPACEDIM == 2)
    ql(i,j,k,QV) = 0.;
#else
    ql(i,j,k,QV) = q(i,j,k-1,QV) + Real(0.5) * dq(i,j,k-1,4);
#endif
#if (NUM_ADV > 0)
        for (int n=0; n < NUM_ADV; n++)
        {
           ql(i,j,k,QFA+n) = q(i,j,k-1,QFA+n) + Real(0.5) * dq(i,j,k-1,QFA+n);
        }
#endif
#if 0
    Real sum = 0.;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real massfrac = q(i,j,k-1,QFS+n)+ 0.5*dq(i,j,k-1,QFS+n);
        massfrac = amrex::max(0., amrex::min(1.,massfrac));
        sum += massfrac;
        ql(i,j,k,QFS+n) = massfrac;
    }
    for (int n=0; n < NUM_SPECIES ; n++)
    {
        ql(i,j,k,QFS+n) = ql(i,j,k,QFS+n)/sum;
    }
#else
    Real sum = 0.;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real specden = q(i,j,k-1,QRHO)*q(i,j,k-1,QFS+n)+ 0.5*(dq(i,j,k-1,QFS+n)
            +q(i,j,k-1,QFS+n)*(dq(i,j,k-1,0)+dq(i,j,k-1,2))/cspeed);
        sum += specden;
        ql(i,j,k,QFS+n) = specden;
    }

    if (sum < small_dens) {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            ql(i,j,k,QFS+n) = q(i,j,k,QFS+n);
        }
    } else {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            ql(i,j,k,QFS+n) = ql(i,j,k,QFS+n)/sum;
        }
    }
    ql(i,j,k,QRHO) = amrex::max(sum, small_dens);
#endif


    ql(i,j,k,QREINT) = q(i,j,k-1,QREINT)+0.5*(dq(i,j,k-1,QREINT)+enth*(dq(i,j,k-1,0)+dq(i,j,k-1,2))/cspeed);



    cspeed = qa(i,j,k,QC);
    enth = (q(i,j,k,QREINT)+ q(i,j,k,QPRES))/ amrex::max(q(i,j,k,QRHO),small_dens);

#if 0
    Real rr = q(i,j,k,QRHO) - Real(0.5) * ( (dq(i,j,k,0)+dq(i,j,k,2))/cspeed + dq(i,j,k,1));
    rr = amrex::max(rr, small_dens);
    qr(i,j,k,QRHO) = rr;
#endif

    Real pr = q(i,j,k,QPRES) - Real(0.5) *  (dq(i,j,k,0)+dq(i,j,k,2))*cspeed;
    pr = amrex::max(pr, small_pres);
    qr(i,j,k,QPRES) = pr;

    qr(i,j,k,QW) = q(i,j,k,QW) - Real(0.5) * ( (dq(i,j,k,2)-dq(i,j,k,0))/q(i,j,k,QRHO));
    qr(i,j,k,QU) = q(i,j,k,QU) - Real(0.5) * dq(i,j,k,3);
#if (AMREX_SPACEDIM == 2)
    qr(i,j,k,QV) = 0.;
#else
    qr(i,j,k,QV) = q(i,j,k,QV) - Real(0.5) * dq(i,j,k,4);
#endif
#if (NUM_ADV > 0)
        for (int n=0; n < NUM_ADV; n++)
        {
           qr(i,j,k,QFA+n) = q(i,j,k,QFA+n) - Real(0.5) * dq(i,j,k,QFA+n);
        }
#endif
#if 0
    sum = 0.;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real massfrac = q(i,j,k,QFS+n) - 0.5*dq(i,j,k,QFS+n);
        massfrac = amrex::max(0., amrex::min(1.,massfrac));
        sum += massfrac;
        qr(i,j,k,QFS+n) = massfrac;
    }
    for (int n=0; n < NUM_SPECIES ; n++)
    {
        qr(i,j,k,QFS+n) = qr(i,j,k,QFS+n)/sum;
    }
#else
    sum = 0.;

    for (int n=0; n < NUM_SPECIES ; n++)
    {
        Real specden = q(i,j,k,QRHO)*q(i,j,k,QFS+n)- 0.5*(dq(i,j,k,QFS+n)
            +q(i,j,k,QFS+n)*(dq(i,j,k,0)+dq(i,j,k,2))/cspeed);
        sum += specden;
        qr(i,j,k,QFS+n) = specden;
    }

    if (sum < small_dens) {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            qr(i,j,k,QFS+n) = q(i,j,k,QFS+n);
        }
    } else {
        for (int n=0; n < NUM_SPECIES ; n++)
        {
            qr(i,j,k,QFS+n) = qr(i,j,k,QFS+n)/sum;
        }
    }
    qr(i,j,k,QRHO) = amrex::max(sum, small_dens);
#endif

    qr(i,j,k,QREINT) = q(i,j,k,QREINT) - 0.5*(dq(i,j,k,QREINT)+enth*(dq(i,j,k,0)+dq(i,j,k,2))/cspeed);

    int dir = 2;
    CAMR_cmpflx(i,j,k,bclo,bchi,domlo,domhi,ql,qr,fz,q3,qa,dir,pmap,small,small_dens,small_pres);
}
#endif

#endif
