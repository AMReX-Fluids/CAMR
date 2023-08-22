#ifndef MOL_HYDRO_EB_K_H
#define MOL_HYDRO_EB_K_H

#include "CAMR.H"
#include <AMReX_FArrayBox.H>
#include "IndexDefines.H"
#include <cmath>

namespace {

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real limiter_eb (amrex::Real dlft, amrex::Real drgt, amrex::Real plm_theta) noexcept
{
    using amrex::Real;

    Real dcen = Real(0.5)*(dlft+drgt);
    Real dsgn = amrex::Math::copysign(Real(1.0), dcen);
    Real slop = plm_theta * amrex::min(amrex::Math::abs(dlft),amrex::Math::abs(drgt));
    Real dlim = (dlft*drgt >= Real(0.0)) ? slop : Real(0.0);
    return dsgn * amrex::min(dlim,amrex::Math::abs(dcen));
}

}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
mol_slope_eb_x (int i, int j, int k,
                amrex::Array4<amrex::Real> const& dq,
                amrex::Array4<amrex::Real const> const& q,
                amrex::Array4<amrex::Real const> const& qa,
                amrex::Array4<amrex::EBCellFlag const> const& flag,
                const amrex::Real small_dens,
                int plm_iorder, amrex::Real plm_theta) noexcept
{
    using amrex::Real;

    Real dlft0,dlft1,dlft2,dlft3,dlft5;
    Real drgt0,drgt1,drgt2,drgt3,drgt5;
#if (AMREX_SPACEDIM == 3)
    Real dlft4;
    Real drgt4;
#endif

    if(plm_iorder == 1)
    {
      for(int n=0; n<QVAR; n++)
        {
             dq(i,j,k,n) = 0.;
        }

    } else {

        Real enth = (q(i,j,k,QREINT)+ q(i,j,k,QPRES))/ amrex::max(q(i,j,k,QRHO),small_dens);
        Real cs2 = qa(i,j,k,QC)*qa(i,j,k,QC);

        // The second test is needed here because outside the domain isConnected can be true
        //     even when the neighbor is covered
        bool left_ok = ( flag(i,j,k).isConnected(-1,0,0) && !flag(i-1,j,k).isCovered() );
        bool rght_ok = ( flag(i,j,k).isConnected( 1,0,0) && !flag(i+1,j,k).isCovered() );

        if (left_ok && rght_ok)
        {
            dlft0 = Real(0.5)*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QU) - q(i-1,j,k,QU));
            dlft1 = (q(i,j,k,QRHO)-q(i-1,j,k,QRHO)) - (q(i,j,k,QPRES) - q(i-1,j,k,QPRES))/cs2;
            dlft2 = Real(0.5)*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QU) - q(i-1,j,k,QU));
            dlft3 = q(i,j,k,QV) - q(i-1,j,k,QV);
#if (AMREX_SPACEDIM == 3)
            dlft4 = q(i,j,k,QW) - q(i-1,j,k,QW);
#endif
            dlft5 = (q(i,j,k,QREINT) - q(i-1,j,k,QREINT))- enth*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/cs2;

            drgt0 = Real(0.5)*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i+1,j,k,QU) - q(i,j,k,QU));
            drgt1 = (q(i+1,j,k,QRHO)-q(i,j,k,QRHO)) - (q(i+1,j,k,QPRES) - q(i,j,k,QPRES))/cs2;
            drgt2 = Real(0.5)*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i+1,j,k,QU) - q(i,j,k,QU));
            drgt3 = q(i+1,j,k,QV) - q(i,j,k,QV);
#if (AMREX_SPACEDIM == 3)
            drgt4 = q(i+1,j,k,QW) - q(i,j,k,QW);
#endif
            drgt5 = (q(i+1,j,k,QREINT) - q(i,j,k,QREINT))- enth*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/cs2;

            dq(i,j,k,0)      = limiter_eb(dlft0, drgt0, plm_theta);
            dq(i,j,k,1)      = limiter_eb(dlft1, drgt1, plm_theta);
            dq(i,j,k,2)      = limiter_eb(dlft2, drgt2, plm_theta);
            dq(i,j,k,3)      = limiter_eb(dlft3, drgt3, plm_theta);
            dq(i,j,k,QREINT) = limiter_eb(dlft5, drgt5, plm_theta);
#if (AMREX_SPACEDIM == 3)
            dq(i,j,k,4)      = limiter_eb(dlft4, drgt4, plm_theta);
#endif

#if (NUM_ADV > 0)
            for (int n=0; n<NUM_ADV; n++)
            {
               dlft0 = q(i  ,j,k,QFA+n) - q(i-1,j,k,QFA+n);
               drgt0 = q(i+1,j,k,QFA+n) - q(i  ,j,k,QFA+n);
               dq(i,j,k,QFA+n) = limiter_eb(dlft0, drgt0, plm_theta);
            }
#endif
#if 0
            for (int n=0; n < NUM_SPECIES ; n++)
            {
               dlft0 = q(i  ,j,k,QFS+n) - q(i-1,j,k,QFS+n);
               drgt0 = q(i+1,j,k,QFS+n) - q(i  ,j,k,QFS+n);
               dq(i,j,k,QFS+n) = limiter_eb(dlft0, drgt0, plm_theta);
            }
#else
            for (int n=0; n < NUM_SPECIES ; n++)
            {
              dlft0 = q(i,j,k,QRHO) * q(i,j,k,QFS+n) - q(i-1,j,k,QRHO)*q(i-1,j,k,QFS+n)
                    -q(i,j,k,QFS+n) * (q(i,j,k,QPRES)- q(i-1,j,k,QPRES))/cs2 ;
              drgt0 = q(i+1,j,k,QRHO)*q(i+1,j,k,QFS+n) - q(i,j,k,QRHO)*q(i,j,k,QFS+n)
                    -q(i,j,k,QFS+n) * (q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/cs2 ;

               dq(i,j,k,QFS+n) = limiter_eb(dlft0, drgt0, plm_theta);
            }
#endif

        // Can't reach to both sides so we zero the slopes
        } else {
            dq(i,j,k,0)      = Real(0.);
            dq(i,j,k,1)      = Real(0.);
            dq(i,j,k,2)      = Real(0.);
            dq(i,j,k,3)      = Real(0.);
#if (AMREX_SPACEDIM == 3)
            dq(i,j,k,4)      = Real(0.);
#endif
            dq(i,j,k,QREINT) = Real(0.);

            for (int n=0; n < NUM_SPECIES ; n++)
            {
               dq(i,j,k,QFS+n) = Real(0.0);
            }
#if (NUM_ADV > 0)
            for (int n=0; n<NUM_ADV; n++)
            {
               dq(i,j,k,QFA+n) = Real(0.0);
            }
#endif
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
mol_slope_eb_y (int i, int j, int k,
                amrex::Array4<amrex::Real> const& dq,
                amrex::Array4<amrex::Real const> const& q,
                amrex::Array4<amrex::Real const> const& qa,
                amrex::Array4<amrex::EBCellFlag const> const& flag,
                const amrex::Real small_dens,
                int plm_iorder, amrex::Real plm_theta) noexcept
{
    using amrex::Real;

    Real dlft0,dlft1,dlft2,dlft3,dlft5;
    Real drgt0,drgt1,drgt2,drgt3,drgt5;
#if (AMREX_SPACEDIM == 3)
    Real dlft4;
    Real drgt4;
#endif

    if (plm_iorder == 1)
    {
        for (int n=0; n<QVAR; n++) {
            dq(i,j,k,n) = 0.;
        }

    } else {

        Real enth = (q(i,j,k,QREINT)+ q(i,j,k,QPRES))/ amrex::max(q(i,j,k,QRHO),small_dens);
        Real cs2 = qa(i,j,k,QC)*qa(i,j,k,QC);

        // The second test is needed here because outside the domain isConnected can be true
        //     even when the neighbor is covered
        bool left_ok = ( flag(i,j,k).isConnected(0,-1,0) && !flag(i,j-1,k).isCovered() );
        bool rght_ok = ( flag(i,j,k).isConnected(0, 1,0) && !flag(i,j+1,k).isCovered() );

        if (left_ok && rght_ok) {

            dlft0 = Real(0.5)*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QV) - q(i,j-1,k,QV));
            dlft1 = (q(i,j,k,QRHO)-q(i,j-1,k,QRHO)) - (q(i,j,k,QPRES) - q(i,j-1,k,QPRES))/cs2;
            dlft2 = Real(0.5)*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QV) - q(i,j-1,k,QV));
            dlft3 =q(i,j,k,QU) - q(i,j-1,k,QU);
#if (AMREX_SPACEDIM == 3)
            dlft4 = q(i,j,k,QW) - q(i,j-1,k,QW);
#endif
            dlft5 = (q(i,j,k,QREINT) - q(i,j-1,k,QREINT))- enth*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/cs2;

            drgt0 = Real(0.5)*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i,j+1,k,QV) - q(i,j,k,QV));
            drgt1 = (q(i,j+1,k,QRHO)-q(i,j,k,QRHO)) - (q(i,j+1,k,QPRES) - q(i,j,k,QPRES))/cs2;
            drgt2 = Real(0.5)*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i,j+1,k,QV) - q(i,j,k,QV));
            drgt3 = q(i,j+1,k,QU) - q(i,j,k,QU);
#if (AMREX_SPACEDIM == 3)
            drgt4 = q(i,j+1,k,QW) - q(i,j,k,QW);
#endif
            drgt5 = (q(i,j+1,k,QREINT) - q(i,j,k,QREINT))- enth*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/cs2;

            dq(i,j,k,0)      = limiter_eb(dlft0, drgt0, plm_theta);
            dq(i,j,k,1)      = limiter_eb(dlft1, drgt1, plm_theta);
            dq(i,j,k,2)      = limiter_eb(dlft2, drgt2, plm_theta);
            dq(i,j,k,3)      = limiter_eb(dlft3, drgt3, plm_theta);
            dq(i,j,k,QREINT) = limiter_eb(dlft5, drgt5, plm_theta);
#if (AMREX_SPACEDIM == 3)
            dq(i,j,k,4)      = limiter_eb(dlft4, drgt4, plm_theta);
#endif

#if (NUM_ADV > 0)
            for (int n=0; n<NUM_ADV; n++)
            {
               dlft0 = q(i,j  ,k,QFA+n) - q(i,j-1,k,QFA+n);
               drgt0 = q(i,j+1,k,QFA+n) - q(i,j  ,k,QFA+n);
               dq(i,j,k,QFA+n) = limiter_eb(dlft0, drgt0, plm_theta);
            }
#endif

#if 0
            for (int n=0; n < NUM_SPECIES ; n++)
            {
               dlft0 = q(i,j  ,k,QFS+n) - q(i,j-1,k,QFS+n);
               drgt0 = q(i,j+1,k,QFS+n) - q(i,j  ,k,QFS+n);
               dq(i,j,k,QFS+n) = limiter_eb(dlft0, drgt0, plm_theta);
            }
#else
            for (int n=0; n < NUM_SPECIES ; n++)
            {
               dlft0 = q(i,j,k,QRHO) * q(i,j,k,QFS+n) - q(i,j-1,k,QRHO)*q(i,j-1,k,QFS+n)
                     -q(i,j,k,QFS+n) * (q(i,j,k,QPRES)- q(i,j-1,k,QPRES))/cs2 ;
               drgt0 = q(i,j+1,k,QRHO)*q(i,j+1,k,QFS+n) - q(i,j,k,QRHO)*q(i,j,k,QFS+n)
                     -q(i,j,k,QFS+n) * (q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/cs2 ;

               dq(i,j,k,QFS+n) = limiter_eb(dlft0, drgt0, plm_theta);
            }
#endif

        // Can't reach to both sides so we zero the slopes
        } else {

            dq(i,j,k,0)      = Real(0.);
            dq(i,j,k,1)      = Real(0.);
            dq(i,j,k,2)      = Real(0.);
            dq(i,j,k,3)      = Real(0.);
#if (AMREX_SPACEDIM == 3)
            dq(i,j,k,4)      = Real(0.);
#endif
            dq(i,j,k,QREINT) = Real(0.);

            for (int n=0; n < NUM_SPECIES ; n++)
            {
               dq(i,j,k,QFS+n) = Real(0.0);
            }
#if (NUM_ADV > 0)
            for (int n=0; n<NUM_ADV; n++)
            {
               dq(i,j,k,QFA+n) = Real(0.0);
            }
#endif
        } // left_ok && rght_ok?
   } // not 1st order
}

#if (AMREX_SPACEDIM == 3)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
mol_slope_eb_z (int i, int j, int k,
                amrex::Array4<amrex::Real> const& dq,
                amrex::Array4<amrex::Real const> const& q,
                amrex::Array4<amrex::Real const> const& qa,
                amrex::Array4<amrex::EBCellFlag const> const& flag,
                const amrex::Real small_dens,
                int plm_iorder, amrex::Real plm_theta) noexcept
{
    using amrex::Real;

    Real dlft0,dlft1,dlft2,dlft3,dlft4,dlft5;
    Real drgt0,drgt1,drgt2,drgt3,drgt4,drgt5;

    if (plm_iorder == 1)
    {
        for (int n=0; n<QVAR; n++) {
             dq(i,j,k,n) = 0.;
        }

    } else {

        Real enth = (q(i,j,k,QREINT)+ q(i,j,k,QPRES))/ amrex::max(q(i,j,k,QRHO),small_dens);
        Real cs2 = qa(i,j,k,QC)*qa(i,j,k,QC);

        // The second test is needed here because outside the domain isConnected can be true
        //     even when the neighbor is covered
        bool left_ok = ( flag(i,j,k).isConnected(0,0,-1) && !flag(i,j,k-1).isCovered() );
        bool rght_ok = ( flag(i,j,k).isConnected(0,0, 1) && !flag(i,j,k+1).isCovered() );

        if (left_ok && rght_ok)
        {
            dlft0 = Real(0.5)*(q(i,j,k,QPRES)-q(i,j,k-1,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QW) - q(i,j,k-1,QW));
            dlft1 = (q(i,j,k,QRHO)-q(i,j,k-1,QRHO)) - (q(i,j,k,QPRES) - q(i,j,k-1,QPRES))/cs2;
            dlft2 = Real(0.5)*(q(i,j,k,QPRES)-q(i,j,k-1,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QW) - q(i,j,k-1,QW));
            dlft3 = q(i,j,k,QU) - q(i,j,k-1,QU);
            dlft4 = q(i,j,k,QV) - q(i,j,k-1,QV);
            dlft5 = (q(i,j,k,QREINT) - q(i,j,k-1,QREINT))- enth*(q(i,j,k,QPRES)-q(i,j,k-1,QPRES))/cs2;

            drgt0 = Real(0.5)*(q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k+1,QW) - q(i,j,k,QW));
            drgt1 = (q(i,j,k+1,QRHO)-q(i,j,k,QRHO)) - (q(i,j,k+1,QPRES) - q(i,j,k,QPRES))/cs2;
            drgt2 = Real(0.5)*(q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k+1,QW) - q(i,j,k,QW));
            drgt3 = q(i,j,k+1,QU) - q(i,j,k,QU);
            drgt4 = q(i,j,k+1,QV) - q(i,j,k,QV);
            drgt5 = (q(i,j,k+1,QREINT) - q(i,j,k,QREINT))- enth*(q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/cs2;

            dq(i,j,k,0) = limiter_eb(dlft0, drgt0, plm_theta);
            dq(i,j,k,1) = limiter_eb(dlft1, drgt1, plm_theta);
            dq(i,j,k,2) = limiter_eb(dlft2, drgt2, plm_theta);
            dq(i,j,k,3) = limiter_eb(dlft3, drgt3, plm_theta);
            dq(i,j,k,4) = limiter_eb(dlft4, drgt4, plm_theta);
            dq(i,j,k,QREINT) = limiter_eb(dlft5, drgt5, plm_theta);

#if (NUM_ADV > 0)
            for (int n=0; n<NUM_ADV; n++)
            {
                dlft0 = q(i,j,k  ,QFA+n) - q(i,j,k-1,QFA+n);
                drgt0 = q(i,j,k+1,QFA+n) - q(i,j,k  ,QFA+n);
                dq(i,j,k,QFA+n) = limiter_eb(dlft0, drgt0, plm_theta);
            }
#endif

#if 0
            for (int n=0; n < NUM_SPECIES ; n++)
            {
               dlft0 = q(i,j,k  ,QFS+n) - q(i,j,k-1,QFS+n);
               drgt0 = q(i,j,k+1,QFS+n) - q(i,j,k  ,QFS+n);
               dq(i,j,k,QFS+n) = limiter_eb(dlft0, drgt0, plm_theta);
            }
#else
            for (int n=0; n < NUM_SPECIES ; n++)
            {
               dlft0 = q(i,j,k,QRHO) * q(i,j,k,QFS+n) - q(i,j,k-1,QRHO)*q(i,j,k-1,QFS+n)
                     -q(i,j,k,QFS+n) * (q(i,j,k,QPRES)- q(i,j,k-1,QPRES))/cs2 ;
               drgt0 = q(i,j,k+1,QRHO)*q(i,j,k+1,QFS+n) - q(i,j,k,QRHO)*q(i,j,k,QFS+n)
                     -q(i,j,k,QFS+n) * (q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/cs2 ;

               dq(i,j,k,QFS+n) = limiter_eb(dlft0, drgt0, plm_theta);
            }
#endif

        // Can't reach to both sides so we zero the slopes
        } else {

            dq(i,j,k,0)      = Real(0.0);
            dq(i,j,k,1)      = Real(0.0);
            dq(i,j,k,2)      = Real(0.0);
            dq(i,j,k,3)      = Real(0.0);
            dq(i,j,k,4)      = Real(0.0);
            dq(i,j,k,QREINT) = Real(0.0);

            for (int n=0; n < NUM_SPECIES ; n++)
            {
               dq(i,j,k,QFS+n) = Real(0.0);
            }
#if (NUM_ADV > 0)
            for (int n=0; n<NUM_ADV; n++)
            {
               dq(i,j,k,QFA+n) = Real(0.0);
            }
#endif
        } // left_ok and rght_ok??
    } // not 1st order
}
#endif
#endif
