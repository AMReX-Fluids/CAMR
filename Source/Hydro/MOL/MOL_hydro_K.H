#ifndef MOL_HYDRO_K_H_
#define MOL_HYDRO_K_H_

#include "IndexDefines.H"
#include "Hydro_cmpflx.H"
#include "Hydro_riemann.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real limiter (amrex::Real dlft, amrex::Real drgt, amrex::Real plm_theta) noexcept
{
    using amrex::Real;

    Real dcen = Real(0.5)*(dlft+drgt);
    Real dsgn = amrex::Math::copysign(Real(1.0), dcen);
    Real slop = plm_theta * amrex::min(amrex::Math::abs(dlft),amrex::Math::abs(drgt));
    Real dlim = (dlft*drgt >= Real(0.0)) ? slop : Real(0.0);
    return dsgn * amrex::min(dlim,amrex::Math::abs(dcen));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
mol_slope_x (int i, int j, int k,
             amrex::Array4<amrex::Real> const& dq,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& qa,
             int plm_iorder, amrex::Real plm_theta, amrex::Real small_dens) noexcept
{
    using amrex::Real;

    if (plm_iorder == 1)
    {
        for(int n=0; n<QVAR; n++)
          {
             dq(i,j,k,n) = 0.;
          }
    } else {
        Real dlft = 0.5*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QU) - q(i-1,j,k,QU));
        Real drgt = 0.5*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i+1,j,k,QU) - q(i,j,k,QU));
        dq(i,j,k,0) = limiter(dlft, drgt, plm_theta);

        Real cs2 = qa(i,j,k,QC)*qa(i,j,k,QC);
#if 0
        dlft = (q(i,j,k,QRHO)-q(i-1,j,k,QRHO)) - (q(i,j,k,QPRES) - q(i-1,j,k,QPRES))/cs2;
        drgt = (q(i+1,j,k,QRHO)-q(i,j,k,QRHO)) - (q(i+1,j,k,QPRES) - q(i,j,k,QPRES))/cs2;
        dq(i,j,k,1) = limiter(dlft, drgt, plm_theta);
#endif


        dlft = 0.5*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QU) - q(i-1,j,k,QU));
        drgt = 0.5*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i+1,j,k,QU) - q(i,j,k,QU));
        dq(i,j,k,2) = limiter(dlft, drgt, plm_theta);

        dlft = q(i,j,k,QV) - q(i-1,j,k,QV);
        drgt = q(i+1,j,k,QV) - q(i,j,k,QV);
        dq(i,j,k,3) = limiter(dlft, drgt, plm_theta);

#if (AMREX_SPACEDIM ==3 )
        dlft = q(i,j,k,QW) - q(i-1,j,k,QW);
        drgt = q(i+1,j,k,QW) - q(i,j,k,QW);
        dq(i,j,k,4) = limiter(dlft, drgt, plm_theta);
#else
        dq(i,j,k,4) = 0.;
#endif

        Real enth = (q(i,j,k,QREINT)+ q(i,j,k,QPRES))/ amrex::max(q(i,j,k,QRHO),small_dens);
        dlft = (q(i,j,k,QREINT) - q(i-1,j,k,QREINT))- enth*(q(i,j,k,QPRES)-q(i-1,j,k,QPRES))/cs2;
        drgt = (q(i+1,j,k,QREINT) - q(i,j,k,QREINT))- enth*(q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/cs2;
        dq(i,j,k,QREINT) = limiter(dlft, drgt, plm_theta);

#if (NUM_ADV > 0)
        for (int n=0; n<NUM_ADV; n++)
        {
           dlft = q(i,j,k,QFA+n) - q(i-1,j,k,QFA+n);
           drgt = q(i+1,j,k,QFA+n) - q(i,j,k,QFA+n);
           dq(i,j,k,QFA+n) = limiter(dlft, drgt, plm_theta);
        }
#endif

#if 0
        for (int n=0; n < NUM_SPECIES ; n++)
        {
           dlft = q(i,j,k,QFS+n) - q(i-1,j,k,QFS+n);
           drgt = q(i+1,j,k,QFS+n) - q(i,j,k,QFS+n);
           dq(i,j,k,QFS+n) = limiter(dlft, drgt, plm_theta);
        }
#else

        for (int n=0; n < NUM_SPECIES ; n++)
        {
           dlft = q(i,j,k,QRHO) * q(i,j,k,QFS+n) - q(i-1,j,k,QRHO)*q(i-1,j,k,QFS+n)
                 -q(i,j,k,QFS+n) * (q(i,j,k,QPRES)- q(i-1,j,k,QPRES))/cs2 ;
           drgt = q(i+1,j,k,QRHO)*q(i+1,j,k,QFS+n) - q(i,j,k,QRHO)*q(i,j,k,QFS+n)
                 -q(i,j,k,QFS+n) * (q(i+1,j,k,QPRES)-q(i,j,k,QPRES))/cs2 ;
           dq(i,j,k,QFS+n) = limiter(dlft, drgt, plm_theta);
        }
#endif

    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
mol_slope_y (int i, int j, int k,
             amrex::Array4<amrex::Real> const& dq,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& qa,
             int plm_iorder, amrex::Real plm_theta, amrex::Real small_dens) noexcept
{
    using amrex::Real;

    if (plm_iorder == 1)
    {
        for(int n=0; n<QVAR; n++)
          {
             dq(i,j,k,n) = 0.;
          }
    } else {

        Real dlft = Real(0.5)*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QV) - q(i,j-1,k,QV));
        Real drgt = Real(0.5)*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i,j+1,k,QV) - q(i,j,k,QV));
        dq(i,j,k,0) = limiter(dlft, drgt, plm_theta);

        Real cs2 = qa(i,j,k,QC)*qa(i,j,k,QC);
        dlft = (q(i,j,k,QRHO)-q(i,j-1,k,QRHO)) - (q(i,j,k,QPRES) - q(i,j-1,k,QPRES))/cs2;
        drgt = (q(i,j+1,k,QRHO)-q(i,j,k,QRHO)) - (q(i,j+1,k,QPRES) - q(i,j,k,QPRES))/cs2;
        dq(i,j,k,1) = limiter(dlft, drgt, plm_theta);

        dlft = Real(0.5)*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QV) - q(i,j-1,k,QV));
        drgt = Real(0.5)*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i,j+1,k,QV) - q(i,j,k,QV));
        dq(i,j,k,2) = limiter(dlft, drgt, plm_theta);

        dlft = q(i,j,k,QU) - q(i,j-1,k,QU);
        drgt = q(i,j+1,k,QU) - q(i,j,k,QU);
        dq(i,j,k,3) = limiter(dlft, drgt, plm_theta);

#if (AMREX_SPACEDIM ==3 )
        dlft = q(i,j,k,QW) - q(i,j-1,k,QW);
        drgt = q(i,j+1,k,QW) - q(i,j,k,QW);
        dq(i,j,k,4) =  limiter(dlft, drgt, plm_theta);
#else
        dq(i,j,k,4) =  0.;
#endif

        Real enth = (q(i,j,k,QREINT)+ q(i,j,k,QPRES))/ amrex::max(q(i,j,k,QRHO),small_dens);
        dlft = (q(i,j,k,QREINT) - q(i,j-1,k,QREINT))- enth*(q(i,j,k,QPRES)-q(i,j-1,k,QPRES))/cs2;
        drgt = (q(i,j+1,k,QREINT) - q(i,j,k,QREINT))- enth*(q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/cs2;
        dq(i,j,k,QREINT) = limiter(dlft, drgt, plm_theta);

#if (NUM_ADV > 0)
        for (int n=0; n<NUM_ADV; n++)
        {
           dlft = q(i,j,k,QFA+n) - q(i,j-1,k,QFA+n);
           drgt = q(i,j+1,k,QFA+n) - q(i,j,k,QFA+n);
           dq(i,j,k,QFA+n) = limiter(dlft, drgt, plm_theta);
        }
#endif
#if 0
        for (int n=0; n < NUM_SPECIES ; n++)
        {
           dlft = q(i,j,k,QFS+n) - q(i,j-1,k,QFS+n);
           drgt = q(i,j+1,k,QFS+n) - q(i,j,k,QFS+n);
           dq(i,j,k,QFS+n) = limiter(dlft, drgt, plm_theta);
        }
#else
        for (int n=0; n < NUM_SPECIES ; n++)
        {
           dlft = q(i,j,k,QRHO) * q(i,j,k,QFS+n) - q(i,j-1,k,QRHO)*q(i,j-1,k,QFS+n)
                 -q(i,j,k,QFS+n) * (q(i,j,k,QPRES)- q(i,j-1,k,QPRES))/cs2 ;
           drgt = q(i,j+1,k,QRHO)*q(i,j+1,k,QFS+n) - q(i,j,k,QRHO)*q(i,j,k,QFS+n)
                 -q(i,j,k,QFS+n) * (q(i,j+1,k,QPRES)-q(i,j,k,QPRES))/cs2 ;
           dq(i,j,k,QFS+n) = limiter(dlft, drgt, plm_theta);
        }
#endif

    }
}

#if (AMREX_SPACEDIM == 3)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
mol_slope_z (int i, int j, int k,
             amrex::Array4<amrex::Real> const& dq,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& qa,
             int plm_iorder, amrex::Real plm_theta, amrex::Real small_dens) noexcept
{
    using amrex::Real;

    if (plm_iorder == 1)
    {
        for(int n=0; n<QVAR; n++)
          {
             dq(i,j,k,n) = 0.;
          }
    } else {

        Real dlft = Real(0.5)*(q(i,j,k,QPRES)-q(i,j,k-1,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QW) - q(i,j,k-1,QW));
        Real drgt = Real(0.5)*(q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) - Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k+1,QW) - q(i,j,k,QW));
        dq(i,j,k,0) = limiter(dlft, drgt, plm_theta);

        Real cs2 = qa(i,j,k,QC)*qa(i,j,k,QC);
        dlft = (q(i,j,k,QRHO)-q(i,j,k-1,QRHO)) - (q(i,j,k,QPRES) - q(i,j,k-1,QPRES))/cs2;
        drgt = (q(i,j,k+1,QRHO)-q(i,j,k,QRHO)) - (q(i,j,k+1,QPRES) - q(i,j,k,QPRES))/cs2;
        dq(i,j,k,1) = limiter(dlft, drgt, plm_theta);

        dlft = Real(0.5)*(q(i,j,k,QPRES)-q(i,j,k-1,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k,QW) - q(i,j,k-1,QW));
        drgt = Real(0.5)*(q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/qa(i,j,k,QC) + Real(0.5)*q(i,j,k,QRHO)*(q(i,j,k+1,QW) - q(i,j,k,QW));
        dq(i,j,k,2) = limiter(dlft, drgt, plm_theta);

        dlft = q(i,j,k,QU) - q(i,j,k-1,QU);
        drgt = q(i,j,k+1,QU) - q(i,j,k,QU);
        dq(i,j,k,3) = limiter(dlft, drgt, plm_theta);

        dlft = q(i,j,k,QV) - q(i,j,k-1,QV);
        drgt = q(i,j,k+1,QV) - q(i,j,k,QV);
        dq(i,j,k,4) = limiter(dlft, drgt, plm_theta);


        Real enth = (q(i,j,k,QREINT)+ q(i,j,k,QPRES))/ amrex::max(q(i,j,k,QRHO),small_dens);
        dlft = (q(i,j,k,QREINT) - q(i,j,k-1,QREINT))- enth*(q(i,j,k,QPRES)-q(i,j,k-1,QPRES))/cs2;
        drgt = (q(i,j,k+1,QREINT) - q(i,j,k,QREINT))- enth*(q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/cs2;
        dq(i,j,k,QREINT) = limiter(dlft, drgt, plm_theta);

#if (NUM_ADV > 0)
        for (int n=0; n<NUM_ADV; n++)
        {
           dlft = q(i,j,k,QFA+n) - q(i,j,k-1,QFA+n);
           drgt = q(i,j,k+1,QFA+n) - q(i,j,k,QFA+n);
           dq(i,j,k,QFA+n) = limiter(dlft, drgt, plm_theta);
        }
#endif

#if 0
        for (int n=0; n < NUM_SPECIES ; n++)
        {
           dlft = q(i,j,k,QFS+n) - q(i,j,k-1,QFS+n);
           drgt = q(i,j,k+1,QFS+n) - q(i,j,k,QFS+n);
           dq(i,j,k,QFS+n) = limiter(dlft, drgt, plm_theta);
        }
#else
        for (int n=0; n < NUM_SPECIES ; n++)
        {
           dlft = q(i,j,k,QRHO) * q(i,j,k,QFS+n) - q(i,j,k-1,QRHO)*q(i,j,k-1,QFS+n)
                 -q(i,j,k,QFS+n) * (q(i,j,k,QPRES)- q(i,j,k-1,QPRES))/cs2 ;
           drgt = q(i,j,k+1,QRHO)*q(i,j,k+1,QFS+n) - q(i,j,k,QRHO)*q(i,j,k,QFS+n)
                 -q(i,j,k,QFS+n) * (q(i,j,k+1,QPRES)-q(i,j,k,QPRES))/cs2 ;
           dq(i,j,k,QFS+n) = limiter(dlft, drgt, plm_theta);
        }
#endif


    }
}
#endif
#endif
