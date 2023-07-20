#include <AMReX_Config.H>
#include <AMReX_REAL.H>
#include "IndexDefines.H"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
eb_divu (int i, int j, int k,
         amrex::Array4<amrex::Real const> const& q,
          amrex::Array4<amrex::Real> const& divu,
          amrex::Array4<amrex::Real const> const& vfrac,
          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxInv)
{
#if (AMREX_SPACEDIM == 2)
    amrex::Real surrounding_vol = vfrac(i,j  ,k  ) + vfrac(i-1,j  ,k  ) +
                                  vfrac(i,j-1,k  ) + vfrac(i-1,j-1,k  );
    amrex::Real cutoff = 4.0 * (1.0 - 1.e-12);
#elif (AMREX_SPACEDIM == 3)
    amrex::Real surrounding_vol = vfrac(i,j  ,k  ) + vfrac(i-1,j  ,k  ) +
                                  vfrac(i,j-1,k  ) + vfrac(i-1,j-1,k  ) +
                                  vfrac(i,j  ,k-1) + vfrac(i-1,j  ,k-1) +
                                  vfrac(i,j-1,k-1) + vfrac(i-1,j-1,k-1);
    amrex::Real cutoff = 8.0 * (1.0 - 1.e-12);
#endif

    divu(i, j, k) = 0.0;

    if (surrounding_vol >= cutoff) {

#if (AMREX_SPACEDIM == 2)
        amrex::Real ux = 0.5 *
                         (q(i, j, k, QU) - q(i - 1, j, k, QU) + q(i, j - 1, k, QU) -
                          q(i - 1, j - 1, k, QU)) * dxInv[0];
        amrex::Real vy = 0.5 *
                         (q(i, j, k, QV) - q(i, j - 1, k, QV) + q(i - 1, j, k, QV) -
                          q(i - 1, j - 1, k, QV)) * dxInv[1];
#elif (AMREX_SPACEDIM == 3)
        amrex::Real ux = 0.25 *
           (q(i, j, k, QU) - q(i - 1, j, k, QU) + q(i, j, k - 1, QU) -
           q(i - 1, j, k - 1, QU) + q(i, j - 1, k, QU) - q(i - 1, j - 1, k, QU) +
           q(i, j - 1, k - 1, QU) - q(i - 1, j - 1, k - 1, QU)) * dxInv[0];

        amrex::Real vy = 0.25 *
          (q(i, j, k, QV) - q(i, j - 1, k, QV) + q(i, j, k - 1, QV) -
           q(i, j - 1, k - 1, QV) + q(i - 1, j, k, QV) - q(i - 1, j - 1, k, QV) +
           q(i - 1, j, k - 1, QV) - q(i - 1, j - 1, k - 1, QV)) * dxInv[1];

        amrex::Real wz = 0.25 *
          (q(i, j, k, QW) - q(i, j, k - 1, QW) + q(i - 1, j, k, QW) -
           q(i - 1, j, k - 1, QW) + q(i, j - 1, k, QW) - q(i, j - 1, k - 1, QW) +
           q(i - 1, j - 1, k, QW) - q(i - 1, j - 1, k - 1, QW)) * dxInv[2];
#endif
      divu(i, j, k) = AMREX_D_TERM(ux, +vy, +wz);
    }
}
