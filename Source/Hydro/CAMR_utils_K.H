#ifndef CAMR_UTILS_H
#define CAMR_UTILS_H

#include "CAMR.H"
#include "IndexDefines.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_divu (
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& q,
  AMREX_D_DECL(const amrex::Real dx,
               const amrex::Real dy,
               const amrex::Real dz),
  amrex::Array4<amrex::Real> const& divu,
  amrex::GpuArray<int,AMREX_SPACEDIM> const& domlo,
  amrex::GpuArray<int,AMREX_SPACEDIM> const& domhi,
  amrex::GpuArray<int,AMREX_SPACEDIM> const& bclo,
  amrex::GpuArray<int,AMREX_SPACEDIM> const& bchi)
{
    int bclo_x = bclo[0]; int domlo_x = domlo[0];
    int bclo_y = bclo[1]; int domlo_y = domlo[1];
    int bchi_x = bchi[0]; int domhi_x = domhi[0];
    int bchi_y = bchi[1]; int domhi_y = domhi[1];

#if (AMREX_SPACEDIM == 2)
    amrex::Real ux, vy;

    if (j == domlo_y && bclo_y == NoSlipWall) {
        ux = ( q(i,j  ,k,QU) - q(i-1,j  ,k,QU) ) / dx;
    } else if (j == domhi_y+1 && bchi_y == NoSlipWall) {
        ux = (q(i,j-1,k,QU) - q(i-1,j-1,k,QU)) / dx;
    } else {
        ux = 0.5 * (q(i,j  ,k,QU) - q(i-1,j  ,k,QU)
                 +  q(i,j-1,k,QU) - q(i-1,j-1,k,QU)) / dx;
    }

    if (i == domlo_x && bclo_x == NoSlipWall) {
      vy = ( q(i  ,j,k,QV) - q(i  ,j-1,k,QV) ) / dy;
    } else if (i == domhi_x+1 && bchi_x == NoSlipWall) {
      vy = ( q(i-1,j,k,QV) - q(i-1,j-1,k,QV) ) / dy;
    } else {
      vy = 0.5 * ( q(i  , j, k, QV) - q(i  ,j-1,k,QV) +
                   q(i-1, j, k, QV) - q(i-1,j-1,k,QV) ) / dy;
    }

#elif (AMREX_SPACEDIM == 3)
    amrex::Real ux, vy, wz;

    int bclo_z = bclo[2]; int domlo_z = domlo[2];
    int bchi_z = bchi[2]; int domhi_z = domhi[2];

    ux = 0.25 * (
        q(i, j  , k  , QU) - q(i-1, j  , k  , QU) +
        q(i, j  , k-1, QU) - q(i-1, j  , k-1, QU) +
        q(i, j-1, k  , QU) - q(i-1, j-1, k  , QU) +
        q(i, j-1, k-1, QU) - q(i-1, j-1, k-1, QU)) / dx;
    vy = 0.25 * (
        q(i  , j, k  , QV) - q(i  , j-1, k  , QV) +
        q(i  , j, k-1, QV) - q(i  , j-1, k-1, QV) +
        q(i-1, j, k  , QV) - q(i-1, j-1, k  , QV) +
        q(i-1, j, k-1, QV) - q(i-1, j-1, k-1, QV)) / dy;
    wz = 0.25 * (
        q(i  , j  , k, QW) - q(i  , j  , k-1, QW) +
        q(i-1, j  , k, QW) - q(i-1, j  , k-1, QW) +
        q(i  , j-1, k, QW) - q(i  , j-1, k-1, QW) +
        q(i-1, j-1, k, QW) - q(i-1, j-1, k-1, QW)) / dz;

    if (i == domlo_x && bclo_x == NoSlipWall) {
        vy = 0.5 * (
            q(i  , j, k  , QV) - q(i  , j-1, k  , QV) +
            q(i  , j, k-1, QV) - q(i  , j-1, k-1, QV) ) / dy;
        wz = 0.5 * (
            q(i  , j  , k, QW) - q(i  , j  , k-1, QW) +
            q(i  , j-1, k, QW) - q(i  , j-1, k-1, QW) ) / dz;
    } else if (i == domhi_x+1 && bchi_x == NoSlipWall) {
        vy = 0.5 * (
            q(i-1, j, k  , QV) - q(i-1, j-1, k  , QV) +
            q(i-1, j, k-1, QV) - q(i-1, j-1, k-1, QV) ) / dy;
        wz = 0.5 * (
            q(i-1, j  , k, QW) - q(i-1, j  , k-1, QW) +
            q(i-1, j-1, k, QW) - q(i-1, j-1, k-1, QW) ) / dz;
    }

    if (j == domlo_y && bclo_y == NoSlipWall) {
        ux = 0.5 * (
            q(i, j  , k  , QU) - q(i-1, j  , k  , QU) +
            q(i, j  , k-1, QU) - q(i-1, j  , k-1, QU) ) / dx;
        wz = 0.5 * (
            q(i  , j  , k, QW) - q(i  , j  , k-1, QW) +
            q(i-1, j  , k, QW) - q(i-1, j  , k-1, QW) ) / dz;
    } else if (j == domhi_y+1 && bchi_y == NoSlipWall) {
        ux = 0.5 * (
            q(i, j-1, k  , QU) - q(i-1, j-1, k  , QU) +
            q(i, j-1, k-1, QU) - q(i-1, j-1, k-1, QU) ) / dx;
        wz = 0.5 * (
            q(i  , j-1, k, QW) - q(i  , j-1, k-1, QW) +
            q(i-1, j-1, k, QW) - q(i-1, j-1, k-1, QW)) / dz;
    }

    if (k == domlo_z && bclo_z == NoSlipWall) {
        ux = 0.5 * (
            q(i, j  , k  , QU) - q(i-1, j  , k  , QU) +
            q(i, j-1, k  , QU) - q(i-1, j-1, k  , QU) ) / dx;
        vy = 0.5 * (
            q(i  , j, k  , QV) - q(i  , j-1, k  , QV) +
            q(i-1, j, k  , QV) - q(i-1, j-1, k  , QV) ) / dy;
    } else if (k == domhi_z+1 && bchi_z == NoSlipWall) {
        ux = 0.5 * (
            q(i, j  , k-1, QU) - q(i-1, j  , k-1, QU) +
            q(i, j-1, k-1, QU) - q(i-1, j-1, k-1, QU) ) / dx;
        vy = 0.5 * (
            q(i  , j, k-1, QV) - q(i  , j-1, k-1, QV) +
            q(i-1, j, k-1, QV) - q(i-1, j-1, k-1, QV)) / dy;
    }

    // Fix ux at y-z corners
    if (j == domlo_y && bclo_y == NoSlipWall &&
        k == domlo_z && bclo_z == NoSlipWall) {
        ux = ( q(i  , j  , k, QU) - q(i-1, j  , k  , QU) ) / dx;
    } else if ( (j == domhi_y+1 && bchi_y == NoSlipWall) &&
                (k == domlo_z   && bclo_z == NoSlipWall) ) {
        ux = ( q(i  , j-1, k, QU) - q(i-1, j-1, k  , QU) ) / dx;
    } else if ( (j == domlo_y   && bclo_y == NoSlipWall) &&
                (k == domhi_z+1 && bchi_z == NoSlipWall) ) {
        ux = ( q(i  , j  , k-1, QU) - q(i-1, j  , k-1, QU) )/ dx;
    } else if ( (j == domhi_y+1 && bchi_y == NoSlipWall) &&
                (k == domhi_z+1 && bchi_z == NoSlipWall) ) {
        ux = ( q(i  , j-1, k-1, QU) - q(i-1, j-1, k-1, QU) )/ dx;
    }

    // Fix vy at x-z corners
    if (i == domlo_x && bclo_x == NoSlipWall &&
        k == domlo_z && bclo_z == NoSlipWall) {
        vy = ( q(i  , j  , k, QV) - q(i  , j-1, k  , QV) ) / dy;
    } else if ( (i == domhi_x+1 && bchi_x == NoSlipWall) &&
                (k == domlo_z   && bclo_z == NoSlipWall) ) {
        vy = ( q(i-1, j  , k, QV) - q(i-1, j-1, k  , QV) ) / dy;
    } else if ( (i == domlo_x   && bclo_x == NoSlipWall) &&
                (k == domhi_z+1 && bchi_z == NoSlipWall) ) {
        vy = ( q(i  , j  , k-1, QV) - q(i  , j-1, k-1, QV) )/ dy;
    } else if ( (i == domhi_x+1 && bchi_x == NoSlipWall) &&
                (k == domhi_z+1 && bchi_z == NoSlipWall) ) {
        vy = ( q(i-1, j  , k-1, QV) - q(i-1, j-1, k-1, QV) )/ dy;
    }

    // Fix wz at x-y corners
    if (i == domlo_x && bclo_x == NoSlipWall &&
        j == domlo_y && bclo_y == NoSlipWall) {
        wz = ( q(i  , j  , k, QW) - q(i  , j  , k-1, QW) ) / dz;
    } else if ( (i == domhi_x+1 && bchi_x == NoSlipWall) &&
                (j == domlo_y   && bclo_y == NoSlipWall) ) {
        wz = ( q(i-1, j  , k, QW) - q(i-1, j  , k-1, QW) ) / dz;
    } else if ( (i == domlo_x   && bclo_x == NoSlipWall) &&
                (j == domhi_y+1 && bchi_y == NoSlipWall) ) {
        wz = ( q(i  , j-1, k, QW) - q(i  , j-1, k-1, QW) )/ dz;
    } else if ( (i == domhi_x+1 && bchi_x == NoSlipWall) &&
                (j == domhi_y+1 && bchi_y == NoSlipWall) ) {
        wz = ( q(i-1, j-1, k, QW) - q(i-1, j-1, k-1, QW) )/ dz;
    }
#endif

  divu(i, j, k) = AMREX_D_TERM(ux, +vy, +wz);
}

// Use interface states from Riemann solver for pdivu.
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_pdivu (
    const int i,
    const int j,
    const int k,
    amrex::Array4<amrex::Real> const& pdivu,
    AMREX_D_DECL(
      amrex::Array4<const amrex::Real> const& q1,
      amrex::Array4<const amrex::Real> const& q2,
      amrex::Array4<const amrex::Real> const& q3),
    AMREX_D_DECL(
      amrex::Array4<const amrex::Real> const& a1,
      amrex::Array4<const amrex::Real> const& a2,
      amrex::Array4<const amrex::Real> const& a3),
    amrex::Array4<const amrex::Real> const& vol)
{
    amrex::ignore_unused(k);
    pdivu(i, j, k) = 0.5 *
                   (AMREX_D_TERM(
                     (q1(i + 1, j, k, GDPRES) + q1(i, j, k, GDPRES)) *
                     (q1(i + 1, j, k, GDU)    * a1(i + 1, j, k) -
                      q1(i    , j, k, GDU)    * a1(i, j, k)),
                    +(q2(i, j + 1, k, GDPRES) + q2(i, j, k, GDPRES)) *
                     (q2(i, j + 1, k, GDV)    * a2(i, j + 1, k) -
                      q2(i, j    , k, GDV)    * a2(i, j, k)),
                    +(q3(i, j, k + 1, GDPRES) + q3(i, j, k, GDPRES)) *
                     (q3(i, j, k + 1, GDW) * a3(i, j, k + 1) -
                      q3(i, j, k    , GDW) * a3(i, j, k)))) / vol(i, j, k);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_norm_spec_flx (
  const int i, const int j, const int k, amrex::Array4<amrex::Real> const& flx)
{
  const int num = NUM_SPECIES + UFS;
  amrex::Real summ, fac;
  summ = 0.0;
  for (int n = UFS; n < num; ++n)
    summ += flx(i, j, k, n);

  fac = (summ != 0.0) ? flx(i, j, k, URHO) / summ : 1.0;
  for (int n = UFS; n < num; ++n)
    flx(i, j, k, n) *= fac;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_ext_flx (
  const int i,
  const int j,
  const int k,
  amrex::Array4<amrex::Real> const& flx,
  amrex::Array4<const amrex::Real> const& a)
{
  amrex::Real area = a(i, j, k);
  for (int n = 0; n < NVAR; ++n)
    flx(i, j, k, n) *= area;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_update (
    const int i,
    const int j,
    const int k,
    amrex::Array4<amrex::Real> const& update,
    const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
    amrex::Array4<const amrex::Real> const& vol,
    amrex::Array4<const amrex::Real> const& pdivu)
{
    amrex::Real volinv = 1.0/vol(i, j, k);
    for (int n = 0; n < NVAR; ++n) {
      update(i, j, k, n) +=
        (AMREX_D_TERM(flx[0](i, j, k, n) - flx[0](i + 1, j, k, n),
                     +flx[1](i, j, k, n) - flx[1](i, j + 1, k, n),
                     +flx[2](i, j, k, n) - flx[2](i, j, k + 1, n))) *
        volinv;
    }
    update(i, j, k, UEINT) -= pdivu(i, j, k);
}
#endif
