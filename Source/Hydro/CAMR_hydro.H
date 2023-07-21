#ifndef _HYDRO_H_
#define _HYDRO_H_

#include "CAMR.H"
#include "IndexDefines.H"

#ifdef AMREX_USE_EB
#include <AMReX_EBFluxRegister.H>
#endif

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_divu(
  const int i,
  const int j,
  const int k,
  amrex::Array4<const amrex::Real> const& q,
  AMREX_D_DECL(const amrex::Real dx,
               const amrex::Real dy,
               const amrex::Real dz),
  amrex::Array4<amrex::Real> const& divu,
  const int* domlo, const int* domhi,
  const int* bclo , const int* bchi)
{

#if (AMREX_SPACEDIM == 2)
    amrex::Real ux, vy;
    int bclo_x = bclo[0]; int domlo_x = domlo[0];
    int bclo_y = bclo[1]; int domlo_y = domlo[1];
    int bchi_x = bchi[0]; int domhi_x = domhi[0];
    int bchi_y = bchi[1]; int domhi_y = domhi[1];

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
            q(i  , j  , k, QW) - q(i  , j  , k-1, QW) +
            q(i  , j-1, k, QW) - q(i  , j-1, k-1, QW) ) / dz;
    } else {
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
    } else {
         ux = 0.25 * (
             q(i, j  , k  , QU) - q(i-1, j  , k  , QU) +
             q(i, j  , k-1, QU) - q(i-1, j  , k-1, QU) +
             q(i, j-1, k  , QU) - q(i-1, j-1, k  , QU) +
             q(i, j-1, k-1, QU) - q(i-1, j-1, k-1, QU)) / dx;
        wz = 0.25 * *
            q(i  , j  , k, QW) - q(i  , j  , k-1, QW) +
            q(i-1, j  , k, QW) - q(i-1, j  , k-1, QW) +
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
    } else {
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
    }
#endif

  divu(i, j, k) = AMREX_D_TERM(ux, +vy, +wz);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_norm_spec_flx(
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
CAMR_ext_flx(
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
CAMR_update(
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
      update(i, j, k, n) += (AMREX_D_TERM(
                               flx[0](i, j, k, n) - flx[0](i + 1, j, k, n),
                              +flx[1](i, j, k, n) - flx[1](i, j + 1, k, n),
                              +flx[2](i, j, k, n) - flx[2](i, j, k + 1, n))) * volinv;

    update(i, j, k, UEINT) -= pdivu(i, j, k);
}

void CAMR_umdrv( bool is_mol,
                 amrex::Box const& bx,
                 amrex::Geometry const& geom,
                 const int* bclo, const int* bchi,
                 amrex::Array4<const amrex::Real> const& uin,
                 amrex::Array4<amrex::Real> const& uout,
                 amrex::Array4<const amrex::Real> const& q,
                 amrex::Array4<const amrex::Real> const& qaux,
                 amrex::Array4<const amrex::Real> const& src_q,
                 const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                 const amrex::Real dt,
                 const int ppm_type,
                 const int use_pslope,
                 const int use_flattening,
                 const int transverse_reset_density,
                 const amrex::Real small,
                 const amrex::Real small_dens,
                 const amrex::Real small_pres,
                 const amrex::Real difmag,
                 const int slope_order,
                 const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
                 const amrex::GpuArray<const amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> a,
                 amrex::Array4<amrex::Real> const& vol);

void adjust_fluxes(
  amrex::Box const& bx,
  amrex::Array4<const amrex::Real> const& u_arr,
  const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
  const amrex::GpuArray<const amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> a,
  amrex::Array4<const amrex::Real> const& div,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> del,
  const int* domlo, const int* domhi, const int* bclo, const int* bchi,
  amrex::Real const difmag);

void CAMR_consup(
  amrex::Box const& bx,
  amrex::Array4<amrex::Real> const& update,
  const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
  amrex::Array4<const amrex::Real> const& vol,
  amrex::Array4<const amrex::Real> const& pdivu);

#ifdef AMREX_USE_EB
void
CAMR_umdrv_eb( amrex::Box const& bx, const amrex::MFIter& mfi,
               amrex::Geometry const& geom,
               const amrex::EBFArrayBoxFactory* ebfact,
               const int* bclo, const int* bchi,
               amrex::Array4<const amrex::Real> const& s_arr,
               amrex::Array4<const amrex::Real> const& q_arr,
               amrex::Array4<      amrex::Real> const& dsdt_arr,
               amrex::Array4<const amrex::Real> const& qaux,
               amrex::Array4<const amrex::Real> const& vfrac,
               amrex::Array4<amrex::EBCellFlag const> const& flag_arr,
               const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
               const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxInv,
               const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flux_arr,
               int as_crse,
               amrex::Array4<amrex::Real> const& drho_as_crse,
               amrex::Array4<int const> const& rrflag_as_crse,
               int as_fine,
               amrex::Array4<amrex::Real> const& dm_as_fine,
               amrex::Array4<int const> const& lev_mask,
               const amrex::Real difmag,
               const amrex::Real dt,
               const amrex::Real small,
               const amrex::Real small_dens,
               const amrex::Real small_pres,
               amrex::BCRec const* bcs_d_ptr,
               const std::string& l_redistribution_type,
               const int l_plm_iorder,
               const int l_eb_weights_type);

void
CAMR_consup_eb( const amrex::Box& bx,
                amrex::Array4<const amrex::Real> const& q_arr,
                amrex::Array4<const amrex::Real> const& qaux_arr,
                amrex::Array4<      amrex::Real> const& divc_arr,
                amrex::Array4<      amrex::Real> const& redistwgt_arr,
                AMREX_D_DECL(amrex::Array4<amrex::Real> const& q1,
                             amrex::Array4<amrex::Real> const& q2,
                             amrex::Array4<amrex::Real> const& q3),
                AMREX_D_DECL(
                amrex::Array4<amrex::Real       const> const& apx,
                amrex::Array4<amrex::Real       const> const& apy,
                amrex::Array4<amrex::Real       const> const& apz),
                AMREX_D_DECL(
                amrex::Array4<amrex::Real       const> const& fcx,
                amrex::Array4<amrex::Real       const> const& fcy,
                amrex::Array4<amrex::Real       const> const& fcz),
                amrex::Array4<const amrex::Real      > const& vfrac,
                amrex::Array4<amrex::EBCellFlag const> const& flag,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxinv,
                const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flux,
                const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flux_tmp,
                const amrex::Real small,
                const amrex::Real small_dens,
                const amrex::Real small_pres,
                const int l_eb_weights_type);
#endif
#endif
