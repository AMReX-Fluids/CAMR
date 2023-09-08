#ifndef HYDRO_H
#define HYDRO_H

#include "IndexDefines.H"

#ifdef AMREX_USE_EB
#include <AMReX_EBFluxRegister.H>
#endif

void hydro_umdrv (bool do_mol,
                  amrex::Box const& bx,
                  amrex::Geometry const& geom,
                  const int* bclo, const int* bchi,
                  amrex::Array4<const amrex::Real> const& uin_arr,
                  amrex::Array4<amrex::Real> const& dsdt_arr,
                  amrex::Array4<const amrex::Real> const& q_arr,
                  amrex::Array4<const amrex::Real> const& qaux_arr,
                  amrex::Array4<const amrex::Real> const& src_q,
                  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                  const amrex::Real dt,
                  const int ppm_type,
                  const int plm_iorder,
                  const int use_pslope,
                  const int use_flattening,
                  const int transverse_reset_density,
                  const amrex::Real small,
                  const amrex::Real small_dens,
                  const amrex::Real small_pres,
                  const amrex::Real smallu,
                  const amrex::Real l_difmag,
                  const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
                  const amrex::GpuArray<const amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> a,
                  amrex::Array4<amrex::Real> const& vol,
                  const PassMap* lpmap);

void adjust_fluxes (
  amrex::Box const& bx,
  amrex::Array4<const amrex::Real> const& u_arr,
  const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
  const amrex::GpuArray<const amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> a,
  amrex::Array4<const amrex::Real> const& div,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> del,
  const int* domlo, const int* domhi, const int* bclo, const int* bchi,
  amrex::Real const difmag);

void hydro_consup (
  amrex::Box const& bx,
  amrex::Array4<amrex::Real> const& update,
  const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flx,
  amrex::Array4<const amrex::Real> const& vol,
  amrex::Array4<const amrex::Real> const& pdivu);

#ifdef AMREX_USE_EB
void
hydro_umdrv_eb (const bool do_mol, amrex::Box const& bx,
                amrex::Box const& bxg_i, const amrex::MFIter& mfi,
                amrex::Geometry const& geom,
                const amrex::EBFArrayBoxFactory* ebfact,
                const int* bclo, const int* bchi,
                amrex::Array4<const amrex::Real> const& uin_arr,
                amrex::Array4<      amrex::Real> const& dsdt_arr,
                amrex::Array4<const amrex::Real> const& q_arr,
                amrex::Array4<const amrex::Real> const& qaux_arr,
                amrex::Array4<const amrex::Real> const& src_q,
                amrex::Array4<const amrex::Real> const& vf_arr,
                amrex::Array4<amrex::EBCellFlag const> const& flag_arr,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxInv,
                const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM>& flux_arr,
                const amrex::GpuArray<const amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> a,
                int as_crse,
                amrex::Array4<amrex::Real> const& drho_as_crse,
                amrex::Array4<int const> const& rrflag_as_crse,
                int as_fine,
                amrex::Array4<amrex::Real> const& dm_as_fine, amrex::Array4<int const> const& lev_mask,
                const amrex::Real dt,
                const int ppm_type,
                const int plm_iorder,
                const int use_pslope,
                const int use_flattening,
                const int transverse_reset_density,
                const amrex::Real small,
                const amrex::Real small_dens,
                const amrex::Real small_pres,
                const amrex::Real smallu,
                const amrex::Real difmag,
                amrex::BCRec const* bcs_d_ptr,
                const std::string& l_redistribution_type,
                const PassMap* lpmap,
                const int l_eb_weights_type);

void adjust_fluxes_eb (
               const amrex::Box& bx,
               amrex::Array4<const amrex::Real> const& q_arr,
               amrex::Array4<const amrex::Real> const& u_arr,
               AMREX_D_DECL(amrex::Array4<amrex::Real const> const& apx,
                            amrex::Array4<amrex::Real const> const& apy,
                            amrex::Array4<amrex::Real const> const& apz ),
               amrex::Array4<const amrex::Real> const& vfrac,
               const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
               const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxInv,
               const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flux,
               const int* domlo, const int* domhi,
               const int*  bclo, const int*  bchi,
               const amrex::Real l_difmag);

void
hydro_consup_eb (const amrex::Box& bx,
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
                 const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flux_tmp,
                 const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flux,
                 const amrex::Real small,
                 const amrex::Real small_dens,
                 const amrex::Real small_pres,
                 const amrex::Real smallu,
                 const int l_eb_weights_type);
#endif
#endif
