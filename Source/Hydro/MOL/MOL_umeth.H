#ifndef MOLHYDRO_H
#define MOLHYDRO_H

#include "IndexDefines.H"

#ifdef AMREX_USE_EB
#include <AMReX_EBMultiFabUtil.H>
#endif

void MOL_umeth (
               const amrex::Box& bx,
               const int* bclo, const int* bchi,
               const int* domlo, const int* domhi,
               amrex::Array4<const amrex::Real> const& q,
               amrex::Array4<const amrex::Real> const& qa,
               AMREX_D_DECL(
                    amrex::Array4<amrex::Real> const& flx1,
                    amrex::Array4<amrex::Real> const& flx2,
                    amrex::Array4<amrex::Real> const& flx3),
               AMREX_D_DECL(
                    amrex::Array4<amrex::Real> const& q1,
                    amrex::Array4<amrex::Real> const& q2,
                    amrex::Array4<amrex::Real> const& q3),
               AMREX_D_DECL(
                    amrex::Array4<const amrex::Real> const& a1,
                    amrex::Array4<const amrex::Real> const& a2,
                    amrex::Array4<const amrex::Real> const& a3),
               amrex::Array4<amrex::Real> const& pdivu,
               amrex::Array4<const amrex::Real> const& vol,
               const amrex::Real small,
               const amrex::Real small_dens,
               const amrex::Real small_pres,
               const amrex::Real smallu,
               const int slope_order,
               const PassMap* lpmap);

#ifdef AMREX_USE_EB
void MOL_umeth_eb (
               const amrex::Box& bx_to_fill,
               const int* bclo, const int* bchi,
               const int* domlo, const int* domhi,
               amrex::Array4<const amrex::Real> const& q_arr,
               amrex::Array4<const amrex::Real> const& qaux_arr,
               AMREX_D_DECL(amrex::Array4<amrex::Real> const& q1,
                            amrex::Array4<amrex::Real> const& q2,
                            amrex::Array4<amrex::Real> const& q3),
               amrex::Array4<amrex::EBCellFlag const> const& flag_arr,
               const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
               const amrex::GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flux_tmp_arr,
               const amrex::Real small,
               const amrex::Real small_dens,
               const amrex::Real small_pres,
               const amrex::Real smallu,
               const int l_plm_iorder,
               const PassMap* lpmap,
               const int l_eb_weights_type);
#endif
#endif
