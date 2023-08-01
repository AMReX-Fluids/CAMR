#ifndef GODUNOV_H
#define GODUNOV_H

#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>

#include "IndexDefines.H"

#ifdef AMREX_USE_EB
#include "AMReX_MultiFabUtil.H"
#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>
#endif

// Host Functions
#if AMREX_SPACEDIM == 3
void Godunov_umeth (
  amrex::Box const& bx,
  const int* bclo,
  const int* bchi,
  const int* domlo,
  const int* domhi,
  amrex::Array4<const amrex::Real> const& q,
  amrex::Array4<const amrex::Real> const& qaux,
  amrex::Array4<const amrex::Real> const& srcQ,
  amrex::Array4<amrex::Real> const& flx1,
  amrex::Array4<amrex::Real> const& flx2,
  amrex::Array4<amrex::Real> const& flx3,
  amrex::Array4<amrex::Real> const& q1,
  amrex::Array4<amrex::Real> const& q2,
  amrex::Array4<amrex::Real> const& q3,
  amrex::Array4<const amrex::Real> const& a1,
  amrex::Array4<const amrex::Real> const& a2,
  amrex::Array4<const amrex::Real> const& a3,
  amrex::Array4<amrex::Real> const& pdivu,
  amrex::Array4<const amrex::Real> const& vol,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> del,
  const amrex::Real dt,
  const amrex::Real small,
  const amrex::Real small_dens,
  const amrex::Real small_pres,
  const int ppm_type,
  const int use_pslope,
  const int use_flattening,
  const int iorder,
  const int transverse_reset_density);

#ifdef AMREX_USE_EB
void Godunov_umeth_eb (
  amrex::Box const& bx,
  const int* bclo,
  const int* bchi,
  const int* domlo,
  const int* domhi,
  amrex::Array4<const amrex::Real> const& q,
  amrex::Array4<const amrex::Real> const& qaux,
  amrex::Array4<const amrex::Real> const& srcQ,
  amrex::Array4<amrex::Real> const& flx1,
  amrex::Array4<amrex::Real> const& flx2,
  amrex::Array4<amrex::Real> const& flx3,
  amrex::Array4<amrex::Real> const& q1,
  amrex::Array4<amrex::Real> const& q2,
  amrex::Array4<amrex::Real> const& q3,
  amrex::Array4<const amrex::Real> const& a1,
  amrex::Array4<const amrex::Real> const& a2,
  amrex::Array4<const amrex::Real> const& a3,
  amrex::Array4<amrex::Real> const& pdivu,
  amrex::Array4<const amrex::Real> const& vfrac,
  amrex::Array4<amrex::EBCellFlag const> const& flag_arr,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> del,
  const amrex::Real dt,
  const amrex::Real small,
  const amrex::Real small_dens,
  const amrex::Real small_pres,
  const int ppm_type,
  const int use_pslope,
  const int use_flattening,
  const int iorder,
  const int transverse_reset_density);
#endif

#elif AMREX_SPACEDIM == 2

void Godunov_umeth (
  amrex::Box const& bx,
  const int* bclo,
  const int* bchi,
  const int* domlo,
  const int* domhi,
  amrex::Array4<const amrex::Real> const& q,
  amrex::Array4<const amrex::Real> const& qaux,
  amrex::Array4<const amrex::Real> const& srcQ,
  amrex::Array4<amrex::Real> const& flx1,
  amrex::Array4<amrex::Real> const& flx2,
  amrex::Array4<amrex::Real> const& q1,
  amrex::Array4<amrex::Real> const& q2,
  amrex::Array4<const amrex::Real> const& a1,
  amrex::Array4<const amrex::Real> const& a2,
  amrex::Array4<amrex::Real> const& pdivu,
  amrex::Array4<const amrex::Real> const& vol,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> del,
  const amrex::Real dt,
  const amrex::Real small,
  const amrex::Real small_dens,
  const amrex::Real small_pres,
  const int ppm_type,
  const int use_pslope,
  const int use_flattening,
  const int iorder,
  const int transverse_reset_density);

#ifdef AMREX_USE_EB
void Godunov_umeth_eb (
  amrex::Box const& bx,
  const int* bclo,
  const int* bchi,
  const int* domlo,
  const int* domhi,
  amrex::Array4<const amrex::Real> const& q,
  amrex::Array4<const amrex::Real> const& qaux,
  amrex::Array4<const amrex::Real> const& srcQ,
  amrex::Array4<amrex::Real> const& flx1,
  amrex::Array4<amrex::Real> const& flx2,
  amrex::Array4<amrex::Real> const& q1,
  amrex::Array4<amrex::Real> const& q2,
  amrex::Array4<const amrex::Real> const& a1,
  amrex::Array4<const amrex::Real> const& a2,
  amrex::Array4<amrex::Real> const& pdivu,
  amrex::Array4<const amrex::Real> const& vol,
  amrex::Array4<const amrex::Real> const& vfrac,
  amrex::Array4<amrex::EBCellFlag const> const& flag_arr,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> del,
  const amrex::Real dt,
  const amrex::Real small,
  const amrex::Real small_dens,
  const amrex::Real small_pres,
  const int ppm_type,
  const int use_pslope,
  const int use_flattening,
  const int iorder,
  const int transverse_reset_density);
#endif
#endif

#endif
