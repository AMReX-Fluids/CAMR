#ifndef TAGGING_H
#define TAGGING_H

#include <AMReX_FArrayBox.H>
#include <AMReX_TagBox.H>
#include "prob_parm.H"

struct TaggingParm
{
  amrex::Real denerr = 1.0e10;
  amrex::Real dengrad = 1.0e10;
  amrex::Real presserr = 1.0e10;
  amrex::Real pressgrad = 1.0e10;
  amrex::Real velerr = 1.0e10;
  amrex::Real velgrad = 1.0e10;
  amrex::Real vorterr = 1.0e10;
  amrex::Real temperr = 1.0e10;
  amrex::Real tempgrad = 1.0e10;
  amrex::Real ftracerr = 1.0e10;
  amrex::Real ftracgrad = 1.0e10;
  amrex::Real vfracerr = 1.0e10;

  int max_denerr_lev = 10;
  int max_dengrad_lev = 10;
  int max_presserr_lev = 10;
  int max_pressgrad_lev = 10;
  int max_velerr_lev = 10;
  int max_velgrad_lev = 10;
  int max_vorterr_lev = 10;
  int max_temperr_lev = 10;
  int max_tempgrad_lev = 10;
  int max_ftracerr_lev = 10;
  int max_ftracgrad_lev = 10;
  int max_vfracerr_lev = 10;
};

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_error(
  const int i,
  const int j,
  const int k,
  amrex::Array4<char> const& tag,
  amrex::Array4<amrex::Real const> const& field,
  const amrex::Real fielderr,
  char tagval) noexcept
{
  // Tag on regions of high field values
  if (field(i, j, k) >= fielderr) {
    tag(i, j, k) = tagval;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_graderror(
  const int i,
  const int j,
  const int k,
  amrex::Array4<char> const& tag,
  amrex::Array4<amrex::Real const> const& field,
  const amrex::Real fieldgrad,
  char tagval) noexcept
{
  // Tag on regions of high field gradient
  AMREX_D_TERM(
      amrex::Real ax = std::abs(field(i + 1, j, k) - field(i, j, k));
                  ax = std::max(ax, std::abs(field(i, j, k) - field(i - 1, j, k)));
    , amrex::Real ay = std::abs(field(i, j + 1, k) - field(i, j, k));
                  ay = std::max(ay, std::abs(field(i, j, k) - field(i, j - 1, k)));
    , amrex::Real az = std::abs(field(i, j, k + 1) - field(i, j, k));
                  az = std::max(az, std::abs(field(i, j, k) - field(i, j, k - 1)));)
#if (AMREX_SPACEDIM == 2)
  if (std::max(ax, ay) >= fieldgrad)
#elif (AMREX_SPACEDIM == 3)
  if (std::max(std::max(ax, ay), az) >= fieldgrad)
#endif
  {
    tag(i, j, k) = tagval;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_abserror(
  const int i,
  const int j,
  const int k,
  amrex::Array4<char> const& tag,
  amrex::Array4<amrex::Real const> const& field,
  const amrex::Real fielderr,
  char tagval) noexcept
{
  // Tag on regions of high field values
  if (std::abs(field(i, j, k)) >= fielderr) {
    tag(i, j, k) = tagval;
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
tag_error_bounds(
  const int i,
  const int j,
  const int k,
  amrex::Array4<char> const& tag,
  amrex::Array4<amrex::Real const> const& field,
  const amrex::Real lbnd,
  const amrex::Real ubnd,
  char tagval) noexcept
{
  // Tag on regions inside bounds
  if ((lbnd < field(i, j, k)) && (field(i, j, k) < ubnd)) {
    tag(i, j, k) = tagval;
  }
}

struct EmptyProbTagStruct
{
  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void set_problem_tags(
    const int /*i*/,
    const int /*j*/,
    const int /*k*/,
    amrex::Array4<char> const& /*tag*/,
    amrex::Array4<amrex::Real const> const& /*field*/,
    char /*tagval*/,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> /*dx*/,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> /*prob_lo*/,
    const amrex::Real /*time*/,
    const int /*level*/,
    ProbParmDevice const& /*d_prob_parm_device*/) noexcept
  {
  }
};

template <typename ProbTagStruct, typename... Args>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
set_problem_tags(Args&&... args)
{
  ProbTagStruct::set_problem_tags(std::forward<Args>(args)...);
}

#endif
