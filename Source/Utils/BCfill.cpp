#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

#include "CAMR.H"
#include "prob.H"

struct PCHypFillExtDir
{
  ProbParmDevice const* lprobparm;

  AMREX_GPU_HOST
  constexpr explicit PCHypFillExtDir(const ProbParmDevice* d_prob_parm)
    : lprobparm(d_prob_parm)
  {
  }

  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& dest,
    const int dcomp,
    const int numcomp,
    amrex::GeometryData const& geomdata,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int /*bcomp*/,
    const int orig_comp) const
  {
    const int* domlo = geomdata.Domain().loVect();
    const int* domhi = geomdata.Domain().hiVect();
    const auto& prob_lo = geomdata.ProbLo();
    const auto& dx = geomdata.CellSize();
    const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
      prob_lo[0] + static_cast<amrex::Real>(iv[0] + 0.5) * dx[0],
      prob_lo[1] + static_cast<amrex::Real>(iv[1] + 0.5) * dx[1],
      prob_lo[2] + static_cast<amrex::Real>(iv[2] + 0.5) * dx[2])};

    const int* bc = bcr->data();

    // We may be asked to fill only a subset of the state -- FillPatch of a
    // single derived component, for instance -- so we must write only
    // components [dcomp, dcomp+numcomp) of dest, which correspond to state
    // components [orig_comp, orig_comp+numcomp).
    AMREX_ASSERT(dcomp + numcomp <= dest.nComp());
    AMREX_ASSERT(orig_comp + numcomp <= NVAR);

    // Note that on a partial fill the entries of s_int outside the range being
    // filled are left at zero, since we have no way to read them here.  That is
    // exact as long as bcnormal treats the components independently, which is
    // true of every problem setup in Exec; a state-coupled bcnormal would need
    // the full state.
    amrex::Real s_int[NVAR] = {0.0};
    amrex::Real s_ext[NVAR] = {0.0};

    // xlo and xhi
    int idir = 0;
    if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domlo[idir], iv[1], iv[2]));
      for (int n = 0; n < numcomp; n++) {
        s_int[orig_comp + n] = dest(loc, dcomp + n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geomdata, *lprobparm);
      for (int n = 0; n < numcomp; n++) {
        dest(iv, dcomp + n) = s_ext[orig_comp + n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domhi[idir], iv[1], iv[2]));
      for (int n = 0; n < numcomp; n++) {
        s_int[orig_comp + n] = dest(loc, dcomp + n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geomdata, *lprobparm);
      for (int n = 0; n < numcomp; n++) {
        dest(iv, dcomp + n) = s_ext[orig_comp + n];
      }
    }
#if AMREX_SPACEDIM > 1
    // ylo and yhi
    idir = 1;
    if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domlo[idir], iv[2]));
      for (int n = 0; n < numcomp; n++) {
        s_int[orig_comp + n] = dest(loc, dcomp + n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geomdata, *lprobparm);
      for (int n = 0; n < numcomp; n++) {
        dest(iv, dcomp + n) = s_ext[orig_comp + n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domhi[idir], iv[2]));
      for (int n = 0; n < numcomp; n++) {
        s_int[orig_comp + n] = dest(loc, dcomp + n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geomdata, *lprobparm);
      for (int n = 0; n < numcomp; n++) {
        dest(iv, dcomp + n) = s_ext[orig_comp + n];
      }
    }
#if AMREX_SPACEDIM == 3
    // zlo and zhi
    idir = 2;
    if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
      for (int n = 0; n < numcomp; n++) {
        s_int[orig_comp + n] = dest(iv[0], iv[1], domlo[idir], dcomp + n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geomdata, *lprobparm);
      for (int n = 0; n < numcomp; n++) {
        dest(iv, dcomp + n) = s_ext[orig_comp + n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
      (iv[idir] > domhi[idir])) {
      for (int n = 0; n < numcomp; n++) {
        s_int[orig_comp + n] = dest(iv[0], iv[1], domhi[idir], dcomp + n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geomdata, *lprobparm);
      for (int n = 0; n < numcomp; n++) {
        dest(iv, dcomp + n) = s_ext[orig_comp + n];
      }
    }
#endif
#endif
  }
};

void
CAMR_bcfill_hyp(
  amrex::Box const& bx,
  amrex::FArrayBox& data,
  const int dcomp,
  const int numcomp,
  amrex::Geometry const& geom,
  const amrex::Real time,
  const amrex::Vector<amrex::BCRec>& bcr,
  const int bcomp,
  const int scomp)
{
  const ProbParmDevice* lprobparm = CAMR::d_prob_parm;
  amrex::GpuBndryFuncFab<PCHypFillExtDir> hyp_bndry_func(
    PCHypFillExtDir{lprobparm});
  hyp_bndry_func(bx, data, dcomp, numcomp, geom, time, bcr, bcomp, scomp);
}

void
CAMR_nullfill(
  amrex::Box const& /*bx*/,
  amrex::FArrayBox& /*data*/,
  const int /*dcomp*/,
  const int /*numcomp*/,
  amrex::Geometry const& /*geom*/,
  const amrex::Real /*time*/,
  const amrex::Vector<amrex::BCRec>& /*bcr*/,
  const int /*bcomp*/,
  const int /*scomp*/)
{
}
