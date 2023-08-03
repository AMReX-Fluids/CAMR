#include "Godunov.H"
#include "CAMR_utils_K.H"
#include "Godunov_utils_2D.H"
#include "Hydro_cmpflx.H"
#include "flatten.H"
#include "PLM.H"
#include "PPM.H"

#if (AMREX_SPACEDIM == 2)
void
Godunov_umeth (
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
  amrex::Array4<const amrex::Real> const& ax,
  amrex::Array4<const amrex::Real> const& ay,
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
  const int l_transverse_reset_density)
{
  BL_PROFILE("CAMR::Godunov_umeth()");
  amrex::Real const dx = del[0];
  amrex::Real const dy = del[1];
  amrex::Real const hdt = 0.5 * dt;
  amrex::Real const hdtdy = 0.5 * dt / dy;
  amrex::Real const hdtdx = 0.5 * dt / dx;

  const int bclx = bclo[0];
  const int bcly = bclo[1];
  const int bchx = bchi[0];
  const int bchy = bchi[1];
  const int dlx = domlo[0];
  const int dly = domlo[1];
  const int dhx = domhi[0];
  const int dhy = domhi[1];

  const amrex::Box& bxg1 = grow(bx, 1);
  const amrex::Box& bxg2 = grow(bx, 2);

  // X data
  int cdir = 0;
  const amrex::Box& xmbx = growHi(bxg2, cdir, 1);
  const amrex::Box& xflxbx = surroundingNodes(grow(bxg2, cdir, -1), cdir);
  amrex::FArrayBox qxm(xmbx, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qxp(bxg2, QVAR, amrex::The_Async_Arena());
  auto const& qxmarr = qxm.array();
  auto const& qxparr = qxp.array();

  // Y data
  cdir = 1;
  const amrex::Box& ymbx = growHi(bxg2, cdir, 1);
  const amrex::Box& yflxbx = surroundingNodes(grow(bxg2, cdir, -1), cdir);
  amrex::FArrayBox qym(ymbx, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qyp(bxg2, QVAR, amrex::The_Async_Arena());
  auto const& qymarr = qym.array();
  auto const& qyparr = qyp.array();

  const PassMap* lpmap = CAMR::d_pass_map;
  if (ppm_type == 0) {
    amrex::ParallelFor(
      bxg2, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

        amrex::Real slope[QVAR];

        amrex::Real flat = 1.0;
        // Calculate flattening in-place
        if (use_flattening == 1) {
          for (int dir_flat = 0; dir_flat < AMREX_SPACEDIM; dir_flat++) {
            flat = std::min(flat, flatten(i, j, k, dir_flat, q));
          }
        }

        // X slopes and interp

        for (int n = 0; n < QVAR; ++n)
        {
          if (n == QPRES && use_pslope)
              slope[n] = plm_pslope(i, j, k, n, 0, q, dx, srcQ, flat, iorder);
          else
              slope[n] = plm_slope(i, j, k, n, 0, q, flat, iorder);
        }
        CAMR_plm_d(
          i, j, k, 0, qxmarr, qxparr, slope, q, qaux(i, j, k, QC), dx, dt,
          small_dens, small_pres, *lpmap);

        // Y slopes and interp
        for (int n = 0; n < QVAR; n++)
        {
          if (n == QPRES && use_pslope)
              slope[n] = plm_pslope(i, j, k, n, 1, q, dy, srcQ, flat, iorder);
          else
              slope[n] = plm_slope(i, j, k, n, 1, q, flat, iorder);
        }
        CAMR_plm_d(
          i, j, k, 1, qymarr, qyparr, slope, q, qaux(i, j, k, QC), dy, dt,
          small_dens, small_pres, *lpmap);
      });

  } else if (ppm_type == 1) {

    // Compute the normal interface states by reconstructing
    // the primitive variables using the piecewise parabolic method
    // and doing characteristic tracing.  We do not apply the
    // transverse terms here.

    int idir = 0;
    trace_ppm(
      bxg2, idir, q, qaux, srcQ, qxmarr, qxparr, bxg2, dt, del, use_flattening,
      small_dens, small_pres, lpmap);

    idir = 1;
    trace_ppm(
      bxg2, idir, q, qaux, srcQ, qymarr, qyparr, bxg2, dt, del, use_flattening,
      small_dens, small_pres, lpmap);

  } else {
    amrex::Error("CAMR::ppm_type must be 0 (PLM) or 1 (PPM)");
  }

  // These are the first flux estimates as per the corner-transport-upwind
  // method X initial fluxes
  cdir = 0;
  amrex::FArrayBox fx(xflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fxarr = fx.array();
  amrex::FArrayBox qgdx(bxg2, NGDNV, amrex::The_Async_Arena());
  auto const& gdtemp = qgdx.array();
  amrex::ParallelFor(xflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qxmarr, qxparr, fxarr, gdtemp, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
  });

  // Y initial fluxes
  cdir = 1;
  amrex::FArrayBox fy(yflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fyarr = fy.array();
  amrex::ParallelFor( yflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qymarr, qyparr, fyarr, q2, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
  });

  // X interface corrections
  cdir = 0;
  const amrex::Box& tybx = grow(bx, cdir, 1);
  amrex::FArrayBox qm(bxg2, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qp(bxg1, QVAR, amrex::The_Async_Arena());
  auto const& qmarr = qm.array();
  auto const& qparr = qp.array();

  amrex::ParallelFor(tybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      CAMR_transd(i, j, k, cdir, qmarr, qparr, qxmarr, qxparr, fyarr, srcQ, qaux, q2, hdt,
                  hdtdy, *lpmap, l_transverse_reset_density, small_pres);
  });

  const amrex::Box& xfxbx = surroundingNodes(bx, cdir);

  // Final Riemann problem X
  amrex::ParallelFor(xfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qmarr, qparr, flx1, q1, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
  });

  // Y interface corrections
  cdir = 1;
  const amrex::Box& txbx = grow(bx, cdir, 1);

  amrex::ParallelFor(txbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      CAMR_transd(i, j, k, cdir, qmarr, qparr, qymarr, qyparr, fxarr, srcQ, qaux, gdtemp,
                  hdt, hdtdx, *lpmap, l_transverse_reset_density, small_pres);
  });

  // Final Riemann problem Y
  const amrex::Box& yfxbx = surroundingNodes(bx, cdir);
  amrex::ParallelFor(yfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qmarr, qparr, flx2, q2, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
  });

  // Construct p div{U}
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    CAMR_pdivu(i, j, k, pdivu, q1, q2, ax, ay, vol);
  });
}
#endif
