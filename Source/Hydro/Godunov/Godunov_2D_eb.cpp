#include "Godunov.H"
#include "Godunov_utils_2D.H"
#include "Hydro_cmpflx.H"
#include "CAMR_utils_K.H"
#include "flatten.H"
#include "PLM.H"
#include "PPM.H"

#if (AMREX_SPACEDIM == 2)

using namespace amrex;

void
Godunov_umeth_eb (
  Box const& bx_to_fill,
  const int* bclo,
  const int* bchi,
  const int* domlo,
  const int* domhi,
  Array4<const amrex::Real> const& q,
  Array4<const amrex::Real> const& qaux,
  Array4<const amrex::Real> const& srcQ,
  Array4<amrex::Real> const& flx1,
  Array4<amrex::Real> const& flx2,
  Array4<amrex::Real> const& q1,
  Array4<amrex::Real> const& q2,
  Array4<const amrex::Real> const& apx,
  Array4<const amrex::Real> const& apy,
  Array4<amrex::EBCellFlag const> const& flag_arr,
  const GpuArray<amrex::Real, AMREX_SPACEDIM> del,
  const Real dt,
  const Real small,
  const Real small_dens,
  const Real small_pres,
  const int ppm_type,
  const int use_pslope,
  const int use_flattening,
  const int iorder,
  const int l_transverse_reset_density)
{
  BL_PROFILE("CAMR::Godunov_umeth_2D_eb()");

  Real const dx = del[0];
  Real const dy = del[1];
  Real const hdt = 0.5 * dt;
  Real const hdtdy = 0.5 * dt / dy;
  Real const hdtdx = 0.5 * dt / dx;

  const int bclx = bclo[0];
  const int bcly = bclo[1];
  const int bchx = bchi[0];
  const int bchy = bchi[1];
  const int dlx = domlo[0];
  const int dly = domlo[1];
  const int dhx = domhi[0];
  const int dhy = domhi[1];

  const Box& bxg1 = grow(bx_to_fill,1);
  const Box& bxg2 = grow(bx_to_fill,2);

  // X data
  int cdir = 0;
  const Box& xmbx = growHi(bxg2, cdir, 1);
  const Box& xflxbx = surroundingNodes(grow(bxg2, cdir, -1), cdir);
  FArrayBox qxm(xmbx, QVAR, amrex::The_Async_Arena());
  FArrayBox qxp(bxg2, QVAR, amrex::The_Async_Arena());
  auto const& qxmarr = qxm.array();
  auto const& qxparr = qxp.array();

  // Y data
  cdir = 1;
  const Box& ymbx = growHi(bxg2, cdir, 1);
  const Box& yflxbx = surroundingNodes(grow(bxg2, cdir, -1), cdir);
  FArrayBox qym(ymbx, QVAR, amrex::The_Async_Arena());
  FArrayBox qyp(bxg2, QVAR, amrex::The_Async_Arena());
  auto const& qymarr = qym.array();
  auto const& qyparr = qyp.array();

  const PassMap* lpmap = CAMR::d_pass_map;

  if (ppm_type == 0)
  {
      ParallelFor(bxg2, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
      {
          if (flag_arr(i,j,k).isCovered()) {
              return;
          }

          Real slope[QVAR];

          Real flat = 1.0;
          // Calculate flattening in-place
          if (use_flattening == 1) {
            for (int dir_flat = 0; dir_flat < AMREX_SPACEDIM; dir_flat++) {
              flat = std::min(flat, flatten_eb(i, j, k, dir_flat, flag_arr, q));
            }
          }

          //
          // X slopes and interp
          //
          for (int n = 0; n < QVAR; ++n)
          {
            if (n == QPRES && use_pslope) {
                slope[n] = plm_pslope_eb(i, j, k, n, 0, flag_arr, q, dx, srcQ, flat, iorder);
            } else {
                slope[n] = plm_slope_eb(i, j, k, n, 0, flag_arr, q, flat, iorder);
            }
          }
          CAMR_plm_d(i, j, k, 0, qxmarr, qxparr, slope, q, qaux(i, j, k, QC), dx, dt,
                     small_dens, small_pres, *lpmap, apx);

          //
          // Y slopes and interp
          //
          for (int n = 0; n < QVAR; n++)
          {
              if (n == QPRES && use_pslope) {
                  slope[n] = plm_pslope_eb(i, j, k, n, 1, flag_arr, q, dy, srcQ, flat, iorder);
              } else {
                  slope[n] = plm_slope_eb(i, j, k, n, 1, flag_arr, q, flat, iorder);
              }
          }
          CAMR_plm_d(i, j, k, 1, qymarr, qyparr, slope, q, qaux(i, j, k, QC), dy, dt,
                     small_dens, small_pres, *lpmap, apy);
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

  // *******************************************************************************
  // These are the first flux estimates as per the corner-transport-upwind
  // method X initial fluxes
  // *******************************************************************************
  cdir = 0;
  FArrayBox fx(xflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fxarr = fx.array();
  FArrayBox qgdx(bxg2, NGDNV, amrex::The_Async_Arena());
  auto const& gdtemp = qgdx.array();
  ParallelFor(
    xflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
        if (!flag_arr(i,j,k).isCovered() && !flag_arr(i-1,j,k).isCovered()) {
            CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qxmarr, qxparr, fxarr, gdtemp, qaux,
                        cdir, *lpmap, small, small_dens, small_pres);
        }
    });

  // *******************************************************************************
  // These are the first flux estimates as per the corner-transport-upwind
  // method Y initial fluxes
  // *******************************************************************************
  cdir = 1;
  FArrayBox fy(yflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fyarr = fy.array();
  ParallelFor( yflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      if (!flag_arr(i,j,k).isCovered() && !flag_arr(i,j-1,k).isCovered()) {
          CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qymarr, qyparr, fyarr, q2, qaux,
                      cdir, *lpmap, small, small_dens, small_pres);
      }
  });

  // *******************************************************************************
  // X interface corrections
  // *******************************************************************************
  cdir = 0;
  const Box& tybx = grow(bx_to_fill, 1);
  FArrayBox qm(bxg2, QVAR, amrex::The_Async_Arena());
  FArrayBox qp(bxg1, QVAR, amrex::The_Async_Arena());
  auto const& qmarr = qm.array();
  auto const& qparr = qp.array();

  ParallelFor(tybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      if (!flag_arr(i,j,k).isCovered()) {
          CAMR_transd(i, j, k, cdir, qmarr, qparr, qxmarr, qxparr, fyarr, srcQ, qaux, q2, hdt,
                      hdtdy, *lpmap, l_transverse_reset_density, small_pres, apx, apy);
      }
  });

  // This box must be grown by one in transverse direction so we can
  // do tangential interpolation when taking divergence later
  const Box& xfxbx = surroundingNodes( grow(bx_to_fill, 1, 1-cdir), cdir);

  // Final Riemann problem X
  ParallelFor(xfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      if (!flag_arr(i,j,k).isCovered() && !flag_arr(i-1,j,k).isCovered()) {
          CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qmarr, qparr, flx1, q1, qaux,
                      cdir, *lpmap, small, small_dens, small_pres);
      }
  });

  // *******************************************************************************
  // Y interface corrections
  // *******************************************************************************
  cdir = 1;
  const Box& txbx = grow(bx_to_fill, 1);

  ParallelFor(txbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      if (!flag_arr(i,j,k).isCovered()) {
          CAMR_transd(i, j, k, cdir, qmarr, qparr, qymarr, qyparr, fxarr, srcQ, qaux, gdtemp,
                      hdt, hdtdx, *lpmap, l_transverse_reset_density, small_pres, apy, apx);
      }
  });

  // This box must be grown by one in transverse direction so we can
  // do tangential interpolation when taking divergence later
  const Box& yfxbx = surroundingNodes( grow(bx_to_fill, 1, 1-cdir), cdir);

  // Final Riemann problem Y
  ParallelFor(yfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      if (!flag_arr(i,j,k).isCovered() && !flag_arr(i,j-1,k).isCovered()) {
          CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qmarr, qparr, flx2, q2, qaux,
                      cdir, *lpmap, small, small_dens, small_pres);
      }
  });
}
#endif
