#include <AMReX_GpuAllocators.H>
#include "Godunov.H"
#include "Godunov_utils_3D.H"
#include "Hydro_cmpflx.H"
#include "CAMR_utils_K.H"
#include "flatten.H"
#include "PLM.H"
#include "PPM.H"

#if (AMREX_SPACEDIM == 3)

using namespace amrex;

void
Godunov_umeth_eb (
  Box const& bx_to_fill,
  const int* bclo,
  const int* bchi,
  const int* domlo,
  const int* domhi,
  Array4<const Real> const& q,
  Array4<const Real> const& qaux,
  Array4<const Real> const& srcQ,
  Array4<Real> const& flx1,
  Array4<Real> const& flx2,
  Array4<Real> const& flx3,
  Array4<Real> const& q1,
  Array4<Real> const& q2,
  Array4<Real> const& q3,
  Array4<const Real> const& apx,
  Array4<const Real> const& apy,
  Array4<const Real> const& apz,
  Array4<EBCellFlag const> const& flag_arr,
  const GpuArray<Real, AMREX_SPACEDIM> del,
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
  BL_PROFILE("CAMR::Godunov_umeth_3D_eb()");

  Real const dx = del[0];
  Real const dy = del[1];
  Real const dz = del[2];
  Real const hdtdx = 0.5 * dt / dx;
  Real const hdtdy = 0.5 * dt / dy;
  Real const hdtdz = 0.5 * dt / dz;
  Real const cdtdx = 1.0 / 3.0 * dt / dx;
  Real const cdtdy = 1.0 / 3.0 * dt / dy;
  Real const cdtdz = 1.0 / 3.0 * dt / dz;
  Real const hdt = 0.5 * dt;

  const int bclx = bclo[0];
  const int bcly = bclo[1];
  const int bclz = bclo[2];
  const int bchx = bchi[0];
  const int bchy = bchi[1];
  const int bchz = bchi[2];
  const int dlx = domlo[0];
  const int dly = domlo[1];
  const int dlz = domlo[2];
  const int dhx = domhi[0];
  const int dhy = domhi[1];
  const int dhz = domhi[2];

  // auto const& bcMaskarr = bcMask.array();
  const Box& bxg1 = grow(bx_to_fill, 1);
  const Box& bxg2 = grow(bx_to_fill, 2);

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

  // Z data
  cdir = 2;
  const Box& zmbx = growHi(bxg2, cdir, 1);
  const Box& zflxbx = surroundingNodes(grow(bxg2, cdir, -1), cdir);
  FArrayBox qzm(zmbx, QVAR, amrex::The_Async_Arena());
  FArrayBox qzp(bxg2, QVAR, amrex::The_Async_Arena());
  auto const& qzmarr = qzm.array();
  auto const& qzparr = qzp.array();

  const PassMap* lpmap = CAMR::d_pass_map;

  // Put the PLM and slopes in the same kernel launch to avoid unnecessary
  // launch overhead
  if (ppm_type == 0) {
    ParallelFor(
      bxg2, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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

        // X slopes and interp
        int idir = 0;
        for (int n = 0; n < QVAR; ++n) {
          if (n == QPRES && use_pslope)
              slope[n] = plm_pslope_eb(i, j, k, n, 0, flag_arr, q, dx, srcQ, flat, iorder);
          else
              slope[n] = plm_slope_eb(i, j, k, n, 0, flag_arr, q, flat, iorder);
        }
        CAMR_plm_d(i, j, k, idir, qxmarr, qxparr, slope, q, qaux(i, j, k, QC), dx, dt,
                   small_dens, small_pres, *lpmap, apx);

        // Y slopes and interp
        idir = 1;
        for (int n = 0; n < QVAR; n++) {
          if (n == QPRES && use_pslope)
              slope[n] = plm_pslope_eb(i, j, k, n, 1, flag_arr, q, dy, srcQ, flat, iorder);
          else
              slope[n] = plm_slope_eb(i, j, k, n, 1, flag_arr, q, flat, iorder);
        }
        CAMR_plm_d(i, j, k, idir, qymarr, qyparr, slope, q, qaux(i, j, k, QC), dy, dt,
                   small_dens, small_pres, *lpmap, apy);

        // Z slopes and interp
        idir = 2;
        for (int n = 0; n < QVAR; ++n) {
          if (n == QPRES && use_pslope)
              slope[n] = plm_pslope_eb(i, j, k, n, 2, flag_arr, q, dz, srcQ, flat, iorder);
          else
              slope[n] = plm_slope_eb(i, j, k, n, 2, flag_arr, q, flat, iorder);
        }
        CAMR_plm_d(i, j, k, idir, qzmarr, qzparr, slope, q, qaux(i, j, k, QC), dz, dt,
                   small_dens, small_pres, *lpmap, apz);
      });
  } else if (ppm_type == 1) {

      // Compute the normal interface states by reconstructing
      // the primitive variables using the piecewise parabolic method
      // and doing characteristic tracing.  We do not apply the
      // transverse terms here.

      trace_ppm(bxg2, 0, q, qaux, srcQ, qxmarr, qxparr, bxg2, dt, del, use_flattening,
                small_dens, small_pres, lpmap);

      trace_ppm(bxg2, 1, q, qaux, srcQ, qymarr, qyparr, bxg2, dt, del, use_flattening,
                small_dens, small_pres, lpmap);

      trace_ppm(bxg2, 2, q, qaux, srcQ, qzmarr, qzparr, bxg2, dt, del, use_flattening,
                small_dens, small_pres, lpmap);

  } else {
      amrex::Error("CAMR::ppm_type must be 0 (PLM) or 1 (PPM)");
  }

  // These are the first flux estimates as per the corner-transport-upwind
  // method X initial fluxes
  cdir = 0;
  FArrayBox fx(xflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fxarr = fx.array();
  FArrayBox qgdx(xflxbx, NGDNV, amrex::The_Async_Arena());
  auto const& gdtempx = qgdx.array();
  ParallelFor(xflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    if (!flag_arr(i, j, k).isCovered() && !flag_arr(i - 1, j, k).isCovered()) {
      CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qxmarr, qxparr, fxarr, gdtempx,
                  qaux, cdir, *lpmap, small, small_dens, small_pres);
    }
  });

  // Y initial fluxes
  cdir = 1;
  FArrayBox fy(yflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fyarr = fy.array();
  FArrayBox qgdy(yflxbx, NGDNV, amrex::The_Async_Arena());
  auto const& gdtempy = qgdy.array();
  ParallelFor(yflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    if (!flag_arr(i, j, k).isCovered() && !flag_arr(i, j - 1, k).isCovered()) {
      CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qymarr, qyparr, fyarr, gdtempy,
                  qaux, cdir, *lpmap, small, small_dens, small_pres);
    }
  });

  // Z initial fluxes
  cdir = 2;
  FArrayBox fz(zflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fzarr = fz.array();
  FArrayBox qgdz(zflxbx, NGDNV, amrex::The_Async_Arena());
  auto const& gdtempz = qgdz.array();
  ParallelFor(zflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    if (!flag_arr(i, j, k).isCovered() && !flag_arr(i, j, k - 1).isCovered()) {
      CAMR_cmpflx(i, j, k, bclz, bchz, dlz, dhz, qzmarr, qzparr, fzarr, gdtempz,
                  qaux, cdir, *lpmap, small, small_dens, small_pres);
    }
  });

  // X interface corrections
  cdir = 0;
  const Box& txbx = grow(bxg1, cdir, 1);
  const Box& txbxm = growHi(txbx, cdir, 1);
  FArrayBox qxym(txbxm, QVAR, amrex::The_Async_Arena());
  FArrayBox qxyp(txbx, QVAR, amrex::The_Async_Arena());
  auto const& qmxy = qxym.array();
  auto const& qpxy = qxyp.array();

  FArrayBox qxzm(txbxm, QVAR, amrex::The_Async_Arena());
  FArrayBox qxzp(txbx, QVAR, amrex::The_Async_Arena());
  auto const& qmxz = qxzm.array();
  auto const& qpxz = qxzp.array();

  ParallelFor(txbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // X|Y
      CAMR_transdo(i, j, k, cdir, 1, qmxy, qpxy, qxmarr, qxparr, fyarr, qaux, gdtempy, cdtdy,
                   *lpmap, l_transverse_reset_density, small_pres, apx, apy);
      // X|Z
      CAMR_transdo(i, j, k, cdir, 2, qmxz, qpxz, qxmarr, qxparr, fzarr, qaux, gdtempz, cdtdz,
                   *lpmap, l_transverse_reset_density, small_pres, apx, apz);
  });

  const Box& txfxbx = surroundingNodes(bxg1, cdir);
  FArrayBox fluxxy(txfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox fluxxz(txfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox gdvxyfab(txfxbx, NGDNV, amrex::The_Async_Arena());
  FArrayBox gdvxzfab(txfxbx, NGDNV, amrex::The_Async_Arena());

  auto const& flxy = fluxxy.array();
  auto const& flxz = fluxxz.array();
  auto const& qxy = gdvxyfab.array();
  auto const& qxz = gdvxzfab.array();

  // Riemann problem X|Y X|Z
  ParallelFor(txfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // X|Y
      CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qmxy, qpxy, flxy, qxy, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
      // X|Z
      CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qmxz, qpxz, flxz, qxz, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
    });
  qxym.clear();
  qxyp.clear();
  qxzm.clear();
  qxzp.clear();

  // Y interface corrections
  cdir = 1;
  const Box& tybx = grow(bxg1, cdir, 1);
  const Box& tybxm = growHi(tybx, cdir, 1);
  FArrayBox qyxm(tybxm, QVAR, amrex::The_Async_Arena());
  FArrayBox qyxp(tybx, QVAR, amrex::The_Async_Arena());
  FArrayBox qyzm(tybxm, QVAR, amrex::The_Async_Arena());
  FArrayBox qyzp(tybx, QVAR, amrex::The_Async_Arena());
  auto const& qmyx = qyxm.array();
  auto const& qpyx = qyxp.array();
  auto const& qmyz = qyzm.array();
  auto const& qpyz = qyzp.array();

  ParallelFor(tybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Y|X
      CAMR_transdo(i, j, k, cdir, 0, qmyx, qpyx, qymarr, qyparr, fxarr, qaux, gdtempx, cdtdx,
                   *lpmap, l_transverse_reset_density, small_pres, apy, apx);
      // Y|Z
      CAMR_transdo(i, j, k, cdir, 2, qmyz, qpyz, qymarr, qyparr, fzarr, qaux, gdtempz, cdtdz,
                   *lpmap, l_transverse_reset_density, small_pres, apy, apz);
  });
  fz.clear();
  qgdz.clear();

  // Riemann problem Y|X Y|Z
  const Box& tyfxbx = surroundingNodes(bxg1, cdir);
  FArrayBox fluxyx(tyfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox fluxyz(tyfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox gdvyxfab(tyfxbx, NGDNV, amrex::The_Async_Arena());
  FArrayBox gdvyzfab(tyfxbx, NGDNV, amrex::The_Async_Arena());

  auto const& flyx = fluxyx.array();
  auto const& flyz = fluxyz.array();
  auto const& qyx = gdvyxfab.array();
  auto const& qyz = gdvyzfab.array();

  ParallelFor(tyfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Y|X
      CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qmyx, qpyx, flyx, qyx, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
      // Y|Z
      CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qmyz, qpyz, flyz, qyz, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
  });
  qyxm.clear();
  qyxp.clear();
  qyzm.clear();
  qyzp.clear();

  // Z interface corrections
  cdir = 2;
  const Box& tzbx = grow(bxg1, cdir, 1);
  const Box& tzbxm = growHi(tzbx, cdir, 1);
  FArrayBox qzxm(tzbxm, QVAR, amrex::The_Async_Arena());
  FArrayBox qzxp(tzbx, QVAR, amrex::The_Async_Arena());
  FArrayBox qzym(tzbxm, QVAR, amrex::The_Async_Arena());
  FArrayBox qzyp(tzbx, QVAR, amrex::The_Async_Arena());

  auto const& qmzx = qzxm.array();
  auto const& qpzx = qzxp.array();
  auto const& qmzy = qzym.array();
  auto const& qpzy = qzyp.array();

  ParallelFor(tzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Z|X
      CAMR_transdo(i, j, k, cdir, 0, qmzx, qpzx, qzmarr, qzparr, fxarr, qaux, gdtempx, cdtdx,
                   *lpmap, l_transverse_reset_density, small_pres, apz, apx);
      // Z|Y
      CAMR_transdo(i, j, k, cdir, 1, qmzy, qpzy, qzmarr, qzparr, fyarr, qaux, gdtempy, cdtdy,
                   *lpmap, l_transverse_reset_density, small_pres, apz, apy);
  });

  fx.clear();
  fy.clear();
  qgdx.clear();
  qgdy.clear();

  // Riemann problem Z|X Z|Y
  const Box& tzfxbx = surroundingNodes(bxg1, cdir);
  FArrayBox fluxzx(tzfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox fluxzy(tzfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox gdvzxfab(tzfxbx, NGDNV, amrex::The_Async_Arena());
  FArrayBox gdvzyfab(tzfxbx, NGDNV, amrex::The_Async_Arena());

  auto const& flzx = fluxzx.array();
  auto const& flzy = fluxzy.array();
  auto const& qzx = gdvzxfab.array();
  auto const& qzy = gdvzyfab.array();

  ParallelFor(tzfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Z|X
      CAMR_cmpflx(i, j, k, bclz, bchz, dlz, dhz, qmzx, qpzx, flzx, qzx, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
      // Z|Y
      CAMR_cmpflx(i, j, k, bclz, bchz, dlz, dhz, qmzy, qpzy, flzy, qzy, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
  });

  qzxm.clear();
  qzxp.clear();
  qzym.clear();
  qzyp.clear();

  // Temp Fabs for Final Fluxes
  FArrayBox qmfab(bxg2, QVAR, amrex::The_Async_Arena());
  FArrayBox qpfab(bxg1, QVAR, amrex::The_Async_Arena());
  auto const& qm = qmfab.array();
  auto const& qp = qpfab.array();

  // X | Y&Z
  cdir = 0;
  const Box& xfxbx = surroundingNodes(bx_to_fill, cdir);
  const Box& tyzbx = grow(bx_to_fill, cdir, 1);
  ParallelFor(tyzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      CAMR_transdd(i, j, k, cdir, qm, qp, qxmarr, qxparr, flyz, flzy, qyz, qzy, qaux, srcQ,
                   hdt, hdtdy, hdtdz, *lpmap, l_transverse_reset_density, small_pres, apx, apy, apz);
  });
  qxm.clear();
  qxp.clear();
  fluxyz.clear();
  gdvyzfab.clear();
  fluxzy.clear();
  gdvzyfab.clear();

  // Final X flux
  ParallelFor(xfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qm, qp, flx1, q1, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
  });

  // Y | X&Z
  cdir = 1;
  const Box& yfxbx = surroundingNodes(bx_to_fill, cdir);
  const Box& txzbx = grow(bx_to_fill, cdir, 1);
  ParallelFor(txzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      CAMR_transdd(i, j, k, cdir, qm, qp, qymarr, qyparr, flxz, flzx, qxz, qzx, qaux, srcQ,
                   hdt, hdtdx, hdtdz, *lpmap, l_transverse_reset_density, small_pres, apy, apx, apz);
  });
  qym.clear();
  qyp.clear();
  fluxxz.clear();
  gdvxzfab.clear();
  fluxzx.clear();
  gdvzxfab.clear();

  // Final Y flux
  ParallelFor(yfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qm, qp, flx2, q2, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
  });

  // Z | X&Y
  cdir = 2;
  const Box& zfxbx = surroundingNodes(bx_to_fill, cdir);
  const Box& txybx = grow(bx_to_fill, cdir, 1);
  ParallelFor(txybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      CAMR_transdd(i, j, k, cdir, qm, qp, qzmarr, qzparr, flxy, flyx, qxy, qyx, qaux, srcQ,
                   hdt, hdtdx, hdtdy, *lpmap, l_transverse_reset_density, small_pres, apz, apx, apy);
  });
  qzm.clear();
  qzp.clear();
  fluxxy.clear();
  gdvxyfab.clear();
  fluxyx.clear();
  gdvyxfab.clear();

  // Final Z flux
  ParallelFor(zfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      CAMR_cmpflx(i, j, k, bclz, bchz, dlz, dhz, qm, qp, flx3, q3, qaux,
                  cdir, *lpmap, small, small_dens, small_pres);
  });
  qmfab.clear();
  qpfab.clear();
}

#endif
