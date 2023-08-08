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

 int cdir;

  Real const dx = del[0];
  Real const dy = del[1];
  Real const dz = del[2];

  Real const hdt = Real(0.5) * dt;
  Real const hdtdx = hdt / dx;
  Real const hdtdy = hdt / dy;
  Real const hdtdz = hdt / dz;
  Real const cdtdx = Real(1.0 / 3.0) * dt / dx;
  Real const cdtdy = Real(1.0 / 3.0) * dt / dy;
  Real const cdtdz = Real(1.0 / 3.0) * dt / dz;

  const int bclx =  bclo[0]; const int bcly =  bclo[1]; const int bclz =  bclo[2];
  const int bchx =  bchi[0]; const int bchy =  bchi[1]; const int bchz =  bchi[2];
  const int  dlx = domlo[0]; const int  dly = domlo[1]; const int  dlz = domlo[2];
  const int  dhx = domhi[0]; const int  dhy = domhi[1]; const int  dhz = domhi[2];

  const Box& bxg1 = grow(bx_to_fill, 1);
  const Box& bxg2 = grow(bx_to_fill, 2);

  // X data
  cdir = 0;
  const Box& xmbx = growHi(bxg2, cdir, 1);
  FArrayBox qxm(xmbx, QVAR, amrex::The_Async_Arena()); auto const& qxmarr = qxm.array();
  FArrayBox qxp(bxg2, QVAR, amrex::The_Async_Arena()); auto const& qxparr = qxp.array();

  // Y data
  cdir = 1;
  const Box& ymbx = growHi(bxg2, cdir, 1);
  FArrayBox qym(ymbx, QVAR, amrex::The_Async_Arena()); auto const& qymarr = qym.array();
  FArrayBox qyp(bxg2, QVAR, amrex::The_Async_Arena()); auto const& qyparr = qyp.array();

  // Z data
  cdir = 2;
  const Box& zmbx = growHi(bxg2, cdir, 1);
  FArrayBox qzm(zmbx, QVAR, amrex::The_Async_Arena()); auto const& qzmarr = qzm.array();
  FArrayBox qzp(bxg2, QVAR, amrex::The_Async_Arena()); auto const& qzparr = qzp.array();


  const PassMap* lpmap = CAMR::d_pass_map;
  //
  // Put the PLM and slopes in the same kernel launch to avoid unnecessary launch overhead
  // Note that we compute the qm and qp values on bxg2
  // -5,-5,-5
  //
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
        // cells -5,-5,-5
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
        // cells -5,-5,-5
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
        // cells -5,-5,-5
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


  // These are the first flux estimates as per the corner-transport-upwind method

  // *************************************************************************************
  // X initial fluxes
  // *************************************************************************************
  cdir = 0;
  const Box& xflxbx = surroundingNodes(grow(bxg2, cdir, -1), cdir);

  FArrayBox fx(xflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fxarr = fx.array();

  FArrayBox qgdx(xflxbx, NGDNV, amrex::The_Async_Arena());
  auto const& gdtempx = qgdx.array();

  // -4,-5,-5
  ParallelFor(xflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    if (apx(i,j,k) > 0.) {
      CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qxmarr, qxparr, fxarr, gdtempx,
                  qaux, cdir, *lpmap, small, small_dens, small_pres);
    }
  });

  // *************************************************************************************
  // Y initial fluxes
  // *************************************************************************************
  cdir = 1;
  const Box& yflxbx = surroundingNodes(grow(bxg2, cdir, -1), cdir);

  FArrayBox fy(yflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fyarr = fy.array();

  FArrayBox qgdy(yflxbx, NGDNV, amrex::The_Async_Arena());
  auto const& gdtempy = qgdy.array();

  // -5,-4,-5
  ParallelFor(yflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    if (apy(i,j,k) > 0.) {
      CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qymarr, qyparr, fyarr, gdtempy,
                  qaux, cdir, *lpmap, small, small_dens, small_pres);
    }
  });

  // *************************************************************************************
  // Z initial fluxes
  // *************************************************************************************
  cdir = 2;
  const Box& zflxbx = surroundingNodes(grow(bxg2, cdir, -1), cdir);

  FArrayBox fz(zflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fzarr = fz.array();

  FArrayBox qgdz(zflxbx, NGDNV, amrex::The_Async_Arena());
  auto const& gdtempz = qgdz.array();

  // -5,-5,-4
  ParallelFor(zflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    if (apz(i,j,k) > 0.) {
      CAMR_cmpflx(i, j, k, bclz, bchz, dlz, dhz, qzmarr, qzparr, fzarr, gdtempz,
                  qaux, cdir, *lpmap, small, small_dens, small_pres);
    }
  });

  // *************************************************************************************
  // X interface corrections
  // *************************************************************************************
  cdir = 0;
  FArrayBox qxym(xmbx, QVAR, amrex::The_Async_Arena());
  FArrayBox qxyp(xmbx, QVAR, amrex::The_Async_Arena());
  auto const& qmxy = qxym.array();
  auto const& qpxy = qxyp.array();

  FArrayBox qxzm(xmbx, QVAR, amrex::The_Async_Arena());
  FArrayBox qxzp(xmbx, QVAR, amrex::The_Async_Arena());
  auto const& qmxz = qxzm.array();
  auto const& qpxz = qxzp.array();

  //
  Box xybx(bxg2); xybx.grow(1,-1);
  // amrex::Print() << "TRANSDO: Y FLUXES CHANGING X " << xybx << std::endl;
  ParallelFor(xybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // fyarr was made on y-faces to -5,-4,-5
      // this loop is over cells to   -5,-4,-5
      if (!flag_arr(i,j,k).isCovered()) {
        // X|Y
        CAMR_transdo(i, j, k, cdir, 1, qmxy, qpxy, qxmarr, qxparr, fyarr, qaux, gdtempy, cdtdy,
                     *lpmap, l_transverse_reset_density, small_pres, apx, apy);
      }
  });

  Box xzbx(bxg2); xzbx.grow(2,-1);
  // amrex::Print() << "TRANSDO: Z FLUXES CHANGING X " << xzbx << std::endl;
  ParallelFor(xzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // fzarr was made on z-faces to -5,-5,-4
      // this loop is over cells to   -5,-5,-4
      if (!flag_arr(i,j,k).isCovered()) {
        // X|Z
        CAMR_transdo(i, j, k, cdir, 2, qmxz, qpxz, qxmarr, qxparr, fzarr, qaux, gdtempz, cdtdz,
                     *lpmap, l_transverse_reset_density, small_pres, apx, apz);
      }
  });

  const Box& txfxbx = surroundingNodes(bxg2, cdir);
  FArrayBox fluxxy(txfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox fluxxz(txfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox gdvxyfab(txfxbx, NGDNV, amrex::The_Async_Arena());
  FArrayBox gdvxzfab(txfxbx, NGDNV, amrex::The_Async_Arena());

  auto const& flxy = fluxxy.array();
  auto const& flxz = fluxxz.array();
  auto const& qxy = gdvxyfab.array();
  auto const& qxz = gdvxzfab.array();


  // Riemann problem X|Y
  Box xycmpbx(surroundingNodes(bxg1,0).grow(2,1));
  // this loop is over x-faces to   -4,-4,-5
  // amrex::Print() << "DOING CMPFLX XY " << xycmpbx << std::endl;
  ParallelFor(xycmpbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // X|Y
      if (apx(i,j,k) > 0.) {
          CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qmxy, qpxy, flxy, qxy, qaux,
                      cdir, *lpmap, small, small_dens, small_pres);
    }
  });

  // Riemann problem X|Z
  Box xzcmpbx(surroundingNodes(bxg1,0).grow(1,1));
  // this loop is over x-faces to   -4,-5,-4
  // amrex::Print() << "DOING CMPFLX XZ " << xzcmpbx << std::endl;
  ParallelFor(xzcmpbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // X|Z
      if (apx(i,j,k) > 0.) {
          CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qmxz, qpxz, flxz, qxz, qaux,
                      cdir, *lpmap, small, small_dens, small_pres);
    }
  });
  qxym.clear();
  qxyp.clear();
  qxzm.clear();
  qxzp.clear();

  // *************************************************************************************
  // Y interface corrections
  // *************************************************************************************

  cdir = 1;
  FArrayBox qyxm(ymbx, QVAR, amrex::The_Async_Arena());
  FArrayBox qyxp(ymbx, QVAR, amrex::The_Async_Arena());
  auto const& qmyx = qyxm.array();
  auto const& qpyx = qyxp.array();

  FArrayBox qyzm(ymbx, QVAR, amrex::The_Async_Arena());
  FArrayBox qyzp(ymbx, QVAR, amrex::The_Async_Arena());
  auto const& qmyz = qyzm.array();
  auto const& qpyz = qyzp.array();

  Box yxbx(bxg2); yxbx.grow(0,-1);
  // amrex::Print() << "TRANSDO: X FLUXES CHANGING Y " << yxbx << std::endl;
  ParallelFor(yxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Y|X
      if (!flag_arr(i, j, k).isCovered()) {
          CAMR_transdo(i, j, k, cdir, 0, qmyx, qpyx, qymarr, qyparr, fxarr, qaux,
                       gdtempx, cdtdx, *lpmap, l_transverse_reset_density,
                       small_pres, apy, apx);
      }
  });

  Box yzbx(bxg2); yzbx.grow(2,-1);
  // amrex::Print() << "TRANSDO: Z FLUXES CHANGING Y " << yzbx << std::endl;
  ParallelFor(yzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Y|Z
      if (!flag_arr(i, j, k).isCovered()) {
          CAMR_transdo(i, j, k, cdir, 2, qmyz, qpyz, qymarr, qyparr, fzarr, qaux,
                       gdtempz, cdtdz, *lpmap, l_transverse_reset_density,
                       small_pres, apy, apz);
    }
  });
  fz.clear();
  qgdz.clear();

  // Riemann problem Y|X Y|Z
  const Box& tyfxbx = surroundingNodes(bxg2, cdir);
  FArrayBox fluxyx(tyfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox fluxyz(tyfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox gdvyxfab(tyfxbx, NGDNV, amrex::The_Async_Arena());
  FArrayBox gdvyzfab(tyfxbx, NGDNV, amrex::The_Async_Arena());

  auto const& flyx = fluxyx.array();
  auto const& flyz = fluxyz.array();
  auto const& qyx = gdvyxfab.array();
  auto const& qyz = gdvyzfab.array();

  // Riemann problem Y|X
  Box yxcmpbx(surroundingNodes(bxg1,1).grow(2,1));
  // amrex::Print() << "DOING CMPFLX YX " << yxcmpbx << std::endl;
  ParallelFor(yxcmpbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Y|X
      if (apy(i,j,k) > 0.) {
           CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qmyx, qpyx, flyx, qyx, qaux,
                       cdir, *lpmap, small, small_dens, small_pres);
      }
  });

  // Riemann problem Y|Z
  Box yzcmpbx(surroundingNodes(bxg1,1).grow(0,1));
  // amrex::Print() << "DOING CMPFLX YZ " << yzcmpbx << std::endl;
  ParallelFor(yzcmpbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Y|Z
      if (apy(i,j,k) > 0.) {
          CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qmyz, qpyz, flyz, qyz, qaux,
                      cdir, *lpmap, small, small_dens, small_pres);
      }
  });

  qyxm.clear();
  qyxp.clear();
  qyzm.clear();
  qyzp.clear();


  // *************************************************************************************
  // Z interface corrections
  // *************************************************************************************
  cdir = 2;

  FArrayBox qzxm(zmbx, QVAR, amrex::The_Async_Arena());
  FArrayBox qzxp(zmbx, QVAR, amrex::The_Async_Arena());
  auto const& qmzx = qzxm.array();
  auto const& qpzx = qzxp.array();

  FArrayBox qzym(zmbx, QVAR, amrex::The_Async_Arena());
  FArrayBox qzyp(zmbx, QVAR, amrex::The_Async_Arena());
  auto const& qmzy = qzym.array();
  auto const& qpzy = qzyp.array();

  Box zxbx(bxg2); zxbx.grow(0,-1);
  // amrex::Print() << "TRANSDO: X FLUXES CHANGING Z " << zxbx << std::endl;
  ParallelFor(zxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Z|X
      if (!flag_arr(i, j, k).isCovered()) {
          CAMR_transdo(i, j, k, cdir, 0, qmzx, qpzx, qzmarr, qzparr, fxarr, qaux,
                       gdtempx, cdtdx, *lpmap, l_transverse_reset_density,
                       small_pres, apz, apx);
      }
  });

  Box zybx(bxg2); zybx.grow(1,-1);
  // amrex::Print() << "TRANSDO: Y FLUXES CHANGING Z " << zybx << std::endl;
  ParallelFor(zybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Z|Y
      if (!flag_arr(i, j, k).isCovered()) {
          CAMR_transdo(i, j, k, cdir, 1, qmzy, qpzy, qzmarr, qzparr, fyarr, qaux,
                       gdtempy, cdtdy, *lpmap, l_transverse_reset_density,
                       small_pres, apz, apy);
      }
  });

  fx.clear();
  fy.clear();
  qgdx.clear();
  qgdy.clear();


  // Riemann problem Z|X Z|Y
  const Box& tzfxbx = surroundingNodes(bxg2, cdir);
  FArrayBox fluxzx(tzfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox fluxzy(tzfxbx, NVAR, amrex::The_Async_Arena());
  FArrayBox gdvzxfab(tzfxbx, NGDNV, amrex::The_Async_Arena());
  FArrayBox gdvzyfab(tzfxbx, NGDNV, amrex::The_Async_Arena());

  auto const& flzx = fluxzx.array();
  auto const& flzy = fluxzy.array();
  auto const& qzx = gdvzxfab.array();
  auto const& qzy = gdvzyfab.array();

  // Riemann problem Z|X
  Box zxcmpbx(surroundingNodes(bxg1,2).grow(1,1));
  // amrex::Print() << "DOING CMPFLX ZX " << zxcmpbx << std::endl;
  ParallelFor(zxcmpbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Z|X
      if (apz(i,j,k) > 0.) {
          CAMR_cmpflx(i, j, k, bclz, bchz, dlz, dhz, qmzx, qpzx, flzx, qzx, qaux,
                      cdir, *lpmap, small, small_dens, small_pres);
      }
  });

  // Riemann problem Z|Y
  Box zycmpbx(surroundingNodes(bxg1,2).grow(0,1));
  // amrex::Print() << "DOING CMPFLX ZY " << zycmpbx << std::endl;
  ParallelFor(zycmpbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      // Z|Y
      if (apz(i,j,k) > 0.) {
          CAMR_cmpflx(i, j, k, bclz, bchz, dlz, dhz, qmzy, qpzy, flzy, qzy, qaux,
                      cdir, *lpmap, small, small_dens, small_pres);
      }
  });

  qzxm.clear();
  qzxp.clear();
  qzym.clear();
  qzyp.clear();

  FArrayBox qmfab(bxg2, QVAR, amrex::The_Async_Arena());
  FArrayBox qpfab(bxg1, QVAR, amrex::The_Async_Arena());
  auto const& qm = qmfab.array();
  auto const& qp = qpfab.array();

  // *************************************************************************************
  // Final X steps
  // *************************************************************************************
  cdir = 0;

  // this loop is over cells to   -4,-4,-4
  // amrex::Print() << "DOING TRANSDD FOR X " << bxg1 << std::endl;
  ParallelFor(bxg1, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      if (!flag_arr(i, j, k).isCovered()) {
          CAMR_transdd(i, j, k, cdir, qm, qp, qxmarr, qxparr, flyz, flzy, qyz, qzy,
                       qaux, srcQ, hdt, hdtdy, hdtdz, *lpmap,
                       l_transverse_reset_density, small_pres, apx, apy, apz);
      }
  });
  qxm.clear();
  qxp.clear();
  fluxyz.clear();
  gdvyzfab.clear();
  fluxzy.clear();
  gdvzyfab.clear();

  // This box must be grown by one in transverse direction so we can
  // do tangential interpolation when taking divergence later
  Box xfbx(surroundingNodes(bx_to_fill,0)); xfbx.grow(1,1); xfbx.grow(2,1);
  ParallelFor(xfbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      if (apx(i,j,k) > 0.) {
          CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qm, qp, flx1, q1, qaux, cdir,
                      *lpmap, small, small_dens, small_pres);
      }
  });

  // *************************************************************************************
  // Final Y steps
  // *************************************************************************************
  cdir = 1;
  // this loop is over cells to   -4,-4,-4
  // amrex::Print() << "DOING TRANSDD FOR Y " << bxg1 << std::endl;
  ParallelFor(bxg1, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      if (!flag_arr(i, j, k).isCovered()) {
          CAMR_transdd(i, j, k, cdir, qm, qp, qymarr, qyparr, flxz, flzx, qxz, qzx,
                       qaux, srcQ, hdt, hdtdx, hdtdz, *lpmap,
                       l_transverse_reset_density, small_pres, apy, apx, apz);
      }
  });
  qym.clear();
  qyp.clear();
  fluxxz.clear();
  gdvxzfab.clear();
  fluxzx.clear();
  gdvzxfab.clear();

  // This box must be grown by one in transverse direction so we can
  // do tangential interpolation when taking divergence later
  Box yfbx(surroundingNodes(bx_to_fill,1)); yfbx.grow(0,1); yfbx.grow(2,1);
  ParallelFor(yfbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      if (apy(i,j,k) > 0.) {
          CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qm, qp, flx2, q2, qaux, cdir,
                      *lpmap, small, small_dens, small_pres);
      }
  });

  // *************************************************************************************
  // Final Z steps
  // *************************************************************************************
  cdir = 2;
  // amrex::Print() << "DOING TRANSDD FOR Y " << bxg1 << std::endl;
  ParallelFor(bxg1, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      if (!flag_arr(i, j, k).isCovered()) {
          CAMR_transdd(i, j, k, cdir, qm, qp, qzmarr, qzparr, flxy, flyx, qxy, qyx,
                       qaux, srcQ, hdt, hdtdx, hdtdy, *lpmap,
                       l_transverse_reset_density, small_pres, apz, apx, apy);
      }
  });
  qzm.clear();
  qzp.clear();
  fluxxy.clear();
  gdvxyfab.clear();
  fluxyx.clear();
  gdvyxfab.clear();

  // This box must be grown by one in transverse direction so we can
  // do tangential interpolation when taking divergence later
  Box zfbx(surroundingNodes(bx_to_fill,2)); zfbx.grow(0,1); zfbx.grow(1,1);
  ParallelFor(zfbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      if (apz(i,j,k) > 0.) {
          CAMR_cmpflx(i, j, k, bclz, bchz, dlz, dhz, qm, qp, flx3, q3, qaux, cdir,
                      *lpmap, small, small_dens, small_pres);
      }
  });
  qmfab.clear();
  qpfab.clear();
}

#endif
