#include <AMReX_GpuAllocators.H>
#include "Godunov.H"
#include "Godunov_utils.H"
#include "CAMR_utils_K.H"
#include "Hydro_cmpflx.H"
#include "PLM.H"
#include "PPM.H"

#if (AMREX_SPACEDIM == 3)
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
  const amrex::Real l_small_dens,
  const amrex::Real l_small_pres,
  const int ppm_type,
  const int use_pslope,
  const int use_flattening,
  const int iorder,
  const int l_transverse_reset_density)
{
  BL_PROFILE("CAMR::Godunov_umeth()");

  amrex::Real const dx = del[0];
  amrex::Real const dy = del[1];
  amrex::Real const dz = del[2];
  amrex::Real const hdtdx = 0.5 * dt / dx;
  amrex::Real const hdtdy = 0.5 * dt / dy;
  amrex::Real const hdtdz = 0.5 * dt / dz;
  amrex::Real const cdtdx = 1.0 / 3.0 * dt / dx;
  amrex::Real const cdtdy = 1.0 / 3.0 * dt / dy;
  amrex::Real const cdtdz = 1.0 / 3.0 * dt / dz;
  amrex::Real const hdt = 0.5 * dt;

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

  // Z data
  cdir = 2;
  const amrex::Box& zmbx = growHi(bxg2, cdir, 1);
  const amrex::Box& zflxbx = surroundingNodes(grow(bxg2, cdir, -1), cdir);
  amrex::FArrayBox qzm(zmbx, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qzp(bxg2, QVAR, amrex::The_Async_Arena());
  auto const& qzmarr = qzm.array();
  auto const& qzparr = qzp.array();

  const PassMap* lpmap = CAMR::d_pass_map;

  // Put the PLM and slopes in the same kernel launch to avoid unnecessary
  // launch overhead
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
        int idir = 0;
        for (int n = 0; n < QVAR; ++n) {
          if (n == QPRES && use_pslope)
              slope[n] = plm_pslope(i, j, k, n, 0, q, dx, srcQ, flat, iorder);
          else
              slope[n] = plm_slope(i, j, k, n, 0, q, flat, iorder);
        }
        CAMR_plm_d(
          i, j, k, idir, qxmarr, qxparr, slope, q, qaux(i, j, k, QC), dx, dt,
          l_small_dens, l_small_pres, *lpmap);

        // Y slopes and interp
        idir = 1;
        for (int n = 0; n < QVAR; n++) {
          if (n == QPRES && use_pslope)
              slope[n] = plm_pslope(i, j, k, n, 1, q, dy, srcQ, flat, iorder);
          else
              slope[n] = plm_slope(i, j, k, n, 1, q, flat, iorder);
        }
        CAMR_plm_d(
          i, j, k, idir, qymarr, qyparr, slope, q, qaux(i, j, k, QC), dy, dt,
          l_small_dens, l_small_pres, *lpmap);

        // Z slopes and interp
        idir = 2;
        for (int n = 0; n < QVAR; ++n) {
          if (n == QPRES && use_pslope)
              slope[n] = plm_pslope(i, j, k, n, 2, q, dz, srcQ, flat, iorder);
          else
              slope[n] = plm_slope(i, j, k, n, 2, q, flat, iorder);
        }
        CAMR_plm_d(
          i, j, k, idir, qzmarr, qzparr, slope, q, qaux(i, j, k, QC), dz, dt,
          l_small_dens, l_small_pres, *lpmap);
      });
  } else if (ppm_type == 1) {
    // Compute the normal interface states by reconstructing
    // the primitive variables using the piecewise parabolic method
    // and doing characteristic tracing.  We do not apply the
    // transverse terms here.

    int idir = 0;
    trace_ppm(
      bxg2, idir, q, qaux, srcQ, qxmarr, qxparr, bxg2, dt, del, use_flattening,
      l_small_dens, l_small_pres, *lpmap);

    idir = 1;
    trace_ppm(
      bxg2, idir, q, qaux, srcQ, qymarr, qyparr, bxg2, dt, del, use_flattening,
      l_small_dens, l_small_pres, *lpmap);

    idir = 2;
    trace_ppm(
      bxg2, idir, q, qaux, srcQ, qzmarr, qzparr, bxg2, dt, del, use_flattening,
      l_small_dens, l_small_pres, *lpmap);

  } else {
    amrex::Error("CAMR::ppm_type must be 0 (PLM) or 1 (PPM)");
  }

  // These are the first flux estimates as per the corner-transport-upwind
  // method X initial fluxes
  cdir = 0;
  amrex::FArrayBox fx(xflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fxarr = fx.array();
  amrex::FArrayBox qgdx(xflxbx, NGDNV, amrex::The_Async_Arena());
  auto const& gdtempx = qgdx.array();
  amrex::ParallelFor(
    xflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      CAMR_cmpflx(
        i, j, k, bclx, bchx, dlx, dhx, qxmarr, qxparr, fxarr, gdtempx, qaux,
        cdir, *lpmap, small, l_small_dens, l_small_pres);
    });

  // Y initial fluxes
  cdir = 1;
  amrex::FArrayBox fy(yflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fyarr = fy.array();
  amrex::FArrayBox qgdy(yflxbx, NGDNV, amrex::The_Async_Arena());
  auto const& gdtempy = qgdy.array();
  amrex::ParallelFor(
    yflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      CAMR_cmpflx(
        i, j, k, bcly, bchy, dly, dhy, qymarr, qyparr, fyarr, gdtempy, qaux,
        cdir, *lpmap, small, l_small_dens, l_small_pres);
    });

  // Z initial fluxes
  cdir = 2;
  amrex::FArrayBox fz(zflxbx, NVAR, amrex::The_Async_Arena());
  auto const& fzarr = fz.array();
  amrex::FArrayBox qgdz(zflxbx, NGDNV, amrex::The_Async_Arena());
  auto const& gdtempz = qgdz.array();
  amrex::ParallelFor(
    zflxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      CAMR_cmpflx(
        i, j, k, bclz, bchz, dlz, dhz, qzmarr, qzparr, fzarr, gdtempz, qaux,
        cdir, *lpmap, small, l_small_dens, l_small_pres);
    });

  // X interface corrections
  cdir = 0;
  const amrex::Box& txbx = grow(bxg1, cdir, 1);
  const amrex::Box& txbxm = growHi(txbx, cdir, 1);
  amrex::FArrayBox qxym(txbxm, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qxyp(txbx, QVAR, amrex::The_Async_Arena());
  auto const& qmxy = qxym.array();
  auto const& qpxy = qxyp.array();

  amrex::FArrayBox qxzm(txbxm, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qxzp(txbx, QVAR, amrex::The_Async_Arena());
  auto const& qmxz = qxzm.array();
  auto const& qpxz = qxzp.array();

  amrex::ParallelFor(txbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    // X|Y
    CAMR_transdo(
      i, j, k, cdir, 1, qmxy, qpxy, qxmarr, qxparr, fyarr, qaux, gdtempy, cdtdy,
      *lpmap, l_transverse_reset_density, l_small_pres);
    // X|Z
    CAMR_transdo(
      i, j, k, cdir, 2, qmxz, qpxz, qxmarr, qxparr, fzarr, qaux, gdtempz, cdtdz,
      *lpmap, l_transverse_reset_density, l_small_pres);
  });

  const amrex::Box& txfxbx = surroundingNodes(bxg1, cdir);
  amrex::FArrayBox fluxxy(txfxbx, NVAR, amrex::The_Async_Arena());
  amrex::FArrayBox fluxxz(txfxbx, NVAR, amrex::The_Async_Arena());
  amrex::FArrayBox gdvxyfab(txfxbx, NGDNV, amrex::The_Async_Arena());
  amrex::FArrayBox gdvxzfab(txfxbx, NGDNV, amrex::The_Async_Arena());

  auto const& flxy = fluxxy.array();
  auto const& flxz = fluxxz.array();
  auto const& qxy = gdvxyfab.array();
  auto const& qxz = gdvxzfab.array();

  // Riemann problem X|Y X|Z
  amrex::ParallelFor(
    txfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      // X|Y
      CAMR_cmpflx(
        i, j, k, bclx, bchx, dlx, dhx, qmxy, qpxy, flxy, qxy, qaux,
        cdir, *lpmap, small, l_small_dens, l_small_pres);
      // X|Z
      CAMR_cmpflx(
        i, j, k, bclx, bchx, dlx, dhx, qmxz, qpxz, flxz, qxz, qaux,
        cdir, *lpmap, small, l_small_dens, l_small_pres);
    });
  qxym.clear();
  qxyp.clear();
  qxzm.clear();
  qxzp.clear();

  // Y interface corrections
  cdir = 1;
  const amrex::Box& tybx = grow(bxg1, cdir, 1);
  const amrex::Box& tybxm = growHi(tybx, cdir, 1);
  amrex::FArrayBox qyxm(tybxm, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qyxp(tybx, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qyzm(tybxm, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qyzp(tybx, QVAR, amrex::The_Async_Arena());
  auto const& qmyx = qyxm.array();
  auto const& qpyx = qyxp.array();
  auto const& qmyz = qyzm.array();
  auto const& qpyz = qyzp.array();

  amrex::ParallelFor(tybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    // Y|X
    CAMR_transdo(
      i, j, k, cdir, 0, qmyx, qpyx, qymarr, qyparr, fxarr, qaux, gdtempx, cdtdx,
      *lpmap, l_transverse_reset_density, l_small_pres);
    // Y|Z
    CAMR_transdo(
      i, j, k, cdir, 2, qmyz, qpyz, qymarr, qyparr, fzarr, qaux, gdtempz, cdtdz,
      *lpmap, l_transverse_reset_density, l_small_pres);
  });
  fz.clear();
  qgdz.clear();

  // Riemann problem Y|X Y|Z
  const amrex::Box& tyfxbx = surroundingNodes(bxg1, cdir);
  amrex::FArrayBox fluxyx(tyfxbx, NVAR, amrex::The_Async_Arena());
  amrex::FArrayBox fluxyz(tyfxbx, NVAR, amrex::The_Async_Arena());
  amrex::FArrayBox gdvyxfab(tyfxbx, NGDNV, amrex::The_Async_Arena());
  amrex::FArrayBox gdvyzfab(tyfxbx, NGDNV, amrex::The_Async_Arena());

  auto const& flyx = fluxyx.array();
  auto const& flyz = fluxyz.array();
  auto const& qyx = gdvyxfab.array();
  auto const& qyz = gdvyzfab.array();

  amrex::ParallelFor(
    tyfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      // Y|X
      CAMR_cmpflx(
        i, j, k, bcly, bchy, dly, dhy, qmyx, qpyx, flyx, qyx, qaux,
        cdir, *lpmap, small, l_small_dens, l_small_pres);
      // Y|Z
      CAMR_cmpflx(
        i, j, k, bcly, bchy, dly, dhy, qmyz, qpyz, flyz, qyz, qaux,
        cdir, *lpmap, small, l_small_dens, l_small_pres);
    });
  qyxm.clear();
  qyxp.clear();
  qyzm.clear();
  qyzp.clear();

  // Z interface corrections
  cdir = 2;
  const amrex::Box& tzbx = grow(bxg1, cdir, 1);
  const amrex::Box& tzbxm = growHi(tzbx, cdir, 1);
  amrex::FArrayBox qzxm(tzbxm, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qzxp(tzbx, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qzym(tzbxm, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qzyp(tzbx, QVAR, amrex::The_Async_Arena());

  auto const& qmzx = qzxm.array();
  auto const& qpzx = qzxp.array();
  auto const& qmzy = qzym.array();
  auto const& qpzy = qzyp.array();

  amrex::ParallelFor(tzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    // Z|X
    CAMR_transdo(
      i, j, k, cdir, 0, qmzx, qpzx, qzmarr, qzparr, fxarr, qaux, gdtempx, cdtdx,
      *lpmap, l_transverse_reset_density, l_small_pres);
    // Z|Y
    CAMR_transdo(
      i, j, k, cdir, 1, qmzy, qpzy, qzmarr, qzparr, fyarr, qaux, gdtempy, cdtdy,
      *lpmap, l_transverse_reset_density, l_small_pres);
  });
  fx.clear();
  fy.clear();
  qgdx.clear();
  qgdy.clear();

  // Riemann problem Z|X Z|Y
  const amrex::Box& tzfxbx = surroundingNodes(bxg1, cdir);
  amrex::FArrayBox fluxzx(tzfxbx, NVAR, amrex::The_Async_Arena());
  amrex::FArrayBox fluxzy(tzfxbx, NVAR, amrex::The_Async_Arena());
  amrex::FArrayBox gdvzxfab(tzfxbx, NGDNV, amrex::The_Async_Arena());
  amrex::FArrayBox gdvzyfab(tzfxbx, NGDNV, amrex::The_Async_Arena());

  auto const& flzx = fluxzx.array();
  auto const& flzy = fluxzy.array();
  auto const& qzx = gdvzxfab.array();
  auto const& qzy = gdvzyfab.array();

  amrex::ParallelFor(
    tzfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      // Z|X
      CAMR_cmpflx(
        i, j, k, bclz, bchz, dlz, dhz, qmzx, qpzx, flzx, qzx, qaux,
        cdir, *lpmap, small, l_small_dens, l_small_pres);
      // Z|Y
      CAMR_cmpflx(
        i, j, k, bclz, bchz, dlz, dhz, qmzy, qpzy, flzy, qzy, qaux,
        cdir, *lpmap, small, l_small_dens, l_small_pres);
    });
  qzxm.clear();
  qzxp.clear();
  qzym.clear();
  qzyp.clear();

  // Temp Fabs for Final Fluxes
  amrex::FArrayBox qmfab(bxg2, QVAR, amrex::The_Async_Arena());
  amrex::FArrayBox qpfab(bxg1, QVAR, amrex::The_Async_Arena());
  auto const& qm = qmfab.array();
  auto const& qp = qpfab.array();

  // X | Y&Z
  cdir = 0;
  const amrex::Box& xfxbx = surroundingNodes(bx, cdir);
  const amrex::Box& tyzbx = grow(bx, cdir, 1);
  amrex::ParallelFor(tyzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    CAMR_transdd(
      i, j, k, cdir, qm, qp, qxmarr, qxparr, flyz, flzy, qyz, qzy, qaux, srcQ,
      hdt, hdtdy, hdtdz, *lpmap, l_transverse_reset_density, l_small_pres);
  });
  qxm.clear();
  qxp.clear();
  fluxyz.clear();
  gdvyzfab.clear();
  fluxzy.clear();
  gdvzyfab.clear();

  // Final X flux
  amrex::ParallelFor(xfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    CAMR_cmpflx(i, j, k, bclx, bchx, dlx, dhx, qm, qp, flx1, q1, qaux,
              cdir, *lpmap, small, l_small_dens, l_small_pres);
  });

  // Y | X&Z
  cdir = 1;
  const amrex::Box& yfxbx = surroundingNodes(bx, cdir);
  const amrex::Box& txzbx = grow(bx, cdir, 1);
  amrex::ParallelFor(txzbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    CAMR_transdd(
      i, j, k, cdir, qm, qp, qymarr, qyparr, flxz, flzx, qxz, qzx, qaux, srcQ,
      hdt, hdtdx, hdtdz, *lpmap, l_transverse_reset_density, l_small_pres);
  });
  qym.clear();
  qyp.clear();
  fluxxz.clear();
  gdvxzfab.clear();
  fluxzx.clear();
  gdvzxfab.clear();

  // Final Y flux
  amrex::ParallelFor(yfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    CAMR_cmpflx(i, j, k, bcly, bchy, dly, dhy, qm, qp, flx2, q2, qaux,
              cdir, *lpmap, small, l_small_dens, l_small_pres);
  });

  // Z | X&Y
  cdir = 2;
  const amrex::Box& zfxbx = surroundingNodes(bx, cdir);
  const amrex::Box& txybx = grow(bx, cdir, 1);
  amrex::ParallelFor(txybx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    CAMR_transdd(
      i, j, k, cdir, qm, qp, qzmarr, qzparr, flxy, flyx, qxy, qyx, qaux, srcQ,
      hdt, hdtdx, hdtdy, *lpmap, l_transverse_reset_density, l_small_pres);
  });
  qzm.clear();
  qzp.clear();
  fluxxy.clear();
  gdvxyfab.clear();
  fluxyx.clear();
  gdvyxfab.clear();

  // Final Z flux
  amrex::ParallelFor(zfxbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    CAMR_cmpflx(i, j, k, bclz, bchz, dlz, dhz, qm, qp, flx3, q3, qaux,
              cdir, *lpmap, small, l_small_dens, l_small_pres);
  });
  qmfab.clear();
  qpfab.clear();

  // Construct p div{U}
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    CAMR_pdivu(
      i, j, k, pdivu, AMREX_D_DECL(q1, q2, q3), AMREX_D_DECL(a1, a2, a3), vol);
  });
}

#endif
