#include "EOS.H"
#include "Derive.H"
#include "CAMR.H"
#include "IndexDefines.H"
#include "CAMR_Constants.H"

void
CAMR_dervelx(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto velx = derfab.array();

#ifdef AMREX_USE_EB
amrex::Real local_small_den = 1.e-20;
#endif

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#ifdef AMREX_USE_EB
    velx(i, j, k) = dat(i, j, k, UMX) / std::max(dat(i, j, k, URHO), local_small_den);
#else
    velx(i, j, k) = dat(i, j, k, UMX) / dat(i, j, k, URHO);
#endif
  });
}

void
CAMR_dervely(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto vely = derfab.array();

#ifdef AMREX_USE_EB
amrex::Real local_small_den = 1.e-20;
#endif

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#ifdef AMREX_USE_EB
    vely(i, j, k) = dat(i, j, k, UMY) / std::max(dat(i, j, k, URHO), local_small_den);
#else
    vely(i, j, k) = dat(i, j, k, UMY) / dat(i, j, k, URHO);
#endif
  });
}

#if (AMREX_SPACEDIM == 3)
void
CAMR_dervelz(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto velz = derfab.array();

#ifdef AMREX_USE_EB
amrex::Real local_small_den = 1.e-20;
#endif

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#ifdef AMREX_USE_EB
    velz(i, j, k) = dat(i, j, k, UMZ) / std::max(dat(i, j, k, URHO), local_small_den);
#else
    velz(i, j, k) = dat(i, j, k, UMZ) / dat(i, j, k, URHO);
#endif
  });
}
#endif

void
CAMR_dermagvel(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto magvel = derfab.array();

#ifdef AMREX_USE_EB
amrex::Real local_small_den = 1.e-20;
#endif

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#ifdef AMREX_USE_EB
    const amrex::Real rhoInv = 1.0 / std::max(dat(i, j, k, URHO), local_small_den);
#else
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);
#endif
    const amrex::Real dat1 = (dat(i, j, k, UMX) * rhoInv);
    const amrex::Real dat2 = (dat(i, j, k, UMY) * rhoInv);
#if (AMREX_SPACEDIM == 2)
    magvel(i, j, k) = sqrt((dat1 * dat1) + (dat2 * dat2));
#else
    const amrex::Real dat3 = (dat(i, j, k, UMZ) * rhoInv);
    magvel(i, j, k) = sqrt((dat1 * dat1) + (dat2 * dat2) + (dat3 * dat3));
#endif
  });
}

void
CAMR_dermagmom(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto magmom = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    magmom(i, j, k) = sqrt(
      dat(i, j, k, UMX) * dat(i, j, k, UMX) +
#if (AMREX_SPACEDIM == 2)
      dat(i, j, k, UMY) * dat(i, j, k, UMY));
#else
      dat(i, j, k, UMY) * dat(i, j, k, UMY) +
      dat(i, j, k, UMZ) * dat(i, j, k, UMZ));
#endif
  });
}

void
CAMR_derkineng(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto kineng = derfab.array();

#ifdef AMREX_USE_EB
amrex::Real local_small_den = 1.e-20;
#endif

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const amrex::Real datxsq = dat(i, j, k, UMX) * dat(i, j, k, UMX);
      const amrex::Real datysq = dat(i, j, k, UMY) * dat(i, j, k, UMY);
#if (AMREX_SPACEDIM == 2)
#ifdef AMREX_USE_EB
      kineng(i, j, k) = 0.5 * (datxsq + datysq) / std::max(dat(i, j, k, URHO), local_small_den);
#else
      kineng(i, j, k) = 0.5 * (datxsq + datysq) / dat(i,j,k,URHO);
#endif
#else
      const amrex::Real datzsq = dat(i, j, k, UMZ) * dat(i, j, k, UMZ);
#ifdef AMREX_USE_EB
      kineng(i, j, k) = 0.5 * (datxsq + datysq + datzsq) / std::max(dat(i, j, k, URHO), local_small_den);
#else
      kineng(i, j, k) = 0.5 * (datxsq + datysq + datzsq) / dat(i,j,k,URHO);
#endif
#endif
  });
}

void
CAMR_dereint1(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  // Compute internal energy from (rho E).
  auto const dat = datfab.const_array();
  auto e = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);
    const amrex::Real ux = dat(i, j, k, UMX) * rhoInv;
    const amrex::Real uy = dat(i, j, k, UMY) * rhoInv;
#if (AMREX_SPACEDIM == 2)
    e(i, j, k) =
      dat(i, j, k, UEDEN) * rhoInv - 0.5 * (ux * ux + uy * uy);
#else
    const amrex::Real uz = dat(i, j, k, UMZ) * rhoInv;
    e(i, j, k) =
      dat(i, j, k, UEDEN) * rhoInv - 0.5 * (ux * ux + uy * uy + uz * uz);
#endif
  });
}

void
CAMR_dereint2(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  // Compute internal energy from (rho e).
  auto const dat = datfab.const_array();
  auto e = derfab.array();

#ifdef AMREX_USE_EB
amrex::Real local_small_den = 1.e-20;
#endif

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#ifdef AMREX_USE_EB
    e(i, j, k) = dat(i, j, k, UEINT) / std::max(dat(i, j, k, URHO), local_small_den);
#else
    e(i, j, k) = dat(i, j, k, UEINT) / dat(i, j, k, URHO);
#endif
  });
}

void
CAMR_derlogden(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto logden = derfab.array();

#ifdef AMREX_USE_EB
amrex::Real local_small_den = 1.e-20;
#endif

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#ifdef AMREX_USE_EB
    logden(i, j, k) = log10(std::max(dat(i, j, k), local_small_den));
#else
    logden(i, j, k) = log10(dat(i, j, k));
#endif
  });
}

void
CAMR_derspec(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto spec = derfab.array();

  amrex::ParallelFor(
    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        amrex::Real rho = dat(i,j,k,0);
#ifdef AMREX_USE_EB
        if (rho <= 0.) {
            spec(i, j, k) = 0.;
        } else
#endif
        spec(i, j, k) = dat(i, j, k, 1) / rho;
    });
}

void
CAMR_dermagvort(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto vort = derfab.array();

  const amrex::Box& gbx = amrex::grow(bx, 1);

  amrex::FArrayBox local(gbx, 3, amrex::The_Async_Arena());
  auto larr = local.array();

#ifdef AMREX_USE_EB
  const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
  const auto& typ = flag_fab.getType(bx);
  if (typ == amrex::FabType::covered) {
    derfab.setVal<amrex::RunOn::Device>(0.0, bx);
    return;
  }
  const auto& flags = flag_fab.const_array();
  const bool all_regular = typ == amrex::FabType::regular;

  amrex::Real local_small_den = 1.e-20;
#endif

  // Convert momentum to velocity.
  amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#ifdef AMREX_USE_EB
    const amrex::Real rhoInv = 1.0 / std::max(dat(i, j, k, URHO), local_small_den);
#else
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);
#endif
    AMREX_D_TERM(larr(i, j, k, 0) = dat(i, j, k, UMX) * rhoInv;
               , larr(i, j, k, 1) = dat(i, j, k, UMY) * rhoInv;
               , larr(i, j, k, 2) = dat(i, j, k, UMZ) * rhoInv;)
  });

  AMREX_D_TERM(const amrex::Real dx = geomdata.CellSizeArray()[0];
             , const amrex::Real dy = geomdata.CellSizeArray()[1];
             , const amrex::Real dz = geomdata.CellSizeArray()[2];);

  // Calculate vorticity.
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)

    // if fab is all regular -> call regular idx and weights
    // otherwise
#ifdef AMREX_USE_EB
    AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
#else
    AMREX_D_TERM(get_idx(i, im, ip);
                 , get_idx(j, jm, jp);
                 , get_idx(k, km, kp);)
#endif
    AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

    AMREX_D_TERM(
      vort(i, j, k) = 0.0 * dx;
      ,
      const amrex::Real vx = wi * (larr(ip, j, k, 1) - larr(im, j, k, 1)) / dx;
      const amrex::Real uy = wj * (larr(i, jp, k, 0) - larr(i, jm, k, 0)) / dy;
      const amrex::Real v3 = vx - uy;
      ,
      const amrex::Real wx = wi * (larr(ip, j, k, 2) - larr(im, j, k, 2)) / dx;
      const amrex::Real wy = wj * (larr(i, jp, k, 2) - larr(i, jm, k, 2)) / dy;
      const amrex::Real uz = wk * (larr(i, j, kp, 0) - larr(i, j, km, 0)) / dz;
      const amrex::Real vz = wk * (larr(i, j, kp, 1) - larr(i, j, km, 1)) / dz;
      const amrex::Real v1 = wy - vz; const amrex::Real v2 = uz - wx;);
    vort(i, j, k) = sqrt(AMREX_D_TERM(0., +v3 * v3, +v1 * v1 + v2 * v2));
  });
}

void
CAMR_derdivu(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto divu = derfab.array();

  const amrex::Box& gbx = amrex::grow(bx, 1);

  amrex::FArrayBox local(gbx, 3, amrex::The_Async_Arena());
  auto larr = local.array();

#ifdef AMREX_USE_EB
  const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
  const auto& typ = flag_fab.getType(bx);
  if (typ == amrex::FabType::covered) {
    derfab.setVal<amrex::RunOn::Device>(0.0, bx);
    return;
  }
  const auto& flags = flag_fab.const_array();
  const bool all_regular = typ == amrex::FabType::regular;

  amrex::Real local_small_den = 1.e-20;
#endif

  // Convert momentum to velocity.
  amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#ifdef AMREX_USE_EB
    const amrex::Real rhoInv = 1.0 / std::max(dat(i, j, k, URHO), local_small_den);
#else
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);
#endif
    AMREX_D_TERM(larr(i, j, k, 0) = dat(i, j, k, UMX) * rhoInv;
               , larr(i, j, k, 1) = dat(i, j, k, UMY) * rhoInv;
               , larr(i, j, k, 2) = dat(i, j, k, UMZ) * rhoInv;)
  });

  AMREX_D_TERM(const amrex::Real dx = geomdata.CellSizeArray()[0];
             , const amrex::Real dy = geomdata.CellSizeArray()[1];
             , const amrex::Real dz = geomdata.CellSizeArray()[2];);

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)
#ifdef AMREX_USE_EB
    AMREX_D_TERM(get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 , get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 , get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
#else
    AMREX_D_TERM(get_idx(i, im, ip);
                 , get_idx(j, jm, jp);
                 , get_idx(k, km, kp);)

#endif
    AMREX_D_TERM(const amrex::Real wi = get_weight(im, ip);
                 , const amrex::Real wj = get_weight(jm, jp);
                 , const amrex::Real wk = get_weight(km, kp);)

    AMREX_D_TERM(
      const amrex::Real uhi = larr(ip, j, k, 0);
      const amrex::Real ulo = larr(im, j, k, 0);
      , const amrex::Real vhi = larr(i, jp, k, 1);
      const amrex::Real vlo = larr(i, jm, k, 1);
      , const amrex::Real whi = larr(i, j, kp, 2);
      const amrex::Real wlo = larr(i, j, km, 2););
    divu(i, j, k) = AMREX_D_TERM(
      wi * (uhi - ulo) / dx, +wj * (vhi - vlo) / dy, +wk * (whi - wlo) / dz);
  });
}

void
CAMR_derenstrophy(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& geomdata,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  // This routine will derive enstrophy  = 1/2 rho (x_vorticity^2 +
  // y_vorticity^2 + z_vorticity^2)
  auto const dat = datfab.const_array();
  auto enstrophy = derfab.array();

  const amrex::Box& gbx = amrex::grow(bx, 1);

  amrex::FArrayBox local(gbx, 3, amrex::The_Async_Arena());
  auto larr = local.array();

#ifdef AMREX_USE_EB
  const auto& flag_fab = amrex::getEBCellFlagFab(datfab);
  const auto& typ = flag_fab.getType(bx);
  if (typ == amrex::FabType::covered) {
    derfab.setVal<amrex::RunOn::Device>(0.0, bx);
    return;
  }
  const auto& flags = flag_fab.const_array();
  const bool all_regular = typ == amrex::FabType::regular;

  amrex::Real local_small_den = 1.e-20;
#endif

  // Convert momentum to velocity.
  amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#ifdef AMREX_USE_EB
    const amrex::Real rhoInv = 1.0 / std::max(dat(i, j, k, URHO), local_small_den);
#else
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);
#endif
    AMREX_D_TERM( larr(i, j, k, 0) = dat(i, j, k, UMX) * rhoInv;,
                  larr(i, j, k, 1) = dat(i, j, k, UMY) * rhoInv;,
                  larr(i, j, k, 2) = dat(i, j, k, UMZ) * rhoInv;);
  });

  AMREX_D_TERM(const amrex::Real dx = geomdata.CellSizeArray()[0];
             , const amrex::Real dy = geomdata.CellSizeArray()[1];
             , const amrex::Real dz = geomdata.CellSizeArray()[2];);

  // Calculate enstrophy.
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    AMREX_D_TERM(int im; int ip;, int jm; int jp;, int km; int kp;)
#ifdef AMREX_USE_EB
    AMREX_D_TERM( get_idx(i, 0, all_regular, flags(i, j, k), im, ip);
                 ,get_idx(j, 1, all_regular, flags(i, j, k), jm, jp);
                 ,get_idx(k, 2, all_regular, flags(i, j, k), km, kp);)
#else
    AMREX_D_TERM( get_idx(i, im, ip);
                 ,get_idx(j, jm, jp);
                 ,get_idx(k, km, kp);)

#endif
    AMREX_D_TERM( const amrex::Real wi = get_weight(im, ip);
                 ,const amrex::Real wj = get_weight(jm, jp);
                 ,const amrex::Real wk = get_weight(km, kp);)

    AMREX_D_TERM(
      enstrophy(i, j, k) = 0.0 * dx;
      ,
      const amrex::Real vx = wi * (larr(ip, j, k, 1) - larr(im, j, k, 1)) / dx;
      const amrex::Real uy = wj * (larr(i, jp, k, 0) - larr(i, jm, k, 0)) / dy;
      const amrex::Real v3 = vx - uy;
      ,
      const amrex::Real wx = wi * (larr(ip, j, k, 2) - larr(im, j, k, 2)) / dx;

      const amrex::Real wy = wj * (larr(i, jp, k, 2) - larr(i, jm, k, 2)) / dy;

      const amrex::Real uz = wk * (larr(i, j, kp, 0) - larr(i, j, km, 0)) / dz;
      const amrex::Real vz = wk * (larr(i, j, kp, 1) - larr(i, j, km, 1)) / dz;

      const amrex::Real v1 = wy - vz; const amrex::Real v2 = uz - wx;);
    enstrophy(i, j, k) = 0.5 * dat(i, j, k, URHO) *
                         (AMREX_D_TERM(0., +v3 * v3, +v1 * v1 + v2 * v2));
  });
}

void
CAMR_dernull(
  const amrex::Box& /*bx*/,
  amrex::FArrayBox& /*derfab*/,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& /*datfab*/,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  // This routine does nothing.
}

void
CAMR_dersoundspeed(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto cfab = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rho = dat(i, j, k, URHO);
#ifdef AMREX_USE_EB
    if (rho <= 0.) {
        cfab(i, j, k) = 0.;
    } else
#endif
    {
    const amrex::Real rhoInv = 1.0 / rho;
    const amrex::Real T = dat(i, j, k, UTEMP);
    amrex::Real massfrac[NUM_SPECIES];
    amrex::Real c;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    amrex::Real gam,pres;
    amrex::Real eint = dat(i, j, k, UEINT)/rho;
    EOS::REY2P(rho,eint,massfrac,pres);
    EOS::REY2Gam(rho,eint,massfrac,gam);
    c = std::sqrt(gam*pres/rho);
   // EOS::RTY2Cs(rho, T, massfrac, c);
    cfab(i, j, k) = c;
    }
  });
}

void
CAMR_dermachnumber(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto mach = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rho = dat(i, j, k, URHO);
#ifdef AMREX_USE_EB
    if (rho <= 0.) {
        mach(i, j, k) = 0.;
    } else
#endif
    {
    const amrex::Real rhoInv = 1.0 / rho;
    const amrex::Real T = dat(i, j, k, UTEMP);
    amrex::Real massfrac[NUM_SPECIES];
    amrex::Real c;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    amrex::Real gam,pres;
    amrex::Real eint = dat(i, j, k, UEINT)/rho;
    EOS::REY2P(rho,eint,massfrac,pres);
    EOS::REY2Gam(rho,eint,massfrac,gam);
    c = std::sqrt(gam*pres/rho);
    //EOS::RTY2Cs(rho, T, massfrac, c);
    const amrex::Real datxsq = dat(i, j, k, UMX) * dat(i, j, k, UMX);
    const amrex::Real datysq = dat(i, j, k, UMY) * dat(i, j, k, UMY);
#if (AMREX_SPACEDIM == 2)
    mach(i, j, k) = sqrt(datxsq + datysq) / dat(i, j, k, URHO) / c;
#else
    const amrex::Real datzsq = dat(i, j, k, UMZ) * dat(i, j, k, UMZ);
    mach(i, j, k) = sqrt(datxsq + datysq + datzsq) / dat(i, j, k, URHO) / c;
#endif
    }
  });
}

void
CAMR_derpres(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto pfab = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rho = dat(i, j, k, URHO);
#ifdef AMREX_USE_EB
    if (rho <= 0.) {
        pfab(i, j, k) = 0.;
    } else
#endif
    {
    const amrex::Real rhoInv = 1.0 / rho;
    amrex::Real e = dat(i, j, k, UEINT) * rhoInv;
    amrex::Real p;
    amrex::Real massfrac[NUM_SPECIES];
    for (int n = 0; n < NUM_SPECIES; ++n) {
      massfrac[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    EOS::REY2P(rho, e, massfrac, p);
    pfab(i, j, k) = p;
    }
  });
}

void
CAMR_dertemp(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  const int /*level*/)
{
  auto const dat = datfab.const_array();
  auto tfab = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    tfab(i, j, k) = dat(i, j, k, UTEMP);
  });
}

void
CAMR_dercp(
  const amrex::Box& bx,
  amrex::FArrayBox& derfab,
  int /*dcomp*/,
  int /*ncomp*/,
  const amrex::FArrayBox& datfab,
  const amrex::Geometry& /*geomdata*/,
  amrex::Real /*time*/,
  const int* /*bcrec*/,
  int /*level*/)
{
  auto const dat = datfab.const_array();
  auto cp_arr = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    const amrex::Real rho = dat(i, j, k, URHO);
#ifdef AMREX_USE_EB
    if (rho <= 0.) {
        cp_arr(i, j, k) = 0.;
    } else
#endif
    {
    amrex::Real mass[NUM_SPECIES];
    const amrex::Real rhoInv = 1.0 / dat(i, j, k, URHO);

    for (int n = 0; n < NUM_SPECIES; n++) {
      mass[n] = dat(i, j, k, UFS + n) * rhoInv;
    }
    amrex::Real cp;
    EOS::RTY2Cp(dat(i, j, k, URHO), dat(i, j, k, UTEMP), mass, cp);
    cp_arr(i, j, k) = cp;
    }
  });
}

