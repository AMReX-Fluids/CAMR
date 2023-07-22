#include "CAMR.H"

void
CAMR::construct_old_grav_source (amrex::Real /*time*/, amrex::Real /*dt*/)
{
  old_sources[grav_src]->setVal(0.0);

  if (!add_grav_src) {
    return;
  }

  const amrex::MultiFab& S_old = get_old_data(State_Type);
  int ng = S_old.nGrow();
  fill_grav_source(S_old, *old_sources[grav_src], ng);

  old_sources[grav_src]->FillBoundary(geom.periodicity());
}

void
CAMR::construct_new_grav_source (amrex::Real /*time*/, amrex::Real /*dt*/)
{
  new_sources[grav_src]->setVal(0.0);

  if (!add_grav_src) {
    return;
  }

  const amrex::MultiFab& S_new = get_new_data(State_Type);
  const amrex::MultiFab& S_old = get_old_data(State_Type);
  int ng = 0;

  fill_gravcorr_source(S_old, S_new, *new_sources[grav_src], ng);
}

void
CAMR::fill_grav_source (const amrex::MultiFab& S,
                        amrex::MultiFab& grav_src,
                        int ng)
{
#ifdef AMREX_USE_EB
  auto const& fact =
    dynamic_cast<amrex::EBFArrayBoxFactory const&>(S.Factory());
  auto const& flags = fact.getMultiEBCellFlagFab();
#endif

  const auto l_const_grav = const_grav;

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(grav_src, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const amrex::Box& bx = mfi.growntilebox(ng);

#ifdef AMREX_USE_EB
    const auto& flag_fab = flags[mfi];
    amrex::FabType typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
      continue;
    }
#endif

    auto const& s   = S.array(mfi);
    auto const& src = grav_src.array(mfi);

    // Evaluate gravity-related source terms (assuming gravity only in z-direction)
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

#if (AMREX_SPACEDIM == 2)
      src(i, j, k, UMY  ) = l_const_grav * s(i,j,k,URHO);  // rho g
      src(i, j, k, UEDEN) = l_const_grav * s(i,j,k,UMY);   // rho u g
#else
      src(i, j, k, UMZ  ) = l_const_grav * s(i,j,k,URHO);  // rho g
      src(i, j, k, UEDEN) = l_const_grav * s(i,j,k,UMZ);   // rho u g
#endif
    });
  }
}

void
CAMR::fill_gravcorr_source(
  const amrex::MultiFab& S_old,
  const amrex::MultiFab& S_new,
  amrex::MultiFab& grav_src, int ng)
{
#ifdef AMREX_USE_EB
  auto const& fact =
    dynamic_cast<amrex::EBFArrayBoxFactory const&>(S_old.Factory());
  auto const& flags = fact.getMultiEBCellFlagFab();
#endif

  const auto l_const_grav = const_grav;

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(grav_src, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const amrex::Box& bx = mfi.growntilebox(ng);

#ifdef AMREX_USE_EB
    const auto& flag_fab = flags[mfi];
    amrex::FabType typ = flag_fab.getType(bx);
    if (typ == amrex::FabType::covered) {
      continue;
    }
#endif

    auto const& sold = S_old.const_array(mfi);
    auto const& snew = S_new.const_array(mfi);
    auto const& src  = grav_src.array(mfi);

    // Evaluate the correction to the gravity source term, i.e. return 1/2(Src_new - Src_old)
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

#if (AMREX_SPACEDIM == 2)
      src(i, j, k, UMY  ) = 0.5 * l_const_grav * (snew(i,j,k,URHO) - sold(i,j,k,URHO));  // rho g
      src(i, j, k, UEDEN) = 0.5 * l_const_grav * (snew(i,j,k,UMY ) - sold(i,j,k,UMY ));  // rho u g
#else
      src(i, j, k, UMZ  ) = 0.5 * l_const_grav * (snew(i,j,k,URHO) - sold(i,j,k,URHO));  // rho g
      src(i, j, k, UEDEN) = 0.5 * l_const_grav * (snew(i,j,k,UMZ ) - sold(i,j,k,UMZ ));  // rho u g
#endif
    });
  }
}
