#include "CAMR.H"
#include "MOL_umeth.H"
#include "CAMR_hydro.H"
#include "hydro_artif_visc.H"

#ifdef AMREX_USE_EB
#include "MOL_hydro_eb_K.H"
#include "MOL_eb_divu_K.H"
#endif

using namespace amrex;

void
adjust_fluxes(
    const Box& bx,
    Array4<const Real> const& u,
    const amrex::GpuArray<const Array4<     Real>, AMREX_SPACEDIM> flx,
    const amrex::GpuArray<const Array4<const Real>, AMREX_SPACEDIM> a,
    Array4<const Real> const& divu,
    const amrex::GpuArray<Real, AMREX_SPACEDIM> del,
    Real const l_difmag)
{
  // Flux alterations
  for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
    Box const& fbx = surroundingNodes(bx, dir);
    const Real dx = del[dir];
    amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

      CAMR_artif_visc(i, j, k, flx[dir], divu, u, dx, l_difmag, dir);

      // Normalize Species Flux
      CAMR_norm_spec_flx(i, j, k, flx[dir]);

      // Make flux extensive
      CAMR_ext_flx(i, j, k, flx[dir], a[dir]);
    });
  } // dir
}

#ifdef AMREX_USE_EB
void
adjust_fluxes_eb(
              const Box& bx,
              Array4<const Real> const& q_arr,
              Array4<const Real> const& u_arr,
              AMREX_D_DECL(
              Array4<Real       const> const& apx,
              Array4<Real       const> const& apy,
              Array4<Real       const> const& apz),
              Array4<const Real      > const& vfrac,
              const GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
              const GpuArray<amrex::Real, AMREX_SPACEDIM> dxinv,
              const GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flux,
              Real l_difmag)
{
    Real areafac;

    // These are the fluxes on face centroids -- they are defined in eb_compute_div
    //    and are the fluxes that go into the flux registers
    AMREX_D_TERM(auto const& fx_arr = flux[0];,
                 auto const& fy_arr = flux[1];,
                 auto const& fz_arr = flux[2];);

    // Compute divu to be used in artificial viscosity
    Box bx_divu = Box(q_arr);  bx_divu.grow(-1); bx_divu.surroundingNodes();
    FArrayBox divu(bx_divu,NVAR);
    divu.setVal<RunOn::Device>(0.0);

    auto const& divu_arr = divu.array();
    amrex::ParallelFor(bx_divu, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (vfrac(i,j,k) > 0.) {
            eb_divu(i, j, k, q_arr, divu_arr, vfrac, dxinv);
        }
    });

    // Flux alterations
    Box const& fbx = Box(fx_arr);
#if (AMREX_SPACEDIM == 2)
    areafac = dx[1];
#elif (AMREX_SPACEDIM == 3)
    areafac = dx[1]*dx[2];
#endif
    amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

          CAMR_artif_visc(i, j, k, fx_arr, divu_arr, u_arr, dx[0], l_difmag, 0);

          // Normalize Species Flux
          if(apx(i,j,k) > 0. ) {
            CAMR_norm_spec_flx(i, j, k, fx_arr);
          }
          // Make flux extensive
          CAMR_ext_flx_eb(i, j, k, fx_arr, areafac, apx);
    });

    // Flux alterations
    Box const& fby = Box(fy_arr);
#if (AMREX_SPACEDIM == 2)
    areafac = dx[0];
#elif (AMREX_SPACEDIM == 3)
    areafac = dx[0]*dx[2];
#endif
    amrex::ParallelFor(fby, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

          CAMR_artif_visc(i, j, k, fy_arr, divu_arr, u_arr, dx[1], l_difmag, 1);

          // Normalize Species Flux
          if(apy(i,j,k) > 0. ) {
              CAMR_norm_spec_flx(i, j, k, fy_arr);
          }

          // Make flux extensive
          CAMR_ext_flx_eb(i, j, k, fy_arr, areafac, apy);
    });

#if (AMREX_SPACEDIM ==3)
     // Flux alterations
     Box const& fbz = Box(fz_arr);
     areafac = dx[1]*dx[0];

    amrex::ParallelFor(fbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

          CAMR_artif_visc(i, j, k, fz_arr, divu_arr, u_arr, dx[2], l_difmag, 2);

          // Normalize Species Flux
          if(apz(i,j,k) > 0. ) {
             CAMR_norm_spec_flx(i, j, k, fz_arr);
          }

          // Make flux extensive
          CAMR_ext_flx_eb(i, j, k, fz_arr, areafac, apz);
      });

#endif

    Gpu::streamSynchronize();
}
#endif
