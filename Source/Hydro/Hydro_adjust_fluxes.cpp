#include "Hydro_utils_K.H"
#include "hydro_artif_visc.H"

#include "AMReX_MultiFab.H"

#ifdef AMREX_USE_EB
#include "Hydro_utils_eb_K.H"
#endif

using namespace amrex;

void
adjust_fluxes (
    const Box& bx,
    Array4<const Real> const& u,
    const amrex::GpuArray<const Array4<     Real>, AMREX_SPACEDIM> flx,
    const amrex::GpuArray<const Array4<const Real>, AMREX_SPACEDIM> a,
    Array4<const Real> const& divu,
    const amrex::GpuArray<Real, AMREX_SPACEDIM> del,
    const int* domlo, const int* domhi,
    const int*  bclo, const int*  bchi,
    Real const l_difmag)
{
  // Flux alterations
  for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
    Box const& fbx = surroundingNodes(bx, dir);
    const Real dx = del[dir];
    int domlo_dir = domlo[dir];
    int domhi_dir = domhi[dir];
    int  bclo_dir =  bclo[dir];
    int  bchi_dir =  bchi[dir];
    amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

      hydro_artif_visc(i, j, k, flx[dir], divu, u, dx, l_difmag, dir,
                       domlo_dir, domhi_dir, bclo_dir, bchi_dir);

      // Normalize Species Flux
      hydro_norm_spec_flx(i, j, k, flx[dir]);

      // Make flux extensive
      hydro_ext_flx(i, j, k, flx[dir], a[dir]);
    });
  } // dir
}

#ifdef AMREX_USE_EB
void
adjust_fluxes_eb (
              const Box& /*bx*/,
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
              const Geometry& geom,
              const int*  bclo, const int*  bchi,
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

    Box nddom = amrex::convert(geom.growPeriodicDomain(16), IntVect(1));
    bx_divu &= nddom;

    auto const& divu_arr = divu.array();
    amrex::ParallelFor(bx_divu, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        eb_divu(i, j, k, q_arr, divu_arr, vfrac, dxinv);
    });

    // Flux alterations
    Box const& fbx = Box(fx_arr);
#if (AMREX_SPACEDIM == 2)
    areafac = dx[1];
#elif (AMREX_SPACEDIM == 3)
    areafac = dx[1]*dx[2];
#endif

    auto const* domlo = geom.Domain().loVect();
    auto const* domhi = geom.Domain().hiVect();

    int domlo_dir = domlo[0];
    int domhi_dir = domhi[0];
    int  bclo_dir =  bclo[0];
    int  bchi_dir =  bchi[0];

    amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
        hydro_artif_visc(i, j, k, fx_arr, divu_arr, u_arr, dx[0], l_difmag, 0,
                        domlo_dir, domhi_dir, bclo_dir, bchi_dir);

        // Normalize Species Flux
        if (apx(i,j,k) > 0. ) {
          hydro_norm_spec_flx(i, j, k, fx_arr);
        }

        // Make flux extensive
        hydro_ext_flx_eb(i, j, k, fx_arr, areafac, apx);
    });

    // Flux alterations
    Box const& fby = Box(fy_arr);
#if (AMREX_SPACEDIM == 2)
    areafac = dx[0];
#elif (AMREX_SPACEDIM == 3)
    areafac = dx[0]*dx[2];
#endif

    domlo_dir = domlo[1];
    domhi_dir = domhi[1];
     bclo_dir =  bclo[1];
     bchi_dir =  bchi[1];

    amrex::ParallelFor(fby, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

          hydro_artif_visc(i, j, k, fy_arr, divu_arr, u_arr, dx[1], l_difmag, 1,
                          domlo_dir, domhi_dir, bclo_dir, bchi_dir);

          // Normalize Species Flux
          if(apy(i,j,k) > 0. ) {
              hydro_norm_spec_flx(i, j, k, fy_arr);
          }

          // Make flux extensive
          hydro_ext_flx_eb(i, j, k, fy_arr, areafac, apy);
    });

#if (AMREX_SPACEDIM ==3)
     // Flux alterations
     Box const& fbz = Box(fz_arr);
     areafac = dx[1]*dx[0];

    domlo_dir = domlo[2];
    domhi_dir = domhi[2];
     bclo_dir =  bclo[2];
     bchi_dir =  bchi[2];

    amrex::ParallelFor(fbz, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

          hydro_artif_visc(i, j, k, fz_arr, divu_arr, u_arr, dx[2], l_difmag, 2,
                          domlo_dir, domhi_dir, bclo_dir, bchi_dir);

          // Normalize Species Flux
          if(apz(i,j,k) > 0. ) {
             hydro_norm_spec_flx(i, j, k, fz_arr);
          }

          // Make flux extensive
          hydro_ext_flx_eb(i, j, k, fz_arr, areafac, apz);
      });

#endif

    Gpu::streamSynchronize();
}
#endif
