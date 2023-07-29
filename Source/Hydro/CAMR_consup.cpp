#include "CAMR.H"
#include "MOL_umeth.H"
#include "CAMR_hydro.H"
#include "CAMR_utils_K.H"

#ifdef AMREX_USE_EB
#include "CAMR_utils_eb_K.H"
#endif

using namespace amrex;

void
CAMR_consup(
    Box const& bx,
    Array4<Real> const& update,
    const amrex::GpuArray<const Array4<     Real>, AMREX_SPACEDIM> flx,
    Array4<const Real> const& vol,
    Array4<const Real> const& pdivu)
{
    // Take divergence of fluxes to define conservative update
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
        CAMR_update(i,j,k,update,flx,vol,pdivu);
    });
}

#ifdef AMREX_USE_EB
void
CAMR_consup_eb( const Box& bx,
                Array4<const Real> const& q_arr,
                Array4<const Real> const& qaux_arr,
                Array4<      Real> const& divc_arr,
                Array4<      Real> const& redistwgt_arr,
                AMREX_D_DECL(Array4<Real> const& q1,
                             Array4<Real> const& q2,
                             Array4<Real> const& q3),
                AMREX_D_DECL(
                Array4<Real       const> const& apx,
                Array4<Real       const> const& apy,
                Array4<Real       const> const& apz),
                AMREX_D_DECL(
                Array4<Real       const> const& fcx,
                Array4<Real       const> const& fcy,
                Array4<Real       const> const& fcz),
                Array4<const Real      > const& vfrac,
                Array4<amrex::EBCellFlag const> const& flag,
                const GpuArray<amrex::Real, AMREX_SPACEDIM> dxinv,
                const GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flux_tmp,
                const GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flux,
                Real small, Real small_dens, Real small_pres,
                const int l_eb_weights_type)
{
    const Box& bxg_i  = Box(divc_arr);

    // These are the fluxes we computed in MOL_umeth_eb and modified in adjust_fluxes_eb
    //  -- they live at face centers
    AMREX_D_TERM(auto const& fx_in_arr = flux_tmp[0];,
                 auto const& fy_in_arr = flux_tmp[1];,
                 auto const& fz_in_arr = flux_tmp[2];);

    // These are the fluxes on face centroids -- they are defined in eb_compute_div
    //    and are the fluxes that go into the flux registers
    AMREX_D_TERM(auto const& fx_out_arr = flux[0];,
                 auto const& fy_out_arr = flux[1];,
                 auto const& fz_out_arr = flux[2];);

    auto const& blo = bx.smallEnd();
    auto const& bhi = bx.bigEnd();

    amrex::ParallelFor(bxg_i, NVAR,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        // This does the divergence but not the redistribution -- we will do that later
        // We do compute the weights here though
        if (vfrac(i,j,k) > 0.) {
            eb_compute_div(i,j,k,n,blo,bhi,q_arr,qaux_arr,divc_arr,
                           AMREX_D_DECL(fx_in_arr ,fy_in_arr ,fz_in_arr),
                           AMREX_D_DECL(fx_out_arr,fy_out_arr,fz_out_arr),
                           flag, vfrac, redistwgt_arr,
                           AMREX_D_DECL(apx, apy, apz),
                           AMREX_D_DECL(fcx, fcy, fcz), dxinv,
                           small, small_dens, small_pres, l_eb_weights_type);
        }
    });

    // Update UEINT update with pdivu term
    amrex::ParallelFor(bxg_i,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (vfrac(i,j,k) > 0.) {
            eb_pdivu(i,j,k,q_arr,
                     AMREX_D_DECL(q1, q2, q3),
                     divc_arr, flag, vfrac,
                     AMREX_D_DECL(apx, apy, apz),
                     AMREX_D_DECL(fcx, fcy, fcz), dxinv);
        }
    });
}
#endif
