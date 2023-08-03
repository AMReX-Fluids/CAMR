#include "CAMR.H"
#include "Godunov.H"
#include "CAMR_hydro.H"
#include "MOL_umeth.H"
#include "IndexDefines.H"
#include "MOL_hydro_eb_K.H"
#include "MOL_riemann_K.H"

#include "AMReX_MultiFabUtil.H"
#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>

#if (AMREX_SPACEDIM == 2)
#include <AMReX_EBMultiFabUtil_2D_C.H>
#elif (AMREX_SPACEDIM == 3)
#include <AMReX_EBMultiFabUtil_3D_C.H>
#endif

using namespace amrex;

void
MOL_umeth_eb (const Box& bx_to_fill,
              const int*  bclo, const int*  bchi,
              const int* domlo, const int* domhi,
              Array4<const Real> const& q_arr,
              Array4<const Real> const& qaux_arr,
              AMREX_D_DECL(Array4<Real> const& q1,
                           Array4<Real> const& q2,
                           Array4<Real> const& q3),
              Array4<const Real      > const& vfrac,
              Array4<amrex::EBCellFlag const> const& flag,
              const GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
              const GpuArray<const amrex::Array4<amrex::Real>, AMREX_SPACEDIM> flux_tmp,
              const amrex::Real small,
              const amrex::Real small_dens,
              const amrex::Real small_pres,
              const int l_plm_iorder,
              const int /*l_eb_weights_type*/)
{
    BL_PROFILE("MOL_umeth_eb()");

    AMREX_D_TERM(const int bclx = bclo[0];,
                 const int bcly = bclo[1];,
                 const int bclz = bclo[2];);
    AMREX_D_TERM(const int bchx = bchi[0];,
                 const int bchy = bchi[1];,
                 const int bchz = bchi[2];);
    AMREX_D_TERM(const int dlx = domlo[0];,
                 const int dly = domlo[1];,
                 const int dlz = domlo[2];);
    AMREX_D_TERM(const int dhx = domhi[0];,
                 const int dhy = domhi[1];,
                 const int dhz = domhi[2];);

    const PassMap* lpmap = CAMR::d_pass_map;

    Real l_plm_theta = 2.0; // [1,2] 1: minmod; 2: van Leer's MC

    // bx_to_fill  = Box(divc_arr);
    const Box& bxg_ii = grow(bx_to_fill,1);

    GpuArray<Real,AMREX_SPACEDIM> dxinv;
    AMREX_D_TERM(dxinv[0] = 1./dx[0];,
                 dxinv[1] = 1./dx[1];,
                 dxinv[2] = 1./dx[2];);

    // ****************************************************************
    // Slopes -- we will compute divc on bxg2 so need slopes on bxg3
    // ****************************************************************
    FArrayBox slopetmp;
    slopetmp.resize(bxg_ii,QVAR);
    auto const& slope = slopetmp.array();

    AMREX_D_TERM(auto const& fx_arr = flux_tmp[0];,
                 auto const& fy_arr = flux_tmp[1];,
                 auto const& fz_arr = flux_tmp[2];);

    // ****************************************************************
    // x-direction
    // ****************************************************************
    amrex::Box xbx = surroundingNodes(bx_to_fill,0); xbx.grow(IntVect(AMREX_D_DECL(0,1,1)));

    amrex::FArrayBox qxm(xbx, QVAR, amrex::The_Async_Arena());
    amrex::FArrayBox qxp(xbx, QVAR, amrex::The_Async_Arena());

    auto const& qxmarr = qxm.array();
    auto const& qxparr = qxp.array();

    amrex::ParallelFor(bxg_ii,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (vfrac(i,j,k) > 0.) {
            mol_slope_eb_x(i, j, k, slope, q_arr, qaux_arr, flag, small_dens, l_plm_iorder, l_plm_theta);
        }
    });

    amrex::ParallelFor(xbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (flag(i,j,k).isConnected(-1,0,0)) {
            mol_riemann_x(i, j, k, fx_arr, slope, q_arr, qaux_arr, q1, qxmarr, qxparr, small, small_dens, small_pres,
                          bclx, bchx, dlx, dhx, *lpmap);
        } else {
           for (int n = 0; n < NGDNV; n++) {
               q1(i,j,k,n) = 0.;
           }
        }
        fx_arr(i,j,k,UTEMP) = 0.0;
    });
    qxm.clear();
    qxp.clear();

    // ****************************************************************
    // y-direction
    // ****************************************************************
    amrex::Box ybx = surroundingNodes(bx_to_fill,1); ybx.grow(IntVect(AMREX_D_DECL(1,0,1)));

    amrex::FArrayBox qym(ybx, QVAR, amrex::The_Async_Arena());
    amrex::FArrayBox qyp(ybx, QVAR, amrex::The_Async_Arena());

    auto const& qymarr = qym.array();
    auto const& qyparr = qyp.array();

    amrex::ParallelFor(bxg_ii,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (vfrac(i,j,k) > 0.) {
            mol_slope_eb_y(i, j, k, slope, q_arr, qaux_arr, flag, small_dens, l_plm_iorder, l_plm_theta);
        }
    });

    amrex::ParallelFor(ybx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (flag(i,j,k).isConnected(0,-1,0)) {
            mol_riemann_y(i, j, k, fy_arr, slope, q_arr, qaux_arr, q2, qymarr, qyparr, small, small_dens, small_pres,
                          bcly, bchy, dly, dhy, *lpmap);
        } else {
           for (int n = 0; n < NGDNV; n++) {
               q2(i,j,k,n) = 0.;
           }
        }
        fy_arr(i,j,k,UTEMP) = 0.0;
    });
    qym.clear();
    qyp.clear();

#if (AMREX_SPACEDIM == 3)
    // ****************************************************************
    // z-direction
    // ****************************************************************
    amrex::Box zbx = surroundingNodes(bx_to_fill,2); zbx.grow(IntVect(1,1,0));

    amrex::FArrayBox qzm(zbx, QVAR, amrex::The_Async_Arena());
    amrex::FArrayBox qzp(zbx, QVAR, amrex::The_Async_Arena());

    auto const& qzmarr = qzm.array();
    auto const& qzparr = qzp.array();
    amrex::ParallelFor(bxg_ii,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (vfrac(i,j,k) > 0.) {
            mol_slope_eb_z(i, j, k, slope, q_arr, qaux_arr, flag, small_dens, l_plm_iorder, l_plm_theta);
        }
    });

    amrex::ParallelFor(zbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (flag(i,j,k).isConnected(0,0,-1))
        {
            mol_riemann_z(i, j, k, fz_arr, slope, q_arr, qaux_arr, q3, qzmarr, qzparr, small, small_dens, small_pres,
                          bclz, bchz, dlz, dhz, *lpmap);
        } else {
           for (int n = 0; n < NGDNV; n++) {
               q3(i,j,k,n) = 0.;
           }
        }
        fz_arr(i,j,k,UTEMP) = 0.0;
    });
    qzm.clear();
    qzp.clear();
#endif
}
