#include "MOL_hydro_K.H"
#include "MOL_riemann_K.H"
#include "Godunov.H"
#include "IndexDefines.H"
#include "AMReX_MultiFabUtil.H"

#ifdef AMREX_USE_EB
#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>
#endif

using namespace amrex;

void
mol_umeth (const Box& bx,
           const int* bclo,
           const int* bchi,
           const int* domlo,
           const int* domhi,
           Array4<const Real> const& q,
           Array4<const Real> const& qa,
           AMREX_D_DECL(Array4<Real> const& fx,
                        Array4<Real> const& fy,
                        Array4<Real> const& fz),
           AMREX_D_DECL(Array4<Real> const& q1,
                        Array4<Real> const& q2,
                        Array4<Real> const& q3),
           AMREX_D_DECL(Array4<const Real> const& a1,
                        Array4<const Real> const& a2,
                        Array4<const Real> const& a3),
           Array4<Real> const& pdivu,
           Array4<const Real> const& vol,
           const amrex::Real small,
           const amrex::Real small_dens,
           const amrex::Real small_pres,
           const int iorder)
{
    BL_PROFILE("CAMR::mol_umeth()");

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

    FArrayBox slopetmp;

    const Box& bxg1 = amrex::grow(bx,1);
    slopetmp.resize(bxg1,QVAR);
    auto const& slope = slopetmp.array();

//    amrex::Real slope[QVAR]

    const PassMap* lpmap = CAMR::d_pass_map;

    Real l_plm_theta = 2.0; // [1,2] 1: minmod; 2: van Leer's MC

    // x-direction
    int cdir = 0;
    const amrex::Box& bxg2 = grow(bx, 2);
    const amrex::Box& xmbx = growHi(bxg2, cdir, 1);
    amrex::FArrayBox qxm(xmbx, QVAR, amrex::The_Async_Arena());
    amrex::FArrayBox qxp(bxg2, QVAR, amrex::The_Async_Arena());
    auto const& qxmarr = qxm.array();
    auto const& qxparr = qxp.array();


    const Box& xslpbx = amrex::grow(bx, cdir, 1);
    amrex::ParallelFor(xslpbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        mol_slope_x(i, j, k, slope, q, qa, iorder, l_plm_theta, small_dens);
//        mol_pred_x(i, j, k, slope, q, qa, iorder, l_plm_theta);
    });
    const Box& xflxbx = amrex::surroundingNodes(bx,cdir);
    amrex::ParallelFor(xflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        mol_riemann_x(i, j, k, fx, slope, q, qa, q1, qxmarr, qxparr, small, small_dens, small_pres,
                      bclx, bchx, dlx, dhx, *lpmap);
    });

    // y-direction
    cdir = 1;
    const amrex::Box& ymbx = growHi(bxg2, cdir, 1);
    amrex::FArrayBox qym(ymbx, QVAR, amrex::The_Async_Arena());
    amrex::FArrayBox qyp(bxg2, QVAR, amrex::The_Async_Arena());
    auto const& qymarr = qym.array();
    auto const& qyparr = qyp.array();

    const Box& yslpbx = amrex::grow(bx, cdir, 1);
    amrex::ParallelFor(yslpbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        mol_slope_y(i, j, k, slope, q, qa, iorder, l_plm_theta, small_dens);
    });
    const Box& yflxbx = amrex::surroundingNodes(bx,cdir);
    amrex::ParallelFor(yflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        mol_riemann_y(i, j, k, fy, slope, q, qa, q2, qymarr, qyparr, small, small_dens, small_pres,
                      bcly, bchy, dly, dhy, *lpmap);
    });

#if (AMREX_SPACEDIM == 3)

    // z-direction
    cdir = 2;
    const amrex::Box& zmbx = growHi(bxg2, cdir, 1);
    amrex::FArrayBox qzm(zmbx, QVAR, amrex::The_Async_Arena());
    amrex::FArrayBox qzp(bxg2, QVAR, amrex::The_Async_Arena());
    auto const& qzmarr = qzm.array();
    auto const& qzparr = qzp.array();

    const Box& zslpbx = amrex::grow(bx, cdir, 1);
    amrex::ParallelFor(zslpbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        mol_slope_z(i, j, k, slope, q, qa, iorder, l_plm_theta, small_dens);
    });
    const Box& zflxbx = amrex::surroundingNodes(bx,cdir);
    amrex::ParallelFor(zflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        mol_riemann_z(i, j, k, fz, slope, q, qa, q3, qzmarr, qzparr, small, small_dens, small_pres,
                      bclz, bchz, dlz, dhz, *lpmap);
    });

#endif

   // Construct p div{U}
   amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
     CAMR_pdivu(i, j, k, pdivu, AMREX_D_DECL(q1, q2, q3), AMREX_D_DECL(a1, a2, a3), vol);
   });

   Gpu::streamSynchronize();
}
