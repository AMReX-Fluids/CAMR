#include "CAMR.H"
#include "CAMR_hydro.H"
#include "MOL_umeth.H"
#include "MOL_hydro_eb_K.H"
#include "CAMR_Constants.H"

#include <AMReX_EB_Redistribution.H>
#include <AMReX_EBMultiFabUtil.H>

using namespace amrex;

#ifdef AMREX_USE_EB
void
CAMR_umdrv_eb( const bool do_mol, Box const& bx, const MFIter& mfi,
               Geometry const& geom,
               const EBFArrayBoxFactory* ebfact,
               const int* bclo, const int* bchi,
               Array4<const Real> const& s_arr,
               Array4<const Real> const& q_arr,
               Array4<      Real> const& dsdt_arr,
               Array4<const Real> const& qaux_arr,
               Array4<const Real> const& src_q,
               Array4<const Real> const& vf_arr,
               Array4<EBCellFlag const> const& flag_arr,
               const GpuArray<Real, AMREX_SPACEDIM> dx,
               const GpuArray<Real, AMREX_SPACEDIM> dxinv,
               const GpuArray<const Array4<Real>, AMREX_SPACEDIM> flux_arr,
               int as_crse,
               Array4<Real> const& drho_as_crse,
               Array4<int const> const& rrflag_as_crse,
               int as_fine,
               Array4<Real> const& dm_as_fine,
               Array4<int const> const& lev_mask,
               const  Real difmag, const  Real dt,
               const Real small,
               const Real small_dens,
               const Real small_pres,
               const BCRec* bcs_d_ptr,
               const std::string& l_redistribution_type,
               const int l_plm_iorder,
               const int ppm_type,
               const int use_pslope,
               const int use_flattening,
               const int transverse_reset_density,
               const int l_eb_weights_type,
               Array4<Real> const& vol)
{
    BL_PROFILE_VAR("CAMR_umdrv_eb()", CAMR_umdrv_eb);

    int ngrow_bx;
    if (l_redistribution_type == "StateRedist") {
       ngrow_bx = 3;
    } else {
       ngrow_bx = 2;
    }
    const Box& bxg_i  = grow(bx,ngrow_bx);
    const Box& bxg_ii = grow(bx,ngrow_bx+1);

    Array4<Real const> AMREX_D_DECL(fcx, fcy, fcz), AMREX_D_DECL(apx, apy, apz), ccc;
    AMREX_D_TERM(fcx = ebfact->getFaceCent()[0]->const_array(mfi);,
                 fcy = ebfact->getFaceCent()[1]->const_array(mfi);,
                 fcz = ebfact->getFaceCent()[2]->const_array(mfi););
    AMREX_D_TERM(apx = ebfact->getAreaFrac()[0]->const_array(mfi);,
                 apy = ebfact->getAreaFrac()[1]->const_array(mfi);,
                 apz = ebfact->getAreaFrac()[2]->const_array(mfi););
    ccc = ebfact->getCentroid().const_array(mfi);

    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

    amrex::FArrayBox qec[AMREX_SPACEDIM];
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
      const Box eboxes = amrex::surroundingNodes(bxg_ii, dir);
      qec[dir].resize(eboxes, NGDNV, amrex::The_Async_Arena());
    }
    amrex::GpuArray<Array4<Real>, AMREX_SPACEDIM> qec_arr{
      {AMREX_D_DECL(qec[0].array(), qec[1].array(), qec[2].array())}};

    // ****************************************************************
    // Quantities for redistribution
    // ****************************************************************
    FArrayBox divc,redistwgt;

    if (l_redistribution_type == "StateRedist") {
             divc.resize(bxg_i,NVAR); // This will hold "dUdt" before redistribution
        redistwgt.resize(bxg_i,NVAR); // This will be "scratch" which holds "Uold + dt*dUdt"
    } else {
             divc.resize(bxg_i,NVAR); // This will hold "dUdt" before redistribution
        redistwgt.resize(bxg_i,1);    // This will hold the weights used in flux redistribution
    }

    // Set to zero just in case
         divc.setVal<RunOn::Device>(0.0);
    redistwgt.setVal<RunOn::Device>(0.0);

    // Because we are going to redistribute, we put the divergence into divc
    //    rather than directly into dsdt_arr
    auto const& divc_arr = divc.array();
    auto const& redistwgt_arr = redistwgt.array();

    // ****************************************************************
    // We need one extra in flux_tmp so we can tangentially interpolate fluxes
    // ****************************************************************
    FArrayBox flux_tmp[AMREX_SPACEDIM];
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        flux_tmp[idim].resize(amrex::surroundingNodes(bxg_ii,idim),NVAR);
        flux_tmp[idim].setVal<RunOn::Device>(0.);
    }

    const GpuArray<const Array4<      Real>, AMREX_SPACEDIM>
      flux_tmp_arr{{AMREX_D_DECL(flux_tmp[0].array(), flux_tmp[1].array(), flux_tmp[2].array())}};

    // ****************************************************************
    // Define divc, the update before redistribution
    // Also construct the redistribution weights for flux redistribution if necessary
    // ****************************************************************
    if (do_mol) {
        mol_umeth_eb(bx, bclo, bchi, domlo, domhi, q_arr, qaux_arr, divc_arr,
                     AMREX_D_DECL(qec_arr[0], qec_arr[1], qec_arr[2]), vf_arr,
                     flag_arr, dx, flux_tmp_arr, small, small_dens, small_pres,
                     l_plm_iorder, l_eb_weights_type);
    } else { // if Godunov
#if AMREX_SPACEDIM == 2
      CAMR_umeth_2D_eb(
#elif AMREX_SPACEDIM == 3
      CAMR_umeth_3D_eb(
#endif
        bx, bclo, bchi, domlo, domhi, q_arr, qaux_arr, src_q,
        AMREX_D_DECL(flux_tmp_arr[0], flux_tmp_arr[1], flux_tmp_arr[2]),
        AMREX_D_DECL(qec_arr[0], qec_arr[1], qec_arr[2]),
        AMREX_D_DECL(apx, apy, apz),
        vol, dx, dt,
        small, small_dens, small_pres, ppm_type, use_pslope, use_flattening,
        l_plm_iorder, transverse_reset_density);

      amrex::Abort("Not implemented yet");
      } // end Godunov

    adjust_fluxes_eb(bx, q_arr, s_arr,
                     AMREX_D_DECL(apx, apy, apz),
                     vf_arr, dx, dxinv, flux_tmp_arr, difmag);

    CAMR_consup_eb(bx, q_arr, qaux_arr,
                   divc_arr, redistwgt_arr,
                   AMREX_D_DECL(qec_arr[0], qec_arr[1], qec_arr[2]),
                   AMREX_D_DECL(apx, apy, apz),
                   AMREX_D_DECL(fcx, fcy, fcz),
                   vf_arr, flag_arr, dxinv,
                   flux_tmp_arr, flux_arr,
                   small, small_dens, small_pres,
                   l_eb_weights_type);

    int l_ncomp = dsdt_arr.nComp();
    int level_mask_not_covered = CAMRConstants::level_mask_notcovered;
    bool use_wts_in_divnc = false;
    ApplyMLRedistribution(bx, l_ncomp,
                          dsdt_arr, divc_arr, s_arr, redistwgt_arr,
                          flag_arr,
                          AMREX_D_DECL(apx, apy, apz),
                          vf_arr,
                          AMREX_D_DECL(fcx, fcy, fcz),
                          ccc, bcs_d_ptr, geom, dt,
                          l_redistribution_type,
                          as_crse, drho_as_crse, rrflag_as_crse,
                          as_fine, dm_as_fine, lev_mask,
                          level_mask_not_covered,
                          use_wts_in_divnc);

  BL_PROFILE_VAR_STOP(CAMR_umdrv_eb);
}
#endif
