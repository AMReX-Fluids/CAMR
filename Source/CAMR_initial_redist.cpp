#include "CAMR.H"
#include "CAMR_hydro.H"
#include "CAMR_utils_K.H"
#include "Godunov.H"
#include "MOL_umeth.H"
#include "CAMR_Constants.H"

#include <AMReX_EB_Redistribution.H>
#include <AMReX_EBMultiFabUtil.H>

using namespace amrex;

#ifdef AMREX_USE_EB
void
CAMR::ApplyInitialMLRedistribution( Box const& bx,
                                    Geometry const& geom,
                                    const EBFArrayBoxFactory* ebfact,
                                    const int* bclo, const int* bchi,
                                    Array4<const Real> const& uin_arr,
                                    Array4<      Real> const& dsdt_arr,
                                    Array4<const Real> const& vf_arr,
                                    Array4<EBCellFlag const> const& flag_arr,
                                    int as_crse, Array4<Real> const& drho_as_crse,
                                    Array4<int const> const& rrflag_as_crse,
                                    int as_fine, Array4<Real> const& dm_as_fine,
                                    Array4<int const> const& lev_mask,
                                    const BCRec* bcs_d_ptr)
{
    BL_PROFILE_VAR("ApplyInitialMLRedist()", ApplyInitialMLRedist);

    const Box& bxg_ii = grow(bxg_i,1);

    Array4<Real const> AMREX_D_DECL(fcx, fcy, fcz), AMREX_D_DECL(apx, apy, apz), ccc;
    AMREX_D_TERM(fcx = ebfact->getFaceCent()[0]->const_array(mfi);,
                 fcy = ebfact->getFaceCent()[1]->const_array(mfi);,
                 fcz = ebfact->getFaceCent()[2]->const_array(mfi););
    AMREX_D_TERM(apx = ebfact->getAreaFrac()[0]->const_array(mfi);,
                 apy = ebfact->getAreaFrac()[1]->const_array(mfi);,
                 apz = ebfact->getAreaFrac()[2]->const_array(mfi););
    ccc = ebfact->getCentroid().const_array(mfi);

    int l_ncomp = dsdt_arr.nComp();
    int level_mask_not_covered = CAMRConstants::level_mask_notcovered;
    bool use_wts_in_divnc = false;

    // We need to set fac_for_redist to 1/2 for MOL because we will
    //    compute this twice per time step, so the contribution of
    //    each needs to be weighted by 1/2
    Real fac_for_redist = (do_mol) ? Real(0.5) : Real(1.0);
    if (redistribution_type == "StateRedist") {
        ApplyInitialMLRedistribution(bx, l_ncomp,
                                     dsdt_arr, divc_arr, uin_arr, redistwgt_arr,
                                     flag_arr,
                                     AMREX_D_DECL(apx, apy, apz),
                                     vf_arr,
                                     AMREX_D_DECL(fcx, fcy, fcz),
                                     ccc, bcs_d_ptr, geom, dt,
                                     l_redistribution_type,
                                     as_crse, drho_as_crse, rrflag_as_crse,
                                     as_fine, dm_as_fine, lev_mask,
                                     level_mask_not_covered,
                                     fac_for_redist);
    }

  BL_PROFILE_VAR_STOP(ApplyInitialMLRedist);
}
#endif
