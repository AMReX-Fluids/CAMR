#include "CAMR.H"
#include "CAMR_hydro.H"
#include "CAMR_ctoprim.H"

using namespace amrex;

void
CAMR::construct_hydro_source (const MultiFab& S,
                              MultiFab& src_to_fill,
                              Real /*time*/,
                              Real dt)
{
    src_to_fill.setVal(0);

    if (verbose) {
        if (do_mol) {
            amrex::Print() << "... Computing MOL-based hydro advance" << std::endl;
        } else {
            amrex::Print() << "... Computing Godunov-based hydro advance" << std::endl;
        }
    }

    Real fac_for_reflux = (do_mol) ? Real(0.5) : Real(1.0);

    AMREX_ASSERT(S.nGrow() == numGrow());

    // Fill the source terms to go into the hydro with only the old-time sources
    int ng = 0;

    for (int n = 0; n < src_list.size(); ++n) {
        MultiFab::Saxpy(sources_for_hydro, 1.0, *old_sources[src_list[n]], 0, 0, NVAR, ng);
    }
    sources_for_hydro.FillBoundary(geom.periodicity());

    int finest_level = parent->finestLevel();

    const auto& dx    = geom.CellSizeArray();

    Real dx1 = dx[0];
    for (int dir = 1; dir < AMREX_SPACEDIM; ++dir) {
      dx1 *= dx[dir];
    }

    std::array<Real, AMREX_SPACEDIM> dxD = {
      {AMREX_D_DECL(dx1, dx1, dx1)}};
    const Real* dxDp = &(dxD[0]);

    MultiFab& S_new = get_new_data(State_Type);

    BL_PROFILE_VAR("CAMR::advance_hydro_umdrv()", PC_UMDRV);

#ifdef AMREX_USE_EB
    const auto& ebfact = dynamic_cast<amrex::EBFArrayBoxFactory const&>(Factory());
#endif

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
#ifdef AMREX_USE_EB
      int ncomp = src_to_fill.nComp();
      FArrayBox dm_as_fine(Box::TheUnitBox(),ncomp);
      FArrayBox fab_drho_as_crse(Box::TheUnitBox(),ncomp);
      IArrayBox fab_rrflag_as_crse(Box::TheUnitBox());
#endif

      amrex::MFItInfo tiling = amrex::TilingIfNotGPU() ? amrex::MFItInfo().EnableTiling(hydro_tile_size) : amrex::MFItInfo();
      for (MFIter mfi(S_new, tiling); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.tilebox();

#ifdef AMREX_USE_EB
        EBCellFlagFab const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
        auto const& flag_arr = flagfab.const_array();

        if (flagfab.getType(bx) != FabType::covered) {
            auto const& vfrac_arr = volfrac->const_array(mfi);
#endif
            const Box& qbx = amrex::grow(bx, numGrow());

            amrex::GpuArray<amrex::FArrayBox, AMREX_SPACEDIM> flux;
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
              const Box& efbx = amrex::surroundingNodes(bx, dir);
              flux[dir].resize(efbx, NVAR, amrex::The_Async_Arena());
              flux[dir].setVal<RunOn::Device>(0.);
            }

            auto const& sarr    = S.const_array(mfi);
            auto const& hyd_src = src_to_fill.array(mfi);

            // Resize Temporary Fabs
            FArrayBox q(qbx, QVAR, amrex::The_Async_Arena());
            FArrayBox qaux(qbx, NQAUX, amrex::The_Async_Arena());
            FArrayBox src_q(qbx, QVAR, amrex::The_Async_Arena());

            // Get Arrays to pass to the gpu.
            auto const& qarr    = q.array();
            auto const& qauxar  = qaux.array();
            auto const& srcqarr = src_q.array();

            BL_PROFILE_VAR("CAMR::ctoprim()", ctop);
            const PassMap* lpmap = d_pass_map;
            const Real small_num        = CAMRConstants::small_num;
            const Real dual_energy_eta  = CAMR::dual_energy_eta1;
            ParallelFor(
              qbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#ifdef AMREX_USE_EB
                if (!flag_arr(i,j,k).isCovered()) {
#endif
                    CAMR_ctoprim(i, j, k, sarr, qarr, qauxar, *lpmap, small_num, dual_energy_eta);
#ifdef AMREX_USE_EB
                } else {
                   for (int n=0; n<QVAR; n++) qarr(i,j,k,n) = 0.;
                }
#endif
            });
            BL_PROFILE_VAR_STOP(ctop);

            const GpuArray<const Array4<      Real>, AMREX_SPACEDIM>
              flx_arr{{AMREX_D_DECL(flux[0].array(), flux[1].array(), flux[2].array())}};
            const amrex::GpuArray<const Array4<const Real>, AMREX_SPACEDIM>
              a{{AMREX_D_DECL(area[0].array(mfi), area[1].array(mfi), area[2].array(mfi))}};

        // Create source terms for primitive variables
        if (!do_mol) {
            const auto& src_in = sources_for_hydro.array(mfi);
            ParallelFor(
              qbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                CAMR_srctoprim(i, j, k, qarr, qauxar, src_in, srcqarr, *lpmap);
              });
        }

#ifdef AMREX_USE_EB
        int ngrow_bx;
        if (redistribution_type == "StateRedist") {
           ngrow_bx = 3;
        } else {
           ngrow_bx = 2;
        }
        const Box& bxg_i  = grow(bx,ngrow_bx);
        if (flagfab.getType(bxg_i) != FabType::regular) {

            EBFluxRegister* fr_as_crse = nullptr;
            if (do_reflux && level < parent->finestLevel()) {
                CAMR& fine_level = getLevel(level+1);
                fr_as_crse = &fine_level.flux_reg;
            }

            EBFluxRegister* fr_as_fine = nullptr;
            if (do_reflux && level > 0) {
                fr_as_fine = &flux_reg;
            }

            int as_crse = (fr_as_crse != nullptr);
            int as_fine = (fr_as_fine != nullptr);

            FArrayBox* p_drho_as_crse = (fr_as_crse) ?
                    fr_as_crse->getCrseData(mfi) : &fab_drho_as_crse;
            const IArrayBox* p_rrflag_as_crse = (fr_as_crse) ?
                   fr_as_crse->getCrseFlag(mfi) : &fab_rrflag_as_crse;

            if (fr_as_fine) {
                const Box dbox1 = geom.growPeriodicDomain(1);
                Box bx_for_dm(amrex::grow(bx,1) & dbox1);
                dm_as_fine.resize(bx_for_dm,ncomp);
                dm_as_fine.setVal<RunOn::Device>(0.0);
            }

            const amrex::StateDescriptor* desc = state[State_Type].descriptor();
            const auto& bcs = desc->getBCs();
            amrex::Gpu::DeviceVector<amrex::BCRec> bcs_d(desc->nComp());
            amrex::Gpu::copy(
              amrex::Gpu::hostToDevice, bcs.begin(), bcs.end(), bcs_d.begin());

            const auto& dxInv = geom.InvCellSizeArray();

            // Return hyd_src - centered at half-time if using Godunov method
            //                - centered at  old-time if using MOL method
            //
            // The dt we pass in here is used if (do_mol == 0), i.e.
            //      in the Godunov prediction, but also if we do StateRedistribution
            //
            CAMR_umdrv_eb(do_mol, bx, bxg_i, mfi, geom, &ebfact,
                          phys_bc.lo(), phys_bc.hi(),
                          sarr, hyd_src, qarr, qauxar, srcqarr,
                          vfrac_arr, flag_arr, dx, dxInv, flx_arr, a,
                          as_crse, p_drho_as_crse->array(), p_rrflag_as_crse->array(),
                          as_fine, dm_as_fine.array(), level_mask.const_array(mfi),
                          dt, ppm_type, plm_iorder, use_pslope,
                          use_flattening, transverse_reset_density,
                          small, small_dens, small_pres, difmag,
                          bcs_d.data(), redistribution_type, eb_weights_type);

            //
            // Here fac_for_reflux = 1.0 if doing Godunov, 0.5 if doing MOL
            //
            if (do_reflux) {
                if (level < finest_level) {
                     getFluxReg(level + 1).CrseAdd(mfi,
                        {{AMREX_D_DECL(&(flux[0]), &(flux[1]), &(flux[2]))}},
                        dxDp, fac_for_reflux*dt, (*volfrac)[mfi],
                        {AMREX_D_DECL(&(*areafrac[0])[mfi], &(*areafrac[1])[mfi], &(*areafrac[2])[mfi])},
                        amrex::RunOn::Device);
                }
                if (level > 0) {
                    getFluxReg(level).FineAdd(mfi,
                       {{AMREX_D_DECL(&(flux[0]), &(flux[1]), &(flux[2]))}},
                       dxDp, fac_for_reflux*dt, (*volfrac)[mfi],
                       {AMREX_D_DECL(&(*areafrac[0])[mfi], &(*areafrac[1])[mfi], &(*areafrac[2])[mfi])},
                       dm_as_fine, amrex::RunOn::Device);
                } // level > 0
            } // do_reflux
        } else {
#endif
            // Return hyd_src - centered at half-time if using Godunov method
            //                - centered at  old-time if using MOL method
            //
            // Note that the dt here is only used if (do_mol == 0), i.e.
            //      in the Godunov prediction
            //
            CAMR_umdrv(do_mol, bx, geom, phys_bc.lo(), phys_bc.hi(),
                       sarr, hyd_src, qarr, qauxar, srcqarr, dx,
                       dt, ppm_type, plm_iorder, use_pslope,
                       use_flattening, transverse_reset_density,
                       small, small_dens, small_pres, difmag,
                       flx_arr, a, volume.array(mfi));

            //
            // Here fac_for_reflux = 1.0 if doing Godunov, 0.5 if doing MOL
            //
            if (do_reflux) {
                if (level < finest_level) {
                    getFluxReg(level + 1).CrseAdd(mfi,
                        {{AMREX_D_DECL(&(flux[0]), &(flux[1]), &(flux[2]))}},
                        dxDp, fac_for_reflux*dt, amrex::RunOn::Device);
                }
                if (level > 0) {
                    getFluxReg(level).FineAdd(mfi,
                       {{AMREX_D_DECL(&(flux[0]), &(flux[1]), &(flux[2]))}},
                       dxDp, fac_for_reflux*dt, amrex::RunOn::Device);
                }
            } // do_reflux

#ifdef AMREX_USE_EB
        } // regular
#endif

#ifdef AMREX_USE_EB
        } // not covered
#endif
      } // mfi
    } // openmp

    BL_PROFILE_VAR_STOP(PC_UMDRV);
}
