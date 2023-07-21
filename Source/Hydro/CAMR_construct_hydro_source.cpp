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
        Array4<EBCellFlag const> const& flag = flagfab.const_array();

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
                if (vfrac_arr(i,j,k) > 0.) {
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
        if (flagfab.getType(bx) != FabType::regular) {

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
                dm_as_fine.resize(amrex::grow(bx,1),ncomp);
                dm_as_fine.setVal<RunOn::Device>(0.0);
            }

            const amrex::StateDescriptor* desc = state[State_Type].descriptor();
            const auto& bcs = desc->getBCs();
            amrex::Gpu::DeviceVector<amrex::BCRec> bcs_d(desc->nComp());
            amrex::Gpu::copy(
              amrex::Gpu::hostToDevice, bcs.begin(), bcs.end(), bcs_d.begin());

            // Return hyd_src - centered at old-time
            const auto& dxInv = geom.InvCellSizeArray();
            CAMR_umdrv_eb(bx, mfi, geom, &ebfact,
                          phys_bc.lo(), phys_bc.hi(),
                          sarr, qarr, hyd_src, qauxar,
                          vfrac_arr, flag, dx, dxInv, flx_arr,
                          as_crse, p_drho_as_crse->array(), p_rrflag_as_crse->array(),
                          as_fine, dm_as_fine.array(), level_mask.const_array(mfi),
                          difmag, dt, small, small_dens, small_pres, bcs_d.data(),
                          redistribution_type, plm_iorder, eb_weights_type);
        } else {
#endif
            // Return hyd_src - centered at half-time if using Godunov method
            //                - centered at  old-time if using MOL method
            CAMR_umdrv(do_mol, bx, geom, phys_bc.lo(), phys_bc.hi(),
                       sarr, hyd_src, qarr, qauxar, srcqarr, dx, dt,
                       ppm_type, use_pslope, use_flattening, transverse_reset_density,
                       small, small_dens, small_pres, difmag, plm_iorder,
                       flx_arr, a, volume.array(mfi));
#ifdef AMREX_USE_EB
        } // regular
#endif

        if (do_reflux) {
          if (level < finest_level) {
#ifdef AMREX_USE_EB
              if (flagfab.getType(amrex::grow(bx,1)) == FabType::regular)
              {
#endif
                   getFluxReg(level + 1).CrseAdd(mfi,
                       {{AMREX_D_DECL(&(flux[0]), &(flux[1]), &(flux[2]))}},
                       dxDp, dt, amrex::RunOn::Device);
#ifdef AMREX_USE_EB
              } else {
                   getFluxReg(level + 1).CrseAdd(mfi,
                      {{AMREX_D_DECL(&(flux[0]), &(flux[1]), &(flux[2]))}},
                      dxDp, dt, (*volfrac)[mfi],
                      {AMREX_D_DECL(&(*areafrac[0])[mfi], &(*areafrac[1])[mfi], &(*areafrac[2])[mfi])},
                      amrex::RunOn::Device);
              }
#endif
          }

          if (level > 0) {
#ifdef AMREX_USE_EB
              if (flagfab.getType(amrex::grow(bx,1)) == FabType::regular)
              {
#endif
                  getFluxReg(level).FineAdd(mfi,
                     {{AMREX_D_DECL(&(flux[0]), &(flux[1]), &(flux[2]))}},
                     dxDp, dt, amrex::RunOn::Device);
#ifdef AMREX_USE_EB
              } else {
                  // BEG HACK HACK HACKH
                  // dm_as_fine.setVal(0.0);
                  // END HACK HACK HACKH

                  getFluxReg(level).FineAdd(mfi,
                     {{AMREX_D_DECL(&(flux[0]), &(flux[1]), &(flux[2]))}},
                     dxDp, dt, (*volfrac)[mfi],
                     {AMREX_D_DECL(&(*areafrac[0])[mfi], &(*areafrac[1])[mfi], &(*areafrac[2])[mfi])},
                     dm_as_fine, amrex::RunOn::Device);
              }
#endif
          } // level > 0
        } // do_reflux

#ifdef AMREX_USE_EB
        } // not covered
#endif
      } // mfi
    } // openmp

    BL_PROFILE_VAR_STOP(PC_UMDRV);
}
