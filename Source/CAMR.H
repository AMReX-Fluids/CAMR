#ifndef COMPFLO_H
#define COMPFLO_H

#include <AMReX_BC_TYPES.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_ErrorList.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>

#ifdef AMREX_USE_EB
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBFluxRegister.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_MultiCutFab.H>
#else
#include <AMReX_YAFluxRegister.H>
#endif

#ifdef AMREX_LAZY
#include <AMReX_Lazy.H>
#endif

#include "MathConstants.H"
#include "Tagging.H"
#include "IndexDefines.H"
#include "prob_parm.H"

enum StateType { State_Type = 0,
                 Source_Type,
                 Work_Estimate_Type,
                 NUM_STATE_TYPE };

// Create storage for all source terms.

enum sources {
  ext_src = 0,
  grav_src,
  num_src };

/*
static amrex::Box
the_same_box(const amrex::Box& b)
{
  return b;
}

static amrex::Box
grow_box_by_one(const amrex::Box& b)
{
  return amrex::grow(b, 1);
}
*/

class CAMR : public amrex::AmrLevel
{
public:
  // Default constructor.  Builds invalid object.
  CAMR();

  CAMR(
    amrex::Amr& papa,
    int lev,
    const amrex::Geometry& level_geom,
    const amrex::BoxArray& bl,
    const amrex::DistributionMapping& dm,
    amrex::Real time);

  // Restart from a checkpoint file.
  void restart(
    amrex::Amr& papa, std::istream& is, bool bReadSpecial = false) override;

  // Call AmrLevel::checkPoint and then add radiation info
  void checkPoint(
    const std::string& dir,
    std::ostream& os,
    amrex::VisMF::How how,
    bool dump_old) override;

  void setPlotVariables() override;

  // Write a plotfile to specified directory.
  void writePlotFile(
    const std::string& dir, std::ostream& os, amrex::VisMF::How how) override;
  void writeSmallPlotFile(
    const std::string& dir, std::ostream& os, amrex::VisMF::How how) override;
  void writeJobInfo(const std::string& dir);
  static void writeBuildInfo(std::ostream& os);

  // Define data descriptors.
  static void variableSetUp();

  // Cleanup data descriptors at end of run.
  static void variableCleanUp();

  // Initialize grid data at problem start-up.
  void initData() override;

  // Initialize stuff whether from init or from restart
  void init_stuff(amrex::Amr& papa,
                  const amrex::Geometry& level_geom,
                  const amrex::BoxArray& bl,
                  const amrex::DistributionMapping& dm);

  // Get problem metrics.
  amrex::MultiFab* Area();
  amrex::MultiFab& Area(int dir);

  // The volume of the multifab.
  amrex::MultiFab& Volume();

  // Set time levels of state data.
  void setTimeLevel(
    amrex::Real time, amrex::Real dt_old, amrex::Real dt_new) override;

  // Initialize data on this level from another CAMR (during regrid).
  void init(amrex::AmrLevel& old) override;

  // Initialize data on this level after regridding if old level did not
  // previously exist
  void init() override;

  // Initialize EB geometry for finest_level and level grids for
  // other levels for the Amr class to do timed load balances.
  int WorkEstType() override { return Work_Estimate_Type; }

  // Proceed with next timestep?
  int okToContinue() override;

  // Advance grids at this level in time.
  amrex::Real
  advance(amrex::Real time, amrex::Real dt, int iteration, int ncycle) override;

  amrex::Real mol_advance(
    amrex::Real time, amrex::Real dt, int amr_iteration, int amr_ncycle);

  amrex::Real CAMR_advance(
    amrex::Real time, amrex::Real dt, int amr_iteration, int amr_ncycle);

  void react(amrex::MultiFab& S);

  void construct_hydro_source(
    const amrex::MultiFab& S,
          amrex::MultiFab& src,
    amrex::Real time,
    amrex::Real dt);

  void construct_old_source(
    int src,
    amrex::Real time,
    amrex::Real dt);

  void construct_new_source(
    int src,
    amrex::Real time,
    amrex::Real dt);

  void construct_old_ext_source(amrex::Real time, amrex::Real dt);
  void construct_new_ext_source(amrex::Real time, amrex::Real dt);

  static void fill_ext_source(
    amrex::Real time,
    amrex::Real dt,
    const amrex::MultiFab& state_old,
    const amrex::MultiFab& S_new,
    amrex::MultiFab& ext_src,
    int ng);

  void construct_old_grav_source(amrex::Real time, amrex::Real dt);
  void construct_new_grav_source(amrex::Real time, amrex::Real dt);

  static void fill_grav_source( const amrex::MultiFab& S_new, amrex::MultiFab& grav_src, int ng);
  static void fill_gravcorr_source( const amrex::MultiFab& S_old, const amrex::MultiFab& S_new,
                                    amrex::MultiFab& grav_src, int ng);

  static void set_active_sources();

  // Estimate time step.
  amrex::Real estTimeStep(amrex::Real dt_old);

  // Compute initial time step.
  amrex::Real initialTimeStep();

  // Compute initial `dt'.
  void computeInitialDt(
    int finest_level,
    int sub_cycle,
    amrex::Vector<int>& n_cycle,
    const amrex::Vector<amrex::IntVect>& ref_ratio,
    amrex::Vector<amrex::Real>& dt_level,
    amrex::Real stop_time) override;

  // Compute new `dt'.
  void computeNewDt(
    int finest_level,
    int sub_cycle,
    amrex::Vector<int>& n_cycle,
    const amrex::Vector<amrex::IntVect>& ref_ratio,
    amrex::Vector<amrex::Real>& dt_min,
    amrex::Vector<amrex::Real>& dt_level,
    amrex::Real stop_time,
    int post_regrid_flag) override;

  // Allocate data at old time.
  void allocOldData() override;

  // Remove data at old time.
  void removeOldData() override;

  // Print information about energy budget.
  void do_energy_diagnostics();

  // Do work after timestep().
  void post_timestep(int iteration) override;

  // Contains operations to be done only after a full coarse timestep.
  void postCoarseTimeStep(amrex::Real cumtime) override;

  // Do work after regrid().
  void post_regrid(int lbase, int new_finest) override;

  // Do work after a restart().
  void post_restart() override;

  // Do work after init().
  void post_init(amrex::Real stop_time) override;

  // Error estimation for regridding.
  void errorEst(
    amrex::TagBoxArray& tags,
    int clearval,
    int tagval,
    amrex::Real time,
    int n_error_buf = 0,
    int ngrow = 0) override;

  // Returns a MultiFab containing the derived data for this level.
  // The user is responsible for deleting this pointer when done
  // with it.  If ngrow>0 the MultiFab is built on the appropriately
  // grown BoxArray.
  std::unique_ptr<amrex::MultiFab>
  derive(const std::string& name, amrex::Real time, int ngrow) override;

  // This version of derive() fills the dcomp'th component of mf with the
  // derived quantity.
  void derive(
    const std::string& name,
    amrex::Real time,
    amrex::MultiFab& mf,
    int dcomp) override;

  static ProbParmDevice* h_prob_parm;
  static ProbParmDevice* d_prob_parm;
  static ProbParmHost* prob_parm_host;
  static TaggingParm* tagging_parm;
  static PassMap* h_pass_map;
  static PassMap* d_pass_map;

  static int numGrow();

  static void reset_internal_energy(amrex::MultiFab& S_new, int ng);

  void computeTemp(amrex::MultiFab& State, int ng);

  static void enforce_consistent_e(amrex::MultiFab& S);

  amrex::Real volWgtSum(
    const std::string& name,
    amrex::Real time,
    bool local = false,
    bool finemask = true);

  amrex::Real volWgtSquaredSum(
    const std::string& name, amrex::Real time, bool local = false);

  amrex::Real volWgtSumMF(
    const amrex::MultiFab& mf,
    int comp,
    bool local = false,
    bool finemask = false);

  amrex::Real volWgtSquaredSumDiff(int comp, amrex::Real time, bool local = false);
  amrex::Real maxDerive(const std::string& name, amrex::Real time, bool local = false);
  amrex::Real minDerive(const std::string& name, amrex::Real time, bool local = false);

  /// Index locations into particle state
  static int pstateVel, pstateT, pstateDia, pstateRho, pstateY, pstateNum;

  // This MultiFab is on the coarser level.  This is useful for the coarser
  // level to mask out the finer level.  We only build this when it is needed.
  // This coarse MultiFab has to live on the fine level because it must be
  // updated even when only the fine level changes.
  amrex::MultiFab fine_mask;
  amrex::MultiFab& build_fine_mask();

#ifdef AMREX_USE_EB
  amrex::MultiFab level_set;

  amrex::iMultiFab level_mask;

  amrex::EBFluxRegister flux_reg;

  amrex::EBFluxRegister& getFluxReg();
  amrex::EBFluxRegister& getFluxReg(int lev);

  amrex::EBFluxRegister pres_reg;

  amrex::EBFluxRegister& getPresReg();
  amrex::EBFluxRegister& getPresReg(int lev);

#else
  amrex::YAFluxRegister flux_reg;

  amrex::YAFluxRegister& getFluxReg();
  amrex::YAFluxRegister& getFluxReg(int lev);

  amrex::YAFluxRegister pres_reg;

  amrex::YAFluxRegister& getPresReg();
  amrex::YAFluxRegister& getPresReg(int lev);

#endif
  void avgDown();
  void avgDown(int state_indx);

// public for CUDA

  static void normalize_species(amrex::MultiFab& S_new);

  static void enforce_min_density(amrex::MultiFab& S_new);

protected:

#ifdef AMREX_USE_EB
    void WriteMyEBSurface ();

    amrex::EBFArrayBoxFactory const&
    EBFactory () const noexcept {
        return static_cast<amrex::EBFArrayBoxFactory const&>(Factory());
    }

    const amrex::MultiFab& volFrac() const { return *volfrac; }
    const amrex::MultiFab* volfrac;

    const amrex::MultiCutFab* bndrycent;
    std::array<const amrex::MultiCutFab*,AMREX_SPACEDIM> areafrac;
    std::array<const amrex::MultiCutFab*,AMREX_SPACEDIM> facecent;
#endif

  // A state array with ghost zones.
  amrex::MultiFab Sborder;

  // Source terms to the hydrodynamics solve.
  amrex::MultiFab sources_for_hydro;

  // Source term representing hydrodynamics update.
  amrex::MultiFab     hydro_source;
  amrex::MultiFab new_hydro_source;

  // Non-hydro source terms.
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> old_sources;
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> new_sources;

  static void read_params();

  static void read_tagging_params();

  CAMR& getLevel(int lev);

  void reflux();

  //
  // Define error estimators
  //
  static void error_setup ();

  void expand_state(amrex::MultiFab& S, const amrex::Real time, const int ng);
  void clean_state(amrex::MultiFab& S);

  void buildMetrics();

  // integrate derived quantities over domain

  amrex::Real
  sumDerive(const std::string& name, amrex::Real time, bool local = false);

  int find_datalog_index(const std::string& logname);

  void sum_integrated_quantities();

  void write_info();

  static void stopJob();

  // The data.
  amrex::MultiFab volume;
  amrex::MultiFab area[AMREX_SPACEDIM];
  amrex::MultiFab dLogArea[1];

  // Static data members.
#include "CAMR_params.H"

  static bool signalStopJob;
  static bool dump_old;
  static int verbose;
  static amrex::BCRec phys_bc;

  // for keeping track of the amount of CPU time used -- this will persist
  // after restarts
  static amrex::Real previousCPUTimeUsed;
  static amrex::Real startCPUTime;

  static amrex::Real getCPUTime();

  static amrex::IntVect hydro_tile_size;

  static int num_state_type;

  static amrex::Vector<std::string> spec_names;

  static amrex::Vector<int> src_list;

  //
  // Holds info for dynamically generated error estimators
  //
  static amrex::Vector<amrex::AMRErrorTag> errtags;
};

void CAMR_bcfill_hyp(
  amrex::Box const& bx,
  amrex::FArrayBox& data,
  const int dcomp,
  const int numcomp,
  amrex::Geometry const& geom,
  const amrex::Real time,
  const amrex::Vector<amrex::BCRec>& bcr,
  const int bcomp,
  const int scomp);

void CAMR_nullfill(
  amrex::Box const& bx,
  amrex::FArrayBox& data,
  const int dcomp,
  const int numcomp,
  amrex::Geometry const& geom,
  const amrex::Real time,
  const amrex::Vector<amrex::BCRec>& bcr,
  const int bcomp,
  const int scomp);

// Inlines

AMREX_FORCE_INLINE
int
CAMR::numGrow()
{
#ifdef AMREX_USE_EB
  int ng = 5;
#else
  int ng = 4;
#endif
  return ng;
}

AMREX_FORCE_INLINE
amrex::MultiFab*
CAMR::Area()
{
  return area;
}

AMREX_FORCE_INLINE
amrex::MultiFab&
CAMR::Area(int dir)
{
  return area[dir];
}

AMREX_FORCE_INLINE
amrex::MultiFab&
CAMR::Volume()
{
  return volume;
}

AMREX_FORCE_INLINE
CAMR&
CAMR::getLevel(int lev)
{
  return *(CAMR*)&parent->getLevel(lev);
}

AMREX_FORCE_INLINE
void
CAMR::stopJob()
{
  signalStopJob = true;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_check_initial_species(
  const int i, const int j, const int k, amrex::Array4<amrex::Real> const& Sfab)
{
  amrex::Real spec_sum = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    spec_sum = spec_sum + Sfab(i, j, k, UFS + n);
  }
  if (std::abs(Sfab(i, j, k, URHO) - spec_sum) > 1.e-8 * Sfab(i, j, k, URHO)) {
    amrex::Abort("Error:: Failed check of initial species summing to 1");
  }
}

//#################################
#ifdef AMREX_USE_EB
//#################################

AMREX_FORCE_INLINE
amrex::EBFluxRegister&
CAMR::getFluxReg()
{
  return flux_reg;
}

AMREX_FORCE_INLINE
amrex::EBFluxRegister&
CAMR::getFluxReg(int lev)
{
  return getLevel(lev).getFluxReg();
}

AMREX_FORCE_INLINE
amrex::EBFluxRegister&
CAMR::getPresReg()
{
  return pres_reg;
}

AMREX_FORCE_INLINE
amrex::EBFluxRegister&
CAMR::getPresReg(int lev)
{
  return getLevel(lev).getPresReg();
}

//#################################
#else
//#################################

AMREX_FORCE_INLINE
amrex::YAFluxRegister&
CAMR::getFluxReg()
{
  return flux_reg;
}

AMREX_FORCE_INLINE
amrex::YAFluxRegister&
CAMR::getFluxReg(int lev)
{
  return getLevel(lev).getFluxReg();
}

AMREX_FORCE_INLINE
amrex::YAFluxRegister&
CAMR::getPresReg()
{
  return pres_reg;
}

AMREX_FORCE_INLINE
amrex::YAFluxRegister&
CAMR::getPresReg(int lev)
{
  return getLevel(lev).getPresReg();
}

//#################################
#endif // EB
//#################################

#endif
