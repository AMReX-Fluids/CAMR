#include <AMReX_ParmParse.H>
#include <AMReX_buildInfo.H>
#include <memory>

#include "CAMR.H"
#include "Derive.H"
#include "IndexDefines.H"
#include "prob.H"

ProbParmDevice* CAMR::d_prob_parm= nullptr;
ProbParmDevice* CAMR::h_prob_parm= nullptr;
ProbParmHost* CAMR::prob_parm_host = nullptr;
TaggingParm* CAMR::tagging_parm = nullptr;
PassMap* CAMR::d_pass_map = nullptr;
PassMap* CAMR::h_pass_map = nullptr;

// Components are:
//                          Interior, Inflow,  Outflow,  Symmetry,     SlipWall,      NoSlipWall
static int scalar_bc[] =   {INT_DIR,  EXT_DIR, FOEXTRAP, REFLECT_EVEN, REFLECT_EVEN,  REFLECT_EVEN};
static int norm_vel_bc[] = {INT_DIR,  EXT_DIR, FOEXTRAP, REFLECT_ODD , REFLECT_ODD,   REFLECT_ODD};
static int tang_vel_bc[] = {INT_DIR,  EXT_DIR, FOEXTRAP, REFLECT_EVEN, REFLECT_EVEN,  REFLECT_ODD};

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

static void
set_scalar_bc(amrex::BCRec& bc, const amrex::BCRec& phys_bc)
{
  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();
  for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
    bc.setLo(dir, scalar_bc[lo_bc[dir]]);
    bc.setHi(dir, scalar_bc[hi_bc[dir]]);
  }
}

static void
set_x_vel_bc(amrex::BCRec& bc, const amrex::BCRec& phys_bc)
{
  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();
  AMREX_D_TERM(
    bc.setLo(0, norm_vel_bc[lo_bc[0]]); bc.setHi(0, norm_vel_bc[hi_bc[0]]);
    , bc.setLo(1, tang_vel_bc[lo_bc[1]]); bc.setHi(1, tang_vel_bc[hi_bc[1]]);
    , bc.setLo(2, tang_vel_bc[lo_bc[2]]); bc.setHi(2, tang_vel_bc[hi_bc[2]]););
}

static void
set_y_vel_bc(amrex::BCRec& bc, const amrex::BCRec& phys_bc)
{
  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();
  AMREX_D_TERM(
    bc.setLo(0, tang_vel_bc[lo_bc[0]]); bc.setHi(0, tang_vel_bc[hi_bc[0]]);
    , bc.setLo(1, norm_vel_bc[lo_bc[1]]); bc.setHi(1, norm_vel_bc[hi_bc[1]]);
    , bc.setLo(2, tang_vel_bc[lo_bc[2]]); bc.setHi(2, tang_vel_bc[hi_bc[2]]););
}

#if (AMREX_SPACEDIM == 3)
static void
set_z_vel_bc(amrex::BCRec& bc, const amrex::BCRec& phys_bc)
{
  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();
  AMREX_D_TERM(
    bc.setLo(0, tang_vel_bc[lo_bc[0]]); bc.setHi(0, tang_vel_bc[hi_bc[0]]);
    , bc.setLo(1, tang_vel_bc[lo_bc[1]]); bc.setHi(1, tang_vel_bc[hi_bc[1]]);
    , bc.setLo(2, norm_vel_bc[lo_bc[2]]); bc.setHi(2, norm_vel_bc[hi_bc[2]]););
}
#endif

void
CAMR::variableSetUp()
{
  // CAMR::variableSetUp is called in the constructor of Amr.cpp, so
  // it should get called every time we start or restart a job

  // initialize the start time for our CPU-time tracker
  startCPUTime = amrex::ParallelDescriptor::second();

  // Output the git commit hashes used to build the executable.

  if (amrex::ParallelDescriptor::IOProcessor()) {
    const char* CAMR_hash = amrex::buildInfoGetGitHash(1);
    const char* amrex_hash = amrex::buildInfoGetGitHash(2);
    const char* CAMRphysics_hash = amrex::buildInfoGetGitHash(3);
    const char* buildgithash = amrex::buildInfoGetBuildGitHash();
    const char* buildgitname = amrex::buildInfoGetBuildGitName();

    if (strlen(CAMR_hash) > 0) {
      amrex::Print() << "\n"
                     << "CAMR git hash: " << CAMR_hash << "\n";
    }
    if (strlen(amrex_hash) > 0) {
      amrex::Print() << "AMReX git hash: " << amrex_hash << "\n";
    }
    if (strlen(CAMRphysics_hash) > 0) {
      amrex::Print() << "CAMRPhysics git hash: " << CAMRphysics_hash << "\n";
    }
    if (strlen(buildgithash) > 0) {
      amrex::Print() << buildgitname << " git hash: " << buildgithash << "\n";
    }

    amrex::Print() << "\n";
  }

  AMREX_ASSERT(desc_lst.size() == 0);

  prob_parm_host = new ProbParmHost;
  h_prob_parm= new ProbParmDevice;
  tagging_parm = new TaggingParm;
  h_pass_map = new PassMap;
  d_prob_parm= static_cast<ProbParmDevice*>(
    amrex::The_Arena()->alloc(sizeof(ProbParmDevice)));
  d_pass_map =
    static_cast<PassMap*>(amrex::The_Arena()->alloc(sizeof(PassMap)));

    // Get options, set phys_bc
    read_params();

    init_pass_map(h_pass_map);

    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, h_pass_map, h_pass_map + 1, d_pass_map);

    amrex::Vector<amrex::Real> center(AMREX_SPACEDIM, 0.0);
    amrex::ParmParse ppc("CAMR");
    ppc.queryarr("center", center, 0, AMREX_SPACEDIM);

    amrex::Interpolater* interp;

#ifdef AMREX_USE_EB
    interp = &amrex::eb_cell_cons_interp;
#else
    interp = &amrex::cell_cons_interp;
#endif

    int ngrow_state = state_nghost;
    AMREX_ASSERT(ngrow_state >= 0);

    bool state_data_extrap = false;
    bool store_in_checkpoint = true;
    desc_lst.addDescriptor(
        State_Type, amrex::IndexType::TheCellType(), amrex::StateDescriptor::Point,
        ngrow_state, NVAR, interp, state_data_extrap, store_in_checkpoint);

    amrex::BCRec bc;

   amrex::Vector<amrex::BCRec> bcs(NVAR);
   amrex::Vector<std::string> name(NVAR);
   set_scalar_bc(bc, phys_bc); bcs[URHO]  = bc; name[URHO] = "density";
   set_x_vel_bc (bc, phys_bc); bcs[UMX]   = bc; name[UMX] = "xmom";
   set_y_vel_bc (bc, phys_bc); bcs[UMY]   = bc; name[UMY] = "ymom";
#if (AMREX_SPACEDIM == 3)
   set_z_vel_bc (bc, phys_bc); bcs[UMZ]   = bc; name[UMZ] = "zmom";
#endif
   set_scalar_bc(bc, phys_bc); bcs[UEDEN] = bc; name[UEDEN] = "rho_E";
   set_scalar_bc(bc, phys_bc); bcs[UEINT] = bc; name[UEINT] = "rho_e";
   set_scalar_bc(bc, phys_bc); bcs[UTEMP] = bc; name[UTEMP] = "Temp";

  /*
  for (int i = 0; i < NUM_ADV; ++i) {
    int cnt = UFA+i;
    char buf[64];
    sprintf(buf, "adv_%d", i);
    set_scalar_bc(bc, phys_bc);
    bcs[cnt] = bc;
    name[cnt] = std::string(buf);
  }
  */

  // Get the species names
  spec_names.resize(NUM_SPECIES);
  spec_names[0] = "F";
  spec_names[1] = "A";
  spec_names[2] = "P";

  if (amrex::ParallelDescriptor::IOProcessor()) {
    amrex::Print() << NUM_SPECIES << " Species: " << std::endl;
    for (int i = 0; i < NUM_SPECIES; i++) {
      amrex::Print() << spec_names[i] << ' ' << ' ';
    }
    amrex::Print() << std::endl;
  }

  for (int i = 0; i < NUM_SPECIES; ++i) {
    int cnt = UFS+i;
    set_scalar_bc(bc, phys_bc);
    bcs[cnt] = bc;
    name[cnt] = "rho_" + spec_names[i];
  }

    amrex::StateDescriptor::BndryFunc bndryfunc1(CAMR_bcfill_hyp);
    bndryfunc1.setRunOnGPU(true);

    desc_lst.setComponent(State_Type, URHO, name, bcs, bndryfunc1);

  num_state_type = desc_lst.size();

  // DEFINE DERIVED QUANTITIES

  // Pressure
  derive_lst.add(
    "pressure", amrex::IndexType::TheCellType(), 1, CAMR_derpres, the_same_box);
  derive_lst.addComponent("pressure", desc_lst, State_Type, URHO, NVAR);

  // Kinetic energy
  derive_lst.add(
    "kineng", amrex::IndexType::TheCellType(), 1, CAMR_derkineng, the_same_box);
  derive_lst.addComponent("kineng", desc_lst, State_Type, URHO, NVAR);

  // Enstrophy
  derive_lst.add(
    "enstrophy", amrex::IndexType::TheCellType(), 1, CAMR_derenstrophy,
    grow_box_by_one);
  derive_lst.addComponent("enstrophy", desc_lst, State_Type, URHO, NVAR);

  // Sound speed (c)
  derive_lst.add(
    "soundspeed", amrex::IndexType::TheCellType(), 1, CAMR_dersoundspeed,
    the_same_box);
  derive_lst.addComponent("soundspeed", desc_lst, State_Type, URHO, NVAR);

  // Mach number(M)
  derive_lst.add(
    "MachNumber", amrex::IndexType::TheCellType(), 1, CAMR_dermachnumber,
    the_same_box);
  derive_lst.addComponent("MachNumber", desc_lst, State_Type, URHO, NVAR);

  // Vorticity
  derive_lst.add(
    "magvort", amrex::IndexType::TheCellType(), 1, CAMR_dermagvort,
    grow_box_by_one);
  derive_lst.addComponent("magvort", desc_lst, State_Type, URHO, NVAR);

  // Div(u)
  derive_lst.add(
    "divu", amrex::IndexType::TheCellType(), 1, CAMR_derdivu, grow_box_by_one);
  derive_lst.addComponent("divu", desc_lst, State_Type, URHO, NVAR);

  // Internal energy as derived from rho*E, part of the state
  derive_lst.add(
    "eint_E", amrex::IndexType::TheCellType(), 1, CAMR_dereint1, the_same_box);
  derive_lst.addComponent("eint_E", desc_lst, State_Type, URHO, NVAR);

  // Internal energy as derived from rho*e, part of the state
  derive_lst.add(
    "eint_e", amrex::IndexType::TheCellType(), 1, CAMR_dereint2, the_same_box);
  derive_lst.addComponent("eint_e", desc_lst, State_Type, URHO, NVAR);

  // Log(density)
  derive_lst.add(
    "logden", amrex::IndexType::TheCellType(), 1, CAMR_derlogden, the_same_box);
  derive_lst.addComponent("logden", desc_lst, State_Type, URHO, NVAR);

  //
  // X from rhoX
  //
  std::string spec_string_F = "X(F)";
  derive_lst.add(spec_string_F,amrex::IndexType::TheCellType(),1,CAMR_derspec,the_same_box);
  derive_lst.addComponent(spec_string_F,desc_lst,State_Type,URHO,1);
  derive_lst.addComponent(spec_string_F,desc_lst,State_Type,UFS,1);
  std::string spec_string_A = "X(A)";
  derive_lst.add(spec_string_A,amrex::IndexType::TheCellType(),1,CAMR_derspec,the_same_box);
  derive_lst.addComponent(spec_string_A,desc_lst,State_Type,URHO,1);
  derive_lst.addComponent(spec_string_A,desc_lst,State_Type,UFS+1,1);
  std::string spec_string_P = "X(P)";
  derive_lst.add(spec_string_P,amrex::IndexType::TheCellType(),1,CAMR_derspec,the_same_box);
  derive_lst.addComponent(spec_string_P,desc_lst,State_Type,URHO,1);
  derive_lst.addComponent(spec_string_P,desc_lst,State_Type,UFS+2,1);

  // Velocities
  derive_lst.add("x_velocity", amrex::IndexType::TheCellType(), 1, CAMR_dervelx, the_same_box);
  derive_lst.addComponent("x_velocity", desc_lst, State_Type, URHO, NVAR);

  derive_lst.add("y_velocity", amrex::IndexType::TheCellType(), 1, CAMR_dervely, the_same_box);
  derive_lst.addComponent("y_velocity", desc_lst, State_Type, URHO, NVAR);

#if (AMREX_SPACEDIM == 3)
  derive_lst.add("z_velocity", amrex::IndexType::TheCellType(), 1, CAMR_dervelz, the_same_box);
  derive_lst.addComponent("z_velocity", desc_lst, State_Type, URHO, NVAR);
#endif

  derive_lst.add(
    "magvel", amrex::IndexType::TheCellType(), 1, CAMR_dermagvel, the_same_box);
  derive_lst.addComponent("magvel", desc_lst, State_Type, URHO, NVAR);

  derive_lst.add(
    "magmom", amrex::IndexType::TheCellType(), 1, CAMR_dermagmom, the_same_box);
  derive_lst.addComponent("magmom", desc_lst, State_Type, URHO, NVAR);

#ifdef AMREX_USE_EB
  derive_lst.add(
    "volfrac", amrex::IndexType::TheCellType(), 1, CAMR_dernull, the_same_box);
  derive_lst.addComponent("volfrac", desc_lst, State_Type, URHO, 1);

  derive_lst.add(
    "vfrac",   amrex::IndexType::TheCellType(), 1, CAMR_dernull, the_same_box);
  derive_lst.addComponent("vfrac", desc_lst, State_Type, URHO, 1);
#endif

  derive_lst.add("cp", amrex::IndexType::TheCellType(), 1, CAMR_dercp, the_same_box);
  derive_lst.addComponent("cp", desc_lst, State_Type, URHO, NVAR);

  // Set list of active sources
  set_active_sources();

  //
  // **************  DEFINE ERROR ESTIMATION QUANTITIES  *************
  //
  error_setup();
}

void
CAMR::variableCleanUp()
{
  delete prob_parm_host;
  delete tagging_parm;
  delete h_prob_parm;
  delete h_pass_map;
  amrex::The_Arena()->free(d_prob_parm);
  amrex::The_Arena()->free(d_pass_map);
}

void
CAMR::set_active_sources()
{
  // optional external source
  if (add_ext_src == 1) {
    src_list.push_back(ext_src);
  }

  // optional gravity source
  if (add_grav_src == 1) {
    src_list.push_back(grav_src);
  }
}
