#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <AMReX_Utility.H>
#include <AMReX_buildInfo.H>
#include <AMReX_ParmParse.H>
#ifdef AMREX_USE_EB
#include <AMReX_EBMultiFabUtil.H>
#endif

#include "CAMR.H"
#include "CAMR_io.H"
#include "IndexDefines.H"

void
CAMR::restart(amrex::Amr& papa, std::istream& is, bool bReadSpecial)
{
  AmrLevel::restart(papa, is, bReadSpecial);

  init_stuff(papa,geom,grids,dmap);

  // get the elapsed CPU time to now;
  if (level == 0 && amrex::ParallelDescriptor::IOProcessor()) {
    // get elapsed CPU time
    std::ifstream CPUFile;
    std::string FullPathCPUFile = parent->theRestartFile();
    FullPathCPUFile += "/CPUtime";
    CPUFile.open(FullPathCPUFile.c_str(), std::ios::in);

    CPUFile >> previousCPUTimeUsed;
    CPUFile.close();

    amrex::Print() << "read CPU time: " << previousCPUTimeUsed << "\n";
  }
}

void
CAMR::checkPoint(
  const std::string& dir,
  std::ostream& os,
  amrex::VisMF::How how,
  bool /*dump_old_default*/)
{
  amrex::AmrLevel::checkPoint(dir, os, how, dump_old);

  if (level == 0 && amrex::ParallelDescriptor::IOProcessor()) {
    {
      // store elapsed CPU time
      std::ofstream CPUFile;
      std::string FullPathCPUFile = dir;
      FullPathCPUFile += "/CPUtime";
      CPUFile.open(FullPathCPUFile.c_str(), std::ios::out);

      CPUFile << std::setprecision(15) << getCPUTime();
      CPUFile.close();
    }

    /* Not implemented for GPU{
            // store any problem-specific stuff
            char * dir_for_pass = new char[dir.size() + 1];
            std::copy(dir.begin(), dir.end(), dir_for_pass);
            dir_for_pass[dir.size()] = '\0';

            int len = dir.size();

            Vector<int> int_dir_name(len);
            for (int j = 0; j < len; j++)
            int_dir_name[j] = (int) dir_for_pass[j];

            AMREX_FORT_PROC_CALL(PROBLEM_CHECKPOINT,problem_checkpoint)(int_dir_name.dataPtr(),
       &len);

            delete [] dir_for_pass;
        }
    */
  }
}

void
CAMR::setPlotVariables()
{
  amrex::AmrLevel::setPlotVariables();

  amrex::ParmParse pp("CAMR");

  bool plot_cost = true;
  pp.query("plot_cost", plot_cost);
  if (plot_cost) {
    amrex::Amr::addDerivePlotVar("WorkEstimate");
  }

  bool plot_rhoy = false;
  pp.query("plot_rhoy", plot_rhoy);
  if (plot_rhoy) {
    for (int i = 0; i < NUM_SPECIES; i++) {
      amrex::Amr::addStatePlotVar(desc_lst[State_Type].name(UFS + i));
    }
  } else {
    for (int i = 0; i < NUM_SPECIES; i++) {
      amrex::Amr::deleteStatePlotVar(desc_lst[State_Type].name(UFS + i));
    }
  }

  bool plot_X = false;
  pp.query("plot_X",plot_X);
  if (plot_X)
  {
      std::string spec_string_F = "X(F)";
      parent->addDerivePlotVar(spec_string_F);
      std::string spec_string_A = "X(A)";
      parent->addDerivePlotVar(spec_string_A);
      std::string spec_string_P = "X(P)";
      parent->addDerivePlotVar(spec_string_P);
  }
}

void
CAMR::writeJobInfo(const std::string& dir)
{
  // job_info file with details about the run
  std::ofstream jobInfoFile;
  std::string FullPathJobInfoFile = dir;
  FullPathJobInfoFile += "/job_info";
  jobInfoFile.open(FullPathJobInfoFile.c_str(), std::ios::out);

  std::string PrettyLine = "==================================================="
                           "============================\n";
  std::string OtherLine = "----------------------------------------------------"
                          "----------------------------\n";
  std::string SkipSpace = "        ";

  // job information
  jobInfoFile << PrettyLine;
  jobInfoFile << " CAMR Job Information\n";
  jobInfoFile << PrettyLine;

  jobInfoFile << "job name: " << job_name << "\n\n";
  jobInfoFile << "inputs file: " << inputs_name << "\n\n";

  jobInfoFile << "number of MPI processes: "
              << amrex::ParallelDescriptor::NProcs() << "\n";
#ifdef _OPENMP
  jobInfoFile << "number of threads:       " << omp_get_max_threads() << "\n";
#endif

  jobInfoFile << "\n";
  jobInfoFile << "CPU time used since start of simulation (CPU-hours): "
              << getCPUTime() / 3600.0;

  jobInfoFile << "\n\n";

  // plotfile information
  jobInfoFile << PrettyLine;
  jobInfoFile << " Plotfile Information\n";
  jobInfoFile << PrettyLine;

  time_t now = time(nullptr);

  // Convert now to tm struct for local timezone
  char time_buffer[128];
  const tm* localtm = localtime(&now);
  strftime(time_buffer, sizeof(time_buffer), "%b %d %Y %H:%M:%S", localtm);
  jobInfoFile << "output data / time: " << time_buffer << std::endl;

  std::string currentDir = amrex::FileSystem::CurrentPath();
  jobInfoFile << "output dir:         " << currentDir << "\n";

  jobInfoFile << "\n\n";

  // build information
  jobInfoFile << PrettyLine;
  jobInfoFile << " Build Information\n";
  jobInfoFile << PrettyLine;

  jobInfoFile << "build date:    " << amrex::buildInfoGetBuildDate() << "\n";
  jobInfoFile << "build machine: " << amrex::buildInfoGetBuildMachine() << "\n";
  jobInfoFile << "build dir:     " << amrex::buildInfoGetBuildDir() << "\n";
  jobInfoFile << "AMReX dir:     " << amrex::buildInfoGetAMReXDir() << "\n";

  jobInfoFile << "\n";

  jobInfoFile << "COMP:          " << amrex::buildInfoGetComp() << "\n";
  jobInfoFile << "COMP version:  " << amrex::buildInfoGetCompVersion() << "\n";
  jobInfoFile << "FCOMP:         " << amrex::buildInfoGetFcomp() << "\n";
  jobInfoFile << "FCOMP version: " << amrex::buildInfoGetFcompVersion() << "\n";

  jobInfoFile << "\n";

  for (int n = 1; n <= amrex::buildInfoGetNumModules(); n++) {
    jobInfoFile << amrex::buildInfoGetModuleName(n) << ": "
                << amrex::buildInfoGetModuleVal(n) << "\n";
  }

  jobInfoFile << "\n";

  const char* githash1 = amrex::buildInfoGetGitHash(1);
  const char* githash2 = amrex::buildInfoGetGitHash(2);
  const char* githash3 = amrex::buildInfoGetGitHash(3);
  if (strlen(githash1) > 0) {
    jobInfoFile << "CAMR       git hash: " << githash1 << "\n";
  }
  if (strlen(githash2) > 0) {
    jobInfoFile << "AMReX       git hash: " << githash2 << "\n";
  }
  if (strlen(githash3) > 0) {
    jobInfoFile << "CAMRPhysics git hash: " << githash3 << "\n";
  }

  const char* buildgithash = amrex::buildInfoGetBuildGitHash();
  const char* buildgitname = amrex::buildInfoGetBuildGitName();
  if (strlen(buildgithash) > 0) {
    jobInfoFile << buildgitname << " git hash: " << buildgithash << "\n";
  }

  jobInfoFile << "\n\n";

  // grid information
  jobInfoFile << PrettyLine;
  jobInfoFile << " Grid Information\n";
  jobInfoFile << PrettyLine;

  int f_lev = parent->finestLevel();

  for (int i = 0; i <= f_lev; i++) {
    jobInfoFile << " level: " << i << "\n";
    jobInfoFile << "   number of boxes = " << parent->numGrids(i) << "\n";
    jobInfoFile << "   maximum zones   = ";
    for (int n = 0; n < AMREX_SPACEDIM; n++) {
      jobInfoFile << parent->Geom(i).Domain().length(n) << " ";
      // jobInfoFile << parent->Geom(i).ProbHi(n) << " ";
    }
    jobInfoFile << "\n\n";
  }

  jobInfoFile << " Boundary conditions\n";
  amrex::Vector<std::string> lo_bc_out(AMREX_SPACEDIM);
  amrex::Vector<std::string> hi_bc_out(AMREX_SPACEDIM);
  amrex::ParmParse pp("CAMR");
  pp.getarr("lo_bc", lo_bc_out, 0, AMREX_SPACEDIM);
  pp.getarr("hi_bc", hi_bc_out, 0, AMREX_SPACEDIM);

  // these names correspond to the integer flags setup in the
  // Setup.cpp

  jobInfoFile << "   -x: " << lo_bc_out[0] << "\n";
  jobInfoFile << "   +x: " << hi_bc_out[0] << "\n";
  if (AMREX_SPACEDIM >= 2) {
    jobInfoFile << "   -y: " << lo_bc_out[1] << "\n";
    jobInfoFile << "   +y: " << hi_bc_out[1] << "\n";
  }
  if (AMREX_SPACEDIM == 3) {
    jobInfoFile << "   -z: " << lo_bc_out[2] << "\n";
    jobInfoFile << "   +z: " << hi_bc_out[2] << "\n";
  }

  jobInfoFile << "\n\n";

  const int mlen = 20;

  jobInfoFile << PrettyLine;
  jobInfoFile << " Species Information\n";
  jobInfoFile << PrettyLine;

  jobInfoFile << std::setw(6) << "index" << SkipSpace << std::setw(mlen + 1)
              << "name" << SkipSpace << std::setw(7) << "A" << SkipSpace
              << std::setw(7) << "Z"
              << "\n";
  jobInfoFile << OtherLine;
  jobInfoFile << "\n\n";

  // runtime parameters
  jobInfoFile << PrettyLine;
  jobInfoFile << " Inputs File Parameters\n";
  jobInfoFile << PrettyLine;

  amrex::ParmParse::dumpTable(jobInfoFile, true);
  jobInfoFile.close();
}

// CAMR::writeBuildInfo
// Similar to writeJobInfo, but the subset of information that makes sense
// without an input file to enable --describe in format similar to CASTRO
void
CAMR::writeBuildInfo(std::ostream& os)
{
  std::string PrettyLine = std::string(78, '=') + "\n";
  // std::string OtherLine = std::string(78, '-') + "\n";
  // std::string SkipSpace = std::string(8, ' ');

  // build information
  os << PrettyLine;
  os << " CAMR Build Information\n";
  os << PrettyLine;

  os << "build date:    " << amrex::buildInfoGetBuildDate() << "\n";
  os << "build machine: " << amrex::buildInfoGetBuildMachine() << "\n";
  os << "build dir:     " << amrex::buildInfoGetBuildDir() << "\n";
  os << "AMReX dir:     " << amrex::buildInfoGetAMReXDir() << "\n";

  os << "\n";

  os << "COMP:          " << amrex::buildInfoGetComp() << "\n";
  os << "COMP version:  " << amrex::buildInfoGetCompVersion() << "\n";

  amrex::Print() << "C++ compiler:  " << amrex::buildInfoGetCXXName() << "\n";
  amrex::Print() << "C++ flags:     " << amrex::buildInfoGetCXXFlags() << "\n";

  os << "\n";

  os << "FCOMP:         " << amrex::buildInfoGetFcomp() << "\n";
  os << "FCOMP version: " << amrex::buildInfoGetFcompVersion() << "\n";

  os << "\n";

  amrex::Print() << "Link flags:    " << amrex::buildInfoGetLinkFlags() << "\n";
  amrex::Print() << "Libraries:     " << amrex::buildInfoGetLibraries() << "\n";

  os << "\n";

  for (int n = 1; n <= amrex::buildInfoGetNumModules(); n++) {
    os << amrex::buildInfoGetModuleName(n) << ": "
       << amrex::buildInfoGetModuleVal(n) << "\n";
  }

  os << "\n";
  const char* githash1 = amrex::buildInfoGetGitHash(1);
  const char* githash2 = amrex::buildInfoGetGitHash(2);
  const char* githash3 = amrex::buildInfoGetGitHash(3);
  if (strlen(githash1) > 0) {
    os << "CAMR       git hash: " << githash1 << "\n";
  }
  if (strlen(githash2) > 0) {
    os << "AMReX       git hash: " << githash2 << "\n";
  }
  if (strlen(githash3) > 0) {
    os << "CAMRPhysics git hash: " << githash3 << "\n";
  }

  const char* buildgithash = amrex::buildInfoGetBuildGitHash();
  const char* buildgitname = amrex::buildInfoGetBuildGitName();
  if (strlen(buildgithash) > 0) {
    os << buildgitname << " git hash: " << buildgithash << "\n";
  }

  os << "\n";
  os << " CAMR Compile time variables: \n";

  os << "\n";
  os << " CAMR Defines: \n";
#ifdef _OPENMP
  os << std::setw(35) << std::left << "_OPENMP " << std::setw(6) << "ON"
     << std::endl;
#else
  os << std::setw(35) << std::left << "_OPENMP " << std::setw(6) << "OFF"
     << std::endl;
#endif

#ifdef MPI_VERSION
  os << std::setw(35) << std::left << "MPI_VERSION " << std::setw(6)
     << MPI_VERSION << std::endl;
#else
  os << std::setw(35) << std::left << "MPI_VERSION " << std::setw(6)
     << "UNDEFINED" << std::endl;
#endif

#ifdef MPI_SUBVERSION
  os << std::setw(35) << std::left << "MPI_SUBVERSION " << std::setw(6)
     << MPI_SUBVERSION << std::endl;
#else
  os << std::setw(35) << std::left << "MPI_SUBVERSION " << std::setw(6)
     << "UNDEFINED" << std::endl;
#endif

#ifdef NUM_ADV
  os << std::setw(35) << std::left << "NUM_ADV=" << NUM_ADV << std::endl;
#else
  os << std::setw(35) << std::left << "NUM_ADV"
     << "is undefined (0)" << std::endl;
#endif

#ifdef AMREX_USE_EB
  os << std::setw(35) << std::left << "AMREX_USE_EB " << std::setw(6) << "ON"
     << std::endl;
#else
  os << std::setw(35) << std::left << "AMREX_USE_EB " << std::setw(6) << "OFF"
     << std::endl;
#endif

#ifdef AMREX_USE_EB
  os << std::setw(35) << std::left << "AMREX_USE_EB " << std::setw(6) << "ON"
     << std::endl;
#else
  os << std::setw(35) << std::left << "AMREX_USE_EB " << std::setw(6) << "OFF"
     << std::endl;
#endif

  os << "\n\n";
}

void
CAMR::writePlotFile(
  const std::string& dir, std::ostream& os, amrex::VisMF::How how)
{
  // The list of indices of State to write to plotfile.
  // first component of pair is state_type,
  // second component of pair is component # within the state_type
  amrex::Vector<std::pair<int, int>> plot_var_map;
  for (int typ = 0; typ < desc_lst.size(); typ++) {
    for (int comp = 0; comp < desc_lst[typ].nComp(); comp++) {
      if (
        amrex::Amr::isStatePlotVar(desc_lst[typ].name(comp)) &&
        desc_lst[typ].getType() == amrex::IndexType::TheCellType()) {
        plot_var_map.push_back(std::pair<int, int>(typ, comp));
      }
    }
  }

  int num_derive = 0;
  std::list<std::string> derive_names;
  const std::list<amrex::DeriveRec>& dlist = derive_lst.dlist();

  for (const auto& it : dlist) {
    if (amrex::Amr::isDerivePlotVar(it.name())) {
      {
        derive_names.push_back(it.name());
        num_derive += it.numDerive();
      }
    }
  }

  const auto n_data_items = static_cast<int>(plot_var_map.size()) + num_derive;

  amrex::Real cur_time = state[State_Type].curTime();

  if (level == 0 && amrex::ParallelDescriptor::IOProcessor()) {
    // The first thing we write out is the plotfile type.
    os << thePlotFileType() << '\n';

    if (n_data_items == 0) {
      amrex::Error("Must specify at least one valid data item to plot");
    }

    os << n_data_items << '\n';

    // Names of variables -- first state, then derived
    const auto pvmap_size = static_cast<int>(plot_var_map.size());
    for (int i = 0; i < pvmap_size; i++) {
      int typ = plot_var_map[i].first;
      int comp = plot_var_map[i].second;
      os << desc_lst[typ].name(comp) << '\n';
    }

    for (const auto& derive_name : derive_names) {
      const amrex::DeriveRec* rec = derive_lst.get(derive_name);
      for (int i = 0; i < rec->numDerive(); i++) {
        os << rec->variableName(i) << '\n';
      }
    }

    os << AMREX_SPACEDIM << '\n';
    os << parent->cumTime() << '\n';
    int f_lev = parent->finestLevel();
    os << f_lev << '\n';
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      os << amrex::DefaultGeometry().ProbLo(i) << ' ';
    }
    os << '\n';
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      os << amrex::DefaultGeometry().ProbHi(i) << ' ';
    }
    os << '\n';
    for (int i = 0; i < f_lev; i++) {
      os << parent->refRatio(i)[0] << ' ';
    }
    os << '\n';
    for (int i = 0; i <= f_lev; i++) {
      os << parent->Geom(i).Domain() << ' ';
    }
    os << '\n';
    for (int i = 0; i <= f_lev; i++) {
      os << parent->levelSteps(i) << ' ';
    }
    os << '\n';
    for (int i = 0; i <= f_lev; i++) {
      for (int k = 0; k < AMREX_SPACEDIM; k++) {
        os << parent->Geom(i).CellSize()[k] << ' ';
      }
      os << '\n';
    }
    os << (int)amrex::DefaultGeometry().Coord() << '\n';
    os << "0\n"; // Write bndry data.

    writeJobInfo(dir);
  }

  // Build the directory to hold the MultiFab at this level.
  // The name is relative to the directory containing the Header file.
  static const std::string BaseName = "/Cell";
  char buf[64];
  snprintf(buf, sizeof buf, "Level_%d", level);
  std::string LevelStr = buf;

  // Now for the full pathname of that directory.
  std::string FullPath = dir;
  if (!FullPath.empty() && FullPath[FullPath.size() - 1] != '/') {
    FullPath += '/';
  }
  FullPath += LevelStr;

  // Only the I/O processor makes the directory if it doesn't already exist.
  if (amrex::ParallelDescriptor::IOProcessor()) {
    if (!amrex::UtilCreateDirectory(FullPath, 0755)) {
      amrex::CreateDirectoryFailed(FullPath);
    }
  }

  // Force other processors to wait till directory is built.
  amrex::ParallelDescriptor::Barrier();

  if (amrex::ParallelDescriptor::IOProcessor()) {
    os << level << ' ' << grids.size() << ' ' << cur_time << '\n';
    os << parent->levelSteps(level) << '\n';

    for (int i = 0; i < grids.size(); ++i) {
      amrex::RealBox gridloc =
        amrex::RealBox(grids[i], geom.CellSize(), geom.ProbLo());
      for (int n = 0; n < AMREX_SPACEDIM; n++) {
        os << gridloc.lo(n) << ' ' << gridloc.hi(n) << '\n';
      }
    }

    // The full relative pathname of the MultiFabs at this level.
    // The name is relative to the Header file containing this name.
    // It's the name that gets written into the Header.
    if (n_data_items > 0) {
      std::string PathNameInHeader = LevelStr;
      PathNameInHeader += BaseName;
      os << PathNameInHeader << '\n';
    }
  }

  // We combine all of the multifabs -- state, derived, etc -- into one
  // multifab -- plotMF.
  // NOTE: we are assuming that each state variable has one component,
  // but a derived variable is allowed to have multiple components.
  int cnt = 0;
  const int nGrow = 0;
  amrex::MultiFab plotMF(
    grids, dmap, n_data_items, nGrow, amrex::MFInfo(), Factory());

  // Cull data from state variables -- use no ghost cells.
  for (int i = 0; i < plot_var_map.size(); i++) {
    int typ = plot_var_map[i].first;
    int comp = plot_var_map[i].second;
    amrex::MultiFab* this_dat = &state[typ].newData();
    amrex::MultiFab::Copy(plotMF, *this_dat, comp, cnt, 1, nGrow);
    cnt++;
  }

  // Cull data from derived variables.
  if (!derive_names.empty()) {
    for (const auto& derive_name : derive_names) {
      const amrex::DeriveRec* rec = derive_lst.get(derive_name);
      int ncomp = rec->numDerive();

      auto derive_dat = derive(derive_name, cur_time, nGrow);
      amrex::MultiFab::Copy(plotMF, *derive_dat, 0, cnt, ncomp, nGrow);
      cnt += ncomp;
    }
  }

#ifdef AMREX_USE_EB
   amrex::EB_set_covered(plotMF,0.0);
	//plotMF.setVal(0.0, cnt, 1, nGrow);
    //amrex::MultiFab::Copy(plotMF,volFrac(),0,cnt,1,nGrow);
	ZeroOutSolidWalls(plotMF);
#endif

  // Use the Full pathname when naming the MultiFab.
  std::string TheFullPath = FullPath;
  TheFullPath += BaseName;
  amrex::VisMF::Write(plotMF, TheFullPath, how, true);
}

void
CAMR::writeSmallPlotFile(
  const std::string& dir, std::ostream& os, amrex::VisMF::How how)
{
  // The list of indices of State to write to plotfile.
  // first component of pair is state_type,
  // second component of pair is component # within the state_type
  amrex::Vector<std::pair<int, int>> plot_var_map;
  for (int typ = 0; typ < desc_lst.size(); typ++) {
    for (int comp = 0; comp < desc_lst[typ].nComp(); comp++) {
      if (
        amrex::Amr::isStateSmallPlotVar(desc_lst[typ].name(comp)) &&
        desc_lst[typ].getType() == amrex::IndexType::TheCellType()) {
        plot_var_map.push_back(std::pair<int, int>(typ, comp));
      }
    }
  }

  const auto n_data_items = static_cast<int>(plot_var_map.size());

  amrex::Real cur_time = state[State_Type].curTime();

  if (level == 0 && amrex::ParallelDescriptor::IOProcessor()) {
    // The first thing we write out is the plotfile type.
    os << thePlotFileType() << '\n';

    if (n_data_items == 0) {
      amrex::Error("Must specify at least one valid data item to plot");
    }

    os << n_data_items << '\n';

    // Names of variables -- first state, then derived
    for (int i = 0; i < plot_var_map.size(); i++) {
      int typ = plot_var_map[i].first;
      int comp = plot_var_map[i].second;
      os << desc_lst[typ].name(comp) << '\n';
    }

    os << AMREX_SPACEDIM << '\n';
    os << parent->cumTime() << '\n';
    int f_lev = parent->finestLevel();
    os << f_lev << '\n';
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      os << amrex::DefaultGeometry().ProbLo(i) << ' ';
    }
    os << '\n';
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      os << amrex::DefaultGeometry().ProbHi(i) << ' ';
    }
    os << '\n';
    for (int i = 0; i < f_lev; i++) {
      os << parent->refRatio(i)[0] << ' ';
    }
    os << '\n';
    for (int i = 0; i <= f_lev; i++) {
      os << parent->Geom(i).Domain() << ' ';
    }
    os << '\n';
    for (int i = 0; i <= f_lev; i++) {
      os << parent->levelSteps(i) << ' ';
    }
    os << '\n';
    for (int i = 0; i <= f_lev; i++) {
      for (int k = 0; k < AMREX_SPACEDIM; k++) {
        os << parent->Geom(i).CellSize()[k] << ' ';
      }
      os << '\n';
    }
    os << (int)amrex::DefaultGeometry().Coord() << '\n';
    os << "0\n"; // Write bndry data.

    // job_info file with details about the run
    writeJobInfo(dir);
  }

  // Build the directory to hold the MultiFab at this level.
  // The name is relative to the directory containing the Header file.
  static const std::string BaseName = "/Cell";
  char buf[64];
  snprintf(buf, sizeof buf, "Level_%d", level);
  std::string LevelStr = buf;

  // Now for the full pathname of that directory.
  std::string FullPath = dir;
  if (!FullPath.empty() && FullPath[FullPath.size() - 1] != '/') {
    FullPath += '/';
  }
  FullPath += LevelStr;

  // Only the I/O processor makes the directory if it doesn't already exist.
  if (amrex::ParallelDescriptor::IOProcessor()) {
    if (!amrex::UtilCreateDirectory(FullPath, 0755)) {
      amrex::CreateDirectoryFailed(FullPath);
    }
  }

  // Force other processors to wait till directory is built.
  amrex::ParallelDescriptor::Barrier();

  if (amrex::ParallelDescriptor::IOProcessor()) {
    os << level << ' ' << grids.size() << ' ' << cur_time << '\n';
    os << parent->levelSteps(level) << '\n';

    for (int i = 0; i < grids.size(); ++i) {
      amrex::RealBox gridloc =
        amrex::RealBox(grids[i], geom.CellSize(), geom.ProbLo());
      for (int n = 0; n < AMREX_SPACEDIM; n++) {
        os << gridloc.lo(n) << ' ' << gridloc.hi(n) << '\n';
      }
    }

    // The full relative pathname of the MultiFabs at this level.
    // The name is relative to the Header file containing this name.
    // It's the name that gets written into the Header.
    if (n_data_items > 0) {
      std::string PathNameInHeader = LevelStr;
      PathNameInHeader += BaseName;
      os << PathNameInHeader << '\n';
    }
  }

  // We combine all of the multifabs -- state, derived, etc -- into one
  // multifab -- plotMF.
  // NOTE: we are assuming that each state variable has one component,
  // but a derived variable is allowed to have multiple components.
  int cnt = 0;
  const int nGrow = 0;
  amrex::MultiFab plotMF(
    grids, dmap, n_data_items, nGrow, amrex::MFInfo(), Factory());

  // Cull data from state variables -- use no ghost cells.
  for (int i = 0; i < plot_var_map.size(); i++) {
    int typ = plot_var_map[i].first;
    int comp = plot_var_map[i].second;
    amrex::MultiFab* this_dat = &state[typ].newData();
    amrex::MultiFab::Copy(plotMF, *this_dat, comp, cnt, 1, nGrow);
    cnt++;
  }

  // Use the Full pathname when naming the MultiFab.
  std::string TheFullPath = FullPath;
  TheFullPath += BaseName;
  amrex::VisMF::Write(plotMF, TheFullPath, how, true);
}
