#include <iomanip>

#include "CAMR.H"

void
CAMR::sum_integrated_quantities()
{
  BL_PROFILE("CAMR::sum_integrated_quantities()");

  if (verbose <= 0) {
    return;
  }

  bool local_flag = true;

  int finest_level = parent->finestLevel();
  amrex::Real time = state[State_Type].curTime();
  amrex::Real mass = 0.0;
#if (AMREX_SPACEDIM == 2)
  amrex::Real mom[2] = {0.0};
#elif (AMREX_SPACEDIM == 3)
  amrex::Real mom[3] = {0.0};
#endif
  amrex::Real rho_e = 0.0;
  amrex::Real rho_K = 0.0;
  amrex::Real rho_E = 0.0;
  amrex::Real temp = 0.0;

  for (int lev = 0; lev <= finest_level; lev++) {
    CAMR& CAMR_lev = getLevel(lev);

    mass += CAMR_lev.volWgtSum("density", time, local_flag);
    mom[0] += CAMR_lev.volWgtSum("xmom", time, local_flag);
    mom[1] += CAMR_lev.volWgtSum("ymom", time, local_flag);
#if (AMREX_SPACEDIM == 3)
    mom[2] += CAMR_lev.volWgtSum("zmom", time, local_flag);
#endif
    rho_e += CAMR_lev.volWgtSum("rho_e", time, local_flag);
    rho_K += CAMR_lev.volWgtSum("kineng", time, local_flag);
    rho_E += CAMR_lev.volWgtSum("rho_E", time, local_flag);

    temp += CAMR_lev.volWgtSum("Temp", time, local_flag);
  }

  if (verbose > 0) {
#if (AMREX_SPACEDIM == 2)
    const int nfoo = 7;
    amrex::Real foo[nfoo] = {mass, mom[0], mom[1],         rho_e, rho_K, rho_E, temp};
#elif (AMREX_SPACEDIM == 3)
    const int nfoo = 8;
    amrex::Real foo[nfoo] = {mass, mom[0], mom[1], mom[2], rho_e, rho_K, rho_E, temp};
#endif

#ifdef AMREX_LAZY
    Lazy::QueueReduction([=]() mutable {
#endif
      amrex::ParallelDescriptor::ReduceRealSum(
        foo, nfoo, amrex::ParallelDescriptor::IOProcessorNumber());
      if (amrex::ParallelDescriptor::IOProcessor()) {
        int i = 0;
        mass = foo[i++];
        mom[0] = foo[i++];
        mom[1] = foo[i++];
#if (AMREX_SPACEDIM == 3)
        mom[2] = foo[i++];
#endif
        rho_e = foo[i++];
        rho_K = foo[i++];
        rho_E = foo[i++];
        temp = foo[i++];

        amrex::Print() << '\n';
        amrex::Print() << "TIME = " << time << " MASS        = " << mass
                       << '\n';
        amrex::Print() << "TIME = " << time << " XMOM        = " << mom[0]
                       << '\n';
        amrex::Print() << "TIME = " << time << " YMOM        = " << mom[1]
                       << '\n';
#if (AMREX_SPACEDIM == 3)
        amrex::Print() << "TIME = " << time << " ZMOM        = " << mom[2]
                       << '\n';
#endif
        amrex::Print() << "TIME = " << time << " RHO*e       = " << rho_e
                       << '\n';
        amrex::Print() << "TIME = " << time << " RHO*K       = " << rho_K
                       << '\n';
        amrex::Print() << "TIME = " << time << " RHO*E       = " << rho_E
                       << '\n';

        const int log_index = find_datalog_index("datalog");
        if (log_index >= 0) {
          std::ostream& data_log1 = parent->DataLog(log_index);
          if (data_log1.good()) {
            const int datwidth = 14;
            if (time == 0.0) {
              data_log1 << std::setw(datwidth) << "          time";
              data_log1 << std::setw(datwidth) << "          mass";
              data_log1 << std::setw(datwidth) << "          xmom";
              data_log1 << std::setw(datwidth) << "          ymom";
#if (AMREX_SPACEDIM == 3)
              data_log1 << std::setw(datwidth) << "          zmom";
#endif
              data_log1 << std::setw(datwidth) << "         rho_K";
              data_log1 << std::setw(datwidth) << "         rho_e";
              data_log1 << std::setw(datwidth) << "         rho_E";
              data_log1 << std::setw(datwidth) << "          temp";
              data_log1 << std::endl;
            }

            // Write the quantities at this time
            const int datprecision = 6;
            data_log1 << std::setw(datwidth) << time;
            data_log1 << std::setw(datwidth) << std::setprecision(datprecision)
                      << mass;
            data_log1 << std::setw(datwidth) << std::setprecision(datprecision)
                      << mom[0];
            data_log1 << std::setw(datwidth) << std::setprecision(datprecision)
                      << mom[1];
#if (AMREX_SPACEDIM == 3)
            data_log1 << std::setw(datwidth) << std::setprecision(datprecision)
                      << mom[2];
#endif
            data_log1 << std::setw(datwidth) << std::setprecision(datprecision)
                      << rho_K;
            data_log1 << std::setw(datwidth) << std::setprecision(datprecision)
                      << rho_e;
            data_log1 << std::setw(datwidth) << std::setprecision(datprecision)
                      << rho_E;
            data_log1 << std::setw(datwidth) << std::setprecision(datprecision)
                      << temp;
            data_log1 << std::endl;
          }
        }
      }
#ifdef AMREX_LAZY
    });
#endif
  }
}
