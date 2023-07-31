
// This file is automatically created by parse_CAMR_params.py.  To update
// or add runtime parameters, please edit _cpp_parameters and then run
// mk_params.sh

int CAMR::state_nghost = 0;
int CAMR::do_reflux = 1;
int CAMR::do_avg_down = 1;
int CAMR::refine_cut_cells = 0;
amrex::Real CAMR::difmag = 0.1;
amrex::Real CAMR::small = 1.e-8;
amrex::Real CAMR::small_dens = 1.e-7;
amrex::Real CAMR::small_temp = 1.e2;
amrex::Real CAMR::small_pres = 1.e-200;
int CAMR::do_mol = 0;
int CAMR::do_react = 0;
int CAMR::add_ext_src = 0;
int CAMR::add_grav_src = 1;
amrex::Real CAMR::const_grav = 0.0;
int CAMR::ppm_type = 0;
int CAMR::ppm_trace_sources = 0;
int CAMR::ppm_temp_fix = 0;
int CAMR::ppm_predict_gammae = 0;
int CAMR::plm_iorder = 4;
int CAMR::use_flattening = 1;
int CAMR::dual_energy_update_E_from_e = 1;
amrex::Real CAMR::dual_energy_eta1 = 1.0e0;
amrex::Real CAMR::dual_energy_eta2 = 1.0e-4;
int CAMR::use_pslope = 0;
int CAMR::fix_mass_flux = 0;
int CAMR::limit_fluxes_on_small_dens = 0;
int CAMR::allow_negative_energy = 1;
int CAMR::allow_small_energy = 1;
int CAMR::transverse_reset_density = 1;
amrex::Real CAMR::eb_boundary_T = 1.0;
int CAMR::eb_isothermal = 1;
int CAMR::eb_noslip = 1;
int CAMR::eb_weights_type = 2;
std::string CAMR::redistribution_type = "FluxRedist";
amrex::Real CAMR::fixed_dt = -1.0;
amrex::Real CAMR::initial_dt = -1.0;
amrex::Real CAMR::dt_cutoff = 0.0;
amrex::Real CAMR::max_dt = 1.e200;
amrex::Real CAMR::cfl = 0.8;
amrex::Real CAMR::init_shrink = 1.0;
amrex::Real CAMR::change_max = 1.1;
amrex::Real CAMR::retry_neg_dens_factor = 1.e-1;
int CAMR::bndry_func_thread_safe = 1;
int CAMR::sum_interval = -1;
amrex::Real CAMR::sum_per = -1.0e0;
int CAMR::hard_cfl_limit = 1;
std::string CAMR::job_name;
int CAMR::write_eb = 0;
