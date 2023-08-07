#include "AMReX_PROB_AMR_F.H"
#include "AMReX_ParmParse.H"
#include "CAMR.H"
#include "prob.H"

extern "C" {
    void amrex_probinit(const int* /*init*/,
                        const int* /*name*/,
                        const int* /*namelen*/,
                        const amrex_real* /*problo*/,
                        const amrex_real* /*probhi*/)
    {
        // Parse params
        amrex::ParmParse pp("prob");

        pp.query("p_l",   CAMR::h_prob_parm->p_l);
        pp.query("p_r",   CAMR::h_prob_parm->p_r);
        pp.query("rho_l", CAMR::h_prob_parm->rho_l);
        pp.query("rho_r", CAMR::h_prob_parm->rho_r);
        pp.query("u_l",   CAMR::h_prob_parm->u_l);
        pp.query("u_r",   CAMR::h_prob_parm->u_r);

        amrex::Gpu::copy(amrex::Gpu::hostToDevice, CAMR::h_prob_parm, CAMR::h_prob_parm+1, CAMR::d_prob_parm);
    }
}
