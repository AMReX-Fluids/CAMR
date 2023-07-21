#ifndef PROB_PARM_H_
#define PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParmDevice
{
    int type = 0;
    amrex::Real p_l = 1.0;
    amrex::Real p_r = 0.1;
    amrex::Real rho_l = 1.0;
    amrex::Real rho_r = 0.125;
    amrex::Real u_l = 0.0;
    amrex::Real u_r = 0.0;
};

struct ProbParmHost
{
    amrex::Gpu::DeviceVector<amrex::Real> scaled_z;
    amrex::Gpu::DeviceVector<amrex::Real> det_density;
    amrex::Gpu::DeviceVector<amrex::Real> det_energy;
    amrex::Gpu::DeviceVector<amrex::Real> det_velocity;

};

#endif
