#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <CAMR.H>

using namespace amrex;

/********************************************************************************
 *                                                                              *
 * Function to create a simple cylinder EB.                                     *
 *                                                                              *
 ********************************************************************************/
void make_eb_plane (const Geometry& geom, int max_coarsening_level, amrex::Real time)
{

    RealArray point{AMREX_D_DECL(static_cast<Real>(0.0 + 0.5*1.25e5*time*time), 0.0, 0.0)};

    RealArray normal{AMREX_D_DECL(-1.0, 0.0, 0.0)};

    EB2::PlaneIF pf(point, normal);

    EB2::GeometryShop<EB2::PlaneIF> gshop(pf);
    EB2::Build(gshop, geom, max_coarsening_level,
               max_coarsening_level, 4, false);
}
