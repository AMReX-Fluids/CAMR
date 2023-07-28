#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_ParmParse.H>

#include <algorithm>

using namespace amrex;

/********************************************************************************
 *                                                                              *
 * Function to create a simple sphere EB.                                     *
 *                                                                              *
 ********************************************************************************/
void make_eb_sphere(const Geometry& geom, int required_coarsening_level)
{
    // Initialise sphere parameters
    bool inside = true;
    Real radius = 0.0002;
    Vector<Real> centervec(3);

    // Get sphere information from inputs file.                               *
    ParmParse pp("sphere");

    pp.query("internal_flow", inside);
    pp.query("radius", radius);
    pp.getarr("center", centervec, 0, 3);
    Array<Real, AMREX_SPACEDIM> center = {AMREX_D_DECL(centervec[0], centervec[1], centervec[2])};

    // Print info about sphere
    amrex::Print() << " " << std::endl;
    amrex::Print() << " Internal Flow: " << inside << std::endl;
    amrex::Print() << " Radius:    " << radius << std::endl;
    amrex::Print() << " Center:    " << centervec[0] << ", " << centervec[1] << ", " << centervec[2]
                   << std::endl;

    // Build the sphere implicit function
    EB2::SphereIF my_sphere(radius, center, inside);

    // Generate GeometryShop
    auto gshop = EB2::makeShop(my_sphere);

    // Build index space
    EB2::Build(gshop, geom, required_coarsening_level, required_coarsening_level);
}
