#include <AMReX_ParmParse.H>

#include <algorithm>
#include <CAMR.H>

using namespace amrex;

void make_eb_regular(const Geometry& geom);
void make_eb_sphere(const Geometry& geom, int required_coarsening_level);
void make_eb_box(const Geometry& geom, int required_coarsening_level);
void make_eb_cylinder(const Geometry& geom, int required_coarsening_level);

void
initialize_EB2 (const Geometry& geom, const int required_coarsening_level,
                const int /*max_coarsening_level*/)
{
   /******************************************************************************
   * CAMR.geometry=<string> specifies the EB geometry. <string> can be one of    *
   * box, cylinder, sphere, spherecube, twocylinders
   ******************************************************************************/

    ParmParse pp("CAMR");

    std::string geom_type;
    pp.query("geometry", geom_type);

   /******************************************************************************
   *                                                                            *
   *  CONSTRUCT EB                                                              *
   *                                                                            *
   ******************************************************************************/

    if(geom_type == "cylinder")
    {
    amrex::Print() << "\n Building cylinder geometry." << std::endl;
        make_eb_cylinder(geom, required_coarsening_level);
    }
    else if(geom_type == "box")
    {
        amrex::Print() << "\n Building box geometry." << std::endl;
        make_eb_box(geom, required_coarsening_level);
    }
    else if(geom_type == "sphere")
    {
    amrex::Print() << "\n Building sphere geometry." << std::endl;
        make_eb_sphere(geom, required_coarsening_level);
    }
    else
    {
    amrex::Print() << "\n No EB geometry declared in inputs => "
                   << " Will build all regular geometry." << std::endl;
        make_eb_regular(geom);
    }
    amrex::Print() << "Done making the geometry ebfactory.\n" << std::endl;
}
