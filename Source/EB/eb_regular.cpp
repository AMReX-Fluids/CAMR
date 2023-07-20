#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>

#include <eb_if.H>

using namespace amrex;

void make_eb_regular(const Geometry& geom)
{
    EB2::AllRegularIF my_regular;
    auto gshop = EB2::makeShop(my_regular);
    EB2::Build(gshop, geom, 0, 100);
}
