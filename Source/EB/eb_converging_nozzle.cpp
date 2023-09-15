#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <CAMR.H>

using namespace amrex;

/********************************************************************************
 *                                                                              *
 * Function to create a converging nozzle EB.                                   *
 *                                                                              *
 ********************************************************************************/
void make_eb_converging_nozzle (const Geometry& geom, int required_coarsening_level)
{
  amrex::Real d_inlet = 8;
  amrex::Real l_inlet = 24;
  amrex::Real l_nozzle = 5;
  amrex::Real d_exit = 5;

  amrex::ParmParse pp("converging_nozzle");
  pp.query("d_inlet", d_inlet);
  pp.query("l_inlet", l_inlet);
  pp.query("l_nozzle", l_nozzle);
  pp.query("d_exit", d_exit);

  amrex::EB2::CylinderIF main(
    0.5 * d_inlet, 0,
    {AMREX_D_DECL(static_cast<amrex::Real>(0.5 * l_inlet), 0, 0)}, true);

  amrex::Real slope_nozzle =
    (0.5 * d_inlet - 0.5 * d_exit) / l_nozzle;
  amrex::Real norm = -1.0 / slope_nozzle;
  amrex::Real nmag = std::sqrt(1 + 1 / (norm * norm));
  amrex::EB2::PlaneIF nozzle_plane(
    {AMREX_D_DECL(0, 0, 0)},
    {AMREX_D_DECL(
      static_cast<amrex::Real>(1.0 / nmag), slope_nozzle / nmag, 0.0)},
    true);
  auto nozzle = amrex::EB2::translate(
    amrex::EB2::rotate(amrex::EB2::lathe(nozzle_plane), 90 * M_PI / 180, 1),
    {AMREX_D_DECL(
      l_inlet + static_cast<amrex::Real>(0.5 * d_inlet / slope_nozzle),
      0, 0)});

  amrex::EB2::CylinderIF exit(
    0.5 * d_exit, 0, {AMREX_D_DECL(l_inlet + l_nozzle, 0, 0)},
    true);
  auto nozzle_exit = amrex::EB2::makeIntersection(nozzle, exit);

  auto polys = amrex::EB2::makeUnion(main, nozzle_exit);
  auto gshop = amrex::EB2::makeShop(polys);
  EB2::Build(gshop, geom, required_coarsening_level, required_coarsening_level);
}
