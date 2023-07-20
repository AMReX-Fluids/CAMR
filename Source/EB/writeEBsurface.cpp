#include <CAMR.H>
#include <AMReX_WriteEBSurface.H>

#if (AMREX_SPACEDIM == 3)
void CAMR::WriteMyEBSurface ()
{
  using namespace amrex;

  amrex::Print() << "Writing the geometry to a vtp file.\n" << std::endl;

  BoxArray & ba            = grids;
  DistributionMapping & dm = dmap;

  const EBFArrayBoxFactory* ebfact = &EBFactory();

  WriteEBSurface(ba,dm,geom,ebfact);
}
#endif
