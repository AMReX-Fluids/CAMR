#include <AMReX_LevelBld.H>

#include "CAMR.H"

class CAMRBld : public amrex::LevelBld
{
  void variableSetUp() override;
  void variableCleanUp() override;
  amrex::AmrLevel* operator()() override;
  amrex::AmrLevel* operator()(
    amrex::Amr& papa,
    int lev,
    const amrex::Geometry& level_geom,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    amrex::Real time) override;
};

CAMRBld CAMR_bld;

amrex::LevelBld*
getLevelBld()
{
  return &CAMR_bld;
}

void
CAMRBld::variableSetUp()
{
  CAMR::variableSetUp();
}

void
CAMRBld::variableCleanUp()
{
  CAMR::variableCleanUp();
}

amrex::AmrLevel*
CAMRBld::operator()()
{
  return new CAMR;
}

amrex::AmrLevel*
CAMRBld::operator()(
  amrex::Amr& papa,
  int lev,
  const amrex::Geometry& level_geom,
  const amrex::BoxArray& ba,
  const amrex::DistributionMapping& dm,
  amrex::Real time)
{
  return new CAMR(papa, lev, level_geom, ba, dm, time);
}
