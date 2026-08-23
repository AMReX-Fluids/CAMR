#include "CAMR.H"
#include "IndexDefines.H"

void
CAMR::construct_old_source(
  int src,
  const amrex::MultiFab& S,
  amrex::Real time,
  amrex::Real dt,
  int ng)
{
  AMREX_ASSERT(src >= 0 && src < num_src);
  AMREX_ASSERT(S.nGrow() >= ng);

  switch (src) {

  case ext_src:
    construct_old_ext_source(S, time, dt, ng);
    break;

  case grav_src:
    construct_old_grav_source(S, time, dt, ng);
    break;
  } // end switch
}

void
CAMR::construct_new_source(
  int src,
  amrex::Real time,
  amrex::Real dt)
{
  AMREX_ASSERT(src >= 0 && src < num_src);

  switch (src) {

  case ext_src:
    construct_new_ext_source(time, dt);
    break;

  case grav_src:
    construct_new_grav_source(time, dt);
    break;

  } // end switch
}
