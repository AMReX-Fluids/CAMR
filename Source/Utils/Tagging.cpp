#include <AMReX_ParmParse.H>

#include "CAMR.H"
#include "Tagging.H"

void
CAMR::read_tagging_params()
{
  // Nothing in CAMR reads TaggingParm: CAMR::errorEst tags only through the
  // amr.refinement_indicators mechanism (CAMR_error.cpp).  Rather than query
  // these keys -- which hides them from AMReX's unused-inputs report and
  // silently ignores them -- refuse them with a pointer to what works.
  amrex::ParmParse pp("tagging");

  static const char* const unsupported[] = {
    "denerr",   "max_denerr_lev",   "dengrad",   "max_dengrad_lev",
    "presserr", "max_presserr_lev", "pressgrad", "max_pressgrad_lev",
    "velerr",   "max_velerr_lev",   "velgrad",   "max_velgrad_lev",
    "vorterr",  "max_vorterr_lev",
    "temperr",  "max_temperr_lev",  "tempgrad",  "max_tempgrad_lev",
    "ftracerr", "max_ftracerr_lev", "ftracgrad", "max_ftracgrad_lev",
    "vfracerr", "max_vfracerr_lev"};

  for (const char* name : unsupported) {
    if (pp.contains(name)) {
      amrex::Abort(std::string("CAMR: input tagging.") + name +
                   " is not supported -- CAMR::errorEst ignores all tagging.*"
                   " keys.  Use amr.refinement_indicators with value_greater /"
                   " value_less / adjacent_difference_greater / vorticity_greater"
                   " (see Source/Utils/CAMR_error.cpp).");
    }
  }
}
