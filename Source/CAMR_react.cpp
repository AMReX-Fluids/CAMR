
#include "CAMR.H"
#include "EOS.H"

using std::string;

using namespace amrex;

void
CAMR::react(MultiFab& S)
{
    BL_PROFILE("CAMR::react()");
 
    if (do_react) {
        amrex::Print() << "This is a stub for reactions but nothing is happening yet" << std::endl;
    }
}
