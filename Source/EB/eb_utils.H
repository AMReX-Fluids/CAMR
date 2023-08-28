#ifndef EB_UTILS_H
#define EB_UTILS_H

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
bool
is_cut_neighborhood (const amrex::IntVect& iv,
                     amrex::Array4<amrex::EBCellFlag const> const& flags,
                     const amrex::IntVect& ng = amrex::IntVect(1))
{
    // Check if there's a cut cell in an ng neighborhood around iv
    bool has_cut_cell = flags(iv).isSingleValued();
    for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
        const auto dimvec = amrex::IntVect::TheDimensionVector(idir);
        for (int n = 1; n <= ng[idir]; n++) {
            has_cut_cell |= flags(iv - n * dimvec).isSingleValued() ||
                            flags(iv + n * dimvec).isSingleValued();
        }
    }
    return has_cut_cell;
}
#endif
