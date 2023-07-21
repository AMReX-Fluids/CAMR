
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
CAMR_artif_visc(
    const int i,
    const int j,
    const int k,
    amrex::Array4<amrex::Real> const& flx,
    amrex::Array4<const amrex::Real> const& divu,
    amrex::Array4<const amrex::Real> const& u,
    amrex::Real const dx,
    amrex::Real const difmag,
    const int dir)
{
    amrex::ignore_unused(k);
    const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
    const amrex::IntVect ivm(iv - amrex::IntVect::TheDimensionVector(dir));
    const amrex::GpuArray<const int, 3> bdim{{dir == 0, dir == 1, dir == 2}};
    const amrex::GpuArray<const int, 3> l_idx{
      {bdim[0] * 0 + bdim[1] * 1 + bdim[2] * 2,
       bdim[0] * 1 + bdim[1] * 0 + bdim[2] * 0,
       bdim[0] * 2 + bdim[1] * 2 + bdim[2] * 1}};

    const amrex::IntVect ivpj(iv + amrex::IntVect::TheDimensionVector(l_idx[1]));
    AMREX_D_TERM(,,const amrex::IntVect ivpk(iv + amrex::IntVect::TheDimensionVector(l_idx[2]));
                 const amrex::IntVect ivpp(
                   iv + amrex::IntVect::TheDimensionVector(l_idx[1]) +
                   amrex::IntVect::TheDimensionVector(l_idx[2])););
    const amrex::Real div =
      difmag *
      std::min( 0.0, AMREX_D_PICK(
               divu(iv), 0.5 * (divu(iv) + divu(ivpj)),
               0.25 * (divu(iv) + divu(ivpj) + divu(ivpk) + divu(ivpp))));

    for (int n = 0; n < NVAR; ++n) {
      if (n != UTEMP) {
        flx(iv, n) += dx * div * (u(iv, n) - u(ivm, n));
      }
    }
    flx(iv, UTEMP) = 0.0;
}
