#include "Godunov.H"
#if (AMREX_SPACEDIM == 2)
#include "Godunov_utils_2D.H"
#elif (AMREX_SPACEDIM == 3)
#include "Godunov_utils_3D.H"
#endif
#include "flatten.H"
#include "PPM.H"

void
trace_ppm(
  const amrex::Box& bx,
  const int idir,
  amrex::Array4<amrex::Real const> const& q_arr,
  amrex::Array4<amrex::Real const> const& q_aux,
  amrex::Array4<amrex::Real const> const& /*srcQ*/,
  amrex::Array4<amrex::Real> const& qm,
  amrex::Array4<amrex::Real> const& qp,
  const amrex::Box& vbx,
  const amrex::Real dt,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx,
  const int use_flattening,
  const amrex::Real small_dens,
  const amrex::Real small_pres,
  PassMap const* pmap)
{
  // here, lo and hi are the range we loop over -- this can include ghost cells
  // vlo and vhi are the bounds of the valid box (no ghost cells)

  //
  // rho : mass density
  // u, v, w : velocities
  // p : gas (hydro) pressure
  // ptot : total pressure (note for pure hydro, this is
  //        just the gas pressure)
  // rhoe_g : gas specific internal energy
  // cgas : sound speed for just the gas contribution
  // cc : total sound speed
  // h_g : gas specific enthalpy / cc**2
  //
  // for pure hydro, we will only consider:
  //    rho, u, v, w, ptot, rhoe_g, cc, h_g
  // amrex::Real hdt = 0.5 * dt;
  amrex::Real dtdx = dt / dx[idir];

  // auto lo = bx.loVect3d();
  // auto hi = bx.hiVect3d();

  auto vlo = vbx.loVect3d();
  auto vhi = vbx.hiVect3d();

  // This does the characteristic tracing to build the interface
  // states using the normal predictor only (no transverse terms).
  //
  // For each zone, we construct Im and Ip arrays -- these are the averages
  // of the various primitive state variables under the parabolic
  // interpolant over the region swept out by one of the 3 different
  // characteristic waves.
  //
  // Im is integrating to the left interface of the current zone
  // (which will be used to build the right ("p") state at that interface)
  // and Ip is integrating to the right interface of the current zone
  // (which will be used to build the left ("m") state at that interface).
  //
  //
  // The choice of reference state is designed to minimize the
  // effects of the characteristic projection.  We subtract the I's
  // off of the reference state, project the quantity such that it is
  // in terms of the characteristic varaibles, and then add all the
  // jumps that are moving toward the interface to the reference
  // state to get the full state on that interface.

  int QUN = 0;
  int QUT = 0;
#if (AMREX_SPACEDIM == 2)
  if (idir == 0) {
    QUN = QU;
    QUT = QV;
  } else if (idir == 1) {
    QUN = QV;
    QUT = QU;
  }
#elif (AMREX_SPACEDIM == 3)
  int QUTT = 0;
  if (idir == 0) {
    QUN = QU;
    QUT = QV;
    QUTT = QW;
  } else if (idir == 1) {
    QUN = QV;
    QUT = QW;
    QUTT = QU;
  } else if (idir == 2) {
    QUN = QW;
    QUT = QU;
    QUTT = QV;
  }
#endif

  // Trace to left and right edges using upwind PPM
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
    const amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
    const amrex::IntVect ivm2(
      iv - 2 * amrex::IntVect::TheDimensionVector(idir));
    const amrex::IntVect ivm1(
      iv - 1 * amrex::IntVect::TheDimensionVector(idir));
    const amrex::IntVect ivp1(
      iv + 1 * amrex::IntVect::TheDimensionVector(idir));
    const amrex::IntVect ivp2(
      iv + 2 * amrex::IntVect::TheDimensionVector(idir));

    amrex::Real cc = std::sqrt(q_aux(iv,QGAMC)* q_arr(iv, QPRES)/q_arr(iv, QRHO));

    amrex::Real un = q_arr(iv, QUN);

    // do the parabolic reconstruction and compute the
    // integrals under the characteristic waves

    amrex::Real flat = 1.0;
    // Calculate flattening in-place
    if (use_flattening == 1) {
      for (int dir_flat = 0; dir_flat < AMREX_SPACEDIM; dir_flat++) {
        flat = std::min(flat, flatten(i, j, k, dir_flat, q_arr));
      }
    }

    amrex::Real Ip[QVAR+1][3];
    amrex::Real Im[QVAR+1][3];

    for (int n = 0; n < QVAR; n++)
    {
        amrex::Real s[5];
        s[im2] = q_arr(ivm2, n);
        s[im1] = q_arr(ivm1, n);
        s[i0] = q_arr(iv, n);
        s[ip1] = q_arr(ivp1, n);
        s[ip2] = q_arr(ivp2, n);
        amrex::Real sm;
        amrex::Real sp;
        ppm_reconstruct(s, flat, sm, sp);
        ppm_int_profile(sm, sp, s[2], un, cc, dtdx, Ip[n], Im[n]);
    }

//  bit of a hack here.  keeping gamrc tracing in qvar
    {
        amrex::Real s[5];
        s[im2] = q_aux(ivm2, QGAMC);
        s[im1] = q_aux(ivm1, QGAMC);
        s[i0]  = q_aux(iv  , QGAMC);
        s[ip1] = q_aux(ivp1, QGAMC);
        s[ip2] = q_aux(ivp2, QGAMC);
        amrex::Real sm;
        amrex::Real sp;
        ppm_reconstruct(s, flat, sm, sp);
        ppm_int_profile(sm, sp, s[2], un, cc, dtdx, Ip[QVAR], Im[QVAR]);
    }

    // We do source term tracing in hydro_transx, hydro_transy, and
    // hydro_transz. So to be consistent we remove the source term
    // tracing here. However, Nyx and Castro do the source term
    // tracing here instead of in the trans routines. If we wanted
    // to do the tracing here, we would have to 1) remove the tracing
    // in the trans routines AND 2) add the tracing in the hydro_plm_x,
    // hydro_plm_y, hydro_plm_z routines.
    //
    // To do the tracing here: Uncomment the chunk below and
    // anything that uses Im_src and Ip_src.

    // // source terms
    // amrex::Real Ip_src[QVAR][3];
    // amrex::Real Im_src[QVAR][3];

    // for (int n = 0; n < QVAR; n++) {
    //   s[im2] = srcQ(ivm2, n);
    //   s[im1] = srcQ(ivm1, n);
    //   s[i0] = srcQ(iv, n);
    //   s[ip1] = srcQ(ivp1, n);
    //   s[ip2] = srcQ(ivp2, n);
    //   ppm_reconstruct(s, flat, sm, sp);
    //   ppm_int_profile(sm, sp, s[i0], un, cc, dtdx, Ip_src[n], Im_src[n]);
    // }

    // Upwind the passive variables
    for (int ipassive = 0; ipassive < NPASSIVE; ++ipassive) {
        const int n = pmap->qpassMap[ipassive];

       // Plus state on face i
       if (
         (idir == 0 && i >= vlo[0]) || (idir == 1 && j >= vlo[1]) ||
         (idir == 2 && k >= vlo[2])) {

         qp(iv, n) = Im[n][1];
       }
       // Minus state on face i+1
       if (
         (idir == 0 && i <= vhi[0]) || (idir == 1 && j <= vhi[1]) ||
         (idir == 2 && k <= vhi[2])) {
         qm(ivp1, n) = Ip[n][1];
       }
    }

    // plus state on face i
    if (
      (idir == 0 && i >= vlo[0]) || (idir == 1 && j >= vlo[1]) ||
      (idir == 2 && k >= vlo[2])) {

      // Set the reference state
      // This will be the fastest moving state to the left --
      // this is the method that Miller & Colella and Colella &
      // Woodward use
      amrex::Real rho_ref = Im[QRHO][0];
      amrex::Real un_ref = Im[QUN][0];

      amrex::Real p_ref = Im[QPRES][0];
      amrex::Real rhoe_g_ref = Im[QREINT][0];
      amrex::Real gam_ref = Im[QVAR][0];

      rho_ref = std::max( rho_ref, small_dens);
      amrex::Real rho_ref_inv = 1.0 / rho_ref;
      p_ref = std::max(p_ref, small_pres);

      // For tracing
      amrex::Real cc_ref = std::sqrt(gam_ref*p_ref/rho_ref);
      amrex::Real csq_ref = cc_ref * cc_ref;
      amrex::Real cc_ref_inv = 1.0 / cc_ref;
      amrex::Real h_g_ref = (p_ref + rhoe_g_ref) * rho_ref_inv / csq_ref;

      // *m are the jumps carried by un-c
      // *p are the jumps carried by un+c

      // Note: for the transverse velocities, the jump is carried
      //       only by the u wave (the contact)

      // we also add the sources here so they participate in the tracing
      amrex::Real dum = un_ref - Im[QUN][0] /*- hdt * Im_src[QUN][0]*/;
      amrex::Real dptotm = p_ref - Im[QPRES][0] /*- hdt * Im_src[QPRES][0]*/;

      amrex::Real drho = rho_ref - Im[QRHO][1] /*- hdt * Im_src[QRHO][1]*/;
      amrex::Real dptot = p_ref - Im[QPRES][1] /*- hdt * Im_src[QPRES][1]*/;
      amrex::Real drhoe_g =
        rhoe_g_ref - Im[QREINT][1] /*- hdt * Im_src[QREINT][1]*/;

      amrex::Real dup = un_ref - Im[QUN][2] /*- hdt * Im_src[QUN][2]*/;
      amrex::Real dptotp = p_ref - Im[QPRES][2] /*- hdt * Im_src[QPRES][2]*/;

      // {rho, u, p, (rho e)} eigensystem

      // These are analogous to the beta's from the original PPM
      // paper (except we work with rho instead of tau).  This is
      // simply (l . dq), where dq = qref - I(q)

      amrex::Real alpham =
        0.5 * (dptotm * rho_ref_inv * cc_ref_inv - dum) * rho_ref * cc_ref_inv;
      amrex::Real alphap =
        0.5 * (dptotp * rho_ref_inv * cc_ref_inv + dup) * rho_ref * cc_ref_inv;
      amrex::Real alpha0r = drho - dptot / csq_ref;
      amrex::Real alpha0e_g = drhoe_g - dptot * h_g_ref;

      if (un-cc > 0.) {
          alpham = 0.;
      } else if (un-cc < 0.) {
          alpham *= -1.0;
      } else {
          alpham *= -0.5;
      }

      if (un+cc > 0.) {
          alphap = 0.;
      } else if (un+cc < 0.) {
          alphap *= -1.0;
      } else {
          alphap *= -0.5;
      }

      if (un > 0.) {
          alpha0r   = 0.;
          alpha0e_g = 0.;
      } else if (un < 0.) {
          alpha0r   *= -1.0;
          alpha0e_g *= -1.0;
      } else {
          alpha0r   *= -0.5;
          alpha0e_g *= -0.5;
      }

      // The final interface states are just
      // q_s = q_ref - sum(l . dq) r
      // note that the a{mpz}right as defined above have the minus already
      qp(iv, QRHO) = rho_ref + alphap + alpham + alpha0r;
      qp(iv, QUN) = un_ref + (alphap - alpham) * cc_ref * rho_ref_inv;
      qp(iv, QPRES) = p_ref + (alphap + alpham) * csq_ref;

      qp(iv, QRHO)  = std::max( qp(iv, QRHO ), small_dens);
      qp(iv, QPRES) = std::max( qp(iv, QPRES), small_pres);
      // Transverse velocities -- there's no projection here, so we
      // don't need a reference state.  We only care about the state
      // traced under the middle wave

      // Recall that I already takes the limit of the parabola
      // in the event that the wave is not moving toward the
      // interface
      qp(iv, QUT) = Im[QUT][1] /*+ hdt * Im_src[QUT][1]*/;
#if (AMREX_SPACEDIM == 3)
      qp(iv, QUTT) = Im[QUTT][1] /*+ hdt * Im_src[QUTT][1]*/;
#endif

      // This allows the (rho e) to take advantage of (pressure > small_pres)
      qp(iv, QREINT) = rhoe_g_ref + (alphap + alpham)*h_g_ref*csq_ref + alpha0e_g;
    }

    // minus state on face i + 1
    if ( (idir == 0 && i <= vhi[0]) || (idir == 1 && j <= vhi[1]) ||
         (idir == 2 && k <= vhi[2]))
    {
      // Set the reference state
      // This will be the fastest moving state to the right
      amrex::Real rho_ref = Ip[QRHO][2];
      amrex::Real un_ref = Ip[QUN][2];

      amrex::Real p_ref = Ip[QPRES][2];
      amrex::Real rhoe_g_ref = Ip[QREINT][2];
      amrex::Real gam_ref = Ip[QVAR][2];

      rho_ref = std::max( rho_ref, small_dens );
      amrex::Real rho_ref_inv = 1.0 / rho_ref;
      p_ref = std::max( p_ref, small_pres );

      // For tracing
      amrex::Real cc_ref = std::sqrt(gam_ref*p_ref/rho_ref);

      amrex::Real csq_ref = cc_ref * cc_ref;
      amrex::Real cc_ref_inv = 1.0 / cc_ref;
      amrex::Real h_g_ref = (p_ref + rhoe_g_ref) * rho_ref_inv / csq_ref;

      // *m are the jumps carried by u-c
      // *p are the jumps carried by u+c

      amrex::Real dum = un_ref - Ip[QUN][0] /*- hdt * Ip_src[QUN][0]*/;
      amrex::Real dptotm = p_ref - Ip[QPRES][0] /*- hdt * Ip_src[QPRES][0]*/;

      amrex::Real drho = rho_ref - Ip[QRHO][1] /*- hdt * Ip_src[QRHO][1]*/;
      amrex::Real dptot = p_ref - Ip[QPRES][1] /*- hdt * Ip_src[QPRES][1]*/;
      amrex::Real drhoe_g =
        rhoe_g_ref - Ip[QREINT][1] /*- hdt * Ip_src[QREINT][1]*/;

      amrex::Real dup = un_ref - Ip[QUN][2] /*- hdt * Ip_src[QUN][2]*/;
      amrex::Real dptotp = p_ref - Ip[QPRES][2] /*- hdt * Ip_src[QPRES][2]*/;

      // {rho, u, p, (rho e)} eigensystem

      // These are analogous to the beta's from the original PPM
      // paper (except we work with rho instead of tau).  This is
      // simply (l . dq), where dq = qref - I(q)

      amrex::Real alpham =
        0.5 * (dptotm * rho_ref_inv * cc_ref_inv - dum) * rho_ref * cc_ref_inv;
      amrex::Real alphap =
        0.5 * (dptotp * rho_ref_inv * cc_ref_inv + dup) * rho_ref * cc_ref_inv;
      amrex::Real alpha0r = drho - dptot / csq_ref;
      amrex::Real alpha0e_g = drhoe_g - dptot * h_g_ref;

      if (un-cc > 0.) {
          alpham *= -1.0;
      } else if (un-cc < 0.) {
          alpham = 0.;
      } else {
          alpham *= -0.5;
      }

      if (un+cc > 0.) {
          alphap *= -1.0;
      } else if (un+cc < 0.) {
          alphap = 0.;
      } else {
          alphap *= -0.5;
      }

      if (un > 0.) {
          alpha0r   *= -1.0;
          alpha0e_g *= -1.0;
      } else if (un < 0.) {
          alpha0r   = 0.;
          alpha0e_g = 0.;
      } else {
          alpha0r   *= -0.5;
          alpha0e_g *= -0.5;
      }

      // The final interface states are just
      // q_s = q_ref - sum (l . dq) r
      // note that the a{mpz}left as defined above have the minus already
      qm(ivp1, QRHO) = rho_ref + alphap + alpham + alpha0r;
      qm(ivp1, QUN) = un_ref + (alphap - alpham) * cc_ref * rho_ref_inv;
      qm(ivp1, QPRES) = p_ref + (alphap + alpham) * csq_ref;

      qm(ivp1, QRHO)  = std::max( qm(ivp1, QRHO), small_dens);
      qm(ivp1, QPRES) = std::max( qm(ivp1, QPRES), small_dens);

      // transverse velocities
      qm(ivp1, QUT) = Ip[QUT][1] /*+ hdt * Ip_src[QUT][1]*/;
#if (AMREX_SPACEDIM == 3)
      qm(ivp1, QUTT) = Ip[QUTT][1] /*+ hdt * Ip_src[QUTT][1]*/;
#endif

      // This allows the (rho e) to take advantage of (pressure >
      // small_pres)
      qm(ivp1, QREINT) = rhoe_g_ref + (alphap + alpham)*h_g_ref*csq_ref + alpha0e_g;
    }
  });
}
