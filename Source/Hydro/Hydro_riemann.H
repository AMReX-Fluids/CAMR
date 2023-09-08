#ifndef HYDRO_RIEMANN_H
#define HYDRO_RIEMANN_H
#include "EOS.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
riemannus(
  const amrex::Real rl_in,
  const amrex::Real ul,
  const amrex::Real vl,
#if (AMREX_SPACEDIM == 3)
  const amrex::Real v2l,
#endif
  const amrex::Real pl_in,
  const amrex::Real rel,
  const amrex::Real rr_in,
  const amrex::Real ur,
  const amrex::Real vr,
#if (AMREX_SPACEDIM == 3)
  const amrex::Real v2r,
#endif
  const amrex::Real pr_in,
  const amrex::Real rer,
  const amrex::Real cav,
  const amrex::Real gamcl,
  const amrex::Real gamcr,
  const amrex::Real small,
  const amrex::Real smallc,
  const amrex::Real small_dens,
  const amrex::Real small_pres,
  amrex::Real& ustar,
  amrex::Real& uflx_rho,
  AMREX_D_DECL(amrex::Real& uflx_u,
               amrex::Real& uflx_v,
               amrex::Real& uflx_w),
  amrex::Real& uflx_eden,
  amrex::Real& uflx_eint,
  AMREX_D_DECL(amrex::Real& qint_iu,
               amrex::Real& qint_iv1,
               amrex::Real& qint_iv2),
  amrex::Real& qint_gdpres,
  amrex::Real& qint_gdgame,
  amrex::Real  smallu)
{
    amrex::Real wsmall = smallc * small_dens;
    amrex::Real rl = std::max(rl_in,small_dens);
    amrex::Real pl = std::max(pl_in,small_pres);
    amrex::Real rr = std::max(rr_in,small_dens);
    amrex::Real pr = std::max(pr_in,small_pres);

    amrex::Real wl = std::max(wsmall, std::sqrt(std::abs(gamcl * pl * rl)));
    amrex::Real wr = std::max(wsmall, std::sqrt(std::abs(gamcr * pr * rr)));

    amrex::Real pstar = ((wr * pl + wl * pr) + wl * wr * (ul - ur)) / (wl + wr);
    pstar = std::max(pstar,small_pres);
    ustar = ((wl * ul + wr * ur) + (pl - pr)) / (wl + wr);

    amrex::Real ro,uo,po,reo,gamco;

    bool mask = std::abs(ustar) < smallu * 0.5 * (std::abs(ul) + std::abs(ur));
    ustar = mask ? 0.0 : ustar;

    if (ustar > 0.){

     ro = rl;
     uo = ul;
     po = pl;
     reo = rel;
     gamco = gamcl;

  } else if (ustar < 0) {

     ro = rr;
     uo = ur;
     po = pr;
     reo = rer;
     gamco = gamcr;

  } else {

     ro = 0.5*(rl+rr);
     uo = 0.5*(ul+ur);
     po = 0.5*(pl+pr);
     reo = 0.5*(rel+rer);
     gamco = 0.5*(gamcl+gamcr);
  }

  ro = std::max(ro,small_dens);
  amrex::Real roinv = 1.0/ro;

  amrex::Real co,co2inv,rstar,cstar,entho,estar,sgnm,spout,spin,ushock,frac,drho;

  co = std::sqrt(std::abs(gamco*po*roinv));
  co = std::max(smallc,co);
  co2inv = 1.0/(co*co);

  drho = (pstar - po)*co2inv;
  rstar = ro + drho;
  rstar = std::max(small_dens,rstar);

  entho = (reo + po)*roinv*co2inv;
  estar = reo + (pstar - po)*entho;
  cstar = std::sqrt(std::abs(gamco*pstar/rstar));
  cstar = std::max(cstar,smallc);

  sgnm = (ustar < 0.) ? -1.0 : 1.0;
  spout = co - sgnm*uo;
  spin = cstar - sgnm*ustar;
  ushock = 0.5*(spin + spout);

  if (pstar-po > 0.0) {
     spin = ushock;
     spout = ushock;
  }

  amrex::Real scr;
  if (spout-spin == 0.0){
     scr = small*cav;
  } else {
     scr = spout-spin;
  }

  frac = (1.0 + (spout + spin)/scr)*0.5;
  frac = std::max(0.0,std::min(1.0,frac));

  mask = ustar > 0.0;
  qint_iv1 = mask ? vl : vr;
#if (AMREX_SPACEDIM == 3)
  qint_iv2 = mask ? v2l : v2r;
#endif

  mask = (ustar == 0.0);
  qint_iv1 = mask ? 0.5 * (vl + vr) : qint_iv1;
#if (AMREX_SPACEDIM == 3)
  qint_iv2 = mask ? 0.5 * (v2l + v2r) : qint_iv2;
#endif

  amrex::Real rho_gd = frac*rstar+(1.0-frac)*ro;
  qint_iu = frac * ustar + (1.0 - frac) * uo;
  qint_gdpres = frac * pstar + (1.0 - frac) * po;
  amrex::Real re_gd = frac*estar + (1.0-frac)*reo;

  mask = (spout < 0.0);
  rho_gd = mask ? ro : rho_gd;
  qint_iu = mask ? uo : qint_iu;
  qint_gdpres = mask ? po : qint_gdpres;
  re_gd = mask ? reo : re_gd;

  mask = (spin >= 0.0);
  rho_gd = mask ? rstar : rho_gd;
  qint_iu = mask ? ustar : qint_iu;
  qint_gdpres = mask ? pstar : qint_gdpres;
  re_gd = mask ? estar : re_gd;

  qint_gdgame = qint_gdpres / re_gd + 1.0;
  qint_gdpres = std::max(qint_gdpres,small_pres);

  uflx_rho = rho_gd * qint_iu;

  uflx_u = uflx_rho * qint_iu + qint_gdpres;
  uflx_v = uflx_rho * qint_iv1;
#if (AMREX_SPACEDIM == 2)
  const amrex::Real rhoetot = re_gd +
    0.5 * rho_gd * (qint_iu * qint_iu + qint_iv1 * qint_iv1);
#elif (AMREX_SPACEDIM == 3)
  uflx_w = uflx_rho * qint_iv2;
  const amrex::Real rhoetot = re_gd +
    0.5 * rho_gd * (qint_iu * qint_iu + qint_iv1 * qint_iv1 + qint_iv2 * qint_iv2);
#endif
  uflx_eden = qint_iu * (rhoetot + qint_gdpres);
  uflx_eint = qint_iu * re_gd;
}

#endif
