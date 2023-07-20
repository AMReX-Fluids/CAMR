#include <AMReX_Config.H>
#include <AMReX_REAL.H>
#include "MOL_hydro_K.H"
#include "MOL_hypflux_K.H"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void eb_compute_div (int i, int j, int k, int n,
                     amrex::IntVect const& blo, amrex::IntVect const& bhi,
                     amrex::Array4<amrex::Real const> const& q,
                     amrex::Array4<amrex::Real const> const& qaux,
                     amrex::Array4<amrex::Real> const& divu,
                     AMREX_D_DECL(amrex::Array4<amrex::Real const> const& u,
                                  amrex::Array4<amrex::Real const> const& v,
                                  amrex::Array4<amrex::Real const> const& w),
                     AMREX_D_DECL(amrex::Array4<amrex::Real> const& fx,
                                  amrex::Array4<amrex::Real> const& fy,
                                  amrex::Array4<amrex::Real> const& fz),
                     amrex::Array4<amrex::EBCellFlag const> const& flag,
                     amrex::Array4<amrex::Real const> const& vfrc,
                     amrex::Array4<amrex::Real      > const& redistwgt,
                     AMREX_D_DECL(amrex::Array4<amrex::Real const> const& apx,
                                  amrex::Array4<amrex::Real const> const& apy,
                                  amrex::Array4<amrex::Real const> const& apz),
                     AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fcx,
                                  amrex::Array4<amrex::Real const> const& fcy,
                                  amrex::Array4<amrex::Real const> const& fcz),
                     amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
                     const amrex::Real small,
                     const amrex::Real small_dens,
                     const amrex::Real small_pres,
                     int eb_weights_type)
{
    AMREX_D_TERM(bool x_high = (i == bhi[0]);,
                 bool y_high = (j == bhi[1]);,
                 bool z_high = (k == bhi[2]));
    bool valid_cell = AMREX_D_TERM( (blo[0] <= i) && (i <= bhi[0]),
                                 && (blo[1] <= j) && (j <= bhi[1]),
                                 && (blo[2] <= k) && (k <= bhi[2]) );

#if (AMREX_SPACEDIM == 2)
    amrex::Real cell_vol_inv = dxinv[0]*dxinv[1];
    if (flag(i,j,k).isCovered())
    {
        divu(i,j,k,n) = 0.0;
        if (valid_cell) {
            fx(i,j,k,n) = 0.;
            fy(i,j,k,n) = 0.;
            if (x_high) {
                fx(i+1,j,k,n) = 0.;
            }
            if (y_high) {
                fy(i,j+1,k,n) = 0.;
            }
        }
    }
    else if (flag(i,j,k).isRegular())
    {
        divu(i,j,k,n) = cell_vol_inv * (u(i+1,j,k,n)-u(i,j,k,n))
            +           cell_vol_inv * (v(i,j+1,k,n)-v(i,j,k,n));
        if (valid_cell) {
            fx(i,j,k,n) = u(i,j,k,n);
            fy(i,j,k,n) = v(i,j,k,n);
            if (x_high) {
                fx(i+1,j,k,n) = u(i+1,j,k,n);
            }
            if (y_high) {
                fy(i,j+1,k,n) = v(i,j+1,k,n);
            }
        }
    }
    else
    {
        amrex::Real fxm = u(i,j,k,n);
        if (apx(i,j,k) != 0.0 && apx(i,j,k) != 1.0) {
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,0)));
            amrex::Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0;
            fxm = (1.0-fracy)*fxm + fracy *u(i,jj,k ,n);
        }
        if (valid_cell) {
            fx(i,j,k,n) = fxm;
        }

        amrex::Real fxp = u(i+1,j,k,n);
        if (apx(i+1,j,k) != 0.0 && apx(i+1,j,k) != 1.0) {
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,0)));
            amrex::Real fracy = flag(i+1,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i+1,j,k,0)) : 0.0;
            fxp = (1.0-fracy)*fxp + fracy *u(i+1,jj,k,n);

        }
        if (valid_cell && x_high) {
            fx(i+1,j,k,n) = fxp;
        }

        amrex::Real fym = v(i,j,k,n);
        if (apy(i,j,k) != 0.0 && apy(i,j,k) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,0)));
            amrex::Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0;
            fym = (1.0-fracx)*fym +  fracx *v(ii,j,k,n);
        }
        if (valid_cell) {
            fy(i,j,k,n) = fym;
        }

        amrex::Real fyp = v(i,j+1,k,n);
        if (apy(i,j+1,k) != 0.0 && apy(i,j+1,k) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,0)));
            amrex::Real fracx = flag(i,j+1,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j+1,k,0)) : 0.0;
            fyp = (1.0-fracx)*fyp + fracx *v(ii,j+1,k,n);
        }
        if (valid_cell && y_high) {
            fy(i,j+1,k,n) = fyp;
        }

        divu(i,j,k,n) = (1.0/vfrc(i,j,k)) * cell_vol_inv *
            ( apx(i+1,j,k)*fxp-apx(i,j,k)*fxm
            + apy(i,j+1,k)*fyp-apy(i,j,k)*fym );

        amrex::GpuArray<amrex::Real,NVAR> flux_hyp_wall;
        for(int nnn=0; nnn<NVAR; nnn++) flux_hyp_wall[nnn]=0.;
        compute_hyp_wallflux(i,j,k,q,qaux,
                             apx(i,j,k),apx(i+1,j,k),apy(i,j,k),apy(i,j+1,k),
                             small,small_dens,small_pres,flux_hyp_wall);

        // Here we assume dx == dy == dz
        divu(i,j,k,n) +=  flux_hyp_wall[n]*dxinv[0]/vfrc(i,j,k);
    }

#else // 3-d starts here

    amrex::Real cell_vol_inv = dxinv[0]*dxinv[1]*dxinv[2];
    if (flag(i,j,k).isCovered())
    {
        divu(i,j,k,n) = 0.0;
        if (valid_cell)
        {
            fx(i,j,k,n) = 0.;
            fy(i,j,k,n) = 0.;
            fz(i,j,k,n) = 0.;
            if (x_high) {
                fx(i+1,j,k,n) = 0.;
            }
            if (y_high) {
                fy(i,j+1,k,n) = 0.;
            }
            if (z_high) {
                fz(i,j,k+1,n) = 0.;
            }
        }
    }
    else if (flag(i,j,k).isRegular())
    {
        divu(i,j,k,n) = cell_vol_inv * (u(i+1,j,k,n)-u(i,j,k,n))
            +           cell_vol_inv * (v(i,j+1,k,n)-v(i,j,k,n))
            +           cell_vol_inv * (w(i,j,k+1,n)-w(i,j,k,n));
        if (valid_cell)
        {
            fx(i,j,k,n) = u(i,j,k,n);
            fy(i,j,k,n) = v(i,j,k,n);
            fz(i,j,k,n) = w(i,j,k,n);
            if (x_high) {
                fx(i+1,j,k,n) = u(i+1,j,k,n);
            }
            if (y_high) {
                fy(i,j+1,k,n) = v(i,j+1,k,n);
            }
            if (z_high) {
                fz(i,j,k+1,n) = w(i,j,k+1,n);
            }
        }
    }
    else
    {
        amrex::Real fxm = u(i,j,k,n);
        if (apx(i,j,k) != 0.0 && apx(i,j,k) != 1.0) {
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,1)));
            amrex::Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0;
            amrex::Real fracz = flag(i,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcx(i,j,k,1)) : 0.0;
            fxm = (1.0-fracy)*(1.0-fracz)*fxm
                +      fracy *(1.0-fracz)*u(i,jj,k ,n)
                +      fracz *(1.0-fracy)*u(i,j ,kk,n)
                +      fracy *     fracz *u(i,jj,kk,n);
        }
        if (valid_cell) {
            fx(i,j,k,n) = fxm;
        }

        amrex::Real fxp = u(i+1,j,k,n);
        if (apx(i+1,j,k) != 0.0 && apx(i+1,j,k) != 1.0) {
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,1)));
            amrex::Real fracy = flag(i+1,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i+1,j,k,0)) : 0.0;
            amrex::Real fracz = flag(i+1,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcx(i+1,j,k,1)) : 0.0;
            fxp = (1.0-fracy)*(1.0-fracz)*fxp
                +      fracy *(1.0-fracz)*u(i+1,jj,k ,n)
                +      fracz *(1.0-fracy)*u(i+1,j ,kk,n)
                +      fracy *     fracz *u(i+1,jj,kk,n);

        }
        if (valid_cell && x_high) {
            fx(i+1,j,k,n) = fxp;
        }

        amrex::Real fym = v(i,j,k,n);
        if (apy(i,j,k) != 0.0 && apy(i,j,k) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,1)));
            amrex::Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0;
            amrex::Real fracz = flag(i,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcy(i,j,k,1)) : 0.0;
            fym = (1.0-fracx)*(1.0-fracz)*fym
                +      fracx *(1.0-fracz)*v(ii,j,k ,n)
                +      fracz *(1.0-fracx)*v(i ,j,kk,n)
                +      fracx *     fracz *v(ii,j,kk,n);
        }
        if (valid_cell) {
            fy(i,j,k,n) = fym;
        }

        amrex::Real fyp = v(i,j+1,k,n);
        if (apy(i,j+1,k) != 0.0 && apy(i,j+1,k) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,1)));
            amrex::Real fracx = flag(i,j+1,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j+1,k,0)) : 0.0;
            amrex::Real fracz = flag(i,j+1,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcy(i,j+1,k,1)) : 0.0;
            fyp = (1.0-fracx)*(1.0-fracz)*fyp
                +      fracx *(1.0-fracz)*v(ii,j+1,k ,n)
                +      fracz *(1.0-fracx)*v(i ,j+1,kk,n)
                +      fracx *     fracz *v(ii,j+1,kk,n);
        }
        if (valid_cell && y_high) {
            fy(i,j+1,k,n) = fyp;
        }

        amrex::Real fzm = w(i,j,k,n);
        if (apz(i,j,k) != 0.0 && apz(i,j,k) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,0)));
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,1)));
            amrex::Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcz(i,j,k,0)) : 0.0;
            amrex::Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcz(i,j,k,1)) : 0.0;

            fzm = (1.0-fracx)*(1.0-fracy)*fzm
                +      fracx *(1.0-fracy)*w(ii,j ,k,n)
                +      fracy *(1.0-fracx)*w(i ,jj,k,n)
                +      fracx *     fracy *w(ii,jj,k,n);
        }
        if (valid_cell) {
            fz(i,j,k,n) = fzm;
        }

        amrex::Real fzp = w(i,j,k+1,n);
        if (apz(i,j,k+1) != 0.0 && apz(i,j,k+1) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k+1,0)));
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k+1,1)));
            amrex::Real fracx = flag(i,j,k+1).isConnected(ii-i,0,0) ? amrex::Math::abs(fcz(i,j,k+1,0)) : 0.0;
            amrex::Real fracy = flag(i,j,k+1).isConnected(0,jj-j,0) ? amrex::Math::abs(fcz(i,j,k+1,1)) : 0.0;
            fzp = (1.0-fracx)*(1.0-fracy)*fzp
                +      fracx *(1.0-fracy)*w(ii,j ,k+1,n)
                +      fracy *(1.0-fracx)*w(i ,jj,k+1,n)
                +      fracx *     fracy *w(ii,jj,k+1,n);
        }
        if (valid_cell && z_high) {
            fz(i,j,k+1,n) = fzp;
        }

        amrex::GpuArray<amrex::Real,NVAR> flux_hyp_wall;
        for(int nnn=0; nnn<NVAR; nnn++) flux_hyp_wall[nnn]=0.;
        compute_hyp_wallflux(i,j,k,q,qaux,
                             apx(i,j,k),apx(i+1,j,k),apy(i,j,k),apy(i,j+1,k),
                             apz(i,j,k),apz(i,j,k+1),small,small_dens,small_pres,flux_hyp_wall);

        // With EB we assume dx == dy == dz
        // NOTE: we have already made the fluxes extensive so we define by dx*dy*dz here ...
        divu(i,j,k,n) = cell_vol_inv / vfrc(i,j,k) * (apx(i+1,j,k)*fxp-apx(i,j,k)*fxm + apy(i,j+1,k)*fyp-apy(i,j,k)*fym
                                                     +apz(i,j,k+1)*fzp-apz(i,j,k)*fzm);
        divu(i,j,k,n) +=  flux_hyp_wall[n]*dxinv[0]/vfrc(i,j,k);
    }
#endif

    // The operations following this assume we have returned the negative of the divergence of fluxes.
    divu(i,j,k,n) *= -1.0;

    // Go ahead and make the redistwgt array here since we'll need it in flux_redistribute
    if (eb_weights_type == 0)
        { redistwgt(i,j,k) = 1.0; }
    else if (eb_weights_type == 1)
        { redistwgt(i,j,k) = q(i,j,k,QRHO)*( q(i,j,k,QREINT) +
#if (AMREX_SPACEDIM == 2)
                        0.5*(q(i,j,k,QU)*q(i,j,k,QU) + q(i,j,k,QV)*q(i,j,k,QV)) ); }
#else
                    0.5*(q(i,j,k,QU)*q(i,j,k,QU) + q(i,j,k,QV)*q(i,j,k,QV) + q(i,j,k,QW)*q(i,j,k,QW)) );
      }
#endif
    else if (eb_weights_type == 2)
        { redistwgt(i,j,k) = q(i,j,k,QRHO); }
    else if (eb_weights_type == 3)
        { redistwgt(i,j,k) = vfrc(i,j,k); }
}
