#include <AMReX_Config.H>
#include <AMReX_REAL.H>
#include "IndexDefines.H"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
eb_pdivu (int i, int j, int k,
          amrex::Array4<amrex::Real const> const& q,
          AMREX_D_DECL(amrex::Array4<amrex::Real const> const& q1,
                       amrex::Array4<amrex::Real const> const& q2,
                       amrex::Array4<amrex::Real const> const& q3),
          amrex::Array4<amrex::Real> const& divu,
          amrex::Array4<amrex::EBCellFlag const> const& flag,
          amrex::Array4<amrex::Real const> const& vfrc,
          AMREX_D_DECL(amrex::Array4<amrex::Real const> const& apx,
                       amrex::Array4<amrex::Real const> const& apy,
                       amrex::Array4<amrex::Real const> const& apz),
          AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fcx,
                       amrex::Array4<amrex::Real const> const& fcy,
                       amrex::Array4<amrex::Real const> const& fcz),
          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv)
{
    amrex::Real pdivu;

#if (AMREX_SPACEDIM == 2)
    if (flag(i,j,k).isRegular())
    {
       pdivu = 0.5 * ((q1(i + 1, j, k, GDPRES) + q1(i, j, k, GDPRES)) *
                      (q1(i + 1, j, k, GDU)  - q1(i, j, k, GDU) ) *dxinv[0]
                     +(q2(i, j + 1, k, GDPRES) + q2(i, j, k, GDPRES)) *
                      (q2(i, j + 1, k, GDV)  - q2(i, j, k, GDV) ) * dxinv[1]);

        divu(i,j,k,UEINT) += pdivu;
    }
    else
    {
        amrex::Real fxm = q1(i,j,k,GDU);
        if (apx(i,j,k) != 0.0 && apx(i,j,k) != 1.0) {
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,0)));
            amrex::Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0;
            fxm = (1.0-fracy)*fxm + fracy *q1(i,jj,k ,GDU);
        }

        amrex::Real fxp = q1(i+1,j,k,GDU);
        if (apx(i+1,j,k) != 0.0 && apx(i+1,j,k) != 1.0) {
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,0)));
            amrex::Real fracy = flag(i+1,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i+1,j,k,0)) : 0.0;
            fxp = (1.0-fracy)*fxp + fracy *q1(i+1,jj,k,GDU);

        }

        amrex::Real fym = q2(i,j,k,GDV);
        if (apy(i,j,k) != 0.0 && apy(i,j,k) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,0)));
            amrex::Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0;
            fym = (1.0-fracx)*fym +  fracx *q2(ii,j,k,GDV);
        }

        amrex::Real fyp = q2(i,j+1,k,GDV);
        if (apy(i,j+1,k) != 0.0 && apy(i,j+1,k) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,0)));
            amrex::Real fracx = flag(i,j+1,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j+1,k,0)) : 0.0;
            fyp = (1.0-fracx)*fyp + fracx *q2(ii,j+1,k,GDV);
        }

        divu(i,j,k,UEINT) += (1.0/vfrc(i,j,k)) * q(i,j,k,QPRES)*
            ( dxinv[0] * (apx(i+1,j,k)*fxp-apx(i,j,k)*fxm)
            + dxinv[1] * (apy(i,j+1,k)*fyp-apy(i,j,k)*fym) );

    }

#else // 3-d starts here

    if (flag(i,j,k).isRegular())
    {
       pdivu = 0.5 * ((q1(i+1,j,k, GDPRES) + q1(i,j,k, GDPRES)) *
                      (q1(i+1,j,k, GDU   ) - q1(i,j,k, GDU   )) *dxinv[0]
                     +(q2(i,j+1,k, GDPRES) + q2(i,j,k, GDPRES)) *
                      (q2(i,j+1,k, GDV   ) - q2(i,j,k, GDV   )) * dxinv[1]
                     +(q2(i,j,k+1, GDPRES) + q2(i,j,k, GDPRES)) *
                      (q2(i,j,k+1, GDW   ) - q2(i,j,k, GDW   )) * dxinv[2]);

        divu(i,j,k,UEINT) += pdivu;
    }
    else
    {
        amrex::Real fxm = q1(i,j,k,GDU);
        if (apx(i,j,k) != 0.0 && apx(i,j,k) != 1.0) {
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0, fcx(i,j,k,1)));
            amrex::Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i,j,k,0)) : 0.0;
            amrex::Real fracz = flag(i,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcx(i,j,k,1)) : 0.0;
            fxm = (1.0-fracy)*(1.0-fracz)*fxm
                +      fracy *(1.0-fracz)*q1(i,jj,k ,GDU)
                +      fracz *(1.0-fracy)*q1(i,j ,kk,GDU)
                +      fracy *     fracz *q1(i,jj,kk,GDU);
        }

        amrex::Real fxp = q1(i+1,j,k,GDU);
        if (apx(i+1,j,k) != 0.0 && apx(i+1,j,k) != 1.0) {
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcx(i+1,j,k,1)));
            amrex::Real fracy = flag(i+1,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcx(i+1,j,k,0)) : 0.0;
            amrex::Real fracz = flag(i+1,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcx(i+1,j,k,1)) : 0.0;
            fxp = (1.0-fracy)*(1.0-fracz)*fxp
                +      fracy *(1.0-fracz)*q1(i+1,jj,k ,GDU)
                +      fracz *(1.0-fracy)*q1(i+1,j ,kk,GDU)
                +      fracy *     fracz *q1(i+1,jj,kk,GDU);

        }

        amrex::Real fym = q2(i,j,k,GDV);
        if (apy(i,j,k) != 0.0 && apy(i,j,k) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j,k,1)));
            amrex::Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j,k,0)) : 0.0;
            amrex::Real fracz = flag(i,j,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcy(i,j,k,1)) : 0.0;
            fym = (1.0-fracx)*(1.0-fracz)*fym
                +      fracx *(1.0-fracz)*q2(ii,j,k ,GDV)
                +      fracz *(1.0-fracx)*q2(i ,j,kk,GDV)
                +      fracx *     fracz *q2(ii,j,kk,GDV);
        }

        amrex::Real fyp = q2(i,j+1,k,GDV);
        if (apy(i,j+1,k) != 0.0 && apy(i,j+1,k) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,0)));
            int kk = k + static_cast<int>(amrex::Math::copysign(1.0,fcy(i,j+1,k,1)));
            amrex::Real fracx = flag(i,j+1,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcy(i,j+1,k,0)) : 0.0;
            amrex::Real fracz = flag(i,j+1,k).isConnected(0,0,kk-k) ? amrex::Math::abs(fcy(i,j+1,k,1)) : 0.0;
            fyp = (1.0-fracx)*(1.0-fracz)*fyp
                +      fracx *(1.0-fracz)*q2(ii,j+1,k ,GDV)
                +      fracz *(1.0-fracx)*q2(i ,j+1,kk,GDV)
                +      fracx *     fracz *q2(ii,j+1,kk,GDV);
        }

        amrex::Real fzm = q3(i,j,k,GDW);
        if (apz(i,j,k) != 0.0 && apz(i,j,k) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,0)));
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k,1)));
            amrex::Real fracx = flag(i,j,k).isConnected(ii-i,0,0) ? amrex::Math::abs(fcz(i,j,k,0)) : 0.0;
            amrex::Real fracy = flag(i,j,k).isConnected(0,jj-j,0) ? amrex::Math::abs(fcz(i,j,k,1)) : 0.0;

            fzm = (1.0-fracx)*(1.0-fracy)*fzm
                +      fracx *(1.0-fracy)*q3(ii,j ,k,GDW)
                +      fracy *(1.0-fracx)*q3(i ,jj,k,GDW)
                +      fracx *     fracy *q3(ii,jj,k,GDW);
        }

        amrex::Real fzp = q3(i,j,k+1,GDW);
        if (apz(i,j,k+1) != 0.0 && apz(i,j,k+1) != 1.0) {
            int ii = i + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k+1,0)));
            int jj = j + static_cast<int>(amrex::Math::copysign(1.0,fcz(i,j,k+1,1)));
            amrex::Real fracx = flag(i,j,k+1).isConnected(ii-i,0,0) ? amrex::Math::abs(fcz(i,j,k+1,0)) : 0.0;
            amrex::Real fracy = flag(i,j,k+1).isConnected(0,jj-j,0) ? amrex::Math::abs(fcz(i,j,k+1,1)) : 0.0;
            fzp = (1.0-fracx)*(1.0-fracy)*fzp
                +      fracx *(1.0-fracy)*q3(ii,j ,k+1,GDW)
                +      fracy *(1.0-fracx)*q3(i ,jj,k+1,GDW)
                +      fracx *     fracy *q3(ii,jj,k+1,GDW);
        }

        divu(i,j,k,UEINT) += (1.0/vfrc(i,j,k)) * q(i,j,k,QPRES)*
            ( dxinv[0] * (apx(i+1,j,k)*fxp-apx(i,j,k)*fxm)
            + dxinv[1] * (apy(i,j+1,k)*fyp-apy(i,j,k)*fym)
            + dxinv[2] * (apz(i,j,k+1)*fzp-apz(i,j,k)*fzm) );
    }
#endif
}
