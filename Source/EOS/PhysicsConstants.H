#ifndef _PHYSICSCONSTANTS_H_
#define _PHYSICSCONSTANTS_H_

#include <AMReX_REAL.H>

struct PhysConstants
{
  static constexpr amrex::Real RU = 8.31446261815324e7;
  static constexpr amrex::Real RUC = 1.98721558317399615845;
  static constexpr amrex::Real PATM = 1.01325e+06;
  static constexpr amrex::Real AIRMW = 26.93;
  static constexpr amrex::Real FUELMW = 28.85;
  static constexpr amrex::Real PRODMW = 29.65;
  static constexpr amrex::Real Avna = 6.022140857e23;
};
#endif
