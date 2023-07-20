#ifndef _INDEX_DEFINES_H_
#define _INDEX_DEFINES_H_

#include <AMReX_REAL.H>
#include <AMReX_Arena.H>
#include <AMReX_GpuMemory.H>

#define Interior   0
#define Inflow     1
#define Outflow    2
#define Symmetry   3
#define SlipWall   4
#define NoSlipWall 5

#define NUM_SPECIES 3

#define URHO 0
#define UMX 1
#define UMY 2

#define QRHO 0
#define QU 1
#define QV 2

#define GDU 1
#define GDV 2

#if (AMREX_SPACEDIM == 2)
#define UEDEN 3
#define UEINT 4
#define UTEMP 5
#define UFA 6

#define QGAME 3
#define QPRES 4
#define QREINT 5
#define QTEMP 6
#define QFA 7

#define GDPRES 3
#define GDGAME 4
#define NGDNV 5

#define NTHERM 6
#define QTHERM 7

#elif (AMREX_SPACEDIM == 3)
#define UMZ 3
#define UEDEN 4
#define UEINT 5
#define UTEMP 6
#define UFA 7

#define QW 3
#define QGAME 4
#define QPRES 5
#define QREINT 6
#define QTEMP 7
#define QFA 8

#define GDW 3
#define GDPRES 4
#define GDGAME 5
#define NGDNV 6

#define NTHERM 7
#define QTHERM 8
#endif

#define NUM_ADV 0

#define QGAMC 0
#define QC 1
#define QCSML 2
#define QDPDR 3
#define QDPDE 4
#define NQAUX 5

#ifndef CAMR_USE_AUX
#define NUM_AUX 0
#endif

#define UFS UFA + NUM_ADV
#define UFX UFS + NUM_SPECIES
#define NVAR NTHERM + NUM_ADV + NUM_SPECIES + NUM_AUX

#define QFS QFA + NUM_ADV
#define QFX QFS + NUM_SPECIES
#define QVAR QTHERM + NUM_ADV + NUM_SPECIES + NUM_AUX
#define NQ QVAR

#define NPASSIVE NUM_SPECIES + NUM_ADV + NUM_AUX

struct PassMap
{
  amrex::GpuArray<int, NPASSIVE> upassMap = {{0}};
  amrex::GpuArray<int, NPASSIVE> qpassMap = {{0}};
};

void init_pass_map(PassMap* pmap);

#endif
