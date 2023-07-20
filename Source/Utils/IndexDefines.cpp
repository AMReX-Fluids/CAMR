#include "IndexDefines.H"

void
init_pass_map(PassMap* pmap)
{
  int curMapIndx = 0;
  for (int i = 0; i < NUM_ADV; ++i) {
    pmap->upassMap[curMapIndx] = i + UFA;
    pmap->qpassMap[curMapIndx] = i + QFA;
    curMapIndx++;
  }
  for (int i = 0; i < NUM_SPECIES; ++i) {
    pmap->upassMap[curMapIndx] = i + UFS;
    pmap->qpassMap[curMapIndx] = i + QFS;
    curMapIndx++;
  }
  for (int i = 0; i < NUM_AUX; ++i) {
    pmap->upassMap[curMapIndx] = i + UFX;
    pmap->qpassMap[curMapIndx] = i + QFX;
    curMapIndx++;
  }
}
