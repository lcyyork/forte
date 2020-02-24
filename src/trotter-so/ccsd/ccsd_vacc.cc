#include "psi4/libpsi4util/process.h"
#include "helpers/timer.h"

#include "trotter-so/trotter_so.h"

using namespace psi;

namespace forte {

void TROTTER_SO::ccsd_vacc(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                           BlockedTensor& T2, BlockedTensor& C2) {
    BlockedTensor temp;

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vacc"});
    temp["v0,a0,c0,c1"] += 1.0 * H2["v0,a0,c0,c1"];
    temp["v0,a0,c0,c1"] += 1.0 * H2["v0,v1,c0,c1"] * T1["a0,v1"];
    C2["v0,a0,c0,c1"] += temp["v0,a0,c0,c1"];
    C2["a0,v0,c0,c1"] -= temp["v0,a0,c0,c1"];
}

} // namespace forte
