#include "psi4/libpsi4util/process.h"
#include "helpers/timer.h"

#include "trotter-so/trotter_so.h"

using namespace psi;

namespace forte {

void TROTTER_SO::ccsd_vvca(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                           BlockedTensor& T2, BlockedTensor& C2) {
    BlockedTensor temp;

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvca"});
    temp["v0,v1,c0,a0"] += 1.0 * H2["v0,v1,c0,a0"];
    temp["v0,v1,c0,a0"] += -1.0 * H2["v0,v1,c0,c1"] * T1["c1,a0"];
    C2["v0,v1,c0,a0"] += temp["v0,v1,c0,a0"];
    C2["v0,v1,a0,c0"] -= temp["v0,v1,c0,a0"];
}

} // namespace forte
