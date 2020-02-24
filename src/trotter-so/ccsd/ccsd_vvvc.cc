#include "psi4/libpsi4util/process.h"
#include "helpers/timer.h"

#include "trotter-so/trotter_so.h"

using namespace psi;

namespace forte {

void TROTTER_SO::ccsd_vvvc(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                           BlockedTensor& T2, BlockedTensor& C2) {
    BlockedTensor temp;

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvc"});
    temp["v0,v1,v2,c0"] += 1.0 * H2["v0,v1,v2,c0"];
    temp["v0,v1,v2,c0"] += 1.0 * H2["v0,v1,c0,c1"] * T1["c1,v2"];
    temp["v0,v1,v2,c0"] += 1.0 * H2["v0,v1,c0,a0"] * T1["a0,v2"];
    temp["v0,v1,v2,c0"] += (-1.0 / 2.0) * H2["v0,v1,c0,c1"] * T1["a0,v2"] * T1["c1,a0"];
    temp["v0,v1,v2,c0"] +=
        (-1.0 / 4.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v2,a0"] * T2["c1,a3,a1,a2"] * L1_["a0,a3"];
    temp["v0,v1,v2,c0"] += (-1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v2,a0"] *
                           T2["c1,a3,a1,a4"] * L2_["a0,a4,a2,a3"];
    temp["v0,v1,v2,c0"] += (-1.0 / 8.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v2,a0"] *
                           T2["c1,a0,a3,a4"] * L2_["a3,a4,a1,a2"];
    temp["v0,v1,v2,c0"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v2,a0"] * T2["c1,a3,a1,a4"] *
                           L1_["a0,a3"] * L1_["a4,a2"];
    temp["v0,v1,v2,c0"] += (-1.0 / 4.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v2,a0"] *
                           T2["c1,a0,a3,a4"] * L1_["a3,a1"] * L1_["a4,a2"];
    C2["v0,v1,v2,c0"] += temp["v0,v1,v2,c0"];
    C2["v0,v1,c0,v2"] -= temp["v0,v1,v2,c0"];
}

} // namespace forte
