#include "psi4/libpsi4util/process.h"
#include "helpers/timer.h"

#include "trotter-so/trotter_so.h"

using namespace psi;

namespace forte {

void TROTTER_SO::ccsd_aacc(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                           BlockedTensor& T2, BlockedTensor& C2) {
    BlockedTensor temp;

    C2["a0,a1,c0,c1"] += 1.0 * H2["a0,a1,c0,c1"];
    C2["a0,a1,c0,c1"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a0,a1,v0,v1"];
    C2["a0,a1,c0,c1"] += 1.0 * H2["v0,a2,c0,c1"] * T2["a0,a1,v0,a2"];
    C2["a0,a1,c0,c1"] += (-1.0 / 2.0) * H2["v0,v1,c0,c1"] * T1["a2,v0"] * T2["a0,a1,v1,a2"];
    C2["a0,a1,c0,c1"] += -1.0 * H2["v0,a2,c0,c1"] * T2["a0,a1,v0,a3"] * L1_["a3,a2"];
    C2["a0,a1,c0,c1"] += 1.0 * H2["v0,v1,c0,c1"] * T1["a2,v0"] * T2["a0,a1,v1,a3"] * L1_["a3,a2"];
    C2["a0,a1,c0,c1"] += (-1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a0,a1,v0,a2"] * T2["a4,a5,v1,a3"] *
                         L2_["a2,a3,a4,a5"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aacc"});
    temp["a0,a1,c0,c1"] += -1.0 * H2["v0,a0,c0,c1"] * T1["a1,v0"];
    temp["a0,a1,c0,c1"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T1["a0,v0"] * T1["a1,v1"];
    temp["a0,a1,c0,c1"] +=
        (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a0,a3,v0,a2"] * T2["a1,a4,v1,a3"] * L1_["a2,a4"];
    temp["a0,a1,c0,c1"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a0,a3,v0,a2"] * T2["a1,a5,v1,a4"] *
                           L2_["a2,a4,a3,a5"];
    temp["a0,a1,c0,c1"] += (-1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a0,a3,v0,a2"] *
                           T2["a1,a5,v1,a4"] * L1_["a2,a5"] * L1_["a4,a3"];
    C2["a0,a1,c0,c1"] += temp["a0,a1,c0,c1"];
    C2["a1,a0,c0,c1"] -= temp["a0,a1,c0,c1"];
}

} // namespace forte
