#include "psi4/libpsi4util/process.h"
#include "helpers/timer.h"

#include "trotter-so/trotter_so.h"

using namespace psi;

namespace forte {

void TROTTER_SO::comm2_O_Ta_C2_cccc(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                                    BlockedTensor& C2) {
    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccc"});
    temp["c0,c1,c2,c3"] += 1.0 * H2["v0,v1,c2,c3"] * T2["c0,a1,v0,a0"] * T2["c1,a2,v1,a1"] * L1_["a0,a2"];
    temp["c0,c1,c2,c3"] += 1.0 * H2["v0,v1,c2,c3"] * T2["c0,a1,v0,a0"] * T2["c1,a3,v1,a2"] * L2_["a0,a2,a1,a3"];
    temp["c0,c1,c2,c3"] += (-1.0 / 2.0) * H2["v0,c0,c2,c3"] * T2["a1,a2,v0,a0"] * T2["c1,a3,a1,a2"] * L1_["a0,a3"];
    temp["c0,c1,c2,c3"] += -1.0 * H2["v0,c0,c2,c3"] * T2["a1,a2,v0,a0"] * T2["c1,a3,a1,a4"] * L2_["a0,a4,a2,a3"];
    temp["c0,c1,c2,c3"] += (-1.0 / 4.0) * H2["v0,c0,c2,c3"] * T2["a1,a2,v0,a0"] * T2["c1,a0,a3,a4"] * L2_["a3,a4,a1,a2"];
    temp["c0,c1,c2,c3"] += 1.0 * H2["v0,a0,c2,c3"] * T2["c0,a2,v0,a1"] * T2["c1,a3,a0,a2"] * L1_["a1,a3"];
    temp["c0,c1,c2,c3"] += 1.0 * H2["v0,a0,c2,c3"] * T2["c0,a2,v0,a1"] * T2["c1,a1,a0,a3"] * L1_["a3,a2"];
    temp["c0,c1,c2,c3"] += 2.0 * H2["v0,a0,c2,c3"] * T2["c0,a2,v0,a1"] * T2["c1,a3,a0,a4"] * L2_["a1,a4,a2,a3"];
    temp["c0,c1,c2,c3"] += -1.0 * H2["v0,a0,c2,c3"] * T2["c0,a2,v0,a1"] * T2["c1,a3,a2,a4"] * L2_["a1,a4,a0,a3"];
    temp["c0,c1,c2,c3"] += (1.0 / 2.0) * H2["v0,a0,c2,c3"] * T2["c0,a2,v0,a1"] * T2["c1,a1,a3,a4"] * L2_["a3,a4,a0,a2"];
    temp["c0,c1,c2,c3"] += 1.0 * H2["a0,a1,c2,c3"] * T2["c0,a2,a0,a3"] * T2["c1,a3,a1,a4"] * L1_["a4,a2"];
    temp["c0,c1,c2,c3"] += 1.0 * H2["a0,a1,c2,c3"] * T2["c0,a2,a0,a3"] * T2["c1,a4,a1,a5"] * L2_["a3,a5,a2,a4"];
    temp["c0,c1,c2,c3"] += -1.0 * H2["a0,a1,c2,c3"] * T2["c0,a2,a0,a3"] * T2["c1,a4,a2,a5"] * L2_["a3,a5,a1,a4"];
    temp["c0,c1,c2,c3"] += (1.0 / 2.0) * H2["a0,a1,c2,c3"] * T2["c0,a2,a0,a3"] * T2["c1,a3,a4,a5"] * L2_["a4,a5,a1,a2"];
    temp["c0,c1,c2,c3"] += -1.0 * H2["v0,v1,c2,c3"] * T2["c0,a1,v0,a0"] * T2["c1,a3,v1,a2"] * L1_["a0,a3"] * L1_["a2,a1"];
    temp["c0,c1,c2,c3"] += 1.0 * H2["v0,c0,c2,c3"] * T2["a1,a2,v0,a0"] * T2["c1,a3,a1,a4"] * L1_["a0,a3"] * L1_["a4,a2"];
    temp["c0,c1,c2,c3"] += (-1.0 / 2.0) * H2["v0,c0,c2,c3"] * T2["a1,a2,v0,a0"] * T2["c1,a0,a3,a4"] * L1_["a3,a1"] * L1_["a4,a2"];
    temp["c0,c1,c2,c3"] += -2.0 * H2["v0,a0,c2,c3"] * T2["c0,a2,v0,a1"] * T2["c1,a3,a0,a4"] * L1_["a1,a3"] * L1_["a4,a2"];
    temp["c0,c1,c2,c3"] += 1.0 * H2["v0,a0,c2,c3"] * T2["c0,a2,v0,a1"] * T2["c1,a3,a2,a4"] * L1_["a1,a3"] * L1_["a4,a0"];
    temp["c0,c1,c2,c3"] += 1.0 * H2["v0,a0,c2,c3"] * T2["c0,a2,v0,a1"] * T2["c1,a1,a3,a4"] * L1_["a3,a0"] * L1_["a4,a2"];
    temp["c0,c1,c2,c3"] += -1.0 * H2["a0,a1,c2,c3"] * T2["c0,a2,a0,a3"] * T2["c1,a4,a1,a5"] * L1_["a3,a4"] * L1_["a5,a2"];
    temp["c0,c1,c2,c3"] += 1.0 * H2["a0,a1,c2,c3"] * T2["c0,a2,a0,a3"] * T2["c1,a4,a2,a5"] * L1_["a3,a4"] * L1_["a5,a1"];
    temp["c0,c1,c2,c3"] += 1.0 * H2["a0,a1,c2,c3"] * T2["c0,a2,a0,a3"] * T2["c1,a3,a4,a5"] * L1_["a4,a1"] * L1_["a5,a2"];
    C2["c0,c1,c2,c3"] += temp["c0,c1,c2,c3"];
    C2["c1,c0,c2,c3"] -= temp["c0,c1,c2,c3"];

    C2["c0,c1,c2,c3"] += 1.0 * H2["v0,v1,c2,c3"] * T1["a0,v0"] * T2["c0,c1,v1,a1"] * L1_["a1,a0"];
    C2["c0,c1,c2,c3"] += -1.0 * H2["v0,v1,c2,c3"] * T2["c0,c1,v0,a0"] * T2["a2,a3,v1,a1"] * L2_["a0,a1,a2,a3"];
    C2["c0,c1,c2,c3"] += 1.0 * H2["v0,a0,c2,c3"] * T1["a1,v0"] * T2["c0,c1,a0,a2"] * L1_["a2,a1"];
    C2["c0,c1,c2,c3"] += -1.0 * H2["v0,a0,c2,c3"] * T2["c0,c1,a0,a4"] * T2["a2,a3,v0,a1"] * L2_["a1,a4,a2,a3"];
    C2["c0,c1,c2,c3"] += 1.0 * H2["v0,a0,c2,c3"] * T2["a2,a3,v0,a1"] * T2["c0,c1,a2,a4"] * L2_["a1,a4,a0,a3"];
}

} // namespace forte
