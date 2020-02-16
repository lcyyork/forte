#include "psi4/libpsi4util/process.h"
#include "helpers/timer.h"

#include "trotter-so/trotter_so.h"

using namespace psi;

namespace forte {

void TROTTER_SO::comm2_O_Ta_C2_vaaa(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                                    BlockedTensor& C2) {
    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vaaa"});
    temp["v0,a0,a1,a2"] += (-1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a0,a3,v1,a1"] * T2["c0,c1,a2,a3"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,v1,c0,a1"] * T1["c0,a3"] * T2["a0,a3,v1,a2"];
    temp["v0,a0,a1,a2"] += (-1.0 / 2.0) * H2["v0,v1,c0,a1"] * T2["a3,a4,v1,a2"] * T2["c0,a0,a3,a4"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,v1,c0,c1"] * T2["c0,a3,v1,a1"] * T2["c1,a0,a2,a4"] * L1_["a4,a3"];
    temp["v0,a0,a1,a2"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a0,a3,v1,a1"] * T2["c0,c1,a2,a4"] * L1_["a4,a3"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,v1,c0,a1"] * T1["a3,v1"] * T2["c0,a0,a2,a4"] * L1_["a4,a3"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,v1,c0,a1"] * T1["c0,a3"] * T2["a0,a4,v1,a2"] * L1_["a3,a4"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,v1,c0,a1"] * T2["a0,a3,v1,a2"] * T2["c0,a4,a5,a6"] * L2_["a5,a6,a3,a4"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,v1,c0,a1"] * T2["a0,a4,v1,a3"] * T2["c0,a5,a2,a4"] * L1_["a3,a5"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,v1,c0,a1"] * T2["a0,a4,v1,a3"] * T2["c0,a3,a2,a5"] * L1_["a5,a4"];
    temp["v0,a0,a1,a2"] += 2.0 * H2["v0,v1,c0,a1"] * T2["a0,a4,v1,a3"] * T2["c0,a5,a2,a6"] * L2_["a3,a6,a4,a5"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,v1,c0,a1"] * T2["c0,a0,a2,a6"] * T2["a4,a5,v1,a3"] * L2_["a3,a6,a4,a5"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,v1,c0,a1"] * T2["a3,a4,v1,a2"] * T2["c0,a0,a3,a5"] * L1_["a5,a4"];
    temp["v0,a0,a1,a2"] += (-1.0 / 2.0) * H2["v0,v1,c0,a1"] * T2["a3,a4,v1,a2"] * T2["c0,a0,a5,a6"] * L2_["a5,a6,a3,a4"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,v1,c0,a3"] * T2["a0,a4,v1,a1"] * T2["c0,a5,a2,a4"] * L1_["a3,a5"];
    temp["v0,a0,a1,a2"] += -2.0 * H2["v0,v1,c0,a3"] * T2["a0,a4,v1,a1"] * T2["c0,a5,a2,a6"] * L2_["a3,a6,a4,a5"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,v1,c0,a3"] * T2["c0,a0,a2,a6"] * T2["a4,a5,v1,a1"] * L2_["a3,a6,a4,a5"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,a0,c0,c1"] * T2["c0,a3,a1,a4"] * T2["c1,a4,a2,a5"] * L1_["a5,a3"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,a0,c0,c1"] * T2["c0,a3,a1,a4"] * T2["c1,a5,a2,a6"] * L2_["a4,a6,a3,a5"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,a3,c0,c1"] * T2["c0,a0,a1,a4"] * T2["c1,a5,a2,a3"] * L1_["a4,a5"];
    temp["v0,a0,a1,a2"] += -2.0 * H2["v0,a3,c0,c1"] * T2["c0,a0,a1,a4"] * T2["c1,a5,a2,a6"] * L2_["a4,a6,a3,a5"];
    temp["v0,a0,a1,a2"] += -2.0 * H2["v0,v1,c0,a1"] * T2["a0,a4,v1,a3"] * T2["c0,a5,a2,a6"] * L1_["a3,a5"] * L1_["a6,a4"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,v1,c0,a1"] * T2["a3,a4,v1,a2"] * T2["c0,a0,a5,a6"] * L1_["a5,a3"] * L1_["a6,a4"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,v1,c0,a3"] * T2["a0,a4,v1,a1"] * T2["c0,a5,a2,a6"] * L1_["a3,a5"] * L1_["a6,a4"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,v1,c0,a3"] * T2["c0,a0,a2,a6"] * T2["a4,a5,v1,a1"] * L1_["a3,a4"] * L1_["a6,a5"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,a0,c0,c1"] * T2["c0,a3,a1,a4"] * T2["c1,a5,a2,a6"] * L1_["a4,a5"] * L1_["a6,a3"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,a3,c0,c1"] * T2["c0,a0,a1,a4"] * T2["c1,a5,a2,a6"] * L1_["a4,a5"] * L1_["a6,a3"];
    C2["v0,a0,a1,a2"] += temp["v0,a0,a1,a2"];
    C2["v0,a0,a2,a1"] -= temp["v0,a0,a1,a2"];
    C2["a0,v0,a1,a2"] -= temp["v0,a0,a1,a2"];
    C2["a0,v0,a2,a1"] += temp["v0,a0,a1,a2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vaaa"});
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,v1,c0,c1"] * T1["c0,v1"] * T2["c1,a0,a1,a2"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,v1,c0,c1"] * T2["c0,a0,v1,a3"] * T2["c1,a3,a1,a2"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,a0,c0,c1"] * T1["c0,a3"] * T2["c1,a3,a1,a2"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,a3,c0,c1"] * T1["c0,a3"] * T2["c1,a0,a1,a2"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,a3,c0,c1"] * T2["c1,a4,a1,a2"] * T2["c0,a0,a3,a4"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,v1,c0,c1"] * T2["c0,a0,v1,a3"] * T2["c1,a4,a1,a2"] * L1_["a3,a4"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,v1,c0,a3"] * T1["a4,v1"] * T2["c0,a0,a1,a2"] * L1_["a3,a4"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,v1,c0,a3"] * T2["a0,a5,v1,a4"] * T2["c0,a4,a1,a2"] * L1_["a3,a5"];
    temp["v0,a0,a1,a2"] += -2.0 * H2["v0,v1,c0,a3"] * T2["a0,a5,v1,a4"] * T2["c0,a6,a1,a2"] * L2_["a3,a4,a5,a6"];
    temp["v0,a0,a1,a2"] += (-1.0 / 2.0) * H2["v0,v1,c0,a3"] * T2["c0,a0,a1,a2"] * T2["a5,a6,v1,a4"] * L2_["a3,a4,a5,a6"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,a0,c0,c1"] * T1["c0,a3"] * T2["c1,a4,a1,a2"] * L1_["a3,a4"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,a0,c0,c1"] * T2["c0,a3,a1,a2"] * T2["c1,a4,a5,a6"] * L2_["a5,a6,a3,a4"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,a3,c0,c1"] * T1["c0,a4"] * T2["c1,a0,a1,a2"] * L1_["a4,a3"];
    temp["v0,a0,a1,a2"] += (1.0 / 2.0) * H2["v0,a3,c0,c1"] * T2["c0,a0,a1,a2"] * T2["c1,a4,a5,a6"] * L2_["a5,a6,a3,a4"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,a3,c0,c1"] * T2["c1,a4,a1,a2"] * T2["c0,a0,a4,a5"] * L1_["a5,a3"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,a3,c0,c1"] * T2["c1,a5,a1,a2"] * T2["c0,a0,a3,a4"] * L1_["a4,a5"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,a3,c0,c1"] * T2["c1,a6,a1,a2"] * T2["c0,a0,a4,a5"] * L2_["a4,a5,a3,a6"];
    temp["v0,a0,a1,a2"] += -1.0 * H2["v0,v1,c0,a3"] * T2["a0,a5,v1,a4"] * T2["c0,a6,a1,a2"] * L1_["a3,a5"] * L1_["a4,a6"];
    temp["v0,a0,a1,a2"] += 1.0 * H2["v0,a3,c0,c1"] * T2["c1,a6,a1,a2"] * T2["c0,a0,a4,a5"] * L1_["a4,a3"] * L1_["a5,a6"];
    C2["v0,a0,a1,a2"] += temp["v0,a0,a1,a2"];
    C2["a0,v0,a1,a2"] -= temp["v0,a0,a1,a2"];
}

} // namespace forte
