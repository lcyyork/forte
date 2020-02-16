#include "psi4/libpsi4util/process.h"
#include "helpers/timer.h"

#include "trotter-so/trotter_so.h"

using namespace psi;

namespace forte {

void TROTTER_SO::comm2_O_Ta_C2_vaca(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                                    BlockedTensor& C2) {
    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vaca"});
    temp["v0,a0,c0,a1"] += -1.0 * H2["v0,v1,c0,c1"] * T1["c1,a2"] * T2["a0,a2,v1,a1"];
    temp["v0,a0,c0,a1"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a2,a3,v1,a1"] * T2["c1,a0,a2,a3"];
    temp["v0,a0,c0,a1"] += 1.0 * H2["v0,v1,c0,c1"] * T1["a2,v1"] * T2["c1,a0,a1,a3"] * L1_["a3,a2"];
    temp["v0,a0,c0,a1"] += 1.0 * H2["v0,v1,c0,c1"] * T1["c1,a2"] * T2["a0,a3,v1,a1"] * L1_["a2,a3"];
    temp["v0,a0,c0,a1"] += 1.0 * H2["v0,v1,c0,c1"] * T2["a0,a2,v1,a1"] * T2["c1,a3,a4,a5"] * L2_["a4,a5,a2,a3"];
    temp["v0,a0,c0,a1"] += -1.0 * H2["v0,v1,c0,c1"] * T2["a0,a3,v1,a2"] * T2["c1,a4,a1,a3"] * L1_["a2,a4"];
    temp["v0,a0,c0,a1"] += -1.0 * H2["v0,v1,c0,c1"] * T2["a0,a3,v1,a2"] * T2["c1,a2,a1,a4"] * L1_["a4,a3"];
    temp["v0,a0,c0,a1"] += -2.0 * H2["v0,v1,c0,c1"] * T2["a0,a3,v1,a2"] * T2["c1,a4,a1,a5"] * L2_["a2,a5,a3,a4"];
    temp["v0,a0,c0,a1"] += -1.0 * H2["v0,v1,c0,c1"] * T2["c1,a0,a1,a5"] * T2["a3,a4,v1,a2"] * L2_["a2,a5,a3,a4"];
    temp["v0,a0,c0,a1"] += -1.0 * H2["v0,v1,c0,c1"] * T2["a2,a3,v1,a1"] * T2["c1,a0,a2,a4"] * L1_["a4,a3"];
    temp["v0,a0,c0,a1"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a2,a3,v1,a1"] * T2["c1,a0,a4,a5"] * L2_["a4,a5,a2,a3"];
    temp["v0,a0,c0,a1"] += 2.0 * H2["v0,v1,c0,c1"] * T2["a0,a3,v1,a2"] * T2["c1,a4,a1,a5"] * L1_["a2,a4"] * L1_["a5,a3"];
    temp["v0,a0,c0,a1"] += 1.0 * H2["v0,v1,c0,c1"] * T2["a2,a3,v1,a1"] * T2["c1,a0,a4,a5"] * L1_["a4,a2"] * L1_["a5,a3"];
    C2["v0,a0,c0,a1"] += temp["v0,a0,c0,a1"];
    C2["v0,a0,a1,c0"] -= temp["v0,a0,c0,a1"];
    C2["a0,v0,c0,a1"] -= temp["v0,a0,c0,a1"];
    C2["a0,v0,a1,c0"] += temp["v0,a0,c0,a1"];
}

} // namespace forte
