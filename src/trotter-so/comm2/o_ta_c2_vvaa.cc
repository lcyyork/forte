#include "psi4/libpsi4util/process.h"
#include "helpers/timer.h"

#include "trotter-so/trotter_so.h"

using namespace psi;

namespace forte {

void TROTTER_SO::comm2_O_Ta_C2_vvaa(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                                    BlockedTensor& C2) {
    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvaa"});
    temp["v0,v1,a0,a1"] += 1.0 * H2["v0,v1,c0,c1"] * T2["c0,a2,a0,a3"] * T2["c1,a3,a1,a4"] * L1_["a4,a2"];
    temp["v0,v1,a0,a1"] += 1.0 * H2["v0,v1,c0,c1"] * T2["c0,a2,a0,a3"] * T2["c1,a4,a1,a5"] * L2_["a3,a5,a2,a4"];
    temp["v0,v1,a0,a1"] += -1.0 * H2["v0,v1,c0,c1"] * T2["c0,a2,a0,a3"] * T2["c1,a4,a1,a5"] * L1_["a3,a4"] * L1_["a5,a2"];
    C2["v0,v1,a0,a1"] += temp["v0,v1,a0,a1"];
    C2["v0,v1,a1,a0"] -= temp["v0,v1,a0,a1"];

    C2["v0,v1,a0,a1"] += -1.0 * H2["v0,v1,c0,c1"] * T1["c0,a2"] * T2["c1,a2,a0,a1"];
    C2["v0,v1,a0,a1"] += 1.0 * H2["v0,v1,c0,c1"] * T1["c0,a2"] * T2["c1,a3,a0,a1"] * L1_["a2,a3"];
    C2["v0,v1,a0,a1"] += -1.0 * H2["v0,v1,c0,c1"] * T2["c0,a2,a0,a1"] * T2["c1,a3,a4,a5"] * L2_["a4,a5,a2,a3"];
}

} // namespace forte
