#include "psi4/libpsi4util/process.h"
#include "helpers/timer.h"

#include "trotter-so/trotter_so.h"

using namespace psi;

namespace forte {

void TROTTER_SO::ccsd_vc(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                         BlockedTensor& C1) {
    BlockedTensor temp;

    C1["v0,c0"] += 1.0 * H1["v0,c0"];
    C1["v0,c0"] += 1.0 * H2["v0,v1,c0,c1"] * T1["c1,v1"];
    C1["v0,c0"] += 1.0 * H2["v0,a0,c0,c1"] * T1["c1,a0"];
    C1["v0,c0"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T1["a0,v1"] * T1["c1,a0"];
    C1["v0,c0"] += 1.0 * H2["v0,v1,c0,a0"] * T1["a1,v1"] * L1_["a0,a1"];
    C1["v0,c0"] += (1.0 / 2.0) * H2["v0,v1,c0,a0"] * T2["a2,a3,v1,a1"] * L2_["a0,a1,a2,a3"];
    C1["v0,c0"] += -1.0 * H2["v0,a0,c0,c1"] * T1["c1,a1"] * L1_["a1,a0"];
    C1["v0,c0"] += (-1.0 / 2.0) * H2["v0,a0,c0,c1"] * T2["c1,a1,a2,a3"] * L2_["a2,a3,a0,a1"];
    C1["v0,c0"] += -1.0 * H2["v0,v1,c0,c1"] * T1["a0,v1"] * T1["c1,a1"] * L1_["a1,a0"];
    C1["v0,c0"] +=
        (-1.0 / 2.0) * H2["v0,v1,c0,c1"] * T1["a0,v1"] * T2["c1,a1,a2,a3"] * L2_["a2,a3,a0,a1"];
    C1["v0,c0"] +=
        (-1.0 / 2.0) * H2["v0,v1,c0,c1"] * T1["c1,a0"] * T2["a2,a3,v1,a1"] * L2_["a0,a1,a2,a3"];
    C1["v0,c0"] +=
        (1.0 / 4.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v1,a0"] * T2["c1,a3,a1,a2"] * L1_["a0,a3"];
    C1["v0,c0"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v1,a0"] * T2["c1,a3,a1,a4"] *
                   L2_["a0,a4,a2,a3"];
    C1["v0,c0"] += (-1.0 / 8.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v1,a0"] * T2["c1,a0,a3,a4"] *
                   L2_["a3,a4,a1,a2"];
    C1["v0,c0"] += (1.0 / 4.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v1,a0"] * T2["c1,a3,a4,a5"] *
                   L3_["a0,a4,a5,a1,a2,a3"];
    C1["v0,c0"] += (-1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v1,a0"] * T2["c1,a3,a1,a4"] *
                   L1_["a0,a3"] * L1_["a4,a2"];
    C1["v0,c0"] += (-1.0 / 4.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v1,a0"] * T2["c1,a0,a3,a4"] *
                   L1_["a3,a1"] * L1_["a4,a2"];
    C1["v0,c0"] += (1.0 / 4.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v1,a0"] * T2["c1,a3,a4,a5"] *
                   L1_["a0,a3"] * L2_["a4,a5,a1,a2"];
    C1["v0,c0"] += -1.0 * H2["v0,v1,c0,c1"] * T2["a1,a2,v1,a0"] * T2["c1,a3,a4,a5"] * L1_["a4,a1"] *
                   L2_["a0,a5,a2,a3"];
    C1["v0,c0"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["a1,a2,v1,a0"] * T2["c1,a3,a4,a5"] *
                   L1_["a0,a3"] * L1_["a4,a1"] * L1_["a5,a2"];
}

} // namespace forte
