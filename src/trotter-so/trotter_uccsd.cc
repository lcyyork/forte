#include "psi4/libpsi4util/process.h"
#include "helpers/timer.h"

#include "trotter_so.h"

using namespace psi;

namespace forte {

void TROTTER_SO::compute_trotter_uccsd(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                                       BlockedTensor& T2, double& C0, BlockedTensor& C1,
                                       BlockedTensor& C2) {
    // scale amplitudes
    T1.scale(1.0 / trotter_level_);
    T2.scale(1.0 / trotter_level_);

    // zero output
    C0 = 0.0;
    C1.zero();
    C2.zero();

    // prepare intermediates
    double X0 = 0.0;
    auto X1 = ambit::BlockedTensor::build(ambit::CoreTensor, "X1", {"gg"});
    auto X2 = ambit::BlockedTensor::build(ambit::CoreTensor, "X2", {"gggg"});
    auto Y1 = ambit::BlockedTensor::build(ambit::CoreTensor, "Y1", {"gg"});
    auto Y2 = ambit::BlockedTensor::build(ambit::CoreTensor, "Y2", {"gggg"});

    Y1["pq"] = H1["pq"];
    Y2["pqrs"] = H2["pqrs"];

    // transform Hamiltonian
    for (int i = 1; i <= trotter_level_; ++i) {
        //        build_ccsd_Hamiltonian(Y1, Y2, T1, T2, X0, X1, X2);
        transform_hamiltonian_recursive(Y1, Y2, T1, T2, X0, X1, X2);

        double Z0 = X0;

        Y1["pq"] = X1["qp"];
        Y2["pqrs"] = X2["rspq"];

        //        build_ccsd_Hamiltonian(Y1, Y2, T1, T2, X0, X1, X2);
        transform_hamiltonian_recursive(Y1, Y2, T1, T2, X0, X1, X2);

        C0 += Z0 + X0;
        if (trotter_sym_) {
            Y1["pq"] = 0.5 * X1["pq"];
            Y1["pq"] += 0.5 * X1["qp"];
            Y2["pqrs"] = 0.5 * X2["pqrs"];
            Y2["pqrs"] += 0.5 * X2["rspq"];
        } else {
            Y1["pq"] = X1["qp"];
            Y2["pqrs"] = X2["rspq"];
        }

        outfile->Printf("\n    Trotter iter. %2d corr. energy: %20.12f = %20.12f + %20.12f", i,
                        Z0 + X0, Z0, X0);
    }

    // symmetrize Hamiltonian
    C1["pq"] = 0.5 * Y1["pq"];
    C1["pq"] += 0.5 * Y1["qp"];
    C2["pqrs"] = 0.5 * Y2["pqrs"];
    C2["pqrs"] += 0.5 * Y2["rspq"];

    // unscale amplitudes
    T1.scale(trotter_level_);
    T2.scale(trotter_level_);
}

void TROTTER_SO::build_ccsd_Hamiltonian(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                                        BlockedTensor& T2, double& C0, BlockedTensor& C1,
                                        BlockedTensor& C2) {
    C0 = 0.0;
    C1["pq"] = H1["pq"];
    C2["pqrs"] = H2["pqrs"];

    C0 += 1.0 * H1["v0,c0"] * T1["c0,v0"];
    C0 += (1.0 / 4.0) * H2["v0,v1,c0,c1"] * T2["c0,c1,v0,v1"];
    C0 += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T1["c0,v0"] * T1["c1,v1"];

    C1["c0,v0"] += 1.0 * H1["v1,v0"] * T1["c0,v1"];
    C1["c0,v0"] += 1.0 * H1["v1,c1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += -1.0 * H1["c0,c1"] * T1["c1,v0"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["v1,v2,v0,c1"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c0,v0,c1"] * T1["c1,v1"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["v1,c0,c1,c2"] * T2["c1,c2,v0,v1"];
    C1["c0,v0"] += -1.0 * H1["v1,c1"] * T1["c0,v1"] * T1["c1,v0"];
    C1["c0,v0"] += 1.0 * H2["v1,v2,v0,c1"] * T1["c0,v1"] * T1["c1,v2"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["v1,v2,c1,c2"] * T1["c0,v1"] * T2["c1,c2,v0,v2"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["v1,v2,c1,c2"] * T1["c1,v0"] * T2["c0,c2,v1,v2"];
    C1["c0,v0"] += 1.0 * H2["v1,v2,c1,c2"] * T1["c1,v1"] * T2["c0,c2,v0,v2"];
    C1["c0,v0"] += 1.0 * H2["v1,c0,c1,c2"] * T1["c1,v0"] * T1["c2,v1"];
    C1["c0,v0"] += -1.0 * H2["v1,v2,c1,c2"] * T1["c0,v1"] * T1["c1,v0"] * T1["c2,v2"];

    C1["v0,v1"] += -1.0 * H1["v0,c0"] * T1["c0,v1"];
    C1["v0,v1"] += 1.0 * H2["v0,v2,v1,c0"] * T1["c0,v2"];
    C1["v0,v1"] += (-1.0 / 2.0) * H2["v0,v2,c0,c1"] * T2["c0,c1,v1,v2"];
    C1["v0,v1"] += -1.0 * H2["v0,v2,c0,c1"] * T1["c0,v1"] * T1["c1,v2"];

    C1["v0,c0"] += 1.0 * H2["v0,v1,c0,c1"] * T1["c1,v1"];

    C1["c0,c1"] += 1.0 * H1["v0,c1"] * T1["c0,v0"];
    C1["c0,c1"] += (1.0 / 2.0) * H2["v0,v1,c1,c2"] * T2["c0,c2,v0,v1"];
    C1["c0,c1"] += -1.0 * H2["v0,c0,c1,c2"] * T1["c2,v0"];
    C1["c0,c1"] += 1.0 * H2["v0,v1,c1,c2"] * T1["c0,v0"] * T1["c2,v1"];

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,c2"] * T2["c1,c2,v1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,c2"] * T1["c0,v2"] * T2["c1,c2,v1,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["v2,v3,c2,c3"] * T2["c0,c2,v0,v2"] * T2["c1,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,c0,v0,c2"] * T1["c1,v2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,c0,c2,c3"] * T1["c2,v0"] * T2["c1,c3,v1,v2"];
    temp["c0,c1,v0,v1"] +=
        (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * T1["c0,v2"] * T1["c1,v3"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,c2,c3"] * T1["c0,v2"] * T1["c2,v0"] * T2["c1,c3,v1,v3"];
    temp["c0,c1,v0,v1"] +=
        (-1.0 / 2.0) * H2["v2,c0,c2,c3"] * T1["c1,v2"] * T1["c2,v0"] * T1["c3,v1"];
    temp["c0,c1,v0,v1"] +=
        (1.0 / 4.0) * H2["v2,v3,c2,c3"] * T1["c0,v2"] * T1["c1,v3"] * T1["c2,v0"] * T1["c3,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H1["v2,v0"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v0,c2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H1["v2,c2"] * T1["c2,v0"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * T1["c2,v1"] * T2["c0,c1,v2,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,c2"] * T1["c2,v2"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,c2,c3"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c0,c1,c2,c3"] * T1["c2,v0"] * T1["c3,v1"];
    temp["c0,c1,v0,v1"] +=
        (1.0 / 4.0) * H2["v2,v3,c2,c3"] * T1["c2,v0"] * T1["c3,v1"] * T2["c0,c1,v2,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,c2,c3"] * T1["c2,v0"] * T1["c3,v2"] * T2["c0,c1,v1,v3"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 1.0 * H1["c0,c2"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,v1"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H1["v2,c2"] * T1["c0,v2"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * T1["c0,v2"] * T1["c1,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,c2,c3"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,c0,c2,c3"] * T1["c1,v2"] * T2["c2,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,c0,c2,c3"] * T1["c2,v2"] * T2["c1,c3,v0,v1"];
    temp["c0,c1,v0,v1"] +=
        (1.0 / 4.0) * H2["v2,v3,c2,c3"] * T1["c0,v2"] * T1["c1,v3"] * T2["c2,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,c2,c3"] * T1["c0,v2"] * T1["c2,v3"] * T2["c1,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c0,c1,c2,c3"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += (1.0 / 4.0) * H2["v2,v3,c2,c3"] * T2["c0,c1,v2,v3"] * T2["c2,c3,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
    temp["c0,c1,v0,c2"] += -1.0 * H2["v1,c0,v0,c2"] * T1["c1,v1"];
    temp["c0,c1,v0,c2"] += 1.0 * H2["v1,c0,c2,c3"] * T2["c1,c3,v0,v1"];
    temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H2["v1,v2,v0,c2"] * T1["c0,v1"] * T1["c1,v2"];
    temp["c0,c1,v0,c2"] += -1.0 * H2["v1,v2,c2,c3"] * T1["c0,v1"] * T2["c1,c3,v0,v2"];
    temp["c0,c1,v0,c2"] += -1.0 * H2["v1,c0,c2,c3"] * T1["c1,v1"] * T1["c3,v0"];
    temp["c0,c1,v0,c2"] +=
        (1.0 / 2.0) * H2["v1,v2,c2,c3"] * T1["c0,v1"] * T1["c1,v2"] * T1["c3,v0"];
    C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
    C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];
    C2["c1,c0,v0,c2"] -= temp["c0,c1,v0,c2"];
    C2["c1,c0,c2,v0"] += temp["c0,c1,v0,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvc"});
    temp["c0,c1,v0,c2"] += 1.0 * H1["v1,c2"] * T2["c0,c1,v0,v1"];
    temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H2["v1,v2,v0,c2"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,c2"] += 1.0 * H2["c0,c1,c2,c3"] * T1["c3,v0"];
    temp["c0,c1,v0,c2"] += (1.0 / 2.0) * H2["v1,v2,c2,c3"] * T1["c3,v0"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,c2"] += -1.0 * H2["v1,v2,c2,c3"] * T1["c3,v1"] * T2["c0,c1,v0,v2"];
    C2["c0,c1,v0,c2"] += temp["c0,c1,v0,c2"];
    C2["c0,c1,c2,v0"] -= temp["c0,c1,v0,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
    temp["v0,c0,v1,v2"] += 1.0 * H2["v0,v3,v1,c1"] * T2["c0,c1,v2,v3"];
    temp["v0,c0,v1,v2"] += -1.0 * H2["v0,c0,v1,c1"] * T1["c1,v2"];
    temp["v0,c0,v1,v2"] += -1.0 * H2["v0,v3,v1,c1"] * T1["c0,v3"] * T1["c1,v2"];
    temp["v0,c0,v1,v2"] += -1.0 * H2["v0,v3,c1,c2"] * T1["c1,v1"] * T2["c0,c2,v2,v3"];
    temp["v0,c0,v1,v2"] += (1.0 / 2.0) * H2["v0,c0,c1,c2"] * T1["c1,v1"] * T1["c2,v2"];
    temp["v0,c0,v1,v2"] +=
        (1.0 / 2.0) * H2["v0,v3,c1,c2"] * T1["c0,v3"] * T1["c1,v1"] * T1["c2,v2"];
    C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
    C2["v0,c0,v2,v1"] -= temp["v0,c0,v1,v2"];
    C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];
    C2["c0,v0,v2,v1"] += temp["v0,c0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvv"});
    temp["v0,c0,v1,v2"] += 1.0 * H1["v0,c1"] * T2["c0,c1,v1,v2"];
    temp["v0,c0,v1,v2"] += 1.0 * H2["v0,v3,v1,v2"] * T1["c0,v3"];
    temp["v0,c0,v1,v2"] += (1.0 / 2.0) * H2["v0,c0,c1,c2"] * T2["c1,c2,v1,v2"];
    temp["v0,c0,v1,v2"] += (1.0 / 2.0) * H2["v0,v3,c1,c2"] * T1["c0,v3"] * T2["c1,c2,v1,v2"];
    temp["v0,c0,v1,v2"] += -1.0 * H2["v0,v3,c1,c2"] * T1["c1,v3"] * T2["c0,c2,v1,v2"];
    C2["v0,c0,v1,v2"] += temp["v0,c0,v1,v2"];
    C2["c0,v0,v1,v2"] -= temp["v0,c0,v1,v2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcvc"});
    temp["v0,c0,v1,c1"] += 1.0 * H2["v0,v2,v1,c1"] * T1["c0,v2"];
    temp["v0,c0,v1,c1"] += -1.0 * H2["v0,v2,c1,c2"] * T2["c0,c2,v1,v2"];
    temp["v0,c0,v1,c1"] += 1.0 * H2["v0,c0,c1,c2"] * T1["c2,v1"];
    temp["v0,c0,v1,c1"] += 1.0 * H2["v0,v2,c1,c2"] * T1["c0,v2"] * T1["c2,v1"];
    C2["v0,c0,v1,c1"] += temp["v0,c0,v1,c1"];
    C2["v0,c0,c1,v1"] -= temp["v0,c0,v1,c1"];
    C2["c0,v0,v1,c1"] -= temp["v0,c0,v1,c1"];
    C2["c0,v0,c1,v1"] += temp["v0,c0,v1,c1"];

    C2["v0,v1,v2,v3"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T2["c0,c1,v2,v3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvv"});
    temp["v0,v1,v2,v3"] += -1.0 * H2["v0,v1,v2,c0"] * T1["c0,v3"];
    temp["v0,v1,v2,v3"] += (1.0 / 2.0) * H2["v0,v1,c0,c1"] * T1["c0,v2"] * T1["c1,v3"];
    C2["v0,v1,v2,v3"] += temp["v0,v1,v2,v3"];
    C2["v0,v1,v3,v2"] -= temp["v0,v1,v2,v3"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvvc"});
    temp["v0,v1,v2,c0"] += 1.0 * H2["v0,v1,c0,c1"] * T1["c1,v2"];
    C2["v0,v1,v2,c0"] += temp["v0,v1,v2,c0"];
    C2["v0,v1,c0,v2"] -= temp["v0,v1,v2,c0"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vccc"});
    temp["v0,c0,c1,c2"] += 1.0 * H2["v0,v1,c1,c2"] * T1["c0,v1"];
    C2["v0,c0,c1,c2"] += temp["v0,c0,c1,c2"];
    C2["c0,v0,c1,c2"] -= temp["v0,c0,c1,c2"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cccc"});
    temp["c0,c1,c2,c3"] += -1.0 * H2["v0,c0,c2,c3"] * T1["c1,v0"];
    temp["c0,c1,c2,c3"] += (1.0 / 2.0) * H2["v0,v1,c2,c3"] * T1["c0,v0"] * T1["c1,v1"];
    C2["c0,c1,c2,c3"] += temp["c0,c1,c2,c3"];
    C2["c1,c0,c2,c3"] -= temp["c0,c1,c2,c3"];

    C2["c0,c1,c2,c3"] += (1.0 / 2.0) * H2["v0,v1,c2,c3"] * T2["c0,c1,v0,v1"];

    if (na_ != 0) {
        C1["a0,v0"] += -1.0 * H1["a0,c0"] * T1["c0,v0"];
        C1["a0,v0"] += -1.0 * H2["v1,a0,v0,c0"] * T1["c0,v1"];
        C1["a0,v0"] += (1.0 / 2.0) * H2["v1,a0,c0,c1"] * T2["c0,c1,v0,v1"];
        C1["a0,v0"] += 1.0 * H2["v1,a0,c0,c1"] * T1["c0,v0"] * T1["c1,v1"];

        C1["a0,c0"] += -1.0 * H2["v0,a0,c0,c1"] * T1["c1,v0"];

        C1["v0,a0"] += -1.0 * H2["v0,v1,c0,a0"] * T1["c0,v1"];

        C1["c0,a0"] += 1.0 * H1["v0,a0"] * T1["c0,v0"];
        C1["c0,a0"] += (-1.0 / 2.0) * H2["v0,v1,c1,a0"] * T2["c0,c1,v0,v1"];
        C1["c0,a0"] += 1.0 * H2["v0,c0,c1,a0"] * T1["c1,v0"];
        C1["c0,a0"] += -1.0 * H2["v0,v1,c1,a0"] * T1["c0,v0"] * T1["c1,v1"];

        C1["a0,a1"] += 1.0 * H2["v0,a0,c0,a1"] * T1["c0,v0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cavv"});
        temp["c0,a0,v0,v1"] += 1.0 * H2["v2,a0,v0,c1"] * T2["c0,c1,v1,v2"];
        temp["c0,a0,v0,v1"] += -1.0 * H2["c0,a0,v0,c1"] * T1["c1,v1"];
        temp["c0,a0,v0,v1"] += -1.0 * H2["v2,a0,v0,c1"] * T1["c0,v2"] * T1["c1,v1"];
        temp["c0,a0,v0,v1"] += -1.0 * H2["v2,a0,c1,c2"] * T1["c1,v0"] * T2["c0,c2,v1,v2"];
        temp["c0,a0,v0,v1"] += (1.0 / 2.0) * H2["c0,a0,c1,c2"] * T1["c1,v0"] * T1["c2,v1"];
        temp["c0,a0,v0,v1"] +=
            (1.0 / 2.0) * H2["v2,a0,c1,c2"] * T1["c0,v2"] * T1["c1,v0"] * T1["c2,v1"];
        C2["c0,a0,v0,v1"] += temp["c0,a0,v0,v1"];
        C2["c0,a0,v1,v0"] -= temp["c0,a0,v0,v1"];
        C2["a0,c0,v0,v1"] -= temp["c0,a0,v0,v1"];
        C2["a0,c0,v1,v0"] += temp["c0,a0,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cavv"});
        temp["c0,a0,v0,v1"] += -1.0 * H1["a0,c1"] * T2["c0,c1,v0,v1"];
        temp["c0,a0,v0,v1"] += 1.0 * H2["v2,a0,v0,v1"] * T1["c0,v2"];
        temp["c0,a0,v0,v1"] += (1.0 / 2.0) * H2["c0,a0,c1,c2"] * T2["c1,c2,v0,v1"];
        temp["c0,a0,v0,v1"] += (1.0 / 2.0) * H2["v2,a0,c1,c2"] * T1["c0,v2"] * T2["c1,c2,v0,v1"];
        temp["c0,a0,v0,v1"] += -1.0 * H2["v2,a0,c1,c2"] * T1["c1,v2"] * T2["c0,c2,v0,v1"];
        C2["c0,a0,v0,v1"] += temp["c0,a0,v0,v1"];
        C2["a0,c0,v0,v1"] -= temp["c0,a0,v0,v1"];

        C2["a0,a1,v0,v1"] += (1.0 / 2.0) * H2["a0,a1,c0,c1"] * T2["c0,c1,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aavv"});
        temp["a0,a1,v0,v1"] += -1.0 * H2["a0,a1,v0,c0"] * T1["c0,v1"];
        temp["a0,a1,v0,v1"] += (1.0 / 2.0) * H2["a0,a1,c0,c1"] * T1["c0,v0"] * T1["c1,v1"];
        C2["a0,a1,v0,v1"] += temp["a0,a1,v0,v1"];
        C2["a0,a1,v1,v0"] -= temp["a0,a1,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aavc"});
        temp["a0,a1,v0,c0"] += 1.0 * H2["a0,a1,c0,c1"] * T1["c1,v0"];
        C2["a0,a1,v0,c0"] += temp["a0,a1,v0,c0"];
        C2["a0,a1,c0,v0"] -= temp["a0,a1,v0,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cavc"});
        temp["c0,a0,v0,c1"] += 1.0 * H2["v1,a0,v0,c1"] * T1["c0,v1"];
        temp["c0,a0,v0,c1"] += -1.0 * H2["v1,a0,c1,c2"] * T2["c0,c2,v0,v1"];
        temp["c0,a0,v0,c1"] += 1.0 * H2["c0,a0,c1,c2"] * T1["c2,v0"];
        temp["c0,a0,v0,c1"] += 1.0 * H2["v1,a0,c1,c2"] * T1["c0,v1"] * T1["c2,v0"];
        C2["c0,a0,v0,c1"] += temp["c0,a0,v0,c1"];
        C2["c0,a0,c1,v0"] -= temp["c0,a0,v0,c1"];
        C2["a0,c0,v0,c1"] -= temp["c0,a0,v0,c1"];
        C2["a0,c0,c1,v0"] += temp["c0,a0,v0,c1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccva"});
        temp["c0,c1,v0,a0"] += -1.0 * H2["v1,c0,v0,a0"] * T1["c1,v1"];
        temp["c0,c1,v0,a0"] += -1.0 * H2["v1,c0,c2,a0"] * T2["c1,c2,v0,v1"];
        temp["c0,c1,v0,a0"] += (1.0 / 2.0) * H2["v1,v2,v0,a0"] * T1["c0,v1"] * T1["c1,v2"];
        temp["c0,c1,v0,a0"] += 1.0 * H2["v1,v2,c2,a0"] * T1["c0,v1"] * T2["c1,c2,v0,v2"];
        temp["c0,c1,v0,a0"] += 1.0 * H2["v1,c0,c2,a0"] * T1["c1,v1"] * T1["c2,v0"];
        temp["c0,c1,v0,a0"] +=
            (-1.0 / 2.0) * H2["v1,v2,c2,a0"] * T1["c0,v1"] * T1["c1,v2"] * T1["c2,v0"];
        C2["c0,c1,v0,a0"] += temp["c0,c1,v0,a0"];
        C2["c0,c1,a0,v0"] -= temp["c0,c1,v0,a0"];
        C2["c1,c0,v0,a0"] -= temp["c0,c1,v0,a0"];
        C2["c1,c0,a0,v0"] += temp["c0,c1,v0,a0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccva"});
        temp["c0,c1,v0,a0"] += 1.0 * H1["v1,a0"] * T2["c0,c1,v0,v1"];
        temp["c0,c1,v0,a0"] += (1.0 / 2.0) * H2["v1,v2,v0,a0"] * T2["c0,c1,v1,v2"];
        temp["c0,c1,v0,a0"] += -1.0 * H2["c0,c1,c2,a0"] * T1["c2,v0"];
        temp["c0,c1,v0,a0"] += (-1.0 / 2.0) * H2["v1,v2,c2,a0"] * T1["c2,v0"] * T2["c0,c1,v1,v2"];
        temp["c0,c1,v0,a0"] += 1.0 * H2["v1,v2,c2,a0"] * T1["c2,v1"] * T2["c0,c1,v0,v2"];
        C2["c0,c1,v0,a0"] += temp["c0,c1,v0,a0"];
        C2["c0,c1,a0,v0"] -= temp["c0,c1,v0,a0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cava"});
        temp["c0,a0,v0,a1"] += 1.0 * H2["v1,a0,v0,a1"] * T1["c0,v1"];
        temp["c0,a0,v0,a1"] += 1.0 * H2["v1,a0,c1,a1"] * T2["c0,c1,v0,v1"];
        temp["c0,a0,v0,a1"] += -1.0 * H2["c0,a0,c1,a1"] * T1["c1,v0"];
        temp["c0,a0,v0,a1"] += -1.0 * H2["v1,a0,c1,a1"] * T1["c0,v1"] * T1["c1,v0"];
        C2["c0,a0,v0,a1"] += temp["c0,a0,v0,a1"];
        C2["c0,a0,a1,v0"] -= temp["c0,a0,v0,a1"];
        C2["a0,c0,v0,a1"] -= temp["c0,a0,v0,a1"];
        C2["a0,c0,a1,v0"] += temp["c0,a0,v0,a1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aava"});
        temp["a0,a1,v0,a2"] += -1.0 * H2["a0,a1,c0,a2"] * T1["c0,v0"];
        C2["a0,a1,v0,a2"] += temp["a0,a1,v0,a2"];
        C2["a0,a1,a2,v0"] -= temp["a0,a1,v0,a2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vavv"});
        temp["v0,a0,v1,v2"] += (1.0 / 2.0) * H2["v0,a0,c0,c1"] * T2["c0,c1,v1,v2"];
        C2["v0,a0,v1,v2"] += temp["v0,a0,v1,v2"];
        C2["a0,v0,v1,v2"] -= temp["v0,a0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vavv"});
        temp["v0,a0,v1,v2"] += -1.0 * H2["v0,a0,v1,c0"] * T1["c0,v2"];
        temp["v0,a0,v1,v2"] += (1.0 / 2.0) * H2["v0,a0,c0,c1"] * T1["c0,v1"] * T1["c1,v2"];
        C2["v0,a0,v1,v2"] += temp["v0,a0,v1,v2"];
        C2["v0,a0,v2,v1"] -= temp["v0,a0,v1,v2"];
        C2["a0,v0,v1,v2"] -= temp["v0,a0,v1,v2"];
        C2["a0,v0,v2,v1"] += temp["v0,a0,v1,v2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vavc"});
        temp["v0,a0,v1,c0"] += 1.0 * H2["v0,a0,c0,c1"] * T1["c1,v1"];
        C2["v0,a0,v1,c0"] += temp["v0,a0,v1,c0"];
        C2["v0,a0,c0,v1"] -= temp["v0,a0,v1,c0"];
        C2["a0,v0,v1,c0"] -= temp["v0,a0,v1,c0"];
        C2["a0,v0,c0,v1"] += temp["v0,a0,v1,c0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcva"});
        temp["v0,c0,v1,a0"] += 1.0 * H2["v0,v2,v1,a0"] * T1["c0,v2"];
        temp["v0,c0,v1,a0"] += 1.0 * H2["v0,v2,c1,a0"] * T2["c0,c1,v1,v2"];
        temp["v0,c0,v1,a0"] += -1.0 * H2["v0,c0,c1,a0"] * T1["c1,v1"];
        temp["v0,c0,v1,a0"] += -1.0 * H2["v0,v2,c1,a0"] * T1["c0,v2"] * T1["c1,v1"];
        C2["v0,c0,v1,a0"] += temp["v0,c0,v1,a0"];
        C2["v0,c0,a0,v1"] -= temp["v0,c0,v1,a0"];
        C2["c0,v0,v1,a0"] -= temp["v0,c0,v1,a0"];
        C2["c0,v0,a0,v1"] += temp["v0,c0,v1,a0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vava"});
        temp["v0,a0,v1,a1"] += -1.0 * H2["v0,a0,c0,a1"] * T1["c0,v1"];
        C2["v0,a0,v1,a1"] += temp["v0,a0,v1,a1"];
        C2["v0,a0,a1,v1"] -= temp["v0,a0,v1,a1"];
        C2["a0,v0,v1,a1"] -= temp["v0,a0,v1,a1"];
        C2["a0,v0,a1,v1"] += temp["v0,a0,v1,a1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vvva"});
        temp["v0,v1,v2,a0"] += -1.0 * H2["v0,v1,c0,a0"] * T1["c0,v2"];
        C2["v0,v1,v2,a0"] += temp["v0,v1,v2,a0"];
        C2["v0,v1,a0,v2"] -= temp["v0,v1,v2,a0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"cacc"});
        temp["c0,a0,c1,c2"] += 1.0 * H2["v0,a0,c1,c2"] * T1["c0,v0"];
        C2["c0,a0,c1,c2"] += temp["c0,a0,c1,c2"];
        C2["a0,c0,c1,c2"] -= temp["c0,a0,c1,c2"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcca"});
        temp["v0,c0,c1,a0"] += 1.0 * H2["v0,v1,c1,a0"] * T1["c0,v1"];
        C2["v0,c0,c1,a0"] += temp["v0,c0,c1,a0"];
        C2["v0,c0,a0,c1"] -= temp["v0,c0,c1,a0"];
        C2["c0,v0,c1,a0"] -= temp["v0,c0,c1,a0"];
        C2["c0,v0,a0,c1"] += temp["v0,c0,c1,a0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"caca"});
        temp["c0,a0,c1,a1"] += 1.0 * H2["v0,a0,c1,a1"] * T1["c0,v0"];
        C2["c0,a0,c1,a1"] += temp["c0,a0,c1,a1"];
        C2["c0,a0,a1,c1"] -= temp["c0,a0,c1,a1"];
        C2["a0,c0,c1,a1"] -= temp["c0,a0,c1,a1"];
        C2["a0,c0,a1,c1"] += temp["c0,a0,c1,a1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccca"});
        temp["c0,c1,c2,a0"] += -1.0 * H2["v0,c0,c2,a0"] * T1["c1,v0"];
        temp["c0,c1,c2,a0"] += (1.0 / 2.0) * H2["v0,v1,c2,a0"] * T1["c0,v0"] * T1["c1,v1"];
        C2["c0,c1,c2,a0"] += temp["c0,c1,c2,a0"];
        C2["c0,c1,a0,c2"] -= temp["c0,c1,c2,a0"];
        C2["c1,c0,c2,a0"] -= temp["c0,c1,c2,a0"];
        C2["c1,c0,a0,c2"] += temp["c0,c1,c2,a0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccca"});
        temp["c0,c1,c2,a0"] += (1.0 / 2.0) * H2["v0,v1,c2,a0"] * T2["c0,c1,v0,v1"];
        C2["c0,c1,c2,a0"] += temp["c0,c1,c2,a0"];
        C2["c0,c1,a0,c2"] -= temp["c0,c1,c2,a0"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"vcaa"});
        temp["v0,c0,a0,a1"] += 1.0 * H2["v0,v1,a0,a1"] * T1["c0,v1"];
        C2["v0,c0,a0,a1"] += temp["v0,c0,a0,a1"];
        C2["c0,v0,a0,a1"] -= temp["v0,c0,a0,a1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccaa"});
        temp["c0,c1,a0,a1"] += -1.0 * H2["v0,c0,a0,a1"] * T1["c1,v0"];
        temp["c0,c1,a0,a1"] += (1.0 / 2.0) * H2["v0,v1,a0,a1"] * T1["c0,v0"] * T1["c1,v1"];
        C2["c0,c1,a0,a1"] += temp["c0,c1,a0,a1"];
        C2["c1,c0,a0,a1"] -= temp["c0,c1,a0,a1"];

        C2["c0,c1,a0,a1"] += (1.0 / 2.0) * H2["v0,v1,a0,a1"] * T2["c0,c1,v0,v1"];

        temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"caaa"});
        temp["c0,a0,a1,a2"] += 1.0 * H2["v0,a0,a1,a2"] * T1["c0,v0"];
        C2["c0,a0,a1,a2"] += temp["c0,a0,a1,a2"];
        C2["a0,c0,a1,a2"] -= temp["c0,a0,a1,a2"];
    }
}

void TROTTER_SO::transform_hamiltonian_recursive(BlockedTensor& H1, BlockedTensor& H2,
                                                 BlockedTensor& T1, BlockedTensor& T2, double& C0,
                                                 BlockedTensor& C1, BlockedTensor& C2) {
    // copy initial one-body Hamiltonian
    C0 = 0.0;
    C1["pq"] = H1["pq"];
    C2["pqrs"] = H2["pqrs"];

    BlockedTensor O1 = ambit::BlockedTensor::build(tensor_type_, "O1", {"gg"});
    BlockedTensor O2 = ambit::BlockedTensor::build(tensor_type_, "O2", {"gggg"});
    O1["pq"] = H1["pq"];
    O2["pqrs"] = H2["pqrs"];

    // iterator variables
    int maxn = foptions_->get_int("DSRG_RSC_NCOMM");
    double ct_threshold = foptions_->get_double("DSRG_RSC_THRESHOLD");
    double X0 = 0.0;
    BlockedTensor X1 = ambit::BlockedTensor::build(tensor_type_, "X1", {"gg"});
    BlockedTensor X2 = ambit::BlockedTensor::build(tensor_type_, "X2", {"gggg"});

    double W0 = 0.0;
    BlockedTensor W1 = ambit::BlockedTensor::build(tensor_type_, "W1", {"gg"});
    BlockedTensor W2 = ambit::BlockedTensor::build(tensor_type_, "W2", {"gggg"});

    // compute Hbar recursively
    for (int n = 1; n <= maxn; ++n) {
        // prefactor before n-nested commutator
        double factor = 1.0 / n;

        H_Ta_C(factor, O1, O2, T1, T2, X0, X1, X2);
        if (n >= 1 and n <= 4) {
            ccsd_hamiltonian(n, O1, O2, T1, T2, X0, X1, X2);
        }

        // add quadratic commutator contribution
        if (foptions_->get_int("TROTTER_RSC_LEVEL") > 1) {
            X0 += W0;
            X1["pq"] += W1["pq"];
            X2["pqrs"] += W2["pqrs"];
            comm2_O_Ta_C(n, O2, T1, T2, W0, W1, W2);
        }

        // add to Hbar
        C0 += X0;
        C1["pq"] += X1["pq"];
        C2["pqrs"] += X2["pqrs"];

        // copy X to O for next level commutator
        O1["pq"] = X1["pq"];
        O2["pqrs"] = X2["pqrs"];

        // test convergence of C
        double norm_C1 = X1.norm();
        double norm_C2 = X2.norm();

        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold) {
            outfile->Printf("\n Recursion ends at n = %d", n);
            break;
        }
    }
}

} // namespace forte
