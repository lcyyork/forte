#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsio/psio.hpp"

#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "omrpt2_so.h"

using namespace psi;

namespace forte {

double OMRPT2_SO::compute_energy() {
    double Eref = compute_reference_energy();

    compute_amplitudes();
    renormalize_ints();

    double Ecorr = compute_correlation_energy();
    double Etotal = Eref + Ecorr;

    print_h2("OO-DSRG-MRPT2 Energy Summary");
    outfile->Printf("\n    OO-DSRG-MRPT2 reference energy:   %20.12f", Eref);
    outfile->Printf("\n    OO-DSRG-MRPT2 correlation energy: %20.12f", Ecorr);
    outfile->Printf("\n    OO-DSRG-MRPT2 total energy:       %20.12f", Etotal);

    return Etotal;
}

double OMRPT2_SO::compute_reference_energy() {
    double Eref = Enuc_ + ints_->frozen_core_energy();

    Eref += H_["mn"] * I_["mn"];
    Eref += 0.5 * V_["c0,c1,c2,c3"] * I_["c0,c2"] * I_["c1,c3"];

    Eref += H_["vu"] * D1_["uv"];
    Eref += V_["vnum"] * D1_["uv"] * I_["mn"];

    Eref += 0.25 * V_["xyuv"] * D2_["uvxy"];

    return Eref;
}

void OMRPT2_SO::compute_amplitudes() {
    outfile->Printf("\n    Computing T2 amplitudes ...");

    T2_ = BTF_->build(tensor_type_, "T2 Amplitudes",
                      {"ccvv", "cavv", "acvv", "aavv", "ccav", "ccva", "ccaa"});
    T2_["ijab"] = V_["ijab"];

    T2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double d = Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]];
        value *= dsrg_source_->compute_renormalized_denominator(d);
    });

    outfile->Printf(" Done.");
}

void OMRPT2_SO::renormalize_ints() {
    outfile->Printf("\n    Computing effective 1st-order integrals ...");

    M2_ = BTF_->build(tensor_type_, "M2 Integrals",
                      {"vvcc", "vvca", "vvac", "vvaa", "avcc", "vacc", "aacc"});
    M2_["abij"] = V_["abij"];

    M2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double d = Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]];
        value *= 1.0 + dsrg_source_->compute_renormalized(d);
    });

    outfile->Printf(" Done.");
}

double OMRPT2_SO::compute_correlation_energy() {
    double Eout = 0.0;

    Eout += 0.25 * M2_["efmn"] * T2_["mnef"];

    Eout += 0.25 * M2_["xymn"] * T2_["mnxy"];
    Eout += 0.125 * M2_["uvmn"] * T2_["mnxy"] * D2_["xyuv"];
    Eout -= 0.5 * M2_["uxmn"] * T2_["mnvx"] * D1_["vu"];

    Eout += 0.5 * M2_["eumn"] * T2_["mneu"];
    Eout -= 0.5 * M2_["eumn"] * T2_["mnev"] * D1_["vu"];

    Eout += 0.5 * M2_["efmv"] * T2_["muef"] * D1_["vu"];

    Eout += 0.125 * M2_["efxy"] * T2_["uvef"] * D2_["xyuv"];

    return Eout;
}

} // namespace forte
