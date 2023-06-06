/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <unistd.h>
#include <algorithm>

#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsio/psio.hpp"

#include "helpers/printing.h"
#include "helpers/timer.h"
#include "sa_mrdsrg.h"

using namespace psi;

namespace forte {

SA_MRDSRG::SA_MRDSRG(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : SADSRG(rdms, scf_info, options, ints, mo_space_info) {
    read_options();
    print_options();
    check_memory();
    startup();
}

void SA_MRDSRG::read_options() {
    corrlv_string_ = foptions_->get_str("CORR_LEVEL");
    bool level_found = false;
    for (const auto& pair : corrlevelmap) {
        if (corrlv_string_ == pair.first) {
            level_found = true;
            break;
        }
    }
    if (not level_found) {
        outfile->Printf("\n  Warning: CORR_LEVEL option %s is not implemented.",
                        corrlv_string_.c_str());
        outfile->Printf("\n  Changed CORR_LEVEL option to LDSRG2_QC");
        corrlv_string_ = "LDSRG2_QC";
        warnings_.push_back(std::make_tuple("Unsupported CORR_LEVEL", "Change to LDSRG2_QC",
                                            "Change options in input.dat"));
    }
    if (corrlv_string_ == "CC2" and (!eri_df_)) {
        throw std::runtime_error("Second-order MRDSRG only available with DF/CD integrals!");
    }

    Hzero_ = foptions_->get_str("DSRG_PT2_H0TH");
    if (Hzero_ != "FDIAG" and Hzero_ != "DYALL" and Hzero_ != "FINK") {
        outfile->Printf("\n  Warning: DSRG_PT2_H0TH option %s is not implemented.", Hzero_.c_str());
        outfile->Printf("\n  Changed DSRG_PT2_H0TH option to FDIAG");
        Hzero_ = "FDIAG";
        warnings_.push_back(std::make_tuple("Unsupported DSRG_PT2_H0TH", "Change to FDIAG",
                                            "Change options in input.dat"));
    }
    Hzero_a1_ = foptions_->get_bool("DSRG_DRESSED_H0TH");

    sequential_Hbar_ = foptions_->get_bool("DSRG_HBAR_SEQ");
    nivo_ = foptions_->get_bool("DSRG_NIVO");

    rsc_ncomm_ = foptions_->get_int("DSRG_RSC_NCOMM");
    rsc_conv_ = foptions_->get_double("DSRG_RSC_THRESHOLD");

    // maxiter_ = foptions_->get_int("DSRG_MAXITER");
    e_conv_ = foptions_->get_double("E_CONVERGENCE");
    r_conv_ = foptions_->get_double("R_CONVERGENCE");

    restart_amps_ = foptions_->get_bool("DSRG_RESTART_AMPS");
    t1_guess_ = foptions_->get_str("DSRG_T1_AMPS_GUESS");
}

void SA_MRDSRG::startup() {
    // prepare integrals
    build_ints();

    // link F_ with Fock_ of SADSRG
    F_ = Fock_;

    // test semi-canonical
    if (!semi_canonical_) {
        outfile->Printf("\n  Orbital invariant formalism will be employed for MR-DSRG.");
        U_ = ambit::BlockedTensor::build(tensor_type_, "U", {"cc", "aa", "vv"});
        Fdiag_ = diagonalize_Fock_diagblocks(U_);
    }

    // determine file names
    chk_filename_prefix_ = psi::PSIOManager::shared_object()->get_default_path() + "forte." +
                           std::to_string(getpid()) + "." +
                           psi::Process::environment.molecule()->name();
    t1_file_chk_.clear();
    t2_file_chk_.clear();
    if (restart_amps_ and (relax_ref_ != "NONE" or brueckner_)) {
        t1_file_chk_ = chk_filename_prefix_ + ".mrdsrg.adapted.t1.bin";
        t2_file_chk_ = chk_filename_prefix_ + ".mrdsrg.adapted.t2.bin";
    }

    t1_file_cwd_ = "forte.mrdsrg.adapted.t1.bin";
    t2_file_cwd_ = "forte.mrdsrg.adapted.t2.bin";
}

void SA_MRDSRG::print_options() {
    // fill in information
    std::vector<std::pair<std::string, int>> calculation_info_int{
        {"Max number of iterations", maxiter_},
        {"Max nested commutators", rsc_ncomm_},
        {"DIIS start", diis_start_},
        {"Min DIIS vectors", diis_min_vec_},
        {"Max DIIS vectors", diis_max_vec_},
        {"DIIS extrapolating freq", diis_freq_},
        {"Number of amplitudes for printing", ntamp_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Flow parameter", s_},
        {"Energy convergence threshold", e_conv_},
        {"Residual convergence threshold", r_conv_},
        {"Recursive single commutator threshold", rsc_conv_},
        {"Taylor expansion threshold", pow(10.0, -double(taylor_threshold_))},
        {"Intruder amplitudes threshold", intruder_tamp_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Correlation level", corrlv_string_},
        {"Integral type", ints_type_},
        {"Source operator", source_},
        {"Reference relaxation", relax_ref_},
        {"3RDM algorithm", L3_algorithm_},
        {"Core-Virtual source type", ccvv_source_},
        {"T1 amplitudes initial guess", t1_guess_}};

    if (internal_amp_ != "NONE") {
        calculation_info_string.emplace_back("Internal amplitudes levels", internal_amp_);
        calculation_info_string.emplace_back("Internal amplitudes selection", internal_amp_select_);
    }

    std::vector<std::pair<std::string, bool>> calculation_info_bool{
        {"Restart amplitudes", restart_amps_},
        {"Sequential DSRG transformation", sequential_Hbar_},
        {"Omit blocks of >= 3 virtual indices", nivo_},
        {"Read amplitudes from current dir", read_amps_cwd_},
        {"Write amplitudes to current dir", dump_amps_cwd_}};

    if (brueckner_) {
        calculation_info_bool.emplace_back("DSRG Brueckner orbitals", brueckner_);
        calculation_info_double.emplace_back("Brueckner convergence", brueckner_conv_);
    }

    if (corrlv_string_ == "CC2") {
        calculation_info_string.emplace_back("Zeroth-order Hamiltonian", Hzero_);
        calculation_info_bool.emplace_back("A1 dressed H0th", Hzero_a1_);
    }

    // print information
    print_selected_options("Computation Information", calculation_info_string,
                           calculation_info_bool, calculation_info_double, calculation_info_int);
}

void SA_MRDSRG::check_memory() {
    if (eri_df_) {
        dsrg_mem_.add_entry("1-electron and 3-index integrals", {"gg", "Lgg"});
    } else {
        dsrg_mem_.add_entry("1- and 2-electron integrals", {"gg", "gggg"});
    }

    dsrg_mem_.add_entry("T1 cluster amplitudes and residuals", {"hp"}, 2);
    dsrg_mem_.add_entry("T2 cluster amplitudes and residuals", {"hhpp"}, 3); // T2, S2, DT2

    if (corrlv_string_ == "CC2") {
        dsrg_mem_.add_entry("1- and 2-body Hbar", {"hhpp", "hp"});
        dsrg_mem_.add_entry("1-body intermediates", {"gg"}, 2);
        dsrg_mem_.add_entry("2-body intermediates", od_two_labels(), 1);
        if (Hzero_ == "DYALL")
            dsrg_mem_.add_entry("Zeroth-order Hamiltonian", {"cc", "aa", "vv", "aaaa"});
        else if (Hzero_ == "FINK")
            dsrg_mem_.add_entry("Zeroth-order Hamiltonian",
                                {"cc", "aa", "vv", "aaaa", "cccc", "caca", "caac", "acca", "acac",
                                 "vcvc", "vccv", "cvvc", "cvcv", "avav", "avva", "vaav", "vava"});
        else
            dsrg_mem_.add_entry("Zeroth-order Hamiltonian", {"cc", "aa", "vv"});
    } else if (corrlv_string_ == "LDSRG2_QC") {
        dsrg_mem_.add_entry("1- and 2-body Hbar", {"hhpp", "hp"});
        dsrg_mem_.add_entry("1- and 2-body intermediates", {"gg", "gggg", "hhpp"});
    } else {
        dsrg_mem_.add_entry("1-body Hbar and intermediates", {"gg"}, 3);
        if (nivo_) {
            dsrg_mem_.add_entry("2-body Hbar and intermediates", nivo_labels(), 3);
        } else {
            dsrg_mem_.add_entry("2-body Hbar and intermediates", {"gggg"}, 3);
        }

        if (sequential_Hbar_) {
            size_t mem_seq =
                eri_df_ ? dsrg_mem_.compute_memory({"Lgg"}) : dsrg_mem_.compute_memory({"gggg"});
            dsrg_mem_.add_entry("Local intermediates for sequential Hbar", mem_seq, false);
        }
    }

    // intermediates used in actual commutator computation
    size_t mem_comm = dsrg_mem_.compute_memory({"hhpp", "ahpp", "hhhp"}) * 2;
    if ((!eri_df_) and (!nivo_)) {
        mem_comm = std::max(mem_comm, dsrg_mem_.compute_memory({"ppph"}));
    }
    dsrg_mem_.add_entry("Local intermediates for commutators", mem_comm, false);

    dsrg_mem_.print("MR-DSRG (" + corrlv_string_ + ")");
}

void SA_MRDSRG::build_ints() {
    // prepare one-electron integrals
    H_ = BTF_->build(tensor_type_, "H", {"gg"});
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->oei_a(i[0], i[1]);
    });

    // prepare two-electron integrals or three-index B
    if (eri_df_) {
        B_ = BTF_->build(tensor_type_, "B 3-idx", {"Lgg"});
        fill_three_index_ints(B_);
    } else {
        V_ = BTF_->build(tensor_type_, "V", {"gggg"});

        for (const std::string& block : V_.block_labels()) {
            auto mo_to_index = BTF_->get_mo_to_index();

            std::vector<size_t> i0 = mo_to_index[block.substr(0, 1)];
            std::vector<size_t> i1 = mo_to_index[block.substr(1, 1)];
            std::vector<size_t> i2 = mo_to_index[block.substr(2, 1)];
            std::vector<size_t> i3 = mo_to_index[block.substr(3, 1)];

            auto Vblock = ints_->aptei_ab_block(i0, i1, i2, i3);
            V_.block(block).copy(Vblock);
        }
    }
}

double SA_MRDSRG::compute_energy() {
    // build initial amplitudes
    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", {"hp"});
    T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
    guess_t(V_, T2_, F_, T1_, B_);

    // get reference energy
    double Etotal = Eref_;

    // compute energy
    if (corrlevelmap[corrlv_string_] == CORR_LV::CC2) {
        Etotal += compute_energy_cc2();
    } else {
        Etotal += compute_energy_ldsrg2();
    }
    //    switch (corrlevelmap[corrlv_string_]) {
    //    case CORR_LV::LDSRG2: {
    //        Etotal += compute_energy_ldsrg2();
    //        break;
    //    }
    //    default: { Etotal += compute_energy_ldsrg2_qc(); }
    //    }

    // if (brueckner_) {
    //     brueckner_orbital_rotation(T1_);
    // }

    return Etotal;
}

double SA_MRDSRG::Hbar_od_norm(const int& n, const std::vector<std::string>& blocks) {
    double norm = 0.0;

    auto T = (n == 1) ? Hbar1_ : Hbar2_;

    for (auto& block : blocks) {
        double norm_block = T.block(block).norm();
        norm += 2.0 * norm_block * norm_block;
    }
    norm = std::sqrt(norm);

    return norm;
}

void SA_MRDSRG::transform_one_body(const std::vector<ambit::BlockedTensor>& oetens,
                                   const std::vector<int>& max_levels) {
    print_h2("Transform One-Electron Operators");

    if (corrlv_string_ == "LDSRG2_QC")
        throw std::runtime_error(
            "Not available for LDSRG2_QC: Try LDSRG2 with DSRG_RSC_NCOMM = 2.");

    int n_tensors = oetens.size();
    Mbar0_ = std::vector<double>(n_tensors, 0.0);
    Mbar1_.resize(n_tensors);
    Mbar2_.resize(n_tensors);
    for (int i = 0; i < n_tensors; ++i) {
        Mbar1_[i] = BTF_->build(tensor_type_, oetens[i].name() + "1", {"aa"});
        if (max_levels[i] > 1)
            Mbar2_[i] = BTF_->build(tensor_type_, oetens[i].name() + "2", {"aaaa"});
    }

    auto max_body = *std::max_element(max_levels.begin(), max_levels.end());
    if (max_body > 1) {
        DT2_["ijab"] = 2.0 * T2_["ijab"] - T2_["ijba"];
    }

    for (int i = 0; i < n_tensors; ++i) {
        local_timer t_local;
        const auto& M = oetens[i];
        print_contents("Transforming " + M.name());
        compute_mbar_ldsrg2(M, max_levels[i], i);
        print_done(t_local.get());
    }
}

ambit::BlockedTensor SA_MRDSRG::rotate_integrals(double& H0, ambit::BlockedTensor& H1,
                                                 ambit::BlockedTensor& H2) {
    auto A1_m = expA1(T1_, false);

    ambit::BlockedTensor U1;
    U1 = BTF_->build(tensor_type_, "Transformer", {"gg"}, true);
    U1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = A1_m->get(i[0], i[1]);
    });

    /// Recompute Hbar0 (ref. energy + T1 correlation), Hbar1 (Fock), and Hbar2 (aptei)
    /// E = 0.5 * ( H["ji"] + F["ji] ) * D1["ij"] + 0.25 * V["xyuv"] * L2["uvxy"]

    // Hbar1 is now "bare H1"
    H1["rs"] = U1["rp"] * H_["pq"] * U1["sq"];

    H0 = 0.0;
    H1.block("cc").iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1])
            H0 += value;
    });
    H0 += 0.5 * H1["uv"] * L1_["vu"];

    // for simplicity, create a core-core density matrix
    BlockedTensor D1c = BTF_->build(tensor_type_, "L1 core", {"cc"});
    for (size_t m = 0, nc = core_mos_.size(); m < nc; ++m) {
        D1c.block("cc").data()[m * nc + m] = 2.0;
    }

    // Hbar1 becomes "Fock"
    ambit::BlockedTensor B;
    if (eri_df_) {
        B = BTF_->build(tensor_type_, "B 3-idx", {"Lgg"}, true);
        B["grs"] = U1["rp"] * B_["gpq"] * U1["sq"];

        BlockedTensor temp = BTF_->build(tensor_type_, "B temp", {"L"}, true);
        temp["g"] = B["gmn"] * D1c["mn"];
        temp["g"] += B["guv"] * L1_["uv"];
        H1["pq"] += temp["g"] * B["gpq"];

        H1["pq"] -= 0.5 * B["gpm"] * B["gnq"] * D1c["mn"];
        H1["pq"] -= 0.5 * B["gpu"] * B["gvq"] * L1_["uv"];
    } else {
        H2["pqrs"] = U1["pt"] * U1["qo"] * V_["t,o,g0,g1"] * U1["r,g0"] * U1["s,g1"];

        H1["pq"] += H2["pnqm"] * D1c["mn"];
        H1["pq"] -= 0.5 * H2["npqm"] * D1c["mn"];

        H1["pq"] += H2["pvqu"] * L1_["uv"];
        H1["pq"] -= 0.5 * H2["vpqu"] * L1_["uv"];
    }

    // compute fully contracted term from T1
    H1.block("cc").iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1])
            H0 += value;
    });
    H0 += 0.5 * H1["uv"] * L1_["vu"];

    if (eri_df_) {
        H0 += 0.5 * B["gux"] * B["gvy"] * L2_["xyuv"];
    } else {
        H0 += 0.5 * H2["uvxy"] * L2_["xyuv"];
    }

    H0 += Efrzc_ + Enuc_ - Eref_;

    return B;
}

} // namespace forte
