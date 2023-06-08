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

#include <algorithm>
#include <cstdio>
#include <sys/stat.h>

#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/disk_io.h"
#include "helpers/timer.h"
#include "helpers/printing.h"
#include "sa_mrdsrg.h"

using namespace psi;

namespace forte {
double SA_MRDSRG::compute_energy_cc2() {
    outfile->Printf("\n\n  ==> Computing Second-Order MRDSRG Energy <==\n");
    outfile->Printf("\n    Zeroth-Order Hamiltonian: %s\n", Hzero_.c_str());

    if (!do_cu3_) {
        outfile->Printf("\n    Skip 3-cumulant contributions in [O2, T2].");
    }

    std::string indent(4, ' ');
    std::string dash(105, '-');
    std::string title;

    title += indent + "              Energy (a.u.)           Non-Diagonal Norm        Amplitude "
                      "RMS         Timings (s)\n";
    title += indent + "       ---------------------------  ---------------------  "
                      "---------------------  -----------------\n";
    title += indent + "Iter.        Corr.         Delta       Hbar1      Hbar2        T1         "
                      "T2        Hbar     Amp.    DIIS\n";
    title += indent + dash;

    outfile->Printf("\n%s", title.c_str());

    // figure out off-diagonal block labels for Hbar1 and Hbar2
    std::vector<std::string> blocks1 = od_one_labels_hp();
    std::vector<std::string> blocks2 = od_two_labels_hhpp();

    // iteration variables
    double Ecorr = 0.0;
    bool converged = false;
    bool err_amps = false;

    setup_cc2_tensors();

    // zeroth-order Hamiltonian
    auto F0 = ambit::BlockedTensor::build(tensor_type_, "F0", {"cc", "aa", "vv"});

    ambit::BlockedTensor V0;
    if (Hzero_ == "DYALL") {
        V0 = ambit::BlockedTensor::build(tensor_type_, "V0", {"aaaa"});
    } else if (Hzero_ == "FINK") {
        V0 = ambit::BlockedTensor::build(tensor_type_, "V0",
                                         {"aaaa", "cccc", "caca", "caac", "acca", "acac", "vcvc",
                                          "vccv", "cvvc", "cvcv", "avav", "avva", "vaav", "vava"});
    }

    if (!Hzero_a1_) {
        F0["pq"] = F_["pq"];
        if (Hzero_ == "DYALL") {
            V0["uvxy"] = B_["gux"] * B_["gvy"];
        } else if (Hzero_ == "FINK") {
            V0["pqrs"] = B_["gpr"] * B_["gqs"];
        }
    }

    // setup DIIS
    if (diis_start_ > 0) {
        diis_manager_init();
    }

    // start iteration
    for (int cycle = 1; cycle <= maxiter_; ++cycle) {
        // use DT2_ as an intermediate used for compute Hbar
        DT2_["ijab"] = 2.0 * T2_["ijab"];
        DT2_["ijab"] -= T2_["ijba"];

        // compute Hbar
        local_timer t_hbar;
        timer hbar("Compute Hbar");

        // Hbar0: ref. energy + T1 correlation
        // Hbar1: Fock
        // Hbar2: aptei
        auto B = rotate_integrals(Hbar0_, Hbar1_, Hbar2_);

        if (Hzero_a1_) {
            F0["pq"] = Hbar1_["pq"];
            if (Hzero_ == "DYALL") {
                V0["uvxy"] = B["gux"] * B["gvy"];
            } else if (Hzero_ == "FINK") {
                V0["pqrs"] = B["gpr"] * B["gqs"];
            }
        }

        C1_.zero();
        C2_.zero();

        // C = [H0th, A2]
        H1_T2_C1(F0, T2_, 1.0, C1_);
        H1_T2_C2(F0, T2_, 1.0, C2_);
        if (Hzero_ == "DYALL" or Hzero_ == "FINK") {
            H2_T2_C1(V0, T2_, DT2_, 1.0, C1_);
            H2_T2_C2(V0, T2_, DT2_, 1.0, C2_);
            if (Hzero_ == "FINK") {
                if (Hzero_a1_)
                    C2_["ijef"] += batched("e", B["g,v0,e"] * B["g,v1,f"] * T2_["i,j,v0,v1"]);
                else
                    C2_["ijef"] += batched("e", B_["g,v0,e"] * B_["g,v1,f"] * T2_["i,j,v0,v1"]);
            }
        }
        O1_["pq"] = C1_["pq"];
        C1_["qp"] += O1_["pq"];
        Hbar2_["ijab"] = C2_["ijab"];
        C2_["abij"] += Hbar2_["ijab"];

        // Hbar_2 = Htilde_2 + [H0th, A2]_2
        Hbar2_["ijab"] += B["gia"] * B["gjb"];

        // Hbar <- [Htilde, A2] + 0.5 * [[H0th, A2], A2]
        O1_.zero();
        C1_["pq"] += 2.0 * Hbar1_["pq"];
        H1_T2_C0(C1_, T2_, 1.0, Hbar0_);
        H1_T2_C1(C1_, T2_, 0.5, O1_);

        V_T2_C0_DF(B, T2_, DT2_, 2.0, Hbar0_);
        V_T2_C1_DF(B, T2_, DT2_, 1.0, O1_);

        H2_T2_C0(C2_, T2_, DT2_, 1.0, Hbar0_);
        H2_T2_C1(C2_, T2_, DT2_, 0.5, O1_);

        Hbar1_["pq"] += O1_["pq"];
        Hbar1_["qp"] += O1_["pq"];

        hbar.stop();
        double Edelta = Hbar0_ - Ecorr;
        Ecorr = Hbar0_;
        double time_hbar = t_hbar.get();

        timer od("Off-diagonal Hbar");
        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar_od_norm(1, blocks1);
        double Hbar2od = Hbar_od_norm(2, blocks2);

        // update amplitudes
        local_timer t_amp;
        update_t();
        double time_amp = t_amp.get();
        od.stop();

        // printing
        outfile->Printf("\n    %4d   %16.12f %10.3e  %10.3e %10.3e  %10.3e %10.3e  %8.3f %8.3f",
                        cycle, Ecorr, Edelta, Hbar1od, Hbar2od, T1rms_, T2rms_, time_hbar,
                        time_amp);

        timer diis("DIIS");
        // DIIS amplitudes
        if (diis_start_ > 0 and cycle >= diis_start_) {
            diis_manager_add_entry();
            outfile->Printf("  S");

            if ((cycle - diis_start_) % diis_freq_ == 0 and
                diis_manager_->subspace_size() >= diis_min_vec_) {
                diis_manager_extrapolate();
                outfile->Printf("/E");
            }
        }
        diis.stop();

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
        converged = (std::fabs(Edelta) < e_conv_ && rms < r_conv_);
        err_amps = (cycle > 5 and std::fabs(rms) > 10.0);

        if (converged or err_amps or cycle == maxiter_) {
            if (err_amps) {
                outfile->Printf(
                    "\n\n    Large RMS for amplitudes. Likely no convergence. Quitting.\n");
            } else {
                Hbar2_["uvxy"] = B["gux"] * B["gvy"];
                DT2_["ijab"] = 2.0 * T2_["ijab"] - T2_["ijba"];
                auto Va = ambit::BlockedTensor::build(tensor_type_, "Va", {"aaaa"});
                H1_T2_C2(C1_, T2_, 0.5, Va);
                V_T2_C2_DF(B, T2_, DT2_, 1.0, Va);
                H2_T2_C2(C2_, T2_, DT2_, 0.5, Va);
                Hbar2_["uvxy"] += Va["uvxy"];
                Hbar2_["xyuv"] += Va["uvxy"];
            }
            break;
        }
    }

    // clean up raw pointers used in DIIS
    if (diis_start_ > 0) {
        diis_manager_cleanup();
    }

    timer final("Summary SO-MRDSRG");
    // print summary
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n\n  ==> Second-Order MRDSRG Energy Summary <==\n");
    std::vector<std::pair<std::string, double>> energy;
    energy.emplace_back("E0 (reference)", Eref_);
    energy.emplace_back("SO-MRDSRG correlation energy", Ecorr);
    energy.emplace_back("SO-MRDSRG total energy", Eref_ + Ecorr);
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %23.15f", str_dim.first.c_str(), str_dim.second);
    }

    // analyze converged amplitudes
    analyze_amplitudes("Final", T1_, T2_);

    // dump amplitudes to disk
    dump_amps_to_disk();

    // fail to converge
    if (!converged) {
        outfile->Printf("\n\n    SO-MRDSRG did not converge in %d iterations!\n", maxiter_);
        if (die_if_not_converged_ or err_amps) {
            clean_checkpoints(); // clean amplitudes in scratch directory
            throw psi::PSIEXCEPTION("The SO-MRDSRG computation does not converge.");
        }
    }
    final.stop();

    Hbar0_ = Ecorr;
    return Ecorr;
}

void SA_MRDSRG::setup_cc2_tensors() {
    BlockedTensor::set_expert_mode(true);

    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", {"gg"});
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", {"hhpp"});

    DT1_ = BTF_->build(tensor_type_, "DT1", {"hp"});
    DT2_ = BTF_->build(tensor_type_, "DT2", {"hhpp"});

    C1_ = BTF_->build(tensor_type_, "C1", {"gg"});
    O1_ = BTF_->build(tensor_type_, "O1", {"gg"});

    C2_ = BTF_->build(tensor_type_, "C2", od_two_labels());
}
} // namespace forte