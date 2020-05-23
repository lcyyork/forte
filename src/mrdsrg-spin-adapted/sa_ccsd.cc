/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include <cctype>

#include "psi4/libdiis/diismanager.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/timer.h"
#include "sa_mrdsrg.h"

using namespace psi;

namespace forte {

double SA_MRDSRG::compute_energy_ccsd() {
    // print title
    outfile->Printf("\n\n  ==> Computing SR-CCSD Energy <==\n");

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

    BlockedTensor::set_expert_mode(true);

    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", {"hp"});
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", {"hhpp"});
    DT1_ = BTF_->build(tensor_type_, "DT1", {"hp"});
    DT2_ = BTF_->build(tensor_type_, "DT2", {"hhpp"});

    // setup DIIS
    if (diis_start_ > 0) {
        diis_manager_init();
    }

    // start iteration
    for (int cycle = 1; cycle <= maxiter_; ++cycle) {
        //        // use DT2_ as an intermediate used for compute Hbar
        //        DT2_["ijab"] = 2.0 * T2_["ijab"];
        //        DT2_["ijab"] -= T2_["ijba"];

        // compute Hbar
        local_timer t_hbar;
        timer hbar("Compute Hbar");
        compute_hbar_ccsd_od(F_, V_, T1_, T2_, Hbar0_, Hbar1_, Hbar2_);
        hbar.stop();
        double Edelta = Hbar0_ - Ecorr;
        Ecorr = Hbar0_;
        double time_hbar = t_hbar.get();

        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar_od_norm(1, blocks1);
        double Hbar2od = Hbar_od_norm(2, blocks2);

        // update amplitudes
        local_timer t_amp;
        update_t();
        double time_amp = t_amp.get();

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
        if (std::fabs(Edelta) < e_conv_ && rms < r_conv_) {
            converged = true;
            break;
        }

        if (cycle == maxiter_) {
            outfile->Printf("\n\n    The computation does not converge in %d iterations!\n",
                            maxiter_);
        }
        if (cycle > 5 and std::fabs(rms) > 10.0) {
            outfile->Printf("\n\n    Large RMS for amplitudes. Likely no convergence. Quitting.\n");
        }
    }

    // clean up raw pointers used in DIIS
    if (diis_start_ > 0) {
        diis_manager_cleanup();
    }

    // print summary
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n\n  ==> SR-CCSD Energy Summary <==\n");
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"SR-CCSD correlation energy", Ecorr});
    energy.push_back({"SR-CCSD total energy", Eref_ + Ecorr});
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %23.15f", str_dim.first.c_str(), str_dim.second);
    }

    // analyze converged amplitudes
    analyze_amplitudes("Final", T1_, T2_);

    // dump amplitudes to disk
    dump_amps_to_cwd();

    // fail to converge
    if (!converged) {
        clean_checkpoints(); // clean amplitudes in scratch directory
        throw psi::PSIEXCEPTION("The SR-CCSD computation does not converge.");
    }

    Hbar0_ = Ecorr;
    return Ecorr;
}

void SA_MRDSRG::compute_hbar_ccsd_od(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                                     BlockedTensor& T2, double& C0, BlockedTensor& C1,
                                     BlockedTensor& C2) {
    C0 = 0.0;
    C1.zero();
    C2.zero();

    C0 += 2.0 * H1["c0,v0"] * T1["c0,v0"];
    C0 += 2.0 * H2["c0,c1,v0,v1"] * T2["c0,c1,v0,v1"];
    C0 += -1.0 * H2["c0,c1,v0,v1"] * T2["c1,c0,v0,v1"];
    C0 += 2.0 * H2["c0,c1,v0,v1"] * T1["c0,v0"] * T1["c1,v1"];
    C0 += -1.0 * H2["c0,c1,v0,v1"] * T1["c0,v1"] * T1["c1,v0"];

    C1["c0,v0"] += H1["c0,v0"];
    C1["c0,v0"] += 1.0 * H1["v1,v0"] * T1["c0,v1"];
    C1["c0,v0"] += 2.0 * H1["c1,v1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += -1.0 * H1["c1,v1"] * T2["c1,c0,v0,v1"];
    C1["c0,v0"] += -1.0 * H1["c1,c0"] * T1["c1,v0"];
    C1["c0,v0"] += 2.0 * H2["v1,v2,v0,c1"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,v2,v0,c1"] * T2["c1,c0,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c0,v0,c1"] * T1["c1,v1"];
    C1["c0,v0"] += 2.0 * H2["c0,v1,v0,c1"] * T1["c1,v1"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,c0"] * T2["c1,c2,v0,v1"];
    C1["c0,v0"] += -2.0 * H2["c1,c2,v1,c0"] * T2["c2,c1,v0,v1"];
    C1["c0,v0"] += -1.0 * H1["c1,v1"] * T1["c0,v1"] * T1["c1,v0"];
    C1["c0,v0"] += 2.0 * H2["v1,v2,v0,c1"] * T1["c0,v1"] * T1["c1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,v2,v0,c1"] * T1["c0,v2"] * T1["c1,v1"];
    C1["c0,v0"] += -2.0 * H2["c1,c2,v1,v2"] * T1["c0,v1"] * T2["c1,c2,v0,v2"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,v2"] * T1["c0,v1"] * T2["c2,c1,v0,v2"];
    C1["c0,v0"] += -2.0 * H2["c1,c2,v1,v2"] * T1["c1,v0"] * T2["c0,c2,v1,v2"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,v2"] * T1["c1,v0"] * T2["c2,c0,v1,v2"];
    C1["c0,v0"] += 4.0 * H2["c1,c2,v1,v2"] * T1["c1,v1"] * T2["c0,c2,v0,v2"];
    C1["c0,v0"] += -2.0 * H2["c1,c2,v1,v2"] * T1["c1,v1"] * T2["c2,c0,v0,v2"];
    C1["c0,v0"] += -2.0 * H2["c1,c2,v1,v2"] * T1["c2,v1"] * T2["c0,c1,v0,v2"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,v2"] * T1["c2,v1"] * T2["c1,c0,v0,v2"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,c0"] * T1["c1,v0"] * T1["c2,v1"];
    C1["c0,v0"] += -2.0 * H2["c1,c2,v1,c0"] * T1["c1,v1"] * T1["c2,v0"];
    C1["c0,v0"] += -2.0 * H2["c1,c2,v1,v2"] * T1["c0,v1"] * T1["c1,v0"] * T1["c2,v2"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,v2"] * T1["c0,v2"] * T1["c1,v0"] * T1["c2,v1"];

    C2["c0,c1,v0,v1"] += H2["c0,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H1["v2,v0"] * T2["c1,c0,v1,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H1["v2,v1"] * T2["c0,c1,v0,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H1["c2,c0"] * T2["c2,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H1["c2,c1"] * T2["c0,c2,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,v1"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,c2"] * T2["c1,c2,v1,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v1,c2"] * T2["c2,c1,v0,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["v2,c1,v0,v1"] * T1["c0,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,c1,v0,c2"] * T2["c2,c0,v1,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,c1,v1,c2"] * T2["c0,c2,v0,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c0,v2,v0,v1"] * T1["c1,v2"];
    C2["c0,c1,v0,v1"] += 2.0 * H2["c0,v2,v0,c2"] * T2["c1,c2,v1,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["c0,v2,v0,c2"] * T2["c2,c1,v1,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v0,c2"] * T1["c2,v1"];
    C2["c0,c1,v0,v1"] += 2.0 * H2["c1,v2,v1,c2"] * T2["c0,c2,v0,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["c1,v2,v1,c2"] * T2["c2,c0,v0,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["c1,c0,v1,c2"] * T1["c2,v0"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,c0,c1"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H1["c2,v2"] * T1["c0,v2"] * T2["c2,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H1["c2,v2"] * T1["c1,v2"] * T2["c0,c2,v0,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H1["c2,v2"] * T1["c2,v0"] * T2["c1,c0,v1,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H1["c2,v2"] * T1["c2,v1"] * T2["c0,c1,v0,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,v1"] * T1["c0,v2"] * T1["c1,v3"];
    C2["c0,c1,v0,v1"] += 2.0 * H2["v2,v3,v0,c2"] * T1["c0,v2"] * T2["c1,c2,v1,v3"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v0,c2"] * T1["c0,v2"] * T2["c2,c1,v1,v3"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v0,c2"] * T1["c0,v3"] * T2["c1,c2,v1,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v0,c2"] * T1["c1,v3"] * T2["c2,c0,v1,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v0,c2"] * T1["c2,v1"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v0,c2"] * T1["c2,v2"] * T2["c1,c0,v1,v3"];
    C2["c0,c1,v0,v1"] += 2.0 * H2["v2,v3,v0,c2"] * T1["c2,v3"] * T2["c1,c0,v1,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v1,c2"] * T1["c0,v3"] * T2["c2,c1,v0,v2"];
    C2["c0,c1,v0,v1"] += 2.0 * H2["v2,v3,v1,c2"] * T1["c1,v2"] * T2["c0,c2,v0,v3"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v1,c2"] * T1["c1,v2"] * T2["c2,c0,v0,v3"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v1,c2"] * T1["c1,v3"] * T2["c0,c2,v0,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v1,c2"] * T1["c2,v0"] * T2["c1,c0,v2,v3"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v1,c2"] * T1["c2,v2"] * T2["c0,c1,v0,v3"];
    C2["c0,c1,v0,v1"] += 2.0 * H2["v2,v3,v1,c2"] * T1["c2,v3"] * T2["c0,c1,v0,v2"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v1,c2"] * T1["c1,v2"] * T1["c2,v0"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,c1,v0,c2"] * T1["c0,v2"] * T1["c2,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["c0,v2,v0,c2"] * T1["c1,v2"] * T1["c2,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["c1,v2,v1,c2"] * T1["c0,v2"] * T1["c2,v0"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T2["c0,c1,v0,v2"] * T2["c3,c2,v1,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T2["c0,c1,v2,v3"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T2["c0,c2,v0,v1"] * T2["c3,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += 4.0 * H2["c2,c3,v2,v3"] * T2["c0,c2,v0,v2"] * T2["c1,c3,v1,v3"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T2["c0,c2,v0,v2"] * T2["c3,c1,v1,v3"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T2["c0,c3,v0,v2"] * T2["c1,c2,v1,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T2["c0,c3,v0,v2"] * T2["c2,c1,v1,v3"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T2["c0,c3,v2,v3"] * T2["c2,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T2["c1,c0,v1,v3"] * T2["c2,c3,v0,v2"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T2["c1,c0,v1,v3"] * T2["c3,c2,v0,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T2["c1,c2,v1,v3"] * T2["c3,c0,v0,v2"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T2["c1,c3,v1,v3"] * T2["c2,c0,v0,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T2["c2,c0,v0,v2"] * T2["c3,c1,v1,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T2["c2,c0,v1,v3"] * T2["c3,c1,v0,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T2["c2,c1,v0,v1"] * T2["c3,c0,v2,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * T1["c1,v2"] * T2["c3,c2,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * T1["c2,v0"] * T2["c1,c3,v1,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * T1["c2,v1"] * T2["c3,c1,v0,v2"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,c0"] * T1["c2,v2"] * T2["c3,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,c0"] * T1["c3,v0"] * T2["c1,c2,v1,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * T1["c3,v0"] * T2["c2,c1,v1,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * T1["c3,v2"] * T2["c2,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c1"] * T1["c0,v2"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c1"] * T1["c2,v0"] * T2["c3,c0,v1,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c1"] * T1["c2,v1"] * T2["c0,c3,v0,v2"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,c1"] * T1["c2,v2"] * T2["c0,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,c1"] * T1["c3,v1"] * T2["c0,c2,v0,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c1"] * T1["c3,v1"] * T2["c2,c0,v0,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c1"] * T1["c3,v2"] * T2["c0,c2,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,c0,c1"] * T1["c2,v0"] * T1["c3,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v0,c2"] * T1["c0,v2"] * T1["c1,v3"] * T1["c2,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v1,c2"] * T1["c0,v3"] * T1["c1,v2"] * T1["c2,v0"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c0,v2"] * T1["c1,v3"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T1["c0,v2"] * T1["c2,v0"] * T2["c1,c3,v1,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c0,v2"] * T1["c2,v0"] * T2["c3,c1,v1,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c0,v2"] * T1["c2,v3"] * T2["c3,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T1["c0,v2"] * T1["c3,v3"] * T2["c2,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c0,v3"] * T1["c2,v0"] * T2["c1,c3,v1,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c0,v3"] * T1["c2,v1"] * T2["c3,c1,v0,v2"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T1["c1,v2"] * T1["c2,v1"] * T2["c0,c3,v0,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c1,v2"] * T1["c2,v1"] * T2["c3,c0,v0,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c1,v2"] * T1["c2,v3"] * T2["c0,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T1["c1,v2"] * T1["c3,v3"] * T2["c0,c2,v0,v1"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c1,v3"] * T1["c2,v0"] * T2["c3,c0,v1,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c1,v3"] * T1["c2,v1"] * T2["c0,c3,v0,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c2,v0"] * T1["c3,v1"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c2,v0"] * T1["c3,v2"] * T2["c1,c0,v1,v3"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T1["c2,v0"] * T1["c3,v3"] * T2["c1,c0,v1,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,v3"] * T1["c2,v1"] * T1["c3,v2"] * T2["c0,c1,v0,v3"];
    C2["c0,c1,v0,v1"] += -2.0 * H2["c2,c3,v2,v3"] * T1["c2,v1"] * T1["c3,v3"] * T2["c0,c1,v0,v2"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * T1["c1,v2"] * T1["c2,v1"] * T1["c3,v0"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c1"] * T1["c0,v2"] * T1["c2,v0"] * T1["c3,v1"];
    C2["c0,c1,v0,v1"] += H2["c2,c3,v2,v3"] * T1["c0,v2"] * T1["c1,v3"] * T1["c2,v0"] * T1["c3,v1"];
}
} // namespace forte
