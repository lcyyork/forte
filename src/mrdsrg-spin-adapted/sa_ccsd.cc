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
        // use DT2_ as an intermediate used for compute Hbar
        DT2_["ijab"] = 2.0 * T2_["ijab"];
        DT2_["ijab"] -= T2_["ijba"];

        // compute Hbar
        local_timer t_hbar;
        timer hbar("Compute Hbar");
        compute_hbar_ccsd_od();
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

void SA_MRDSRG::compute_hbar_ccsd_od() {

    auto tilde_tau = BTF_->build(tensor_type_, "tilde_tau", {"ccvv"});
    auto tau = BTF_->build(tensor_type_, "tau", {"ccvv"});

    tau["ijab"] = T1_["ia"] * T1_["jb"];

    tilde_tau["ijab"] = T2_["ijab"];
    tilde_tau["ijab"] += 0.5 * tau["ijab"];

    tau["ijab"] += T2_["ijab"];

    // energy
    Hbar0_ = 2.0 * F_["ia"] * T1_["ia"];

    Hbar0_ += 2.0 * V_["ijab"] * tau["ijab"];
    Hbar0_ -= V_["ijba"] * tau["ijab"];

    auto W1 = BTF_->build(tensor_type_, "W1", {"cc", "vv", "cv"});

    W1["ae"] += F_["ae"];
    for (size_t e = 0, nv = mo_space_info_->size("RESTRICTED_UOCC"); e < nv; ++e) {
        W1.block("vv").data()[e * nv + e] = 0.0;
    }

    W1["ae"] -= 0.5 * T1_["ma"] * F_["me"];

    W1["ae"] += 2.0 * T1_["mf"] * V_["mafe"];
    W1["ae"] -= T1_["mf"] * V_["maef"];

    W1["ae"] -= 2.0 * tilde_tau["mnfa"] * V_["mnfe"];
    W1["ae"] += tilde_tau["mnaf"] * V_["mnfe"];

    W1["mi"] += F_["mi"];
    for (size_t m = 0, nc = mo_space_info_->size("RESTRICTED_DOCC"); m < nc; ++m) {
        W1.block("cc").data()[m * nc + m] = 0.0;
    }

    W1["mi"] += 0.5 * T1_["ie"] * F_["me"];

    W1["mi"] += 2.0 * T1_["ne"] * V_["mnie"];
    W1["mi"] -= T1_["ne"] * V_["mnei"];

    W1["mi"] += 2.0 * tilde_tau["inef"] * V_["mnef"];
    W1["mi"] -= tilde_tau["inef"] * V_["mnfe"];

    W1["me"] += F_["me"];

    W1["me"] += 2.0 * T1_["nf"] * V_["mnef"];
    W1["me"] -= T1_["nf"] * V_["mnfe"];

    auto W2 = BTF_->build(tensor_type_, "W2", {"cccc", "cvvc", "cvcv", "vvvv"});

    W2["mnij"] += V_["mnij"];
    W2["mnij"] += T1_["je"] * V_["mnie"];
    W2["mnij"] += T1_["ie"] * V_["ejmn"];
    W2["mnij"] += 0.5 * tau["ijef"] * V_["mnef"];

    W2["abef"] += V_["abef"];
    W2["abef"] += T1_["mb"] * V_["amef"];
    W2["abef"] += T1_["ma"] * V_["mbef"];
    W2["abef"] += 0.5 * tau["mnab"] * V_["mnef"];

    W2["mbej"] += V_["mbej"];
    W2["mbej"] += T1_["jf"] * V_["mbef"];
    W2["mbej"] -= T1_["nb"] * V_["ejmn"];
    W2["mbej"] += T2_["njfb"] * V_["mnef"];
    W2["mbej"] -= 0.5 * T2_["njfb"] * V_["mnfe"];
    W2["mbej"] -= 0.5 * T2_["jnfb"] * V_["mnef"];
    W2["mbej"] -= T1_["jf"] * T1_["nb"] * V_["mnef"];

    W2["mbje"] += V_["mbje"];
    W2["mbje"] += T1_["jf"] * V_["femb"];
    W2["mbje"] -= T1_["nb"] * V_["jemn"];
    W2["mbje"] -= 0.5 * T2_["jnfb"] * V_["mnfe"];
    W2["mbje"] -= T1_["jf"] * T1_["nb"] * V_["femn"];

    // amplitudes
    Hbar1_["ia"] = F_["ia"];

    Hbar1_["ia"] += T1_["ie"] * W1["ae"];

    Hbar1_["ia"] -= T1_["ma"] * W1["mi"];

    Hbar1_["ia"] += 2.0 * T2_["imae"] * W1["me"];
    Hbar1_["ia"] -= T2_["imea"] * W1["me"];

    Hbar1_["ia"] -= T1_["nf"] * V_["naif"];
    Hbar1_["ia"] += 2.0 * T1_["nf"] * V_["anif"];

    Hbar1_["ia"] += 2.0 * T2_["imef"] * V_["amef"];
    Hbar1_["ia"] -= T2_["imfe"] * V_["amef"];

    Hbar1_["ia"] -= 2.0 * T2_["nmea"] * V_["nmei"];
    Hbar1_["ia"] += T2_["mnea"] * V_["nmei"];

    Hbar2_["ijab"] = V_["ijab"];

    Hbar2_["ijab"] += T2_["ijae"] * W1["be"];
    Hbar2_["ijab"] -= 0.5 * T2_["ijae"] * T1_["mb"] * W1["me"];

    Hbar2_["ijab"] += T2_["ijeb"] * W1["ae"];
    Hbar2_["ijab"] -= 0.5 * T2_["ijeb"] * T1_["ma"] * W1["me"];

    Hbar2_["ijab"] -= T2_["imab"] * W1["mj"];
    Hbar2_["ijab"] -= 0.5 * T2_["imab"] * T1_["je"] * W1["me"];

    Hbar2_["ijab"] -= T2_["mjab"] * W1["mi"];
    Hbar2_["ijab"] -= 0.5 * T2_["mjab"] * T1_["ie"] * W1["me"];

    Hbar2_["ijab"] += tau["mnab"] * W2["mnij"];

    Hbar2_["ijab"] += tau["ijef"] * W2["abef"];
//    Hbar2_["ijab"] += tau["ijef"] * V_["abef"];
//    Hbar2_["ijab"] -= tau["ijef"] * T1_["ma"] * V_["mbef"];
//    Hbar2_["ijab"] -= tau["ijef"] * T1_["mb"] * V_["amef"];
//    Hbar2_["ijab"] += 0.5 * tau["ijef"] * tau["mnab"] * V_["mnef"];

    Hbar2_["ijab"] += 2.0 * T2_["imae"] * W2["mbej"];
    Hbar2_["ijab"] -= T2_["imea"] * W2["mbej"];
    Hbar2_["ijab"] -= T2_["imae"] * W2["mbje"];

    Hbar2_["ijab"] -= T2_["imeb"] * W2["maje"];

    Hbar2_["ijab"] -= T2_["mjae"] * W2["mbie"];

    Hbar2_["ijab"] += 2.0 * T2_["mjeb"] * W2["maei"];
    Hbar2_["ijab"] -= T2_["mjeb"] * W2["maie"];
    Hbar2_["ijab"] -= T2_["jmeb"] * W2["maei"];

    Hbar2_["ijab"] -= T1_["ie"] * T1_["ma"] * V_["ejmb"];
    Hbar2_["ijab"] -= T1_["ie"] * T1_["mb"] * V_["ejam"];
    Hbar2_["ijab"] -= T1_["je"] * T1_["ma"] * V_["mbie"];
    Hbar2_["ijab"] -= T1_["je"] * T1_["mb"] * V_["amie"];

    Hbar2_["ijab"] += T1_["ie"] * V_["ejab"];
    Hbar2_["ijab"] += T1_["je"] * V_["abie"];

    Hbar2_["ijab"] -= T1_["ma"] * V_["mbij"];
    Hbar2_["ijab"] -= T1_["mb"] * V_["ijam"];
}
} // namespace forte
