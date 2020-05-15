/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <map>
#include <vector>
#include <sys/stat.h>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"

#include "base_classes/mo_space_info.h"
#include "base_classes/scf_info.h"
#include "helpers/disk_io.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "cc.h"

using namespace psi;

namespace forte {

std::unique_ptr<CC_SO> make_cc_so(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                  std::shared_ptr<ForteOptions> options,
                                  std::shared_ptr<ForteIntegrals> ints,
                                  std::shared_ptr<MOSpaceInfo> mo_space_info) {
    return std::make_unique<CC_SO>(rdms, scf_info, options, ints, mo_space_info);
}

CC_SO::CC_SO(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
             std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : DynamicCorrelationSolver(rdms, scf_info, options, ints, mo_space_info),
      BTF_(new BlockedTensorFactory()), tensor_type_(ambit::CoreTensor) {
    print_method_banner({"Spin-Orbital Coupled Cluster Using Generated Equations", "Chenyang Li"});
    startup();
}

CC_SO::~CC_SO() {}

std::shared_ptr<ActiveSpaceIntegrals> CC_SO::compute_Heff_actv() {
    throw psi::PSIEXCEPTION(
        "Computing active-space Hamiltonian is not yet implemented for spin-orbital code.");
}

void CC_SO::startup() {
    // recompute reference energy from RDMs
    Eref_ = compute_Eref_from_rdms(rdms_, ints_, mo_space_info_);

    // read options
    read_options();

    // set up MO space
    setup_mo_space();

    // set up integrals
    setup_integrals();

    // build Fock matrix
    build_Fock();
    print_orbital_energies();

    // print options
    print_options();

    // default prefix for amplitudes dump
    std::string corr_lowercase(corr_level_);
    std::transform(corr_lowercase.begin(), corr_lowercase.end(), corr_lowercase.begin(), ::tolower);
    file_prefix_ = "forte." + corr_lowercase;
}

void CC_SO::read_options() {
    corr_level_ = foptions_->get_str("CC_LEVEL");

    do_triples_ = corr_level_.find("CCSDT") != std::string::npos;

    trotter_level_ = foptions_->get_int("CCSD_TROTTER_LEVEL");
    trotter_sym_ = foptions_->get_bool("CCSD_TROTTER_SYMM");

    e_convergence_ = foptions_->get_double("E_CONVERGENCE");
    r_convergence_ = foptions_->get_double("R_CONVERGENCE");
    maxiter_ = foptions_->get_int("MAXITER");

    ntamp_ = foptions_->get_int("NTAMP");

    read_amps_ = foptions_->get_bool("CC_READ_AMPS");
    dump_amps_ = foptions_->get_bool("CC_DUMP_AMPS");
}

void CC_SO::setup_mo_space() {
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    // test for SOCC
    auto socc = scf_info_->soccpi();
    auto actv_dim = mo_space_info_->dimension("ACTIVE");
    if (socc != actv_dim) {
        throw PSIEXCEPTION("Inconsistent dimension for singly occupied orbitals.");
    }

    size_t nsocc = mo_space_info_->size("ACTIVE");
    size_t twice_ms = std::round(2.0 * foptions_->get_double("MS"));
    if (nsocc != twice_ms) {
        throw PSIEXCEPTION("Not high-spin configuration. Please change Ms.");
    }

    // orbital spaces
    acore_sos_ = mo_space_info_->corr_absolute_mo("GENERALIZED HOLE");
    avirt_sos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");

    // put all beta behind alpha
    size_t mo_shift = mo_space_info_->size("CORRELATED");
    for (size_t idx : mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC")) {
        bcore_sos_.push_back(idx + mo_shift);
    }
    for (size_t idx : mo_space_info_->corr_absolute_mo("GENERALIZED PARTICLE")) {
        bvirt_sos_.push_back(idx + mo_shift);
    }

    // spin orbital indices
    core_sos_ = acore_sos_;
    virt_sos_ = avirt_sos_;
    core_sos_.insert(core_sos_.end(), bcore_sos_.begin(), bcore_sos_.end());
    virt_sos_.insert(virt_sos_.end(), bvirt_sos_.begin(), bvirt_sos_.end());

    // size of each spin orbital space
    ncore_ = core_sos_.size();
    nvirt_ = virt_sos_.size();
    nso_ = ncore_ + nvirt_;
    nmo_ = nso_ / 2;

    BTF_->add_mo_space("c", "i,j,k,l,m,n,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9", core_sos_, NoSpin);
    BTF_->add_mo_space("v", "a,b,c,d,e,f,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9", virt_sos_, NoSpin);

    BTF_->add_composite_mo_space("g", "p,q,r,s,t,o,g0,g1,g2,g3,g4,g5,g6,g7,g8,g9", {"c", "v"});
}

void CC_SO::setup_integrals() {
    // prepare one-electron integrals
    H_ = BTF_->build(tensor_type_, "H", {"gg"});
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] < nmo_ && i[1] < nmo_) {
            value = ints_->oei_a(i[0], i[1]);
        }
        if (i[0] >= nmo_ && i[1] >= nmo_) {
            value = ints_->oei_b(i[0] - nmo_, i[1] - nmo_);
        }
    });

    // prepare two-electron integrals
    V_ = BTF_->build(tensor_type_, "V", {"gggg"});
    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        bool spin0 = i[0] < nmo_;
        bool spin1 = i[1] < nmo_;
        bool spin2 = i[2] < nmo_;
        bool spin3 = i[3] < nmo_;
        if (spin0 && spin1 && spin2 && spin3) {
            value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
        }
        if ((!spin0) && (!spin1) && (!spin2) && (!spin3)) {
            value = ints_->aptei_bb(i[0] - nmo_, i[1] - nmo_, i[2] - nmo_, i[3] - nmo_);
        }
        if (spin0 && (!spin1) && spin2 && (!spin3)) {
            value = ints_->aptei_ab(i[0], i[1] - nmo_, i[2], i[3] - nmo_);
        }
        if (spin1 && (!spin0) && spin3 && (!spin2)) {
            value = ints_->aptei_ab(i[1], i[0] - nmo_, i[3], i[2] - nmo_);
        }
        if (spin0 && (!spin1) && spin3 && (!spin2)) {
            value = -ints_->aptei_ab(i[0], i[1] - nmo_, i[3], i[2] - nmo_);
        }
        if (spin1 && (!spin0) && spin2 && (!spin3)) {
            value = -ints_->aptei_ab(i[1], i[0] - nmo_, i[2], i[3] - nmo_);
        }
    });
}

void CC_SO::build_Fock() {
    // build Fock matrix (initial guess of one-body Hamiltonian)
    F_ = BTF_->build(tensor_type_, "Fock", {"gg"});
    F_["pq"] = H_["pq"];

    auto K1 = BTF_->build(tensor_type_, "Kronecker delta", {"cc"});
    (K1.block("cc")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = (i[0] == i[1] ? 1.0 : 0.0);
    });
    F_["pq"] += V_["pjqi"] * K1["ij"];

    // obtain diagonal elements of Fock matrix
    Fd_ = std::vector<double>(nso_);
    F_.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
            if (i[0] == i[1]) {
                Fd_[i[0]] = value;
            }
        });
}

void CC_SO::print_orbital_energies() {
    size_t nc_a = acore_sos_.size();
    size_t nc_b = bcore_sos_.size();
    print_h2("Orbital Energies");
    outfile->Printf("\n     MO     Alpha           Beta");
    outfile->Printf("\n    ---------------------------------");
    for (size_t i = 0; i < nmo_; ++i) {
        outfile->Printf("\n    %3zu %11.6f(%d) %11.6f(%d)", i + 1, Fd_[i], i < nc_a, Fd_[i + nmo_],
                        i < nc_b);
    }
    outfile->Printf("\n    ---------------------------------");
}

void CC_SO::print_options() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info_int{
        {"Max Iteration", maxiter_}, {"Number of Printed T Amplitudes", ntamp_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Energy Convergence", e_convergence_}, {"Residue Convergence", r_convergence_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Correlation Level", corr_level_}, {"Integral Type", foptions_->get_str("INT_TYPE")}};

    auto true_false_string = [](bool x) {
        if (x) {
            return std::string("TRUE");
        } else {
            return std::string("FALSE");
        }
    };

    if (corr_level_ == "CCSD_TROTTER") {
        calculation_info_int.push_back({"Trotter level", trotter_level_});
        calculation_info_string.push_back(
            {"Symmetrize Trotter at each step", true_false_string(trotter_sym_)});
    }

    calculation_info_string.push_back({"Read amplitudes from disk", true_false_string(read_amps_)});
    calculation_info_string.push_back({"Dump amplitudes to disk", true_false_string(dump_amps_)});

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-35s %20s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-35s %20.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_int) {
        outfile->Printf("\n    %-35s %20d", str_dim.first.c_str(), str_dim.second);
    }
}

double CC_SO::compute_energy() {
    if (corr_level_ == "CCSD" or corr_level_ == "CCSDT" or
        corr_level_ == "CCSDT_1A" or corr_level_ == "CCSDT_1B") {
        DT1_ = BTF_->build(tensor_type_, "T1 Residuals", {"cv"});
        DT2_ = BTF_->build(tensor_type_, "T2 Residuals", {"ccvv"});
    } else if(corr_level_ == "CCSD_TROTTER") {
        DT1_ = BTF_->build(tensor_type_, "T1 Residuals", {"cv"});
        DT2_ = BTF_->build(tensor_type_, "T2 Residuals", {"ccvv"});
        Hbar1_ = BTF_->build(tensor_type_, "Hbar1", {"gg"});
        Hbar2_ = BTF_->build(tensor_type_, "Hbar2", {"gggg"});
    } else {
        throw PSIEXCEPTION("Not Implemented yet!");
    }

    // build initial amplitudes
    print_h2("Build Initial Cluster Amplitudes");
    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", {"cv"});
    T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"ccvv"});
    guess_t2();
    guess_t1();

    if (do_triples_) {
        if (corr_level_ == "CCSDT" or corr_level_ == "CCSDT_1A" or corr_level_ == "CCSDT_1B") {
            DT3_ = BTF_->build(tensor_type_, "Hbar3", {"cccvvv"});
        } else {
            Hbar3_ = BTF_->build(tensor_type_, "Hbar3", {"gggggg"});
        }

        T3_ = BTF_->build(tensor_type_, "T3 Amplitudes", {"cccvvv"});
        guess_t3();
    }

    // setup DIIS
    if (diis_start_ > 0) {
        diis_manager_init();
    }

    // iteration variables
    double Etotal = Eref_;
    bool converged = false;

    // start iteration
    outfile->Printf("\n\n  ==> Start Iterations <==\n");
    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "------------------------------------------------------------");
    outfile->Printf("\n           Cycle     Energy (a.u.)     Delta(E)  "
                    "|Hbar1|_N  |Hbar2|_N    |T1|    |T2|    |T3|  max(T1) max(T2) max(T3) DIIS");
    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "------------------------------------------------------------");

    for (int cycle = 1; cycle <= maxiter_; ++cycle) {
        if (corr_level_ == "CCSD") {
            compute_ccsd_amp(F_, V_, T1_, T2_, Hbar0_, DT1_, DT2_);
        } else if (corr_level_ == "CCSD_TROTTER") {
            compute_ccsd_trotter(F_, V_, T1_, T2_, Hbar0_, Hbar1_, Hbar2_);
            DT1_["ia"] = Hbar1_["ia"];
            DT2_["ijab"] = Hbar2_["ijab"];
        } else if (corr_level_ == "CCSDT") {
            compute_ccsdt_amp(F_, V_, T1_, T2_, T3_, Hbar0_, DT1_, DT2_, DT3_);
        } else if (corr_level_ == "CCSDT_1A" or corr_level_ == "CCSDT_1B") {
            compute_ccsdt1_amp(F_, V_, T1_, T2_, T3_, Hbar0_, DT1_, DT2_, DT3_);
        } else {
            throw PSIEXCEPTION("Unknown correlation level for CC_SO");
        }

        double Edelta = Eref_ + Hbar0_ - Etotal;
        Etotal = Eref_ + Hbar0_;

        // norm of non-diagonal Hbar
        double Hbar1Nnorm = DT1_.norm();
        double Hbar2Nnorm = DT2_.norm();

        outfile->Printf("\n      @CC %4d %20.12f %11.3e %10.3e %10.3e %7.4f "
                        "%7.4f %7.4f %7.4f %7.4f %7.4f",
                        cycle, Etotal, Edelta, Hbar1Nnorm, Hbar2Nnorm, T1norm_, T2norm_, T3norm_,
                        T1max_, T2max_, T3max_);

        update_t2();
        update_t1();
        if (do_triples_) {
            update_t3();
        }

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

        // test convergence
        double rms = std::max(std::max(rms_t1_, rms_t2_), rms_t3_);
        if (std::fabs(Edelta) < e_convergence_ && rms < r_convergence_) {
            converged = true;
        }

        if (converged) {
            break;
        }
    }

    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");
    outfile->Printf("\n\n\n    %s Energy Summary", corr_level_.c_str());
    outfile->Printf("\n    Correlation energy      = %25.15f", Etotal - Eref_);
    outfile->Printf("\n  * Total energy            = %25.15f\n", Etotal);

    // clean up raw pointers used in DIIS
    if (diis_start_ > 0) {
        diis_manager_cleanup();
    }

    // write amplitudes to files
    if (dump_amps_) {
        // default name: file_prefix + "." + name + ".master.txt";
        write_disk_BT(T1_, "t1", file_prefix_);
        write_disk_BT(T2_, "t2", file_prefix_);
        if (do_triples_) {
            write_disk_BT(T3_, "t3", file_prefix_);
        }
    }

    if (not converged) {
        throw PSIEXCEPTION("CC_SO computation did not converged.");
    }

    psi::Process::environment.globals["CURRENT ENERGY"] = Etotal;

    return Etotal;
}

void CC_SO::rotate_hamiltonian(double& Eeff, BlockedTensor& Fnew, BlockedTensor& Vnew) {
    ambit::BlockedTensor A1 = BTF_->build(tensor_type_, "A1 Amplitudes", {"gg"});
    A1["ia"] = T1_["ia"];
    A1["ai"] -= T1_["ia"];

    psi::SharedMatrix A1_m(new psi::Matrix("A1", nso_, nso_));
    A1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        A1_m->set(i[0], i[1], value);
    });

    // >=3 is required for high energy convergence
    A1_m->expm(3);

    ambit::BlockedTensor U1 = BTF_->build(tensor_type_, "Transformer", {"gg"});
    U1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = A1_m->get(i[0], i[1]);
    });

    // Recompute Hbar0 (ref. energy + T1 correlation), Fnew (Fock), and Vnew (aptei)
    // E = 0.5 * ( H["ji"] + F["ji] ) * D1["ij"]

    Fnew["rs"] = U1["rp"] * H_["pq"] * U1["sq"];

    Eeff = 0.0;
    Fnew.block("cc").iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1]) {
            Eeff += 0.5 * value;
        }
    });

    Vnew["g0,g1,g2,g3"] = U1["g0,g4"] * U1["g1,g5"] * V_["g4,g5,g6,g7"] * U1["g2,g6"] * U1["g3,g7"];

    auto K1 = BTF_->build(tensor_type_, "Kronecker delta", {"cc"});
    (K1.block("cc")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = (i[0] == i[1] ? 1.0 : 0.0);
    });
    Fnew["pq"] += Vnew["pjqi"] * K1["ij"];

    // compute fully contracted term from T1
    Fnew.block("cc").iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1]) {
            Eeff += 0.5 * value;
        }
    });

    Eeff += Efrzc_ + Enuc_ - Eref_;
}
} // namespace forte
