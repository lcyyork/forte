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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"

#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "trotter_so.h"

using namespace psi;

namespace forte {

std::unique_ptr<TROTTER_SO> make_trotter_so(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                            std::shared_ptr<ForteOptions> options,
                                            std::shared_ptr<ForteIntegrals> ints,
                                            std::shared_ptr<MOSpaceInfo> mo_space_info) {
    return std::make_unique<TROTTER_SO>(rdms, scf_info, options, ints, mo_space_info);
}

TROTTER_SO::TROTTER_SO(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                       std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : DynamicCorrelationSolver(rdms, scf_info, options, ints, mo_space_info),
      BTF_(new BlockedTensorFactory()), tensor_type_(ambit::CoreTensor) {
    print_method_banner({"Spin-Orbital Trotter Multireference CC & DSRG", "Chenyang Li"});
    startup();
    print_summary();
}

TROTTER_SO::~TROTTER_SO() {}

std::shared_ptr<ActiveSpaceIntegrals> TROTTER_SO::compute_Heff_actv() {
    double Edsrg = Eref_ + Hbar0_;

    // scalar from H1 and H2
    Edsrg -= Hbar1_["vu"] * L1_["uv"];
    Edsrg += 0.5 * L1_["uv"] * Hbar2_["vyux"] * L1_["xy"];
    Edsrg -= 0.25 * Hbar2_["xyuv"] * L2_["uvxy"];

    // Hbar1
    Hbar1_["uv"] -= Hbar2_["uxvy"] * L1_["yx"];

    // create spin-integrated integrals
    size_t na_mo = na_ / 2;

    auto H1a = ambit::Tensor::build(tensor_type_, "H1a", std::vector<size_t>(2, na_mo));
    H1a.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t index = i[0] * na_ + i[1];
        value = (Hbar1_.block("aa")).data()[index];
    });

    auto H1b = ambit::Tensor::build(tensor_type_, "H1b", std::vector<size_t>(2, na_mo));
    H1b.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t index = (i[0] + na_mo) * na_ + (i[1] + na_mo);
        value = (Hbar1_.block("aa")).data()[index];
    });

    auto myPow = [](size_t x, size_t p) {
        size_t i = 1;
        for (size_t j = 1; j <= p; j++)
            i *= x;
        return i;
    };

    auto H2aa = ambit::Tensor::build(tensor_type_, "H2aa", std::vector<size_t>(4, na_mo));
    H2aa.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t index = 0;
        for (int m = 0; m < 4; ++m) {
            index += i[m] * myPow(na_, 3 - m);
        }
        value = (Hbar2_.block("aaaa")).data()[index];
    });

    auto H2ab = ambit::Tensor::build(tensor_type_, "H2ab", std::vector<size_t>(4, na_mo));
    H2ab.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t i0 = i[0];
        size_t i1 = i[1] + na_mo;
        size_t i2 = i[2];
        size_t i3 = i[3] + na_mo;
        size_t index = 0;
        index = i0 * myPow(na_, 3) + i1 * myPow(na_, 2) + i2 * myPow(na_, 1) + i3;
        value = (Hbar2_.block("aaaa")).data()[index];
    });

    auto H2bb = ambit::Tensor::build(tensor_type_, "H2bb", std::vector<size_t>(4, na_mo));
    H2bb.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t index = 0;
        for (int m = 0; m < 4; ++m) {
            index += (i[m] + na_mo) * myPow(na_, 3 - m);
        }
        value = (Hbar2_.block("aaaa")).data()[index];
    });

    // create FCIIntegral shared_ptr
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints =
        std::make_shared<ActiveSpaceIntegrals>(ints_, mo_space_info_->get_corr_abs_mo("ACTIVE"),
                                               mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));
    fci_ints->set_active_integrals(H2aa, H2ab, H2bb);
    fci_ints->set_restricted_one_body_operator(H1a.data(), H1b.data());
    fci_ints->set_scalar_energy(Edsrg - ints_->nuclear_repulsion_energy() - Efrzc_);

    return fci_ints;
}

void TROTTER_SO::startup() {
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    Eref_ = compute_Eref_from_rdms(rdms_, ints_, mo_space_info_);
    Efrzc_ = ints_->frozen_core_energy();

    corr_level_ = foptions_->get_str("TROTTER_CORR_LEVEL");
    trotter_level_ = foptions_->get_int("TROTTER_LEVEL");
    trotter_sym_ = foptions_->get_bool("TROTTER_SYMM");

    maxiter_ = foptions_->get_int("MAXITER");
    e_conv_ = foptions_->get_double("E_CONVERGENCE");
    r_conv_ = foptions_->get_double("R_CONVERGENCE");

    s_ = foptions_->get_double("DSRG_S");
    int taylor_threshold = foptions_->get_int("TAYLOR_THRESHOLD");

    source_ = foptions_->get_str("SOURCE");
    if (source_ == "STANDARD") {
        dsrg_source_ = std::make_shared<STD_SOURCE>(s_, taylor_threshold);
    } else if (source_ == "LABS") {
        dsrg_source_ = std::make_shared<LABS_SOURCE>(s_, taylor_threshold);
    } else if (source_ == "DYSON") {
        dsrg_source_ = std::make_shared<DYSON_SOURCE>(s_, taylor_threshold);
    } else {
        throw psi::PSIEXCEPTION("Source operator not support.");
    }

    ntamp_ = foptions_->get_int("NTAMP");
    intruder_tamp_ = foptions_->get_double("INTRUDER_TAMP");

    // orbital spaces
    acore_sos = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    aactv_sos = mo_space_info_->get_corr_abs_mo("ACTIVE");
    avirt_sos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // put all beta behind alpha
    size_t mo_shift = mo_space_info_->size("CORRELATED");

    for (size_t idx : acore_sos)
        bcore_sos.push_back(idx + mo_shift);
    for (size_t idx : aactv_sos)
        bactv_sos.push_back(idx + mo_shift);
    for (size_t idx : avirt_sos)
        bvirt_sos.push_back(idx + mo_shift);

    // spin orbital indices
    core_sos_ = acore_sos;
    actv_sos_ = aactv_sos;
    virt_sos_ = avirt_sos;
    core_sos_.insert(core_sos_.end(), bcore_sos.begin(), bcore_sos.end());
    actv_sos_.insert(actv_sos_.end(), bactv_sos.begin(), bactv_sos.end());
    virt_sos_.insert(virt_sos_.end(), bvirt_sos.begin(), bvirt_sos.end());

    // size of each spin orbital space
    nc_ = core_sos_.size();
    na_ = actv_sos_.size();
    nv_ = virt_sos_.size();
    nh_ = na_ + nc_;
    np_ = na_ + nv_;
    nso_ = nh_ + nv_;
    size_t nmo = nso_ / 2;
    size_t na_mo = na_ / 2;

    BTF_->add_mo_space("c", "m,n,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9", core_sos_, NoSpin);
    BTF_->add_mo_space("a", "u,v,w,x,y,z,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9", actv_sos_, NoSpin);
    BTF_->add_mo_space("v", "e,f,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9", virt_sos_, NoSpin);

    BTF_->add_composite_mo_space("h", "i,j,k,l,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9", {"c", "a"});
    BTF_->add_composite_mo_space("p", "a,b,c,d,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9", {"a", "v"});
    BTF_->add_composite_mo_space("g", "p,q,r,s,t,o,g0,g1,g2,g3,g4,g5,g6,g7,g8,g9", {"c", "a", "v"});

    // prepare one-electron integrals
    H_ = BTF_->build(tensor_type_, "H", {"gg"});
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] < nmo && i[1] < nmo) {
            value = ints_->oei_a(i[0], i[1]);
        }
        if (i[0] >= nmo && i[1] >= nmo) {
            value = ints_->oei_b(i[0] - nmo, i[1] - nmo);
        }
    });

    // prepare two-electron integrals
    V_ = BTF_->build(tensor_type_, "V", {"gggg"});
    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        bool spin0 = i[0] < nmo;
        bool spin1 = i[1] < nmo;
        bool spin2 = i[2] < nmo;
        bool spin3 = i[3] < nmo;
        if (spin0 && spin1 && spin2 && spin3) {
            value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
        }
        if ((!spin0) && (!spin1) && (!spin2) && (!spin3)) {
            value = ints_->aptei_bb(i[0] - nmo, i[1] - nmo, i[2] - nmo, i[3] - nmo);
        }
        if (spin0 && (!spin1) && spin2 && (!spin3)) {
            value = ints_->aptei_ab(i[0], i[1] - nmo, i[2], i[3] - nmo);
        }
        if (spin1 && (!spin0) && spin3 && (!spin2)) {
            value = ints_->aptei_ab(i[1], i[0] - nmo, i[3], i[2] - nmo);
        }
        if (spin0 && (!spin1) && spin3 && (!spin2)) {
            value = -ints_->aptei_ab(i[0], i[1] - nmo, i[3], i[2] - nmo);
        }
        if (spin1 && (!spin0) && spin2 && (!spin3)) {
            value = -ints_->aptei_ab(i[1], i[0] - nmo, i[2], i[3] - nmo);
        }
    });

    // prepare density matrices
    L1_ = BTF_->build(tensor_type_, "Gamma1", {"hh"});
    (L1_.block("cc")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = (i[0] == i[1] ? 1.0 : 0.0);
    });
    (rdms_.g1a()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index = i[0] * na_ + i[1];
        (L1_.block("aa")).data()[index] = value;
    });
    (rdms_.g1b()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index = (i[0] + na_mo) * na_ + (i[1] + na_mo);
        (L1_.block("aa")).data()[index] = value;
    });

    auto myPow = [](size_t x, size_t p) {
        size_t i = 1;
        for (size_t j = 1; j <= p; j++)
            i *= x;
        return i;
    };

    // prepare two-body density cumulant
    L2_ = BTF_->build(tensor_type_, "Lambda2", {"aaaa"});
    (rdms_.L2aa()).citerate([&](const std::vector<size_t>& i, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            size_t index = 0;
            for (int m = 0; m < 4; ++m) {
                index += i[m] * myPow(na_, 3 - m);
            }
            (L2_.block("aaaa")).data()[index] = value;
        }
    });
    (rdms_.L2bb()).citerate([&](const std::vector<size_t>& i, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            size_t index = 0;
            for (int m = 0; m < 4; ++m) {
                index += (i[m] + na_mo) * myPow(na_, 3 - m);
            }
            (L2_.block("aaaa")).data()[index] = value;
        }
    });
    (rdms_.L2ab()).citerate([&](const std::vector<size_t>& i, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            size_t i0 = i[0];
            size_t i1 = i[1] + na_mo;
            size_t i2 = i[2];
            size_t i3 = i[3] + na_mo;
            size_t index = 0;
            index = i0 * myPow(na_, 3) + i1 * myPow(na_, 2) + i2 * myPow(na_, 1) + i3;
            (L2_.block("aaaa")).data()[index] = value;
            index = i1 * myPow(na_, 3) + i0 * myPow(na_, 2) + i3 * myPow(na_, 1) + i2;
            (L2_.block("aaaa")).data()[index] = value;
            index = i1 * myPow(na_, 3) + i0 * myPow(na_, 2) + i2 * myPow(na_, 1) + i3;
            (L2_.block("aaaa")).data()[index] = -value;
            index = i0 * myPow(na_, 3) + i1 * myPow(na_, 2) + i3 * myPow(na_, 1) + i2;
            (L2_.block("aaaa")).data()[index] = -value;
        }
    });
    outfile->Printf("\n    Norm of L2_: %12.8f.", L2_.norm());

    // prepare three-body density cumulant
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        L3_ = BTF_->build(tensor_type_, "Lambda3", {"aaaaaa"});
        (rdms_.L3aaa()).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t index = 0;
                for (int m = 0; m < 6; ++m) {
                    index += i[m] * myPow(na_, 5 - m);
                }
                (L3_.block("aaaaaa")).data()[index] = value;
            }
        });
        (rdms_.L3bbb()).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t index = 0;
                for (int m = 0; m < 6; ++m) {
                    index += (i[m] + na_mo) * myPow(na_, 5 - m);
                }
                (L3_.block("aaaaaa")).data()[index] = value;
            }
        });
        (rdms_.L3aab()).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                // original: a[0]a[1]b[2]; permutation: a[0]b[2]a[1] (-1),
                // b[2]a[0]a[1] (+1)
                std::vector<size_t> upper(3);
                std::vector<std::vector<size_t>> uppers;
                upper[0] = i[0];
                upper[1] = i[1];
                upper[2] = i[2] + na_mo;
                uppers.push_back(upper);
                upper[0] = i[0];
                upper[2] = i[1];
                upper[1] = i[2] + na_mo;
                uppers.push_back(upper);
                upper[1] = i[0];
                upper[2] = i[1];
                upper[0] = i[2] + na_mo;
                uppers.push_back(upper);
                std::vector<size_t> lower(3);
                std::vector<std::vector<size_t>> lowers;
                lower[0] = i[3];
                lower[1] = i[4];
                lower[2] = i[5] + na_mo;
                lowers.push_back(lower);
                lower[0] = i[3];
                lower[2] = i[4];
                lower[1] = i[5] + na_mo;
                lowers.push_back(lower);
                lower[1] = i[3];
                lower[2] = i[4];
                lower[0] = i[5] + na_mo;
                lowers.push_back(lower);

                for (int m = 0; m < 3; ++m) {
                    std::vector<size_t> u = uppers[m];
                    size_t iu = 0;
                    for (int mi = 0; mi < 3; ++mi)
                        iu += u[mi] * myPow(na_, 5 - mi);
                    for (int n = 0; n < 3; ++n) {
                        std::vector<size_t> l = lowers[n];
                        size_t index = iu;
                        for (int ni = 0; ni < 3; ++ni)
                            index += l[ni] * myPow(na_, 2 - ni);
                        (L3_.block("aaaaaa")).data()[index] = value * pow(-1.0, m + n);
                    }
                }
            }
        });
        (rdms_.L3abb()).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                // original: a[0]b[1]b[2]; permutation: b[1]a[0]b[2] (-1),
                // b[1]b[2]a[0] (+1)
                std::vector<size_t> upper(3);
                std::vector<std::vector<size_t>> uppers;
                upper[0] = i[0];
                upper[1] = i[1] + na_mo;
                upper[2] = i[2] + na_mo;
                uppers.push_back(upper);
                upper[1] = i[0];
                upper[0] = i[1] + na_mo;
                upper[2] = i[2] + na_mo;
                uppers.push_back(upper);
                upper[2] = i[0];
                upper[0] = i[1] + na_mo;
                upper[1] = i[2] + na_mo;
                uppers.push_back(upper);
                std::vector<size_t> lower(3);
                std::vector<std::vector<size_t>> lowers;
                lower[0] = i[3];
                lower[1] = i[4] + na_mo;
                lower[2] = i[5] + na_mo;
                lowers.push_back(lower);
                lower[1] = i[3];
                lower[0] = i[4] + na_mo;
                lower[2] = i[5] + na_mo;
                lowers.push_back(lower);
                lower[2] = i[3];
                lower[0] = i[4] + na_mo;
                lower[1] = i[5] + na_mo;
                lowers.push_back(lower);

                for (int m = 0; m < 3; ++m) {
                    std::vector<size_t> u = uppers[m];
                    size_t iu = 0;
                    for (int mi = 0; mi < 3; ++mi)
                        iu += u[mi] * myPow(na_, 5 - mi);
                    for (int n = 0; n < 3; ++n) {
                        std::vector<size_t> l = lowers[n];
                        size_t index = iu;
                        for (int ni = 0; ni < 3; ++ni)
                            index += l[ni] * myPow(na_, 2 - ni);
                        (L3_.block("aaaaaa")).data()[index] = value * pow(-1.0, m + n);
                    }
                }
            }
        });
        outfile->Printf("\n    Norm of L3_: %12.8f.", L3_.norm());
    }

    // build Fock matrix (initial guess of one-body Hamiltonian)
    F_ = BTF_->build(tensor_type_, "Fock", {"gg"});
    F_["pq"] = H_["pq"];
    F_["pq"] += V_["pjqi"] * L1_["ij"];

    // obtain diagonal elements of Fock matrix
    Fd_ = std::vector<double>(nso_);
    F_.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
            if (i[0] == i[1]) {
                Fd_[i[0]] = value;
            }
        });
}

void TROTTER_SO::print_summary() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{{"Trotter level", trotter_level_},
                                                              {"Max iteration", maxiter_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Flow parameter", s_}, {"Energy convergence", e_conv_}, {"Residual convergence", r_conv_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Correlation level", corr_level_},
        {"Integral type", foptions_->get_str("INT_TYPE")},
        {"Source operator", source_},
        {"Trotter symmetrized", trotter_sym_ ? "TRUE" : "FALSE"}};

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-40s %10d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %10.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %10s", str_dim.first.c_str(), str_dim.second.c_str());
    }
}

void TROTTER_SO::guess_t2() {
    local_timer timer;
    std::string str = "Computing T2 amplitudes     ...";
    outfile->Printf("\n    %-35s", str.c_str());

    T2_["ijab"] = V_["ijab"];

    T2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= dsrg_source_->compute_renormalized_denominator(Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] -
                                                                Fd_[i[3]]);
    });

    // zero internal amplitudes
    T2_.block("aaaa").zero();

    // norm and max
    t2_max_ = T2_.norm(0);
    t2_norm_ = T2_.norm();

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void TROTTER_SO::guess_t1() {
    local_timer timer;
    std::string str = "Computing T1 amplitudes     ...";
    outfile->Printf("\n    %-35s", str.c_str());

    // use simple single-reference guess
    T1_["ia"] = F_["ia"];
    T1_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= dsrg_source_->compute_renormalized_denominator(Fd_[i[0]] - Fd_[i[1]]);
    });

    // zero internal amplitudes
    T1_.block("aa").zero();

    // norm and max
    t1_max_ = T1_.norm(0);
    t1_norm_ = T1_.norm();

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void TROTTER_SO::update_t2() {
    // create a temp for Hbar2
    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
    temp["uvxy"] = Hbar2_["uvxy"];

    // compute DT2 = Hbar2 * (1 - exp(-s * D * D)) / D - T2 * exp(-s * D * D)
    BlockedTensor DT2 = ambit::BlockedTensor::build(tensor_type_, "DT2", {"hhpp"});
    DT2["ijab"] = Hbar2_["ijab"];
    DT2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double delta = Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]];
        value *= dsrg_source_->compute_renormalized_denominator(delta);
    });

    // copy T2 to Hbar2
    Hbar2_["ijab"] = T2_["ijab"];

    // scale T2 by exp(-s * D * D)
    T2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double delta = Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]];
        value *= std::exp(-s_ * delta * delta);
    });

    DT2["ijab"] -= T2_["ijab"];
    DT2.block("aaaa").zero();
    t2_rms_ = DT2.norm();

    T2_["ijab"] = Hbar2_["ijab"] + DT2["ijab"];

    // norm and max
    t2_max_ = T2_.norm(0);
    t2_norm_ = T2_.norm();

    Hbar2_["uvxy"] = temp["uvxy"];
}

void TROTTER_SO::update_t1() {
    BlockedTensor R1 = ambit::BlockedTensor::build(tensor_type_, "R1", {"hp"});
    R1["ia"] = T1_["ia"];
    R1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= (Fd_[i[0]] - Fd_[i[1]]);
    });
    R1["ia"] += Hbar1_["ia"];
    R1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= dsrg_source_->compute_renormalized_denominator(Fd_[i[0]] - Fd_[i[1]]);
    });

    // zero internal amplitudes
    R1.block("aa").zero();

    BlockedTensor D1 = ambit::BlockedTensor::build(tensor_type_, "DT1", {"hp"});
    D1["ia"] = R1["ia"] - T1_["ia"];
    t1_rms_ = D1.norm();

    T1_["ia"] = R1["ia"];

    // norm and max
    t1_max_ = T1_.norm(0);
    t1_norm_ = T1_.norm();
}

double TROTTER_SO::compute_energy() {
    // copy initial one-body Hamiltonian to Hbar1
    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", {"gg"});
    Hbar1_["pq"] = F_["pq"];

    // copy initial two-body Hamiltonian to Hbar2
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", {"gggg"});
    Hbar2_["pqrs"] = V_["pqrs"];

    // build initial amplitudes
    outfile->Printf("\n\n  ==> Build Initial Amplitude from DSRG-MRPT2 <==\n");
    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", {"hp"});
    T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
    guess_t2();
    guess_t1();

    // iteration variables
    double Etotal = Eref_;
    bool converged = false;

    // start iteration
    outfile->Printf("\n\n  ==> Start Iterations <==\n");
    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");
    outfile->Printf("\n           Cycle     Energy (a.u.)     Delta(E)  "
                    "|Hbar1|_N  |Hbar2|_N    |T1|    |T2|    max(T1)    max(T2)");
    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");

    for (int cycle = 0; cycle <= maxiter_; ++cycle) {
        if (corr_level_ == "CCSD") {
            compute_trotter_uccsd(F_, V_, T1_, T2_, Hbar0_, Hbar1_, Hbar2_);
        } else {
            outfile->Printf("Not implemented %s", corr_level_.c_str());
        }

        double Edelta = Eref_ + Hbar0_ - Etotal;
        Etotal = Eref_ + Hbar0_;

        // norm of non-diagonal Hbar
        BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hv", "ca"});
        temp["ia"] = Hbar1_["ia"];
        double Hbar1Nnorm = temp.norm();

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpv", "hhva", "hcaa", "ccaa"});
        temp["ijab"] = Hbar2_["ijab"];
        double Hbar2Nnorm = temp.norm();
        for (const std::string block : temp.block_labels()) {
            temp.block(block).reset();
        }

        outfile->Printf("\n      @CC %4d %20.12f %11.3e %10.3e %10.3e %7.4f "
                        "%7.4f %10.6f %10.6f",
                        cycle, Etotal, Edelta, Hbar1Nnorm, Hbar2Nnorm, t1_norm_, t2_norm_, t1_max_,
                        t2_max_);

        // update amplitudes
        update_t2();
        update_t1();

        // test convergence
        double rms = std::max(t1_rms_, t2_rms_);
        if (std::fabs(Edelta) < e_conv_ && rms < r_conv_) {
            converged = true;
            break;
        }
    }

    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");
    outfile->Printf("\n\n\n    Trotter %s Energy Summary", corr_level_.c_str());
    outfile->Printf("\n    Correlation energy      = %25.15f", Etotal - Eref_);
    outfile->Printf("\n  * Total energy            = %25.15f\n", Etotal);

    if (not converged) {
        throw PSIEXCEPTION("TROTTER_SO computation did not converged.");
    }

    psi::Process::environment.globals["CURRENT ENERGY"] = Etotal;

    return Etotal;
}

} // namespace forte
