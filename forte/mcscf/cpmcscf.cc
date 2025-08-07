/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/psi4-dec.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "base_classes/state_info.h"
#include "base_classes/forte_options.h"
#include "base_classes/active_space_solver.h"
#include "mrdsrg-helper/dsrg_transformed.h"
#include "helpers/printing.h"

#include "cpmcscf.h"

namespace forte {

using namespace ambit;

CPMCSCF_SOLVER::CPMCSCF_SOLVER(std::shared_ptr<ForteIntegrals> ints,
                               std::shared_ptr<ActiveSpaceSolver> as_solver,
                               std::shared_ptr<ForteOptions> options,
                               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ints_(ints), as_solver_(as_solver), options_(options), mo_space_info_(mo_space_info) {

    // make sure all states of SA-MCSCF are available in the current ActiveSpaceSolver
    state_weights_map_ = make_state_weights_map(options, mo_space_info, true);
    auto state_energy_map = as_solver->state_energies_map();
    for (const auto& [state, weights] : state_weights_map_) {
        if (state_energy_map.find(state) == state_energy_map.end()) {
            throw std::runtime_error(
                "Not all states of GRAD_AVG_STATE are found in ActiveSpaceSolver");
        } else {
            if (weights.size() > state_energy_map[state].size()) {
                throw std::runtime_error(
                    "Not all roots of GRAD_AVG_STATE are found in ActiveSpaceSolver");
            }
        }
    }

    // read target state
    py::list target = options->get_gen_list("GARD_TARGET_STATE");
    if (target.size() != 3) {
        psi::outfile->Printf("\n  Error: invalid input of GARD_TARGET_STATE.");
        psi::outfile->Printf("\n  Each entry should take an array of three numbers.");
        throw std::runtime_error("Invalid input of GARD_TARGET_STATE");
    }
    int irrep = py::cast<int>(target[0]);
    int multi = py::cast<int>(target[1]);
    for (const auto& [state, e] : as_solver->state_energies_map()) {
        if (state.irrep() == irrep and state.multiplicity() == multi) {
            target_root_ = {state, py::cast<int>(target[2])};
            break;
        }
    }

    auto int_type = ints_->integral_type();
    eri_df_ = (int_type == DF or int_type == DiskDF or int_type == Cholesky);
    psi::outfile->Printf("\n ERI_DF? %s", eri_df_ ? "TRUE" : "FALSE");

    // set up MO and CI spaces
    set_mo_space();

    // SA-RDMs
    init_rdms();

    // various intermediates
    init_ints();

    // compute the active reference energy of the target state
    target_e0_actv_ = compute_target_eref_actv();
    target_ec_actv_ = 0.0;
    psi::outfile->Printf("\n target e0 = %.15f", target_e0_actv_);

    // test build sigma
    // auto state_ndets_map = as_solver_->state_space_size_map();
    // for (const auto& [state, weights] : state_weights_map_) {
    //     auto ndets = state_ndets_map[state];
    //     auto& xc = xc_[state].data();
    //     for (size_t root = 0, nroots = weights.size() - 1; root < nroots; ++root) {
    //         auto ci = as_solver_->ci_wfn(state, root);
    //         psi::C_DCOPY(ndets, ci->pointer(), 1, &xc[(root + 1) * ndets], 1);
    //     }
    //     auto ci = as_solver_->ci_wfn(state, weights.size() - 1);
    //     psi::C_DCOPY(ndets, ci->pointer(), 1, &xc[0], 1);
    //     xc_[state].print();
    // }
    // xo_["p_,q_"] = A_["p_,q_"];
    // xo_["p_,q_"] -= A_["q_,p_"];
    // build_sigma();

    // test solve using target state CASCI
    auto Ft = ambit::BlockedTensor::build(tensor_type_, "Ft", {"g_,g_"});
    auto rdms =
        as_solver_->rdms_from_disk(target_root_.first, target_root_.second, 2, RDMsType::spin_free);

    auto D1 = BlockedTensor::build(tensor_type_, "D1", {"a_,a_"});
    auto d1 = D1.block("a_,a_");
    d1("pq") += 0.5 * rdms->SF_G1()("pq");
    d1("pq") += 0.5 * rdms->SF_G1()("qp");

    auto D2 = BlockedTensor::build(tensor_type_, "D2", {"a_,a_,a_,a_"});
    auto d2 = D2.block("a_,a_,a_,a_");
    d2("pqrs") += 0.25 * rdms->SF_G2()("pqrs");
    d2("pqrs") += 0.25 * rdms->SF_G2()("qpsr");
    d2("pqrs") += 0.25 * rdms->SF_G2()("rspq");
    d2("pqrs") += 0.25 * rdms->SF_G2()("srqp");

    Ft["p_,q_"] = Fc_["p_,q_"];
    Ft["p_,q_"] += D1["u_,v_"] * G_["p_,u_,q_,v_"];
    Ft["p_,q_"] -= 0.5 * D1["u_,v_"] * G_["p_,u_,v_,q_"];

    auto A = BlockedTensor::build(tensor_type_, "A", {"g_,g_"});
    A["p_,m_"] = 2.0 * Ft["p_,m_"];
    A["p_,u_"] = Fc_["p_,v_"] * D1_["v_,u_"];
    A["p_,u_"] += G_["p_,v_,x_,y_"] * D2["x_,y_,u_,v_"];

    bo_["p_,q_"] = 2.0 * A["p_,q_"];
    bo_["p_,q_"] -= 2.0 * A["q_,p_"];

    for (const auto& [state, weights] : state_weights_map_) {
        if (state != target_root_.first)
            continue;
        auto& bc_data = bc_[state].data();
        for (size_t root = 0, nroots = weights.size(); root < nroots; ++root) {
            if (root != target_root_.second)
                continue;
            auto target_ci = as_solver_->ci_wfn(state, root);
            auto ndets = target_ci->dim();
            auto sigma = std::make_shared<psi::Vector>("b0", ndets);
            as_solver_->generalized_sigma(state, target_ci, sigma);
            psi::C_DCOPY(ndets, sigma->pointer(), 1, &bc_data[root * ndets], 1);
        }
    }
    bo_.print();
    for (const auto& [state, weights] : state_weights_map_) {
        bc_.at(state).print();
    }

    solve();
}

void CPMCSCF_SOLVER::set_mo_space() {
    label_to_mos_.clear();
    label_to_mos_["c"] = mo_space_info_->absolute_mo("INACTIVE_DOCC");
    label_to_mos_["v"] = mo_space_info_->absolute_mo("INACTIVE_UOCC");
    label_to_mos_["a"] = mo_space_info_->absolute_mo("ACTIVE");

    BlockedTensor::set_expert_mode(true);
    BlockedTensor::add_mo_space("c_", "m_,n_", label_to_mos_["c"], NoSpin);
    BlockedTensor::add_mo_space("a_", "t_,u_,v_,w_,y_,x_,z_", label_to_mos_["a"], NoSpin);
    BlockedTensor::add_mo_space("v_", "a_,b_,c_,d_", label_to_mos_["v"], NoSpin);
    BlockedTensor::add_composite_mo_space("o_", "i_,j_,k_,l_", {"c_", "a_"});
    BlockedTensor::add_composite_mo_space("g_", "p_,q_,r_,s_", {"c_", "a_", "v_"});

    if (eri_df_) {
        std::vector<size_t> aux_mos(ints_->nthree());
        for (size_t i = 0; i < ints_->nthree(); ++i)
            aux_mos[i] = i;
        label_to_mos_["L"] = aux_mos;
        BlockedTensor::add_mo_space("L_", "g_,", aux_mos, NoSpin);
    }

    auto nmo = mo_space_info_->size("ALL");
    auto nmopi = mo_space_info_->dimension("ALL");
    mos_rel_.resize(nmo);
    int nirrep = mo_space_info_->nirrep();
    for (int h = 0, offset = 0; h < nirrep; ++h) {
        for (int i = 0; i < nmopi[h]; ++i) {
            mos_rel_[i + offset] = std::make_pair(h, i);
        }
        offset += nmopi[h];
    }
}

void CPMCSCF_SOLVER::init_rdms() {
    auto rdms =
        as_solver_->compute_average_rdms(state_weights_map_, 2, RDMsType::spin_free, false, true);
    D1_ = BlockedTensor::build(tensor_type_, "D1", {"a_,a_"});
    auto D1 = D1_.block("a_,a_");
    D1("pq") += 0.5 * rdms->SF_G1()("pq");
    D1("pq") += 0.5 * rdms->SF_G1()("qp");

    D2_ = BlockedTensor::build(tensor_type_, "D2", {"a_,a_,a_,a_"});
    auto D2 = D2_.block("a_,a_,a_,a_");
    D2("pqrs") += 0.25 * rdms->SF_G2()("pqrs");
    D2("pqrs") += 0.25 * rdms->SF_G2()("qpsr");
    D2("pqrs") += 0.25 * rdms->SF_G2()("rspq");
    D2("pqrs") += 0.25 * rdms->SF_G2()("srqp");

    M1_ = BlockedTensor::build(tensor_type_, "GD1", {"a_,a_"});
    M2_ = BlockedTensor::build(tensor_type_, "GD2", {"a_,a_,a_,a_"});
}

void CPMCSCF_SOLVER::init_ints() {
    // build fock
    compute_fock();

    // TODO: build 2-e integrals with at least two active indices
    G_ = BlockedTensor::build(tensor_type_, "g2", {"g_,g_,g_,g_"});
    if (eri_df_) {
        Q_ = BlockedTensor::build(tensor_type_, "B", {"L_,g_,g_"});
        for (const std::string& block : Q_.block_labels()) {
            auto i0 = label_to_mos_[block.substr(0, 1)];
            auto i1 = label_to_mos_[block.substr(2, 1)];
            auto i2 = label_to_mos_[block.substr(4, 1)];
            std::string _block = block.substr(0, 2);
            for (size_t i : {2, 4})
                _block += "," + block.substr(i, 2);
            Q_.block(_block).copy(ints_->three_integral_block(i0, i1, i2));
        }
        G_["p_,q_,r_,s_"] = Q_["g_,p_,r_"] * Q_["g_,q_,s_"];
    } else {
        for (const std::string& block : G_.block_labels()) {
            auto i0 = label_to_mos_[block.substr(0, 1)];
            auto i1 = label_to_mos_[block.substr(2, 1)];
            auto i2 = label_to_mos_[block.substr(4, 1)];
            auto i3 = label_to_mos_[block.substr(6, 1)];
            std::string _block = block.substr(0, 2);
            for (size_t i : {2, 4, 6})
                _block += "," + block.substr(i, 2);
            G_.block(_block).copy(ints_->aptei_ab_block(i0, i1, i2, i3));
        }
    }

    // // build total fock
    // Ft_["p_,q_"] = Fc_["p_,q_"];
    // Ft_["p_,q_"] += D1_["u_,v_"] * G_["p_,u_,q_,v_"];
    // Ft_["p_,q_"] -= 0.5 * D1_["u_,v_"] * G_["p_,u_,v_,q_"];

    // build orbital response matrix A
    A_ = BlockedTensor::build(tensor_type_, "A", {"g_,g_"});
    A_["p_,m_"] = 2.0 * Ft_["p_,m_"];
    A_["p_,u_"] = Fc_["p_,v_"] * D1_["v_,u_"];
    A_["p_,u_"] += G_["p_,v_,x_,y_"] * D2_["x_,y_,u_,v_"];

    // build diagonal Hessian
    compute_hess_diag();

    // initialize x and other intermediates
    xo_ = ambit::BlockedTensor::build(tensor_type_, "xo", {"g_,g_"});
    bo_ = ambit::BlockedTensor::build(tensor_type_, "bo", {"g_,g_"});

    for (const auto& [state, ndets] : as_solver_->state_space_size_map()) {
        auto nroots = state_weights_map_[state].size();
        xc_[state] = ambit::Tensor::build(tensor_type_, "xc", {nroots, ndets});
        bc_[state] = ambit::Tensor::build(tensor_type_, "bc", {nroots, ndets});
    }

    L_ = BlockedTensor::build(tensor_type_, "L", {"g_,g_,g_,g_"});
    L_["p_,q_,r_,s_"] = 4.0 * G_["p_,q_,r_,s_"];
    L_["p_,q_,r_,s_"] -= G_["p_,q_,s_,r_"];
    L_["p_,q_,r_,s_"] -= G_["p_,r_,q_,s_"];

    Z1_ = ambit::BlockedTensor::build(tensor_type_, "Z1", {"a_,a_"});
    Z2_ = ambit::BlockedTensor::build(tensor_type_, "Z2", {"a_,a_,a_,a_"});
}

void CPMCSCF_SOLVER::compute_fock() {
    Fc_ = ambit::BlockedTensor::build(tensor_type_, "Fc", {"g_,g_"});
    Ft_ = ambit::BlockedTensor::build(tensor_type_, "Ft", {"g_,g_"});
    Fd_ = std::vector<double>(mo_space_info_->size("ALL"));

    auto fill_fock = [&](ambit::BlockedTensor F, std::shared_ptr<psi::Matrix> f) {
        F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            auto [h0, p] = mos_rel_[i[0]];
            auto [h1, q] = mos_rel_[i[1]];
            value = (h0 == h1 ? 1.0 : 0.0) * f->get(h0, p, q);
        });
    };

    auto nirrep = mo_space_info_->nirrep();
    auto doccpi = mo_space_info_->dimension("INACTIVE_DOCC");
    auto [fa, _, ec] = ints_->make_fock_inactive(psi::Dimension(nirrep), doccpi);
    fill_fock(Fc_, fa);

    const auto& d1_data = D1_.block("a_,a_").data();
    auto actvpi = mo_space_info_->dimension("ACTIVE");
    auto nactv = actvpi.sum();
    auto rdm1 = std::make_shared<psi::Matrix>("1RDM", actvpi, actvpi);
    for (size_t h = 0, offset = 0; h < nirrep; ++h) {
        for (int u = 0; u < actvpi[h]; ++u) {
            size_t nu = u + offset;
            for (int v = 0; v < actvpi[h]; ++v) {
                rdm1->set(h, u, v, d1_data[nu * nactv + v + offset]);
            }
        }
        offset += actvpi[h];
    }
    fa->add(ints_->make_fock_active_restricted(rdm1));
    fill_fock(Ft_, fa);

    // fill in diagonal Fock in Pitzer ordering
    for (const std::string& space : {"c", "a", "v"}) {
        std::string block = space + "_," + space + "_";
        auto mos = label_to_mos_[space];
        for (size_t i = 0, size = mos.size(); i < size; ++i) {
            Fd_[mos[i]] = Ft_.block(block).data()[i * size + i];
        }
    }
}

double CPMCSCF_SOLVER::compute_target_eref_actv() {
    double e0 = 0.0;
    auto [state, root] = target_root_;
    auto rdms = as_solver_->rdms_from_disk(state, root, 2, RDMsType::spin_free);
    e0 += Fc_.block("a_,a_")("u_,v_") * rdms->SF_G1()("v_,u_");
    e0 += 0.5 * G_.block("a_,a_,a_,a_")("u_,v_,x_,y_") * rdms->SF_G2()("x_,y_,u_,v_");
    return e0;
}

// ambit::BlockedTensor CPMCSCF_SOLVER::compute_orb_grad(ambit::BlockedTensor D1,
// ambit::BlockedTensor D2) {
//     auto A = BlockedTensor::build(tensor_type_, "A", {"g_,g_"});
//     A["p_,m_"] = 2.0 * Ft_["p_,m_"];
//     A["p_,u_"] = Fc_["p_,v_"] * D1["v_,u_"];
//     A["p_,u_"] += G_["p_,v_,x_,y_"] * D2["x_,y_,u_,v_"];
//     return A;
// }

void CPMCSCF_SOLVER::compute_hess_diag() {
    // modified diagonal orbital Hessian from Theor. Chem. Acc. 97, 88-95 (1997)
    Ho_ = ambit::BlockedTensor::build(tensor_type_, "Ho", {"g_,g_"});

    // core-virtual block
    Ho_.block("v_,c_").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["v"][i[0]];
        auto i1 = label_to_mos_["c"][i[1]];
        value = 4.0 * (Fd_[i0] - Fd_[i1]);
    });
    Ho_.block("c_,v_")("pq") = Ho_.block("v_,c_")("qp");

    // active-virtual block
    auto nactv = mo_space_info_->size("ACTIVE");
    auto& D1data = D1_.block("a_,a_").data();
    auto& Adata = A_.block("a_,a_").data();
    Ho_.block("v_,a_").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["v"][i[0]];
        auto i1 = i[1] * nactv + i[1];
        value = 2.0 * (Fd_[i0] * D1data[i1] - Adata[i1]);
    });
    Ho_.block("a_,v_")("pq") = Ho_.block("v_,a_")("qp");

    // core-active block
    Ho_.block("a_,c_").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["a"][i[0]];
        auto i1 = label_to_mos_["c"][i[1]];
        auto i1p = i[0] * nactv + i[0];
        value = 4.0 * (Fd_[i0] - Fd_[i1]);
        value += 2.0 * (Fd_[i1] * D1data[i1p] - Adata[i1p]);
    });
    Ho_.block("c_,a_")("pq") = Ho_.block("a_,c_")("qp");

    // active-active block [SI of J. Chem. Phys. 152, 074102 (2020)]
    auto nactv2 = nactv * nactv;
    auto nactv3 = nactv * nactv2;
    auto& Fdata = Fc_.block("a_,a_").data();
    auto& Gdata = G_.block("a_,a_,a_,a_").data();
    auto& D2data = D2_.block("a_,a_,a_,a_").data();

    auto g2 = ambit::BlockedTensor::build(tensor_type_, "g2_actv", {"a_,a_,a_"});
    auto d2 = ambit::BlockedTensor::build(tensor_type_, "d2_actv", {"a_,a_,a_"});
    auto Guu = ambit::BlockedTensor::build(tensor_type_, "Guu", {"a_,a_"});
    auto Guv = ambit::BlockedTensor::build(tensor_type_, "Guv", {"a_,a_"});

    // <ux|uy> -> uxy
    g2.block("a_,a_,a_").iterate([&](const std::vector<size_t>& i, double& value) {
        value = Gdata[i[0] * nactv3 + i[1] * nactv2 + i[0] * nactv + i[2]];
    });
    // <vx,vy> -> vxy
    d2.block("a_,a_,a_").iterate([&](const std::vector<size_t>& i, double& value) {
        value = D2data[i[0] * nactv3 + i[1] * nactv2 + i[0] * nactv + i[2]];
    });
    Guu["u_,v_"] = g2["u_,x_,y_"] * d2["v_,x_,y_"];

    // <uu|xy> -> uxy
    g2.block("a_,a_,a_").iterate([&](const std::vector<size_t>& i, double& value) {
        value = Gdata[i[0] * nactv3 + i[0] * nactv2 + i[1] * nactv + i[2]];
    });
    // <vv|xy> -> vxy
    d2.block("a_,a_,a_").iterate([&](const std::vector<size_t>& i, double& value) {
        value = D2data[i[0] * nactv3 + i[0] * nactv2 + i[1] * nactv + i[2]];
    });
    Guu["u_,v_,"] += 2.0 * g2["u_,x_,y_"] * d2["v_,x_,y_"];

    Guu.block("a_,a_").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = i[0] * nactv + i[0];
        auto i1 = i[1] * nactv + i[1];
        value += Fdata[i0] * D1data[i1];
    });

    Guv["u_,v_"] = Fc_["u_,v_"] * D1_["v_,u_"];
    Guv["u_,v_"] += G_["u_,x_,v_,y_"] * D2_["v_,x_,u_,y_"];
    Guv["u_,v_"] += 2.0 * G_["u_,v_,x_,y_"] * D2_["v_,u_,x_,y_"];

    Ho_["u_,v_"] = 2.0 * Guu["u_,v_"];
    Ho_["u_,v_"] += 2.0 * Guu["v_,u_"];
    Ho_["u_,v_"] -= 2.0 * Guv["u_,v_"];
    Ho_["u_,v_"] -= 2.0 * Guv["v_,u_"];

    Ho_.block("a_,a_").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = i[0] * nactv + i[0];
        auto i1 = i[1] * nactv + i[1];
        value -= 2.0 * (Adata[i0] + Adata[i1]);
    });

    // diagonal CI Hessian
    for (const auto& [state, ndets] : as_solver_->state_space_size_map()) {
        Hc_[state] = as_solver_->space_energies(state);
    }
}

void CPMCSCF_SOLVER::apply_Minv(ambit::BlockedTensor ro, std::map<StateInfo, ambit::Tensor>& rc) {
    for (const std::string& block :
         {"c_,a_", "c_,v_", "a_,v_", "a_,c_", "v_,c_", "v_,a_", "a_,a_"}) {
        auto& ro_data = ro.block(block).data();
        auto& Ho_data = Ho_.block(block).data();
        for (size_t i = 0, size = ro_data.size(); i < size; ++i) {
            if (std::fabs(Ho_data[i]) > 1.0e-10) {
                ro_data[i] /= Ho_data[i];
            }
        }
    }
    for (auto& [state, r] : rc) {
        auto nroots = r.dim(0);
        auto ndets = r.dim(1);
        auto& rdata = r.data();
        const auto& weights = state_weights_map_[state];
        for (size_t root = 0, shift = 0; root < nroots; ++root) {
            auto w = weights[root];
            if (w < 1.0e-15)
                continue;
            for (size_t I = 0; I < ndets; ++I)
                rdata[I + shift] /= w * Hc_.at(state)[I];
            shift += ndets;
        }
    }
}

void CPMCSCF_SOLVER::set_bc(const std::map<StateInfo, ambit::Tensor>& bc) {
    for (const auto& [state, b] : bc) {
        if (bc_.find(state) == bc_.end())
            throw std::runtime_error("Unexpected StateInfo in the input bc");
        if (bc_[state].dims() != b.dims())
            throw std::runtime_error("Unexpected dimensions for the input bc");
        bc_[state]("pq") = b("pq");
    }
}

void CPMCSCF_SOLVER::set_bo_aa(ambit::Tensor b_aa) { set_bo_block("a_,a_", b_aa); }

void CPMCSCF_SOLVER::set_bo_av(ambit::Tensor b_av) { set_bo_block("a_,v_", b_av); }

void CPMCSCF_SOLVER::set_bo_ca(ambit::Tensor b_ca) { set_bo_block("c_,a_", b_ca); }

void CPMCSCF_SOLVER::set_bo_cv(ambit::Tensor b_cv) { set_bo_block("c_,v_", b_cv); }

void CPMCSCF_SOLVER::set_bo_block(const std::string& block, ambit::Tensor b) {
    std::string b_str = block.substr(0, 1) + block.substr(3, 1);
    if (b.dims() != bo_.block(block).dims())
        throw std::runtime_error("Unexpected dimensions for the input b_" + b_str);
    bo_.block(block)("pq") = b("pq");
}

void CPMCSCF_SOLVER::solve() {
    // project out CI for bc
    project_ci(bc_);
    for (const auto& [state, weights] : state_weights_map_) {
        bc_.at(state).print();
    }

    // GMRES(n)
    bool converged = false;
    double r_conv = 1.0e-10;
    int maxiter = 1;
    int mmiter = 5;
    std::vector<ambit::BlockedTensor> Qo(mmiter + 1);
    std::vector<std::map<StateInfo, ambit::Tensor>> Qc(mmiter + 1);
    for (int i = 0; i < mmiter; ++i) {
        Qo[i] = ambit::BlockedTensor::build(tensor_type_, "Qo", {"g_,g_"});
        for (const auto& [state, ndets] : as_solver_->state_space_size_map()) {
            auto nroots = state_weights_map_[state].size();
            Qc[i][state] = ambit::Tensor::build(tensor_type_, "Qc", {nroots, ndets});
        }
    }

    // restart (macro iterations)
    for (int iter = 0; iter < maxiter; ++iter) {
        std::vector<double> sn(mmiter), cs(mmiter), beta(mmiter + 1);
        std::vector<double> H(mmiter * mmiter + mmiter); // (mmiter + 1) x mmiter

        project_ci(xc_);
        for (const auto& [state, weights] : state_weights_map_) {
            xc_.at(state).print();
        }

        // initial residual
        build_sigma(xo_, xc_, Qo[0], Qc[0]);
        axpy(1.0, bo_, bc_, Qo[0], Qc[0]);
        apply_Minv(Qo[0], Qc[0]);
        beta[0] = normalize(Qo[0], Qc[0]);

        auto state_ndets_map = as_solver_->state_space_size_map();
        for (const auto& [state, weights] : state_weights_map_) {
            psi::outfile->Printf("\n ==> state %s", state.str_short().c_str());
            auto ndets = state_ndets_map[state];
            auto& xc = Qc[0][state].data();
            for (size_t root = 0, size = weights.size(); root < size; ++root) {
                auto ci = as_solver_->ci_wfn(state, root);
                double dot = psi::C_DDOT(ndets, ci->pointer(), 1, &xc[root * ndets], 1);
                psi::outfile->Printf("\n inner product between Q[0] and ci of root %zu: %20.15f",
                                     root, dot);
            }
        }

        // micro iterations
        int k = 0;
        do {
            // Arnoldi
            project_ci(Qc[k + 1]);
            build_sigma(Qo[k], Qc[k], Qo[k + 1], Qc[k + 1]);
            apply_Minv(Qo[k + 1], Qc[k + 1]);
            for (int j = 0; j < k; ++j) {
                auto Hjk = vector_dot(Qo[j], Qc[j], Qo[k + 1], Qc[k + 1]);
                H[j * mmiter + k] = Hjk;
                axpy(-Hjk, Qo[j], Qc[j], Qo[k + 1], Qc[k + 1]);
            }
            H[(k + 1) * mmiter + k] = normalize(Qo[k + 1], Qc[k + 1]);

            for (const auto& [state, weights] : state_weights_map_) {
                psi::outfile->Printf("\n ==> state %s", state.str_short().c_str());
                auto ndets = state_ndets_map[state];
                auto& xc = Qc[k + 1][state].data();
                for (size_t root = 0, size = weights.size(); root < size; ++root) {
                    auto ci = as_solver_->ci_wfn(state, root);
                    double dot = psi::C_DDOT(ndets, ci->pointer(), 1, &xc[root * ndets], 1);
                    psi::outfile->Printf(
                        "\n inner product between Q[%d] and ci of root %zu: %20.15f", k + 1, root,
                        dot);
                }
            }

            // H to upper triangular via Givens rotation
            for (int j = 0; j < k; ++j) {
                auto ia = j * mmiter + k, ib = (j + 1) * mmiter + k;
                auto a = H[ia], b = H[ib];
                H[ib] = -sn[j] * a + cs[j] * b;
                H[ia] = cs[j] * a + sn[j] * b;
            }
            auto ih = k * mmiter + k, ig = (k + 1) * mmiter + k;
            auto h = H[ih], g = H[ig];
            auto rho = std::sqrt(h * h + g * g);
            sn[k] = g / rho, cs[k] = h / rho;
            H[ih] = cs[k] * h + sn[k] * g;
            H[ig] = 0.0;
            beta[k + 1] = -sn[k] * beta[k];
            beta[k] = cs[k] * beta[k];

            // test convergence
            psi::outfile->Printf("\n  macro %2d  micro %2d  error %13.6e", iter, k, beta[k + 1]);
            if (fabs(beta[++k]) < r_conv) { // we increase k by 1 here!!!
                converged = true;
                break;
            }
        } while (k < mmiter);

        // solve upper triangular system
        std::vector<double> Hk(k * k);
        for (int m = 0; m < k; ++m) {
            for (int n = m; n < k; ++n) {
                Hk[m * k + n] = H[m * mmiter + n];
            }
        }
        psi::C_DTRSV('U', 'N', 'N', k, Hk.data(), k, beta.data(), 1);

        // apply results
        for (int i = 0; i < k; ++i) {
            Qo[i].scale(beta[i]);
            for (auto& [state, tensor] : Qc[i]) {
                tensor.scale(beta[i]);
            }
            axpy(1.0, Qo[i], Qc[i], xo_, xc_);
        }

        if (converged)
            break;
    }

    if (not converged)
        throw std::runtime_error("GMRES not converged");
}

void CPMCSCF_SOLVER::build_sigma(ambit::BlockedTensor qo, std::map<StateInfo, ambit::Tensor>& qc,
                                 ambit::BlockedTensor so, std::map<StateInfo, ambit::Tensor>& sc) {
    // assume ca, cv, av blocks of qo are available
    qo.block("a_,c_")("pq") = -1.0 * qo.block("c_,a_")("qp");
    qo.block("v_,c_")("pq") = -1.0 * qo.block("c_,v_")("qp");
    qo.block("v_,a_")("pq") = -1.0 * qo.block("a_,v_")("qp");

    // orbital response for orbital constraints
    so["p_,q_"] = A_["p_,i_"] * qo["i_,q_"];

    so["p_,m_"] += 2.0 * qo["m_,r_"] * Ft_["r_,p_"];
    so["p_,m_"] += 2.0 * qo["n_,r_"] * L_["r_,m_,n_,p_"];

    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"a_,g_"});
    temp["v_,r_"] = qo["u_,r_"] * D1_["v_,u_"];
    so["p_,m_"] += temp["v_,r_"] * L_["r_,m_,v_,p_"];

    so["p_,u_"] += temp["u_,r_"] * Fc_["r_,p_"];

    temp["x_,p_"] = qo["m_,r_"] * L_["r_,x_,m_,p_"];
    so["p_,u_"] += temp["x_,p_"] * D1_["x_,u_"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"a_,a_,a_,g_"});
    temp["x_,y_,u_,r_"] = qo["z_,r_"] * D2_["x_,y_,z_,u_"];
    so["p_,u_"] += G_["r_,p_,x_,y_"] * temp["x_,y_,u_,r_"];
    so["p_,u_"] += G_["r_,p_,x_,y_"] * temp["x_,u_,y_,r_"];
    so["p_,u_"] += G_["r_,x_,p_,y_"] * temp["u_,y_,x_,r_"];

    // Zo_["p_,u_"] += xo_["m_,r_"] * L_["r_,x_,m_,p_"] * D1_["x_,u_"];
    // Zo_["p_,u_"] += xo_["z_,r_"] * Fc_["r_,p_"] * D1_["u_,z_"];
    // Zo_["p_,u_"] += xo_["z_,r_"] * G_["r_,p_,x_,y_"] * D2_["x_,y_,z_,u_"];
    // Zo_["p_,u_"] += xo_["z_,r_"] * G_["r_,p_,x_,y_"] * D2_["x_,u_,z_,y_"];
    // Zo_["p_,u_"] += xo_["z_,r_"] * G_["r_,x_,p_,y_"] * D2_["u_,y_,z_,x_"];

    // CI response for orbital constraints
    auto temp1 = ambit::BlockedTensor::build(tensor_type_, "temp", {"a_,a_"});
    auto temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"a_,a_,a_,a_"});

    // TODO: need to modify for canonical active orbitals
    temp1["u_,v_"] = qo["u_,p_"] * Fc_["p_,v_"];
    temp1["u_,v_"] += 2.0 * qo["m_,p_"] * G_["p_,u_,m_,v_"];
    temp1["u_,v_"] -= qo["m_,p_"] * G_["p_,u_,v_,m_"];
    temp2["u_,v_,x_,y_"] = qo["u_,p_"] * G_["p_,v_,x_,y_"];

    Z1_["u_,v_"] = temp1["u_,v_"];
    Z1_["u_,v_"] += temp1["v_,u_"];

    Z2_["u_,v_,x_,y_"] = temp2["u_,v_,x_,y_"];
    Z2_["u_,v_,x_,y_"] += temp2["x_,y_,u_,v_"];
    Z2_["u_,v_,x_,y_"] += temp2["v_,u_,y_,x_"];
    Z2_["u_,v_,x_,y_"] += temp2["y_,x_,v_,u_"];

    auto Zaa = ambit::BlockedTensor::build(tensor_type_, "Z2aa", {"a_,a_,a_,a_"});
    Zaa["u_,v_,x_,y_"] = 0.25 * Z2_["u_,v_,x_,y_"];
    Zaa["u_,v_,x_,y_"] -= 0.25 * Z2_["v_,u_,x_,y_"];

    auto Z1a = Z1_.block("a_,a_");
    auto Z2aa = Zaa.block("a_,a_,a_,a_");
    auto Z2ab = Z2_.block("a_,a_,a_,a_");
    auto Zints = std::make_shared<DressedQuantity>(0.0, Z1a, Z1a, Z2aa, Z2ab, Z2aa);
    Z1_.print();
    Z2_.print();

    // add_sigma
    double zero_weight = 1.0e-15;
    auto state_ndets_map = as_solver_->state_space_size_map();
    for (const auto& [state, weights] : state_weights_map_) {
        auto ndets = state_ndets_map[state];
        auto& Zc = sc[state].data();
        for (size_t root = 0, nroots = weights.size(); root < nroots; ++root) {
            if (weights[root] < zero_weight)
                continue;
            auto iter_begin = Zc.begin() + root * ndets;
            std::span<double> sigma(iter_begin, iter_begin + ndets);
            as_solver_->add_sigma_kbody(state, root, weights[root], Zints, sigma);
        }
        sc[state].print();
    }

    // orbital response for CI constraints (assume orthogonality between xc and ci)
    M1_.zero();
    M2_.zero();
    auto M1a = M1_.block("a_,a_");
    auto M2a = M2_.block("a_,a_,a_,a_");

    for (const auto& [state, weights] : state_weights_map_) {
        const auto& xc = qc[state].data();
        auto ndets = state_ndets_map[state];
        for (size_t root = 0, nroots = weights.size(); root < nroots; ++root) {
            if (weights[root] < zero_weight)
                continue;
            auto iter_begin = xc.begin() + root * ndets;
            std::span<const double> X(iter_begin, iter_begin + ndets);
            auto grdms = as_solver_->grdms(state, root, X, 2, RDMsType::spin_free, true);
            grdms->scale(weights[root]);
            M1a("uv") += grdms->SF_G1()("uv");
            M1a("uv") += grdms->SF_G1()("vu");
            M2a("uvxy") += grdms->SF_G2()("uvxy");
            M2a("uvxy") += grdms->SF_G2()("vuyx");
            M2a("uvxy") += grdms->SF_G2()("xyuv");
            M2a("uvxy") += grdms->SF_G2()("yxvu");
        }
    }

    so["p_,m_"] += G_["m_,u_,p_,v_"] * M1_["v_,u_"];
    so["p_,z_"] += Fc_["p_,v_"] * M1_["v_,z_"];
    so["p_,z_"] += 0.5 * G_["p_,v_,x_,y_"] * M2_["x_,y_,z_,v_"];

    // CI response for CI constraints
    for (const auto& [state, weights] : state_weights_map_) {
        auto ndets = state_ndets_map[state];
        auto xsub = std::make_shared<psi::Vector>(ndets);
        auto sigma = std::make_shared<psi::Vector>(ndets);
        auto& Zc = sc[state].data();
        auto& xc = qc[state].data();
        for (size_t i = 0, nroots = weights.size(); i < nroots; ++i) {
            if (weights[i] < zero_weight)
                continue;
            psi::C_DCOPY(ndets, &xc[i * ndets], 1, xsub->pointer(), 1);
            as_solver_->generalized_sigma(state, xsub, sigma);
            sigma->set_name("sigma" + std::to_string(i));
            sigma->print();
            psi::C_DAXPY(ndets, weights[i], sigma->pointer(), 1, &Zc[i * ndets], 1);
        }
    }

    // CI normalization condition
    for (const auto& [state, weights] : state_weights_map_) {
        auto ndets = state_ndets_map[state];
        auto& Zc = sc[state].data();
        for (size_t root = 0, size = weights.size(); root < size; ++root) {
            double iota = 0.0;
            if (state == target_root_.first and root == target_root_.second) {
                iota += 2.0 * (target_e0_actv_ + target_ec_actv_);
            }

            double zfactor = 2.0 * weights[root];
            auto rdms = as_solver_->rdms_from_disk(state, root, 2, RDMsType::spin_free);
            iota += zfactor * temp1.block("a_,a_")("uv") * rdms->SF_G1()("vu");
            iota += zfactor * temp2.block("a_,a_,a_,a_")("uvxy") * rdms->SF_G2()("xyuv");

            // auto grdms = as_solver_->grdms_from_disk(state, root, 2, RDMsType::spin_free);
            // iota += weights[root] * Fc_.block("a_,a_")("uv") * rdms->SF_G1()("vu");
            // iota += 0.5 * weights[root] * G_.block("a_,a_,a_,a_")("uvxy") *
            // rdms->SF_G2()("xyuv");

            auto ci = as_solver_->ci_wfn(state, root);
            psi::C_DAXPY(ndets, -iota, ci->pointer(), 1, &Zc[root * ndets], 1);
        }
    }

    // project ci out of sigma
    project_ci(sc);

    // subtract transpose of orbital response
    temp = ambit::BlockedTensor::build(tensor_type_, "Zo", {"g_,g_"});
    temp["p_,q_"] = so["p_,q_"];
    so["p_,q_"] -= temp["q_,p_"];

    so.print();
}

void CPMCSCF_SOLVER::project_ci(std::map<StateInfo, ambit::Tensor>& vecs, bool all) {
    if (all) {
        // TODO
    } else {
        auto state_ndets_map = as_solver_->state_space_size_map();
        for (const auto& [state, weights] : state_weights_map_) {
            auto ndets = state_ndets_map[state];
            auto& xc = vecs[state].data();
            for (size_t root = 0, size = weights.size(); root < size; ++root) {
                auto ci = as_solver_->ci_wfn(state, root);
                double dot = psi::C_DDOT(ndets, ci->pointer(), 1, &xc[root * ndets], 1);
                psi::C_DAXPY(ndets, -dot, ci->pointer(), 1, &xc[root * ndets], 1);
            }
        }
    }
}

void CPMCSCF_SOLVER::set_ecorr_actv(double ec_actv) { target_ec_actv_ = ec_actv; }

double CPMCSCF_SOLVER::normalize(ambit::BlockedTensor vo, std::map<StateInfo, ambit::Tensor>& vc) {
    double norm = vector_dot(vo, vc, vo, vc);
    norm = std::sqrt(norm);
    vo.scale(1.0 / norm);
    for (auto& [state, tensor] : vc) {
        tensor.scale(1.0 / norm);
    }
    return norm;
}

double CPMCSCF_SOLVER::vector_dot(const ambit::BlockedTensor& vo,
                                  const std::map<StateInfo, ambit::Tensor>& vc,
                                  const ambit::BlockedTensor& wo,
                                  const std::map<StateInfo, ambit::Tensor>& wc) {
    double out = 0.0;

    // only consider ca, cv, av blocks of the orbital part
    out += vo.block("c_,a_")("pq") * wo.block("c_,a_")("pq");
    out += vo.block("c_,v_")("pq") * wo.block("c_,v_")("pq");
    out += vo.block("a_,v_")("pq") * wo.block("a_,v_")("pq");

    for (auto& [state, tensor] : wc) {
        out += tensor("pq") * vc.at(state)("pq");
    }

    return out;
}

void CPMCSCF_SOLVER::axpy(double a, const ambit::BlockedTensor& xo,
                          const std::map<StateInfo, ambit::Tensor>& xc, ambit::BlockedTensor yo,
                          std::map<StateInfo, ambit::Tensor>& yc) {
    yo["p_,q_"] += a * xo["p_,q_"];
    for (const auto& [state, tensor] : xc) {
        yc[state]("pq") += a * tensor("pq");
    }
}

// CPMCSCF::CPMCSCF(std::shared_ptr<MCSCF_ORB_GRAD> orb_grad,
//                  std::shared_ptr<ActiveSpaceSolver> as_solver,
//                  std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo>
//                  mo_space_info)
//     : orb_grad_(orb_grad), as_solver_(as_solver), options_(options),
//     mo_space_info_(mo_space_info) {

// }

// void CPMCSCF::setup_mos() {
//     /// TODO: currently assume no frozen orbitals

//     label_to_mos_.clear();
//     label_to_cmos_.clear();

//     nirrep_ = mo_space_info_->nirrep();

//     nsopi_ = ints_->nsopi();
//     nmopi_ = mo_space_info_->dimension("ALL");
//     ndoccpi_ = mo_space_info_->dimension("INACTIVE_DOCC");
//     nactvpi_ = mo_space_info_->dimension("ACTIVE");

//     actv_mos_ = mo_space_info_->absolute_mo("ACTIVE");
//     core_mos_ = mo_space_info_->absolute_mo("INACTIVE_DOCC");

//     label_to_mos_["c"] = core_mos_;
//     label_to_mos_["a"] = actv_mos_;
//     label_to_mos_["v"] = mo_space_info_->absolute_mo("INACTIVE_UOCC");

//     // ncmopi_ = mo_space_info_->dimension("ALL");
//     // nfrzvpi_ = psi::Dimension(nirrep_);
//     // nfrzcpi_ = psi::Dimension(nirrep_);

//     // label_to_mos_["f"] = std::vector<size_t>();
//     // label_to_mos_["u"] = std::vector<size_t>();

//     // label_to_cmos_["c"] = mo_space_info_->corr_absolute_mo("INACTIVE_DOCC");
//     // label_to_cmos_["v"] = mo_space_info_->corr_absolute_mo("INACTIVE_UOCC");
//     // label_to_cmos_["a"] = mo_space_info_->corr_absolute_mo("ACTIVE");

//     nso_ = nsopi_.sum();
//     nmo_ = nmopi_.sum();
//     // ncmo_ = ncmopi_.sum();
//     nactv_ = nactvpi_.sum();
//     // nfrzc_ = nfrzcpi_.sum();

//     // in Pitzer ordering
//     mos_rel_.resize(nmo_);
//     for (int h = 0, offset = 0; h < nirrep_; ++h) {
//         for (int i = 0; i < nmopi_[h]; ++i) {
//             mos_rel_[i + offset] = std::make_pair(h, i);
//         }
//         offset += nmopi_[h];
//     }

//     // in Pitzer ordering
//     mos_rel_space_.resize(nmo_);
//     for (const std::string& space : {"c", "a", "v"}) {
//         const auto& mos = label_to_mos_[space];
//         for (size_t p = 0, size = mos.size(); p < size; ++p) {
//             mos_rel_space_[mos[p]] = std::make_pair(space, p);
//         }
//     }

//     // set up ambit spaces
//     BlockedTensor::reset_mo_spaces();
//     BlockedTensor::set_expert_mode(true);

//     // BlockedTensor::add_mo_space("f", "I,J", label_to_mos_["f"], NoSpin);
//     BlockedTensor::add_mo_space("c", "i,j", core_mos_, NoSpin);
//     BlockedTensor::add_mo_space("a", "t,u,v,w,y,x,z", actv_mos_, NoSpin);
//     BlockedTensor::add_mo_space("v", "a,b", label_to_mos_["v"], NoSpin);
//     // BlockedTensor::add_mo_space("u", "A,B", label_to_mos_["u"], NoSpin);

//     // BlockedTensor::add_composite_mo_space("F", "M,N", {"f", "u"});
//     BlockedTensor::add_composite_mo_space("g", "p,q,r,s", {"c", "a", "v"});
//     // BlockedTensor::add_composite_mo_space("G", "P,Q,R,S", {"f", "c", "a", "v", "u"});
// }

// void CPMCSCF::read_options() {
//     print_ = options_->get_int("PRINT");
//     debug_print_ = options_->get_bool("MCSCF_DEBUG_PRINTING");

//     ints_cutoff_ = options_->get_double("INTS_TOLERANCE");

//     internal_rot_ = options_->get_bool("MCSCF_INTERNAL_ROT");

//     ortho_trans_algo_ = options_->get_str("MCSCF_ORB_ORTHO_TRANS");

//     // zero rotations
//     zero_rots_.resize(nirrep_);
//     auto zero_rots = options_->get_gen_list("MCSCF_ZERO_ROT");

//     if (not zero_rots.empty()) {
//         for (size_t i = 0, npairs = zero_rots.size(); i < npairs; ++i) {
//             py::list pair = zero_rots[i];
//             if (pair.size() != 3) {
//                 outfile->Printf("\n  Error: invalid input of MCSCF_ZERO_ROT.");
//                 outfile->Printf("\n  Each entry should take an array of three numbers.");
//                 throw std::runtime_error("Invalid input of MCSCF_ZERO_ROT");
//             }

//             int irrep = py::cast<int>(pair[0]);
//             if (irrep >= nirrep_ or irrep < 0) {
//                 outfile->Printf("\n  Error: invalid irrep in MCSCF_ZERO_ROT.");
//                 outfile->Printf("\n  Check the input irrep (start from 0) not to exceed %d",
//                                 nirrep_ - 1);
//                 throw std::runtime_error("Invalid irrep in MCSCF_ZERO_ROT");
//             }

//             int i1 = py::cast<int>(pair[1]) - 1;
//             int i2 = py::cast<int>(pair[2]) - 1;
//             size_t n = nmopi_[irrep];
//             if (static_cast<size_t>(i1) >= n or i1 < 0 or static_cast<size_t>(i2) >= n or i2 < 0)
//             {
//                 outfile->Printf("\n  Error: invalid orbital indices in MCSCF_ZERO_ROT.");
//                 outfile->Printf("\n  The input orbital indices (start from 1) should not exceed "
//                                 "%zu (number of orbitals in irrep %d)",
//                                 n, irrep);
//                 throw std::runtime_error("Invalid orbital indices in MCSCF_ZERO_ROT");
//             }

//             zero_rots_[irrep][i1].emplace(i2);
//             zero_rots_[irrep][i2].emplace(i1);
//         }
//     }

//     auto frza_rot = options_->get_int_list("MCSCF_ACTIVE_FROZEN_ORBITAL");
//     auto actv_rel_mos = mo_space_info_->relative_mo("ACTIVE");
//     if (not frza_rot.empty()) {
//         for (size_t u : frza_rot) {
//             if (u >= nactv_) {
//                 outfile->Printf("\n  Error: invalid indices in MCSCF_ACTIVE_FROZEN_ORBITAL.");
//                 outfile->Printf("\n  Active orbitals include all of those in GAS1-GAS6");
//                 outfile->Printf("\n  Input indices (0 based wrt active) should not exceed %zu.",
//                                 nactv_ - 1);
//                 throw std::runtime_error("Invalid indices in MCSCF_ACTIVE_FROZEN_ORBITAL");
//             }

//             // zero between orbital u and all others
//             int irrep = actv_rel_mos[u].first;
//             auto nu = actv_rel_mos[u].second;
//             for (int p = 0; p < nmopi_[irrep]; ++p) {
//                 zero_rots_[irrep][nu].emplace(p);
//                 zero_rots_[irrep][p].emplace(nu);
//             }
//         }
//     }

//     if (debug_print_ and !zero_rots_.empty()) {
//         print_h2("Orbital Rotations Ignored (User Defined)");
//         outfile->Printf("\n    Both irrep and indices are zero-based.\n");
//         for (int h = 0; h < nirrep_; ++h) {
//             for (const auto& index_map : zero_rots_[h]) {
//                 auto p = index_map.first;
//                 for (const auto& q : index_map.second) {
//                     if (p <= q) {
//                         outfile->Printf("\n    irrep: %d, pair: (%4zu,%4zu)", h, p, q);
//                     }
//                 }
//             }
//         }
//     }
// }

// CPSCF_SOLVER::CPSCF_SOLVER(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::JK> JK,
//                            std::shared_ptr<psi::Matrix> C, std::shared_ptr<psi::Matrix> b,
//                            std::shared_ptr<psi::Vector> edocc, std::shared_ptr<psi::Vector>
//                            euocc)
//     : options_(options), cpscf_(JK, C, b, edocc, euocc) {}

// bool CPSCF_SOLVER::solve() {
//     // recast Ax = b to minimization of 0.5 * x^T A x - b^T x

//     auto lbfgs_param = std::make_shared<LBFGS_PARAM>();
//     lbfgs_param->epsilon = options_->get_double("CPSCF_CONVERGENCE");
//     lbfgs_param->maxiter = options_->get_int("CPSCF_MAXITER");
//     lbfgs_param->print = options_->get_int("PRINT");
//     lbfgs_param->max_dir = 0.2;
//     lbfgs_param->step_length_method = LBFGS_PARAM::STEP_LENGTH_METHOD::MAX_CORRECTION;

//     LBFGS lbfgs(lbfgs_param);

//     print_h2("Solving CP-SCF Equation");

//     x_ = std::make_shared<psi::Vector>("CPSCF x", cpscf_.vdimpi());
//     lbfgs.minimize(cpscf_, x_);

//     return lbfgs.converged();
// }

// std::shared_ptr<psi::Matrix> CPSCF_SOLVER::x() { return cpscf_.vec_to_mat(x_); }

// CPSCF::CPSCF(std::shared_ptr<psi::JK> JK, std::shared_ptr<psi::Matrix> C,
//              std::shared_ptr<psi::Matrix> b, std::shared_ptr<psi::Vector> edocc,
//              std::shared_ptr<psi::Vector> euocc)
//     : JK_(JK), C_(C), edocc_(edocc), euocc_(euocc) {

//     // set up basic stuff
//     nirrep_ = C->nirrep();

//     if (b->nirrep() != nirrep_)
//         throw std::runtime_error("Inconsistent nirrep for b vector");
//     if (edocc->nirrep() != nirrep_ or euocc->nirrep() != nirrep_)
//         throw std::runtime_error("Inconsistent nirrep for orbital energies vectors");

//     ndoccpi_ = edocc->dimpi();
//     nuoccpi_ = euocc->dimpi();

//     if (ndoccpi_ + nuoccpi_ != C->colspi())
//         throw std::runtime_error("Inconsistent number of MOs in C and input orbital energies");

//     Cdocc_ = std::make_shared<psi::Matrix>("C_docc (CPSCF)", C->rowspi(), ndoccpi_);
//     Cuocc_ = std::make_shared<psi::Matrix>("C_uocc (CPSCF)", C->rowspi(), nuoccpi_);
//     for (int h = 0; h < nirrep_; ++h) {
//         for (int i = 0; i < ndoccpi_[h]; ++i) {
//             Cdocc_->set_column(h, i, C->get_column(h, i));
//         }
//         for (int a = 0; a < nuoccpi_[h]; ++a) {
//             Cuocc_->set_column(h, a, C->get_column(h, a + ndoccpi_[h]));
//         }
//     }

//     std::vector<int> dims;
//     for (int h = 0; h < nirrep_; ++h) {
//         dims.push_back(ndoccpi_[h] * nuoccpi_[h]);
//     }
//     vdims_ = psi::Dimension(dims);

//     b_ = mat_to_vec(b);
// }

// double CPSCF::evaluate(std::shared_ptr<psi::Vector> x, std::shared_ptr<psi::Vector> g, bool do_g)
// {
//     auto X = vec_to_mat(x);

//     // contract X with Roothaan-Bagus supermatrix
//     // AX_{ai} = [4 * (ai|bj) - (ab|ji) - (aj|bi)] * X_{bj}
//     auto Cdressed = psi::linalg::doublet(Cuocc_, X, false, false);

//     JK_->set_do_K(true);
//     std::vector<std::shared_ptr<psi::Matrix>>& Cls = JK_->C_left();
//     std::vector<std::shared_ptr<psi::Matrix>>& Crs = JK_->C_right();
//     Cls.clear();
//     Crs.clear();
//     Cls.push_back(Cdressed);
//     Crs.push_back(Cdocc_);
//     JK_->compute();

//     auto J = JK_->J()[0];
//     J->scale(4.0);
//     J->subtract(JK_->K()[0]);
//     J->subtract((JK_->K()[0])->transpose());

//     auto AX = psi::linalg::triplet(Cuocc_, J, Cdocc_, true, false, false);

//     // add contribution of orbital energy difference: AX_{ai} += (e_a - e_i) * X_{ai}
//     for (int h = 0; h < nirrep_; ++h) {
//         for (int a = 0; a < nuoccpi_[h]; ++a) {
//             for (int i = 0; i < ndoccpi_[h]; ++i) {
//                 double value = (euocc_->get(h, a) - edocc_->get(h, i)) * X->get(h, a, i);
//                 AX->add(h, a, i, value);
//             }
//         }
//     }

//     // reshape Ax to vector
//     auto ax = mat_to_vec(AX);

//     // compute scalar: 0.5 * x^T A x - b^T x
//     double fx = 0.5 * x->vector_dot(*ax) - b_->vector_dot(*x);

//     if (do_g) {
//         g->copy(*ax);
//         g->subtract(*b_);
//     }

//     return fx;
// }

// void CPSCF::hess_diag(std::shared_ptr<psi::Vector>, std::shared_ptr<psi::Vector> h0) {
//     // just use orbital energy difference
//     for (int h = 0; h < nirrep_; ++h) {
//         for (int a = 0; a < nuoccpi_[h]; ++a) {
//             for (int i = 0; i < ndoccpi_[h]; ++i) {
//                 double value = euocc_->get(h, a) - edocc_->get(h, i);
//                 h0->set(h, a * ndoccpi_[h] + i, value);
//             }
//         }
//     }
// }

// void CPSCF::test_vec_dim(std::shared_ptr<psi::Vector> v) {
//     if (v->dimpi() != vdims_) {
//         psi::outfile->Printf("\n  Expected dimension:    ");
//         vdims_.print();
//         psi::outfile->Printf("\n  Input vector dimension:");
//         (v->dimpi()).print();
//         throw std::runtime_error("Invalid dimension of vector");
//     }
// }

// void CPSCF::test_mat_dim(std::shared_ptr<psi::Matrix> M) {
//     if (M->rowspi() != nuoccpi_ or M->colspi() != ndoccpi_) {
//         psi::outfile->Printf("\n  Expected dimension:\n");
//         psi::outfile->Printf("    row:    ");
//         nuoccpi_.print();
//         psi::outfile->Printf("    colulmn:");
//         ndoccpi_.print();
//         psi::outfile->Printf("\n  Input matrix dimension:\n");
//         psi::outfile->Printf("    row:    ");
//         (M->rowspi()).print();
//         psi::outfile->Printf("    colulmn:");
//         (M->colspi()).print();
//         throw std::runtime_error("Matrix dimension is not virtual by occupied!");
//     }
// }

// std::shared_ptr<psi::Vector> CPSCF::mat_to_vec(std::shared_ptr<psi::Matrix> M) {
//     test_mat_dim(M);

//     auto v = std::make_shared<psi::Vector>("V", vdims_);

//     for (int h = 0; h < nirrep_; ++h) {
//         for (int a = 0; a < nuoccpi_[h]; ++a) {
//             for (int i = 0; i < ndoccpi_[h]; ++i) {
//                 v->set(h, a * ndoccpi_[h] + i, M->get(h, a, i));
//             }
//         }
//     }

//     return v;
// }

// std::shared_ptr<psi::Matrix> CPSCF::vec_to_mat(std::shared_ptr<psi::Vector> v) {
//     test_vec_dim(v);

//     auto M = std::make_shared<psi::Matrix>("M", nuoccpi_, ndoccpi_);

//     for (int h = 0; h < nirrep_; ++h) {
//         for (int a = 0; a < nuoccpi_[h]; ++a) {
//             for (int i = 0; i < ndoccpi_[h]; ++i) {
//                 M->set(h, a, i, v->get(h, a * ndoccpi_[h] + i));
//             }
//         }
//     }

//     return M;
// }

} // namespace forte
