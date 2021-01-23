/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <numeric>

#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/timer.h"
#include "sadsrg.h"

using namespace psi;

namespace forte {

double SADSRG::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;
    E += 2.0 * H1["am"] * T1["ma"];

    auto temp = ambit::BlockedTensor::build(tensor_type_, "Temp110", {"aa"});
    temp["uv"] += H1["ev"] * T1["ue"];
    temp["uv"] -= H1["um"] * T1["mv"];

    E += L1_["vu"] * temp["uv"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("110", timer.get());
    return E;
}

double SADSRG::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;
    auto temp = ambit::BlockedTensor::build(tensor_type_, "Temp120", {"aaaa"});
    temp["uvxy"] += H1["ex"] * T2["uvey"];
    temp["uvxy"] -= H1["vm"] * T2["muyx"];

    E += L2_["xyuv"] * temp["uvxy"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("120", timer.get());
    return E;
}

double SADSRG::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;

    auto temp = ambit::BlockedTensor::build(tensor_type_, "Temp120", {"aaaa"});
    temp["uvxy"] += H2["evxy"] * T1["ue"];
    temp["uvxy"] -= H2["uvmy"] * T1["mx"];

    E += L2_["xyuv"] * temp["uvxy"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("210", timer.get());
    return E;
}

std::vector<double> SADSRG::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2,
                                     const double& alpha, double& C0) {
    local_timer timer;

    std::vector<double> Eout{0.0, 0.0, 0.0};
    double E = 0.0;

    // [H2, T2] (C_2)^4 from ccvv
    E += H2["efmn"] * S2["mnef"];

    // [H2, T2] (C_2)^4 L1 from cavv
    E += H2["efmu"] * S2["mvef"] * L1_["uv"];

    // [H2, T2] (C_2)^4 L1 from ccav
    E += H2["vemn"] * S2["mnue"] * Eta1_["uv"];

    Eout[0] += E;

    // other terms involving T2 with at least two active indices
    auto Esmall = H2_T2_C0_T2small(H2, T2, S2);

    for (int i = 0; i < 3; ++i) {
        E += Esmall[i];
        Eout[i] += Esmall[i];
        Eout[i] *= alpha;
    }

    // multiply prefactor and copy to C0
    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("220", timer.get());
    return Eout;
}

std::vector<double> SADSRG::H2_T2_C0_T2small(BlockedTensor& H2, BlockedTensor& T2,
                                             BlockedTensor& S2) {
    /**
     * Note the following blocks should be available in memory.
     * H2: vvaa, aacc, avca, avac, vaaa, aaca
     * T2: aavv, ccaa, caav, acav, aava, caaa
     * S2: aavv, ccaa, caav, acav, aava, caaa
     */

    double E1 = 0.0, E2 = 0.0, E3 = 0.0;

    // [H2, T2] L1 from aavv
    E1 += 0.25 * H2["efxu"] * S2["yvef"] * L1_["uv"] * L1_["xy"];

    // [H2, T2] L1 from ccaa
    E1 += 0.25 * H2["vymn"] * S2["mnux"] * Eta1_["uv"] * Eta1_["xy"];

    // [H2, T2] L1 from caav
    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp_caav", {"aaaa"});
    temp["uxyv"] += 0.5 * H2["vemx"] * S2["myue"];
    temp["uxyv"] += 0.5 * H2["vexm"] * S2["ymue"];
    E1 += temp["uxyv"] * Eta1_["uv"] * L1_["xy"];

    // [H2, T2] L1 from caaa and aaav
    temp.zero();
    temp.set_name("temp_aaav_caaa");
    temp["uxyv"] += 0.25 * H2["evwx"] * S2["zyeu"] * L1_["wz"];
    temp["uxyv"] += 0.25 * H2["vzmx"] * S2["myuw"] * Eta1_["wz"];
    E1 += temp["uxyv"] * Eta1_["uv"] * L1_["xy"];

    // <[Hbar2, T2]> C_4 (C_2)^2
    temp.zero();
    temp.set_name("temp_H2T2C0_L2");

    // HH
    temp["uvxy"] += 0.5 * H2["uvmn"] * T2["mnxy"];
    temp["uvxy"] += 0.5 * H2["uvmw"] * T2["mzxy"] * L1_["wz"];

    // PP
    temp["uvxy"] += 0.5 * H2["efxy"] * T2["uvef"];
    temp["uvxy"] += 0.5 * H2["ezxy"] * T2["uvew"] * Eta1_["wz"];

    // HP
    temp["uvxy"] += H2["uexm"] * S2["vmye"];
    temp["uvxy"] -= H2["uemx"] * T2["vmye"];
    temp["uvxy"] -= H2["vemx"] * T2["muye"];

    // HP with Gamma1
    temp["uvxy"] += 0.5 * H2["euwx"] * S2["zvey"] * L1_["wz"];
    temp["uvxy"] -= 0.5 * H2["euxw"] * T2["zvey"] * L1_["wz"];
    temp["uvxy"] -= 0.5 * H2["evxw"] * T2["uzey"] * L1_["wz"];

    // HP with Eta1
    temp["uvxy"] += 0.5 * H2["wumx"] * S2["mvzy"] * Eta1_["wz"];
    temp["uvxy"] -= 0.5 * H2["uwmx"] * T2["mvzy"] * Eta1_["wz"];
    temp["uvxy"] -= 0.5 * H2["vwmx"] * T2["muyz"] * Eta1_["wz"];

    E2 += temp["uvxy"] * L2_["uvxy"];

    // <[Hbar2, T2]> C_6 C_2
    if (do_cu3_) {
        E3 += H2.block("vaaa")("ewxy") * T2.block("aava")("uvez") * rdms_.SF_L3()("xyzuwv");
        E3 -= H2.block("aaca")("uvmz") * T2.block("caaa")("mwxy") * rdms_.SF_L3()("xyzuwv");
    }

    return {E1, E2, E3};
}

void SADSRG::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
                      BlockedTensor& C1) {
    local_timer timer;

    C1["ip"] += alpha * H1["ap"] * T1["ia"];
    C1["qa"] -= alpha * H1["qi"] * T1["ia"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("111", timer.get());
}

void SADSRG::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                      BlockedTensor& C1) {
    local_timer timer;

    C1["ia"] += 2.0 * alpha * H1["bm"] * T2["imab"];
    C1["ia"] -= alpha * H1["bm"] * T2["miab"];

    C1["ia"] += alpha * H1["bu"] * T2["ivab"] * L1_["uv"];
    C1["ia"] -= 0.5 * alpha * H1["bu"] * T2["viab"] * L1_["uv"];

    C1["ia"] -= alpha * H1["vj"] * T2["ijau"] * L1_["uv"];
    C1["ia"] += 0.5 * alpha * H1["vj"] * T2["jiau"] * L1_["uv"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("121", timer.get());
}

void SADSRG::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                      BlockedTensor& C1) {
    local_timer timer;

    C1["qp"] += 2.0 * alpha * T1["ma"] * H2["qapm"];
    C1["qp"] -= alpha * T1["ma"] * H2["aqpm"];

    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp211", {"av"});
    temp["ye"] = T1["xe"] * L1_["yx"];
    C1["qp"] += alpha * temp["ye"] * H2["qepy"];
    C1["qp"] -= 0.5 * alpha * temp["ye"] * H2["eqpy"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp211", {"ca"});
    temp["mv"] = T1["mu"] * L1_["uv"];
    C1["qp"] -= alpha * temp["mv"] * H2["qvpm"];
    C1["qp"] += 0.5 * alpha * temp["mv"] * H2["vqpm"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("211", timer.get());
}

void SADSRG::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                      BlockedTensor& C1) {
    local_timer timer;

    // [Hbar2, T2] (C_2)^3 and C_4 * C_2 2:2 -> C1 particle contractions
    C1["ir"] += alpha * H2["abrm"] * S2["imab"];

    C1["ir"] += 0.5 * alpha * L1_["uv"] * S2["ivab"] * H2["abru"];

    C1["ir"] -= 0.5 * alpha * L1_["uv"] * S2["imub"] * H2["vbrm"];
    C1["ir"] -= 0.5 * alpha * L1_["uv"] * S2["miub"] * H2["bvrm"];

    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp221G2", {"aaaa"});
    temp["uvxy"] = L2_["uvxy"];
    temp["uvxy"] += L1_["ux"] * L1_["vy"];
    C1["ir"] -= 0.5 * alpha * H2["uarx"] * T2["ivya"] * temp["xyvu"];

    temp["uvxy"] -= 0.5 * L1_["uy"] * L1_["vx"];
    C1["ir"] += 0.5 * alpha * T2["ijxy"] * H2["uvrj"] * temp["xyuv"];

    temp["xyuv"] = L2_["xyuv"];
    temp["xyuv"] -= 0.5 * L1_["xv"] * L1_["yu"];
    C1["ir"] += 0.5 * alpha * H2["aurx"] * S2["ivay"] * temp["xyuv"];
    C1["ir"] -= 0.5 * alpha * H2["uarx"] * T2["ivay"] * temp["xyuv"];

    // [Hbar2, T2] (C_2)^3 and C_4 * C_2 2:2 -> C1 hole contractions
    C1["pa"] -= alpha * H2["peij"] * S2["ijae"];

    C1["pa"] -= 0.5 * alpha * Eta1_["uv"] * S2["ijau"] * H2["pvij"];

    C1["pa"] += 0.5 * alpha * Eta1_["uv"] * S2["vjae"] * H2["peuj"];
    C1["pa"] += 0.5 * alpha * Eta1_["uv"] * S2["jvae"] * H2["peju"];

    temp["xyuv"] = L2_["xyuv"];
    temp["xyuv"] += Eta1_["xu"] * Eta1_["yv"];
    C1["pa"] += 0.5 * alpha * H2["puxi"] * T2["viay"] * temp["xyvu"];

    temp["xyuv"] -= 0.5 * Eta1_["xv"] * Eta1_["yu"];
    C1["pa"] -= 0.5 * alpha * H2["pbxy"] * T2["uvab"] * temp["xyuv"];

    temp["xyuv"] = L2_["xyuv"];
    temp["xyuv"] -= 0.5 * Eta1_["xv"] * Eta1_["yu"];
    C1["pa"] -= 0.5 * alpha * H2["puix"] * S2["ivay"] * temp["xyuv"];
    C1["pa"] += 0.5 * alpha * H2["puxi"] * T2["ivay"] * temp["xyuv"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    C1["jb"] += 0.5 * alpha * H2["avxy"] * S2["ujab"] * L2_["xyuv"];

    C1["jb"] -= 0.5 * alpha * H2["uviy"] * S2["ijxb"] * L2_["xyuv"];

    C1["qs"] += alpha * H2["eqxs"] * T2["uvey"] * L2_["xyuv"];
    C1["qs"] -= 0.5 * alpha * H2["eqsx"] * T2["uvey"] * L2_["xyuv"];

    C1["qs"] -= alpha * H2["uqms"] * T2["mvxy"] * L2_["xyuv"];
    C1["qs"] += 0.5 * alpha * H2["uqsm"] * T2["mvxy"] * L2_["xyuv"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("221", timer.get());
}

void SADSRG::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                      BlockedTensor& C2) {
    local_timer timer;

    C2["ijpb"] += alpha * T2["ijab"] * H1["ap"];
    C2["jibp"] += alpha * T2["ijab"] * H1["ap"];

    C2["qjab"] -= alpha * T2["ijab"] * H1["qi"];
    C2["jqba"] -= alpha * T2["ijab"] * H1["qi"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("122", timer.get());
}

void SADSRG::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                      BlockedTensor& C2) {
    local_timer timer;

    C2["irpq"] += alpha * T1["ia"] * H2["arpq"];
    C2["riqp"] += alpha * T1["ia"] * H2["arpq"];

    C2["rsaq"] -= alpha * T1["ia"] * H2["rsiq"];
    C2["srqa"] -= alpha * T1["ia"] * H2["rsiq"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("212", timer.get());
}

void SADSRG::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                      BlockedTensor& C2) {
    local_timer timer;

    // H2, T2, and C2 should have the 2-fold symmetry: pqrs = qpsr,
    // which must also apply to blocks, e.g., cavc exists <=> accv exists.
    std::unordered_set<std::string> C2blocks;
    for (const auto& block : C2.block_labels()) {
        C2blocks.insert(block);
    }
    bool symmetry = std::all_of(C2blocks.begin(), C2blocks.end(), [&](const std::string& block) {
        auto block_swap =
            block.substr(1, 1) + block.substr(0, 1) + block.substr(3, 1) + block.substr(2, 1);
        return C2blocks.find(block_swap) != C2blocks.end();
    });
    if (not symmetry) {
        throw psi::PSIEXCEPTION("Symmetry (pqrs = qpsr) not detected in C2.");
    }

    // particle-particle contractions
    H2_T2_C2_PP(H2, T2, alpha, C2);

    // hole-hole contractions
    H2_T2_C2_HH(H2, T2, alpha, C2);

    // hole-particle contractions
    H2_T2_C2_PH(H2, T2, S2, alpha, C2);

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("222", timer.get());
}

void SADSRG::H2_T2_C2_PP(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2) {
    C2["ijrs"] += alpha * H2["abrs"] * T2["ijab"];

    // we want to store an intermediate to prevent repeated computation for the following
    // C2["ijrs"] -= 0.5 * alpha * L1_["xy"] * T2["ijxb"] * H2["ybrs"];
    // C2["jisr"] -= 0.5 * alpha * L1_["xy"] * T2["ijxb"] * H2["ybrs"];

    // separate C2 into small blocks that fit in memory

    // 1. filter out irrelavent C2 blocks
    std::vector<std::string> C2blocks;
    std::unordered_set<std::string> C2blocks_ij;
    for (const std::string& block : C2.block_labels()) {
        if (block.substr(0, 1) != virt_label_ and block.substr(1, 1) != virt_label_) {
            C2blocks.push_back(block);
            C2blocks_ij.insert(block.substr(0, 2));
        }
    }

    // 2. figure out the blocks for intermediate X2["ijyb"] = L1_["xy"] * T2["ijxb"]
    std::unordered_map<std::string, std::vector<std::string>> blocks_ij_to_yb;
    for (const auto& block : T2.block_labels()) {
        auto ij = block.substr(0, 2);
        if (C2blocks_ij.find(ij) != C2blocks_ij.end() and block.substr(2, 1) == actv_label_) {
            blocks_ij_to_yb[ij].push_back(block.substr(2, 2));
        }
    }

    // 3. separate C2 into vectors of blocks, where each vector fits in memory
    std::vector<std::string> large_blocks;
    auto block_batches = separate_blocks(C2blocks, large_blocks, [&](const std::string& block) {
        // the intermediate when building C2["ijrs"]: X2["ijyb"] = L1_["xy"] * T2["ijxb"]
        auto ij = block.substr(0, 2);
        const auto& blocks_yb = blocks_ij_to_yb[ij];
        return std::accumulate(blocks_yb.begin(), blocks_yb.end(), 0,
                               [&](size_t x, const std::string& yb) {
                                   return x + dsrg_mem_.compute_n_elements(ij + yb);
                               });
    });

    // loop over batches of blocks that fit in memory
    for (const auto& blocks : block_batches) {
        auto temp = BlockedTensor::build(tensor_type_, "temp222PP", blocks);
        temp["ijrs"] = 0.5 * alpha * L1_["xy"] * T2["ijxb"] * H2["ybrs"];
        C2["ijrs"] -= temp["ijrs"];
        C2["jisr"] -= temp["ijrs"];
    }

    if (large_blocks.size() == 0)
        return;

    // for super large block, we batch over index i

    // 1. classify C2 large blocks based on index i
    std::unordered_map<std::string, std::vector<std::string>> C2big_i_to_jrs;
    for (const auto& block : large_blocks) {
        C2big_i_to_jrs[block.substr(0, 1)].push_back(block.substr(1, 3));
    }

    // 2. classify T2 blocks based on index i
    std::unordered_map<std::string, std::vector<std::string>> T2_i_to_jxb;
    for (const auto& ij_yb : blocks_ij_to_yb) {
        auto i = ij_yb.first.substr(0, 1);
        auto j = ij_yb.first.substr(1, 1);
        for (const auto& yb : ij_yb.second) {
            T2_i_to_jxb[i].push_back(j + yb);
        }
    }

    // 3. batching
    for (const auto& block_pair : C2big_i_to_jrs) {
        const auto& block_i = block_pair.first;

        auto T2sub = BlockedTensor::build(tensor_type_, "T2sub222PP", T2_i_to_jxb[block_i]);
        auto temp = BlockedTensor::build(tensor_type_, "temp222PP_L", block_pair.second);

        for (size_t i = 0, i_size = label_to_spacemo_[block_i[0]].size(); i < i_size; ++i) {
            // T2 slice
            fill_slice3_from_tensor4(T2, T2sub, block_i, i, T2_i_to_jxb[block_i]);

            // contraction
            temp["jrs"] = 0.5 * alpha * L1_["xy"] * T2sub["jxb"] * H2["ybrs"];

            // fill in C2
            axpy_slice3_to_tensor4_with_sym(C2, temp, -1.0, block_i, i, block_pair.second);
        }
    }
}

void SADSRG::H2_T2_C2_HH(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2) {
    C2["pqab"] += alpha * H2["pqij"] * T2["ijab"];

    // we want to store an intermediate to prevent repeated computation for the following
    // C2["pqab"] -= 0.5 * alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"];
    // C2["qpba"] -= 0.5 * alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"];

    // separate C2 into small blocks that fit in memory

    // 1. filter out irrelavent C2 blocks
    std::vector<std::string> C2blocks;
    std::unordered_set<std::string> C2blocks_ab, C2blocks_pq;
    for (const std::string& block : C2.block_labels()) {
        if (block.substr(2, 1) != core_label_ and block.substr(3, 1) != core_label_) {
            C2blocks.push_back(block);
            C2blocks_pq.insert(block.substr(0, 2));
            C2blocks_ab.insert(block.substr(2, 2));
        }
    }

    // 2. figure out the blocks for intermediate X2["xjab"] = Eta1_["xy"] * T2["yjab"]
    std::unordered_map<std::string, std::vector<std::string>> blocks_ab_to_xj;
    for (const auto& block : T2.block_labels()) {
        auto ab = block.substr(2, 2);
        if (C2blocks_ab.find(ab) != C2blocks_ab.end() and block.substr(0, 1) == actv_label_) {
            blocks_ab_to_xj[ab].push_back(block.substr(0, 2));
        }
    }

    // 3. separate C2 into vectors of blocks, where each vector fits in memory
    std::vector<std::string> large_blocks;
    auto block_batches = separate_blocks(C2blocks, large_blocks, [&](const std::string& block) {
        // the intermediate when building C2["pqab"]: X2["xjab"] = Eta1_["xy"] * T2["yjab"]
        auto ab = block.substr(2, 2);
        const auto& blocks_ab = blocks_ab_to_xj[ab];
        return std::accumulate(blocks_ab.begin(), blocks_ab.end(), 0,
                               [&](size_t x, const std::string& xj) {
                                   return x + dsrg_mem_.compute_n_elements(xj + ab);
                               });
    });

    // loop over batches of blocks that fit in memory
    for (const auto& blocks : block_batches) {
        auto temp = BlockedTensor::build(tensor_type_, "temp222HH", blocks);
        temp["pqab"] = 0.5 * alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"];
        C2["pqab"] -= temp["pqab"];
        C2["qpba"] -= temp["pqab"];
    }

    if (large_blocks.size() == 0)
        return;

    // for super large block, we batch over index p

    // 1. classify C2 large blocks based on index p
    std::unordered_map<std::string, std::vector<std::string>> C2big_p_to_qab;
    for (const auto& block : large_blocks) {
        C2big_p_to_qab[block.substr(0, 1)].push_back(block.substr(1, 3));
    }

    // 2. classify H2 blocks based on index p
    std::unordered_map<std::string, std::vector<std::string>> H2_p_to_qxj;
    for (const auto& block : H2.block_labels()) {
        if (C2blocks_pq.find(block.substr(0, 2)) != C2blocks_pq.end() and
            block.substr(2, 1) == actv_label_ and block.substr(3, 1) != virt_label_) {
            H2_p_to_qxj[block.substr(0, 1)].push_back(block.substr(1, 3));
        }
    }

    // 3. batching
    for (const auto& block_pair : C2big_p_to_qab) {
        const auto& block_p = block_pair.first;

        auto H2sub = BlockedTensor::build(tensor_type_, "H2sub222HH", H2_p_to_qxj[block_p]);
        auto temp = BlockedTensor::build(tensor_type_, "temp222HH_L", block_pair.second);

        for (size_t p = 0, p_size = label_to_spacemo_[block_p[0]].size(); p < p_size; ++p) {
            // H2 slice
            fill_slice3_from_tensor4(H2, H2sub, block_p, p, H2_p_to_qxj[block_p]);

            // contraction
            temp["qab"] = 0.5 * alpha * H2sub["qxj"] * Eta1_["xy"] * T2["yjab"];

            // fill in C2
            axpy_slice3_to_tensor4_with_sym(C2, temp, -1.0, block_p, p, block_pair.second);
        }
    }
}

void SADSRG::H2_T2_C2_PH(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2,
                         const double& alpha, BlockedTensor& C2) {
    /* we want to store two intermediates to prevent repeated computation for the following
     * temp["qjsb"] += alpha * H2["aqms"] * S2["mjab"];
     * temp["qjsb"] -= alpha * H2["aqsm"] * T2["mjab"];
     * temp["qjsb"] += 0.5 * alpha * L1_["xy"] * S2["yjab"] * H2["aqxs"];
     * temp["qjsb"] -= 0.5 * alpha * L1_["xy"] * T2["yjab"] * H2["aqsx"];
     * temp["qjsb"] -= 0.5 * alpha * L1_["xy"] * S2["ijxb"] * H2["yqis"];
     * temp["qjsb"] += 0.5 * alpha * L1_["xy"] * T2["ijxb"] * H2["yqsi"];
     * C2["qjsb"] += temp["qjsb"];
     * C2["jqbs"] += temp["qjsb"];
     *
     * temp["qjbs"] -= alpha * H2["aqsm"] * T2["mjba"];
     * temp["qjbs"] -= 0.5 * alpha * L1_["xy"] * T2["yjba"] * H2["aqsx"];
     * temp["qjbs"] += 0.5 * alpha * L1_["xy"] * T2["ijbx"] * H2["yqsi"];
     * C2["qjbs"] += temp["qjbs"];
     * C2["jqsb"] += temp["qjbs"];
     */

    // separate C2_qjsb and C2_qjbs into small blocks that fit in memory

    // 1. filter out irrelavent C2 blocks
    std::vector<std::string> C2blocks_qjsb, C2blocks_qjbs;
    std::unordered_set<std::string> C2blocks_qjsb_jb, C2blocks_qjbs_jb;
    for (const std::string& block : C2.block_labels()) {
        auto j = block.substr(1, 1);
        if (j != virt_label_) {
            if (block.substr(3, 1) != core_label_) {
                C2blocks_qjsb.push_back(block);
                C2blocks_qjsb_jb.insert(j + block.substr(3, 1));
            }
            if (block.substr(2, 1) != core_label_) {
                C2blocks_qjbs.push_back(block);
                C2blocks_qjbs_jb.insert(j + block.substr(2, 1));
            }
        }
    }

    // 2. figure out the blocks for intermediate
    // for C2_qjsb: Y2["xjab"] = L1_["xy"] * T2["yjab"]
    // for C2_qjbs: Y2["xjba"] = L1_["xy"] * T2["yjba"]
    std::unordered_map<std::string, std::vector<std::string>> blocks_xjab_jb_to_xa;
    std::unordered_map<std::string, std::vector<std::string>> blocks_xjba_jb_to_xa;
    for (const std::string& block : T2.block_labels()) {
        auto x = block.substr(0, 1);
        auto j = block.substr(1, 1);
        auto a = block.substr(2, 1);
        auto b = block.substr(3, 1);
        if (x == actv_label_) {
            if (C2blocks_qjsb_jb.find(j + b) != C2blocks_qjsb_jb.end()) {
                blocks_xjab_jb_to_xa[j + b].push_back(x + a);
            }
            if (C2blocks_qjbs_jb.find(j + a) != C2blocks_qjbs_jb.end()) {
                blocks_xjba_jb_to_xa[j + a].push_back(x + b);
            }
        }
    }

    // 3. separate C2_qjsb into vectors of blocks, where each vector fits in memory
    std::vector<std::string> large_blocks_qjsb;
    auto block_batches_qjsb =
        separate_blocks(C2blocks_qjsb, large_blocks_qjsb, [&](const std::string& block) {
            // largest intermediate forming C2["qjsb"]: Y2["xjab"] = L1_["xy"] * T2["yjab"]
            auto j = block.substr(1, 1);
            auto b = block.substr(3, 1);
            const auto& blocks_xa = blocks_xjab_jb_to_xa[j + b];
            return std::accumulate(blocks_xa.begin(), blocks_xa.end(), 0,
                                   [&](size_t t, const std::string& xa) {
                                       return t + dsrg_mem_.compute_n_elements(j + b + xa);
                                   });
        });

    // 4. separate C2_qjbs into vectors of blocks, where each vector fits in memory
    std::vector<std::string> large_blocks_qjbs;
    auto block_batches_qjbs =
        separate_blocks(C2blocks_qjbs, large_blocks_qjbs, [&](const std::string& block) {
            // largest intermediate forming C2["qjsb"]: Y2["xjba"] = L1_["xy"] * T2["yjba"]
            auto j = block.substr(1, 1);
            auto b = block.substr(2, 1);
            const auto& blocks_xa = blocks_xjba_jb_to_xa[j + b];
            return std::accumulate(blocks_xa.begin(), blocks_xa.end(), 0,
                                   [&](size_t t, const std::string& xa) {
                                       return t + dsrg_mem_.compute_n_elements(j + b + xa);
                                   });
        });

    // loop over batches of blocks that fit in memory
    for (const auto& blocks : block_batches_qjsb) {
        auto temp = BlockedTensor::build(tensor_type_, "temp222HP", blocks);
        temp["qjsb"] += alpha * H2["aqms"] * S2["mjab"];
        temp["qjsb"] -= alpha * H2["aqsm"] * T2["mjab"];
        temp["qjsb"] += 0.5 * alpha * L1_["xy"] * S2["yjab"] * H2["aqxs"];
        temp["qjsb"] -= 0.5 * alpha * L1_["xy"] * T2["yjab"] * H2["aqsx"];
        temp["qjsb"] -= 0.5 * alpha * L1_["xy"] * S2["ijxb"] * H2["yqis"];
        temp["qjsb"] += 0.5 * alpha * L1_["xy"] * T2["ijxb"] * H2["yqsi"];
        C2["qjsb"] += temp["qjsb"];
        C2["jqbs"] += temp["qjsb"];
    }

    for (const auto& blocks : block_batches_qjbs) {
        auto temp = BlockedTensor::build(tensor_type_, "temp222HP", blocks);
        temp["qjbs"] -= alpha * H2["aqsm"] * T2["mjba"];
        temp["qjbs"] -= 0.5 * alpha * L1_["xy"] * T2["yjba"] * H2["aqsx"];
        temp["qjbs"] += 0.5 * alpha * L1_["xy"] * T2["ijbx"] * H2["yqsi"];
        C2["qjbs"] += temp["qjbs"];
        C2["jqsb"] += temp["qjbs"];
    }

    if (large_blocks_qjbs.size() == 0 and large_blocks_qjsb.size())
        return;

    /* For super large blocks, we consider S2- and T2-related terms differently.
     * For S2 related terms, we batch over index j and consider C2_jqbs instead of C2_qjsb.
     * For T2 related terms, we batch over index q.
     */

    // S2 related terms

    // 1. classify C2_jqbs large blocks based on index j
    std::unordered_map<std::string, std::vector<std::string>> C2big_j_to_qbs;
    for (const auto& block : large_blocks_qjsb) {
        auto q = block.substr(0, 1);
        auto j = block.substr(1, 1);
        auto s = block.substr(2, 1);
        auto b = block.substr(3, 1);
        C2big_j_to_qbs[j].push_back(q + b + s);
    }

    // 2. classify S2 blocks based on index j
    std::unordered_map<std::string, std::vector<std::string>> S2_j_to_iba;
    for (const auto& block : S2.block_labels()) {
        auto j = block.substr(0, 1);
        auto b = block.substr(2, 1);
        if (C2blocks_qjsb_jb.find(j + b) != C2blocks_qjsb_jb.end()) {
            S2_j_to_iba[j].push_back(block.substr(1, 3));
        }
    }

    // 3. batching
    for (const auto& block_pair : C2big_j_to_qbs) {
        const auto& block_j = block_pair.first;

        auto S2sub = BlockedTensor::build(tensor_type_, "T2sub222PP", S2_j_to_iba[block_j]);
        auto temp = BlockedTensor::build(tensor_type_, "temp222PP_L", block_pair.second);

        for (size_t j = 0, j_size = label_to_spacemo_[block_j[0]].size(); j < j_size; ++j) {
            // T2 slice
            fill_slice3_from_tensor4(S2, S2sub, block_j, j, S2_j_to_iba[block_j]);

            // contraction
            temp["qbs"] = alpha * H2["aqms"] * S2sub["mba"];
            temp["qbs"] += 0.5 * alpha * L1_["xy"] * S2sub["yba"] * H2["aqxs"];
            temp["qbs"] -= 0.5 * alpha * L1_["xy"] * S2sub["ibx"] * H2["yqis"];

            // fill in C2
            axpy_slice3_to_tensor4_with_sym(C2, temp, 1.0, block_j, j, block_pair.second);
        }
    }

    /* For T2 related terms, we classify C2_qjsb and C2_qjbs into three parts:
     * 1) blocks both belong to qjsb and qjbs
     * 2) blocks uniquely belong to qjsb
     * 3) blocks uniquely belong to qjbs
     * This is done to minimize the times to build intermediates:
     * X2sub["(q)ays"] = H2sub["(q)axs"] * L1_["xy"]
     * and Y2sub["(q)xis"] = H2sub["(q)yis"] * L1_["xy"]
     */

    // figure out blocks intersection and difference
    std::sort(large_blocks_qjbs.begin(), large_blocks_qjbs.end());
    std::sort(large_blocks_qjsb.begin(), large_blocks_qjsb.end());
    std::vector<std::string> large_blocks_common(36), large_blocks_sb(36), large_blocks_bs(36);

    auto it_intersect = std::set_intersection(large_blocks_qjbs.begin(), large_blocks_qjbs.end(),
                                              large_blocks_qjsb.begin(), large_blocks_qjsb.end(),
                                              large_blocks_common.begin());
    large_blocks_common.resize(it_intersect - large_blocks_common.begin());

    auto it_diff_bs = std::set_difference(large_blocks_qjbs.begin(), large_blocks_qjbs.end(),
                                          large_blocks_common.begin(), large_blocks_common.end(),
                                          large_blocks_bs.begin());
    large_blocks_bs.resize(it_diff_bs - large_blocks_bs.begin());

    auto it_diff_sb = std::set_difference(large_blocks_qjsb.begin(), large_blocks_qjsb.end(),
                                          large_blocks_common.begin(), large_blocks_common.end(),
                                          large_blocks_sb.begin());
    large_blocks_sb.resize(it_diff_sb - large_blocks_sb.begin());

    // blocks that are in common

    // 1. classify C2 large blocks based on index q
    std::unordered_map<std::string, std::vector<std::string>> C2big_q_to_jcb;
    std::unordered_set<std::string> C2blocks_qs;
    for (const auto& block : large_blocks_common) {
        auto q = block.substr(0, 1);
        C2big_q_to_jcb[q].push_back(block.substr(1, 3));
        C2blocks_qs.insert(q + block.substr(2, 1));
        C2blocks_qs.insert(q + block.substr(3, 1));
    }

    // 2. classify H2 large blocks based on index q
    auto form_H2_q_to_ais = [&]() {
        std::unordered_map<std::string, std::vector<std::string>> q_to_ais;
        for (const auto& block : H2.block_labels()) {
            auto q = block.substr(0, 1);
            if (C2blocks_qs.find(q + block.substr(3, 1)) != C2blocks_qs.end()) {
                q_to_ais[q].push_back(block.substr(1, 3));
            }
        }
        return q_to_ais;
    };
    auto H2_q_to_ais = form_H2_q_to_ais();

    // 3. batching
    for (const auto& block_pair : C2big_q_to_jcb) {
        const auto& block_q = block_pair.first;

        auto H2sub = BlockedTensor::build(tensor_type_, "H2sub222HP", H2_q_to_ais[block_q]);
        auto X2sub = BlockedTensor::build(tensor_type_, "X2sub222HP", H2_q_to_ais[block_q]);
        auto temp = BlockedTensor::build(tensor_type_, "temp222HP_L", block_pair.second);

        for (size_t q = 0, q_size = label_to_spacemo_[block_q[0]].size(); q < q_size; ++q) {
            // H2 slice
            fill_slice3_from_tensor4(H2, H2sub, block_q, q, H2_q_to_ais[block_q]);

            // contraction
            temp["jsb"] = -1.0 * alpha * H2sub["ams"] * T2["mjab"];
            temp["jbs"] -= alpha * H2sub["ams"] * T2["mjba"];

            X2sub["ays"] = H2sub["axs"] * L1_["xy"];
            temp["jsb"] -= 0.5 * alpha * T2["yjab"] * X2sub["ays"];
            temp["jbs"] -= 0.5 * alpha * T2["yjba"] * X2sub["ays"];

            X2sub["xis"] = H2sub["yis"] * L1_["xy"];
            temp["jsb"] += 0.5 * alpha * T2["ijxb"] * X2sub["xis"];
            temp["jbs"] += 0.5 * alpha * T2["ijbx"] * X2sub["xis"];

            // fill in C2
            axpy_slice3_to_tensor4_with_sym(C2, temp, 1.0, block_q, q, block_pair.second);
        }
    }

    // blocks that are unique in qjsb

    // 1. classify C2_qjsb large blocks based on index q
    std::unordered_map<std::string, std::vector<std::string>> C2big_q_to_jsb;
    C2blocks_qs.clear();
    for (const auto& block : large_blocks_sb) {
        auto q = block.substr(0, 1);
        C2big_q_to_jsb[q].push_back(block.substr(1, 3));
        C2blocks_qs.insert(q + block.substr(2, 1));
    }

    // 2. classify H2 large blocks based on index q
    H2_q_to_ais = form_H2_q_to_ais();

    // 3. batching
    for (const auto& block_pair : C2big_q_to_jsb) {
        const auto& block_q = block_pair.first;

        auto H2sub = BlockedTensor::build(tensor_type_, "H2sub222HP", H2_q_to_ais[block_q]);
        auto temp = BlockedTensor::build(tensor_type_, "temp222HP_L", block_pair.second);

        for (size_t q = 0, q_size = label_to_spacemo_[block_q[0]].size(); q < q_size; ++q) {
            // H2 slice
            fill_slice3_from_tensor4(H2, H2sub, block_q, q, H2_q_to_ais[block_q]);

            // contraction
            temp["jsb"] = -1.0 * alpha * H2sub["ams"] * T2["mjab"];
            temp["jsb"] -= 0.5 * alpha * T2["yjab"] * H2sub["axs"] * L1_["xy"];
            temp["jsb"] += 0.5 * alpha * T2["ijxb"] * H2sub["yis"] * L1_["xy"];

            // fill in C2
            axpy_slice3_to_tensor4_with_sym(C2, temp, 1.0, block_q, q, block_pair.second);
        }
    }

    // blocks that are unique in qjbs

    // 1. classify C2_qjbs large blocks based on index q
    std::unordered_map<std::string, std::vector<std::string>> C2big_q_to_jbs;
    C2blocks_qs.clear();
    for (const auto& block : large_blocks_bs) {
        auto q = block.substr(0, 1);
        C2big_q_to_jbs[q].push_back(block.substr(1, 3));
        C2blocks_qs.insert(q + block.substr(3, 1));
    }

    // 2. classify H2 large blocks based on index q
    H2_q_to_ais = form_H2_q_to_ais();

    // 3. batching
    for (const auto& block_pair : C2big_q_to_jbs) {
        const auto& block_q = block_pair.first;

        auto H2sub = BlockedTensor::build(tensor_type_, "H2sub222HP", H2_q_to_ais[block_q]);
        auto temp = BlockedTensor::build(tensor_type_, "temp222HP_L", block_pair.second);

        for (size_t q = 0, q_size = label_to_spacemo_[block_q[0]].size(); q < q_size; ++q) {
            // H2 slice
            fill_slice3_from_tensor4(H2, H2sub, block_q, q, H2_q_to_ais[block_q]);

            // contraction
            temp["jbs"] = -1.0 * alpha * H2sub["ams"] * T2["mjba"];
            temp["jbs"] -= 0.5 * alpha * H2sub["axs"] * L1_["xy"] * T2["yjba"];
            temp["jbs"] += 0.5 * alpha * H2sub["yis"] * L1_["xy"] * T2["ijbx"];

            // fill in C2
            axpy_slice3_to_tensor4_with_sym(C2, temp, 1.0, block_q, q, block_pair.second);
        }
    }
}

void SADSRG::V_T1_C0_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;

    auto temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp120", {"Laa"});
    temp["gux"] += B["gex"] * T1["ue"];
    temp["gux"] -= B["gum"] * T1["mx"];

    E += L2_["xyuv"] * temp["gux"] * B["gvy"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("210", timer.get());
}

std::vector<double> SADSRG::V_T2_C0_DF(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& S2,
                                       const double& alpha, double& C0) {
    local_timer timer;

    std::vector<double> Eout{0.0, 0.0, 0.0};
    double E = 0.0;

    // [H2, T2] (C_2)^4 from ccvv, cavv, and ccav
    auto temp = ambit::BlockedTensor::build(tensor_type_, "temp_220", {"Lvc"});
    temp["gem"] += B["gfn"] * S2["mnef"];
    temp["gem"] += B["gfu"] * S2["mvef"] * L1_["uv"];
    temp["gem"] += B["gvn"] * S2["nmue"] * Eta1_["uv"];
    E += temp["gem"] * B["gem"];
    Eout[0] += E;

    // form H2 for other blocks that fits memory
    std::vector<std::string> blocks{"aacc", "aaca", "vvaa", "vaaa", "avac", "avca"};
    auto H2 = ambit::BlockedTensor::build(tensor_type_, "temp_H2", blocks);
    H2["abij"] = B["gai"] * B["gbj"];

    auto Esmall = H2_T2_C0_T2small(H2, T2, S2);

    for (int i = 0; i < 3; ++i) {
        Eout[i] += Esmall[i];
    }
    E += Esmall[0] + Esmall[1] + Esmall[2];

    // multiply prefactor and copy to C0
    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("220", timer.get());
    return Eout;
}

void SADSRG::V_T1_C1_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha,
                        BlockedTensor& C1) {
    local_timer timer;

    auto temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp211", {"L"});
    temp["g"] += 2.0 * alpha * T1["ma"] * B["gam"];
    temp["g"] += alpha * T1["xe"] * L1_["yx"] * B["gey"];
    temp["g"] -= alpha * T1["mu"] * L1_["uv"] * B["gvm"];
    C1["qp"] += temp["g"] * B["gqp"];

    temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp211", {"Lgc"});
    temp["gpm"] -= alpha * T1["ma"] * B["gap"];
    temp["gpm"] += 0.5 * alpha * T1["mu"] * L1_["uv"] * B["gvp"];
    C1["qp"] += temp["gpm"] * B["gqm"];

    C1["qp"] -= 0.5 * alpha * T1["xe"] * L1_["yx"] * B["gep"] * B["gqy"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("211", timer.get());
}

void SADSRG::V_T2_C1_DF(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                        BlockedTensor& C1) {
    local_timer timer;

    // [Hbar2, T2] (C_2)^3 and C_4 * C_2 2:2 -> C1 particle contractions
    auto temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp221", {"Lhp"});

    temp["gia"] += alpha * B["gbm"] * S2["imab"];

    temp["gia"] += 0.5 * alpha * L1_["uv"] * S2["ivab"] * B["gbu"];

    temp["giv"] -= 0.5 * alpha * L1_["uv"] * S2["imub"] * B["gbm"];
    temp["gia"] -= 0.5 * alpha * L1_["uv"] * S2["miua"] * B["gvm"];

    auto G2 = ambit::BlockedTensor::build(tensor_type_, "DFtemp221G2", {"aaaa"});
    G2["xyuv"] = L2_["xyuv"];
    G2["xyuv"] += L1_["xu"] * L1_["yv"];
    temp["giu"] -= 0.5 * alpha * B["gax"] * T2["ivya"] * G2["xyvu"];

    G2["xyuv"] -= 0.5 * L1_["xv"] * L1_["yu"];
    temp["giu"] += 0.5 * alpha * T2["ijxy"] * G2["xyuv"] * B["gvj"];

    G2["xyuv"] = L2_["xyuv"];
    G2["xyuv"] -= 0.5 * L1_["xv"] * L1_["yu"];
    temp["gia"] += 0.5 * alpha * B["gux"] * S2["ivay"] * G2["xyuv"];
    temp["giu"] -= 0.5 * alpha * B["gax"] * T2["ivay"] * G2["xyuv"];

    C1["ir"] += temp["gia"] * B["gar"];

    // [Hbar2, T2] (C_2)^3 and C_4 * C_2 2:2 -> C1 hole contractions
    temp.zero();

    temp["gia"] -= alpha * B["gej"] * S2["ijae"];

    temp["gia"] -= 0.5 * alpha * Eta1_["uv"] * S2["ijau"] * B["gvj"];

    temp["gua"] += 0.5 * alpha * Eta1_["uv"] * S2["vjae"] * B["gej"];
    temp["gia"] += 0.5 * alpha * Eta1_["uv"] * S2["ivae"] * B["geu"];

    G2["xyuv"] = L2_["xyuv"];
    G2["xyuv"] += Eta1_["xu"] * Eta1_["yv"];
    temp["gxa"] += 0.5 * alpha * B["gui"] * T2["viay"] * G2["xyvu"];

    G2["xyuv"] -= 0.5 * Eta1_["xv"] * Eta1_["yu"];
    temp["gxa"] -= 0.5 * alpha * G2["xyuv"] * T2["uvab"] * B["gby"];

    G2["xyuv"] = L2_["xyuv"];
    G2["xyuv"] -= 0.5 * Eta1_["xv"] * Eta1_["yu"];
    temp["gia"] -= 0.5 * alpha * B["gux"] * S2["ivay"] * G2["xyuv"];
    temp["gxa"] += 0.5 * alpha * B["gui"] * T2["ivay"] * G2["xyuv"];

    C1["pa"] += temp["gia"] * B["gpi"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp221", {"Laa"});
    temp["gxu"] = B["gvy"] * L2_["xyuv"];
    C1["jb"] += 0.5 * alpha * B["gax"] * S2["ujab"] * temp["gxu"];
    C1["jb"] -= 0.5 * alpha * B["gui"] * S2["ijxb"] * temp["gxu"];

    temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp221", {"L"});
    temp["g"] += alpha * B["gex"] * T2["uvey"] * L2_["xyuv"];
    temp["g"] -= alpha * B["gum"] * T2["mvxy"] * L2_["xyuv"];
    C1["qs"] += temp["g"] * B["gqs"];

    C1["qs"] -= 0.5 * alpha * B["ges"] * B["gqx"] * T2["uvey"] * L2_["xyuv"];

    C1["qs"] += 0.5 * alpha * B["gus"] * B["gqm"] * T2["mvxy"] * L2_["xyuv"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("221", timer.get());
}

void SADSRG::V_T1_C2_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha,
                        BlockedTensor& C2) {
    local_timer timer;

    C2["irpq"] += alpha * T1["ia"] * B["gap"] * B["grq"];
    C2["riqp"] += alpha * T1["ia"] * B["gap"] * B["grq"];
    C2["rsaq"] -= alpha * T1["ia"] * B["gri"] * B["gsq"];
    C2["srqa"] -= alpha * T1["ia"] * B["gri"] * B["gsq"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("212", timer.get());
}

void SADSRG::V_T2_C2_DF(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                        BlockedTensor& C2) {
    local_timer timer;

    // H2, T2, and C2 should have the 2-fold symmetry: pqrs = qpsr,
    // which must also apply to blocks, e.g., cavc exists <=> accv exists.
    std::unordered_set<std::string> C2blocks;
    for (const auto& block : C2.block_labels()) {
        C2blocks.insert(block);
    }
    bool symmetry = std::all_of(C2blocks.begin(), C2blocks.end(), [&](const std::string& block) {
        auto block_swap =
            block.substr(1, 1) + block.substr(0, 1) + block.substr(3, 1) + block.substr(2, 1);
        return C2blocks.find(block_swap) != C2blocks.end();
    });
    if (not symmetry) {
        throw psi::PSIEXCEPTION("Symmetry (pqrs = qpsr) not detected in C2.");
    }

    // particle-particle contractions
    H2_T2_C2_PP_DF(B, T2, alpha, C2);
//    C2["ijrs"] += batched("r", alpha * B["gar"] * B["gbs"] * T2["ijab"]);

//    C2["ijrs"] -= batched("r", 0.5 * alpha * B["gyr"] * B["gbs"] * L1_["xy"] * T2["ijxb"]);
//    C2["ijrs"] -= batched("s", 0.5 * alpha * B["gys"] * B["gbr"] * L1_["xy"] * T2["jixb"]);

    //    std::vector<std::string> C2blocks;
    //    for (const std::string& block : C2.block_labels()) {
    //        if (block[0] == virt_label_[0] or block[1] == virt_label_[0])
    //            continue;
    //        C2blocks.push_back(block);
    //    }

    //    auto temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222", C2blocks);
    //    temp["ijrs"] += batched("r", L1_["xy"] * T2["ijxb"] * B["gyr"] * B["gbs"]);

    //    C2["ijrs"] -= 0.5 * alpha * temp["ijrs"];
    //    C2["jisr"] -= 0.5 * alpha * temp["ijrs"];

    // hole-hole contractions
    H2_T2_C2_HH_DF(B, T2, alpha, C2);
//    C2["pqab"] += batched("p", alpha * B["gpi"] * B["gqj"] * T2["ijab"]);

//    C2["pqab"] -= batched("p", 0.5 * alpha * B["gpx"] * B["gqj"] * Eta1_["xy"] * T2["yjab"]);
//    C2["pqab"] -= batched("q", 0.5 * alpha * B["gqx"] * B["gpj"] * Eta1_["xy"] * T2["yjba"]);

    //    std::vector<std::string> Vblocks;
    //    for (const std::string& block : C2.block_labels()) {
    //        if (block[2] == core_label_[0] or block[3] == core_label_[0])
    //            continue;
    //        std::string s = block.substr(0, 2) + "hh";
    //        if (std::find(Vblocks.begin(), Vblocks.end(), s) == Vblocks.end())
    //            Vblocks.push_back(s);
    //    }

    //    temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222", Vblocks);
    //    temp["pqij"] = B["gpi"] * B["gqj"];

    //    C2["pqab"] += alpha * temp["pqij"] * T2["ijab"];

    //    C2["pqab"] -= 0.5 * alpha * Eta1_["xy"] * T2["yjab"] * temp["pqxj"];
    //    C2["qpba"] -= 0.5 * alpha * Eta1_["xy"] * T2["yjab"] * temp["pqxj"];

    // hole-particle contractions
    auto temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222", {"Lhp"});
    temp["gjb"] += alpha * B["gam"] * S2["mjab"];
    temp["gjb"] += 0.5 * alpha * L1_["xy"] * S2["yjab"] * B["gax"];
    temp["gjb"] -= 0.5 * alpha * L1_["xy"] * S2["ijxb"] * B["gyi"];

    C2["qjsb"] += temp["gjb"] * B["gqs"];
    C2["jqbs"] += temp["gjb"] * B["gqs"];

    for (const std::string& block : temp.block_labels()) {
        temp.block(block).reset();
    }

    // exchange like terms
    V_T2_C2_DF_PH_X(B, T2, alpha, C2);

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("222", timer.get());
}

void SADSRG::H2_T2_C2_PP_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                            BlockedTensor& C2) {
    C2["ijrs"] += batched("r", alpha * B["gar"] * B["gbs"] * T2["ijab"]);

    // separate C2 into small blocks that fit in memory

    // 1. filter out irrelavent C2 blocks
    std::vector<std::string> C2blocks;
    std::unordered_set<std::string> C2blocks_ij;
    for (const std::string& block : C2.block_labels()) {
        if (block.substr(0, 1) != virt_label_ and block.substr(1, 1) != virt_label_) {
            C2blocks.push_back(block);
            C2blocks_ij.insert(block.substr(0, 2));
        }
    }

    // 2. figure out block labels for index b for given ij
    std::unordered_map<std::string, std::vector<std::string>> T2blocks_ij_to_b;
    for (const auto& block : T2.block_labels()) {
        auto ij = block.substr(0, 2);
        if (C2blocks_ij.find(ij) != C2blocks_ij.end() and block.substr(2, 1) == actv_label_) {
            T2blocks_ij_to_b[ij].push_back(block.substr(3, 1));
        }
    }

    // 3. separate C2 into vectors of blocks, where each vector fits in memory
    std::vector<std::string> large_blocks;
    auto block_batches = separate_blocks(C2blocks, large_blocks, [&](const std::string& block) {
        // the intermediate when building C2["ijrs"]: V2["(r)ybs"] = B["gy(r)"] * B["gbs"]
        auto s = block.substr(3, 1);
        const auto& blocks_b = T2blocks_ij_to_b[block.substr(0, 2)];
        return std::accumulate(blocks_b.begin(), blocks_b.end(), 0,
                               [&](size_t x, const std::string& b) {
                               return x + dsrg_mem_.compute_n_elements(s + b + "a");
                               });
    });

    // loop over batches of blocks that fit in memory
    for (const auto& blocks : block_batches) {
        auto temp = BlockedTensor::build(tensor_type_, "temp222PP", blocks);
        temp["ijrs"] += batched("r", 0.5 * alpha * B["gyr"] * B["gbs"] * L1_["xy"] * T2["ijxb"]);
        C2["ijrs"] -= temp["ijrs"];
        C2["jisr"] -= temp["ijrs"];
    }

    // for super large blocks, we batch over index r

    // 1. classify C2 large blocks based on index r
    std::unordered_map<std::string, std::vector<std::string>> C2big_r_to_ijs;
    for (const auto& block : large_blocks) {
        C2big_r_to_ijs[block.substr(2, 1)].push_back(block.substr(0, 2) + block.substr(3, 1));
    }

    size_t actv_size = actv_mos_.size();

    for (const auto& block_pair : C2big_r_to_ijs) {
        const auto& block_r = block_pair.first;

        auto& Bdata = B.block("La" + block_r).data();
        auto Bsub = BlockedTensor::build(tensor_type_, "Bsub222PPDF", {"La"});
        auto temp = BlockedTensor::build(tensor_type_, "temp222PPDF", block_pair.second);

        for (size_t r = 0, r_size = label_to_spacemo_[block_r[0]].size(); r < r_size; ++r) {
            // B slice
            Bsub.block("La").iterate([&](const std::vector<size_t> idx, double& value){
                value = Bdata[idx[0] * r_size * actv_size + idx[1] * r_size + r];
            });

            // contraction
            temp["ijs"] = 0.5 * alpha * L1_["xy"] * Bsub["gy"] * B["gbs"] * T2["ijxb"];

            // add to C2
            for (const auto& block_ijs : block_pair.second) {
                auto block_i = block_ijs.substr(0, 1);
                auto block_j = block_ijs.substr(1, 1);
                auto block_s = block_ijs.substr(2, 1);

                auto i_size = label_to_spacemo_[block_ijs[0]].size();
                auto j_size = label_to_spacemo_[block_ijs[1]].size();
                auto s_size = label_to_spacemo_[block_ijs[2]].size();

                auto rs_size = r_size * s_size;

                // C2["ij(r)s"] -= temp["ijs"];
                auto jrs_size = j_size * rs_size;
                auto& Cijrs_data = C2.block(block_i + block_j + block_r + block_s).data();
                temp.block(block_ijs).citerate([&](const std::vector<size_t>& id, const double& value){
                    Cijrs_data[id[0] * jrs_size + id[1] * rs_size + r * s_size + id[2]] -= value;
                });

                // C2["jis(r)"] -= temp["ijs"];
                auto irs_size = i_size * rs_size;
                auto& Cjisr_data = C2.block(block_j + block_i + block_s + block_r).data();
                temp.block(block_ijs).citerate([&](const std::vector<size_t>& id, const double& value){
                    Cjisr_data[id[1] * irs_size + id[0] * rs_size + id[2] * r_size + r] -= value;
                });
            }
        }
    }
}

void SADSRG::H2_T2_C2_HH_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                            BlockedTensor& C2) {
    C2["pqab"] += batched("p", alpha * B["gpi"] * B["gqj"] * T2["ijab"]);

    C2["pqab"] -= batched("p", 0.5 * alpha * B["gpx"] * B["gqj"] * Eta1_["xy"] * T2["yjab"]);
    C2["pqab"] -= batched("q", 0.5 * alpha * B["gqx"] * B["gpj"] * Eta1_["xy"] * T2["yjba"]);
}

void SADSRG::V_T2_C2_DF_PH_X(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                             BlockedTensor& C2) {
    C2["qjsb"] -= batched("q", alpha * B["gas"] * B["gqm"] * T2["mjab"]);
    C2["qjsb"] -= batched("q", 0.5 * alpha * L1_["xy"] * T2["yjab"] * B["gas"] * B["gqx"]);
    C2["qjsb"] += batched("q", 0.5 * alpha * L1_["xy"] * T2["ijxb"] * B["gys"] * B["gqi"]);

    C2["jqbs"] -= batched("q", alpha * B["gas"] * B["gqm"] * T2["mjab"]);
    C2["jqbs"] -= batched("q", 0.5 * alpha * L1_["xy"] * T2["yjab"] * B["gas"] * B["gqx"]);
    C2["jqbs"] += batched("q", 0.5 * alpha * L1_["xy"] * T2["ijxb"] * B["gys"] * B["gqi"]);

    C2["jqsb"] -= batched("q", alpha * B["gas"] * B["gqm"] * T2["mjba"]);
    C2["jqsb"] -= batched("q", 0.5 * alpha * L1_["xy"] * T2["yjba"] * B["gas"] * B["gqx"]);
    C2["jqsb"] += batched("q", 0.5 * alpha * L1_["xy"] * T2["ijbx"] * B["gys"] * B["gqi"]);

    C2["qjbs"] -= batched("q", alpha * B["gas"] * B["gqm"] * T2["mjba"]);
    C2["qjbs"] -= batched("q", 0.5 * alpha * L1_["xy"] * T2["yjba"] * B["gas"] * B["gqx"]);
    C2["qjbs"] += batched("q", 0.5 * alpha * L1_["xy"] * T2["ijbx"] * B["gys"] * B["gqi"]);

    //    std::vector<std::string> qjsb_small, qjsb_large, jqsb_small, jqsb_large;

    //    for (const std::string& block : C2.block_labels()) {
    //        auto j = block.substr(1, 1);
    //        auto b = block.substr(3, 1);

    //        if (j != virt_label_ and b != core_label_) {
    //            if (std::count(block.begin(), block.end(), 'v') > 2) {
    //                qjsb_large.push_back(block);
    //            } else {
    //                qjsb_small.push_back(block);
    //            }
    //        }

    //        j = block.substr(0, 1);
    //        if (j != virt_label_ and b != core_label_) {
    //            if (std::count(block.begin(), block.end(), 'v') > 2) {
    //                jqsb_large.push_back(block);
    //            } else {
    //                jqsb_small.push_back(block);
    //            }
    //        }
    //    }

    //    auto temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", qjsb_small);
    //    temp["qjsb"] -= alpha * B["gas"] * B["gqm"] * T2["mjab"];
    //    temp["qjsb"] -= 0.5 * alpha * L1_["xy"] * T2["yjab"] * B["gas"] * B["gqx"];
    //    temp["qjsb"] += 0.5 * alpha * L1_["xy"] * T2["ijxb"] * B["gys"] * B["gqi"];

    //    C2["qjsb"] += temp["qjsb"];
    //    C2["jqbs"] += temp["qjsb"];

    //    temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", jqsb_small);
    //    temp["jqsb"] -= alpha * B["gas"] * B["gqm"] * T2["mjba"];
    //    temp["jqsb"] -= 0.5 * alpha * L1_["xy"] * T2["yjba"] * B["gas"] * B["gqx"];
    //    temp["jqsb"] += 0.5 * alpha * L1_["xy"] * T2["ijbx"] * B["gys"] * B["gqi"];

    //    C2["jqsb"] += temp["jqsb"];
    //    C2["qjbs"] += temp["jqsb"];

    //    if (qjsb_large.size() != 0) {
    //        C2["e,j,f,v0"] -= batched("e", alpha * B["g,a,f"] * B["g,e,m"] * T2["m,j,a,v0"]);
    //        C2["j,e,v0,f"] -= batched("e", alpha * B["g,a,f"] * B["g,e,m"] * T2["m,j,a,v0"]);

    //        temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", {"ahpv"});
    //        temp["xjae"] = L1_["xy"] * T2["yjae"];
    //        C2["e,j,f,v0"] -= batched("e", 0.5 * alpha * temp["x,j,a,v0"] * B["g,a,f"] *
    //        B["g,e,x"]); C2["j,e,v0,f"] -= batched("e", 0.5 * alpha * temp["x,j,a,v0"] *
    //        B["g,a,f"] * B["g,e,x"]);

    //        temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", {"hhav"});
    //        temp["ijye"] = L1_["xy"] * T2["ijxe"];
    //        C2["e,j,f,v0"] += batched("e", 0.5 * alpha * temp["i,j,y,v0"] * B["g,y,f"] *
    //        B["g,e,i"]); C2["j,e,v0,f"] += batched("e", 0.5 * alpha * temp["i,j,y,v0"] *
    //        B["g,y,f"] * B["g,e,i"]);
    //    }

    //    if (jqsb_large.size() != 0) {
    //        C2["j,e,f,v0"] -= batched("e", alpha * B["g,a,f"] * B["g,e,m"] * T2["m,j,v0,a"]);
    //        C2["e,j,v0,f"] -= batched("e", alpha * B["g,a,f"] * B["g,e,m"] * T2["m,j,v0,a"]);

    //        temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", {"ahvp"});
    //        temp["xjea"] = L1_["xy"] * T2["yjea"];
    //        C2["j,e,f,v0"] -= batched("e", 0.5 * alpha * temp["x,j,v0,a"] * B["g,a,f"] *
    //        B["g,e,x"]); C2["e,j,v0,f"] -= batched("e", 0.5 * alpha * temp["x,j,v0,a"] *
    //        B["g,a,f"] * B["g,e,x"]);

    //        temp = ambit::BlockedTensor::build(tensor_type_, "DFtemp222PHX", {"hhva"});
    //        temp["ijey"] = L1_["xy"] * T2["ijex"];
    //        C2["j,e,f,v0"] += batched("e", 0.5 * alpha * temp["i,j,v0,y"] * B["g,y,f"] *
    //        B["g,e,i"]); C2["e,j,v0,f"] += batched("e", 0.5 * alpha * temp["i,j,v0,y"] *
    //        B["g,y,f"] * B["g,e,i"]);
    //    }
}

void SADSRG::H_A_Ca(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                    BlockedTensor& S2, const double& alpha, BlockedTensor& C1, BlockedTensor& C2) {
    // set up G2["pqrs"] = 2 * H2["pqrs"] - H2["pqsr"]
    auto G2 = ambit::BlockedTensor::build(tensor_type_, "G2H", {"avac", "aaac", "avaa"});
    G2["pqrs"] = 2.0 * H2["pqrs"] - H2["pqsr"];

    H_A_Ca_small(H1, H2, G2, T1, T2, S2, alpha, C1, C2);

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "tempHACa", {"aa"});
    temp["wz"] += H2["efzm"] * S2["wmef"];
    temp["wz"] -= H2["wemn"] * S2["mnze"];

    C1["uv"] += alpha * temp["uv"];
    C1["vu"] += alpha * temp["uv"];
}

void SADSRG::H_A_Ca_small(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& G2,
                          BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& S2,
                          const double& alpha, BlockedTensor& C1, BlockedTensor& C2) {
    /**
     * The following blocks should be available in memory:
     * G2: avac, aaac, avaa
     * H2: vvaa, aacc, avca, avac, vaaa, aaca, aaaa
     * T2: aavv, ccaa, caav, acav, aava, caaa, aaaa
     * S2: the same as T2
     */

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "tempHACa", {"aa"});

    temp["uv"] += H1["ev"] * T1["ue"];
    temp["uv"] -= H1["um"] * T1["mv"];

    H_T_C1a_smallG(G2, T1, T2, temp);

    H_T_C1a_smallS(H1, H2, T2, S2, temp);

    C1["uv"] += alpha * temp["uv"];
    C1["vu"] += alpha * temp["uv"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aaaa"});

    H_T_C2a_smallS(H1, H2, T1, T2, S2, temp);

    C2["uvxy"] += alpha * temp["uvxy"];
    C2["xyuv"] += alpha * temp["uvxy"];
}

void SADSRG::H_T_C1a_smallG(BlockedTensor& G2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C1) {
    /**
     * The following blocks should be available in memory:
     * G2: avac, aaac, avaa
     * T2: aava, caaa
     */

    C1["uv"] += T1["ma"] * G2["uavm"];
    C1["uv"] += 0.5 * T1["xe"] * L1_["yx"] * G2["uevy"];
    C1["uv"] -= 0.5 * T1["mx"] * L1_["xy"] * G2["uyvm"];

    C1["wz"] += 0.5 * G2["wezx"] * T2["uvey"] * L2_["xyuv"];
    C1["wz"] -= 0.5 * G2["wuzm"] * T2["mvxy"] * L2_["xyuv"];
}

void SADSRG::H_T_C1a_smallS(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T2,
                            BlockedTensor& S2, BlockedTensor& C1) {
    /**
     * The following blocks should be available in memory:
     * H2: vvaa, aacc, avca, avac, vaaa, aaca, aaaa
     * T2: aavv, ccaa, caav, acav, aava, caaa, aaaa
     * S2: the same as T2
     */

    // C1["uv"] += H1["bm"] * S2["umvb"];
    C1["uv"] += H1["em"] * S2["umve"];
    C1["uv"] += H1["xm"] * S2["muxv"];

    // C1["uv"] += 0.5 * H1["bx"] * S2["uyvb"] * L1_["xy"];
    C1["uv"] += 0.5 * H1["ex"] * S2["yuev"] * L1_["xy"];
    C1["uv"] += 0.5 * H1["wx"] * S2["uyvw"] * L1_["xy"];

    // C1["uv"] -= 0.5 * H1["yj"] * S2["ujvx"] * L1_["xy"];
    C1["uv"] -= 0.5 * H1["ym"] * S2["muxv"] * L1_["xy"];
    C1["uv"] -= 0.5 * H1["yw"] * S2["uwvx"] * L1_["xy"];

    C1["wz"] += H2["uemz"] * S2["mwue"];
    C1["wz"] += H2["uezm"] * S2["wmue"];
    C1["wz"] += H2["vumz"] * S2["mwvu"];

    C1["wz"] -= H2["wemu"] * S2["muze"];
    C1["wz"] -= H2["weum"] * S2["umze"];
    C1["wz"] -= H2["ewvu"] * S2["vuez"];

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aaaa"});

    // temp["wzuv"] += 0.5 * S2["wvab"] * H2["abzu"];
    temp["wzuv"] += 0.5 * S2["wvef"] * H2["efzu"];
    temp["wzuv"] += 0.5 * S2["wvex"] * H2["exzu"];
    temp["wzuv"] += 0.5 * S2["vwex"] * H2["exuz"];
    temp["wzuv"] += 0.5 * S2["wvxy"] * H2["xyzu"];

    // temp["wzuv"] -= 0.5 * S2["wmub"] * H2["vbzm"];
    temp["wzuv"] -= 0.5 * S2["wmue"] * H2["vezm"];
    temp["wzuv"] -= 0.5 * S2["mwxu"] * H2["xvmz"];

    // temp["wzuv"] -= 0.5 * S2["mwub"] * H2["bvzm"];
    temp["wzuv"] -= 0.5 * S2["mwue"] * H2["vemz"];
    temp["wzuv"] -= 0.5 * S2["mwux"] * H2["vxmz"];

    temp["wzuv"] += 0.25 * S2["jwxu"] * L1_["xy"] * H2["yvjz"];
    temp["wzuv"] -= 0.25 * S2["ywbu"] * L1_["xy"] * H2["bvxz"];
    temp["wzuv"] -= 0.25 * S2["wybu"] * L1_["xy"] * H2["bvzx"];
    C1["wz"] += temp["wzuv"] * L1_["uv"];

    temp.zero();

    // temp["wzuv"] -= 0.5 * S2["ijzu"] * H2["wvij"];
    temp["wzuv"] -= 0.5 * S2["mnzu"] * H2["wvmn"];
    temp["wzuv"] -= 0.5 * S2["mxzu"] * H2["wvmx"];
    temp["wzuv"] -= 0.5 * S2["mxuz"] * H2["vwmx"];
    temp["wzuv"] -= 0.5 * S2["xyzu"] * H2["wvxy"];

    // temp["wzuv"] += 0.5 * S2["vjze"] * H2["weuj"];
    temp["wzuv"] += 0.5 * S2["vmze"] * H2["weum"];
    temp["wzuv"] += 0.5 * S2["xvez"] * H2["ewxu"];

    // temp["wzuv"] += 0.5 * S2["jvze"] * H2["weju"];
    temp["wzuv"] += 0.5 * S2["mvze"] * H2["wemu"];
    temp["wzuv"] += 0.5 * S2["vxez"] * H2["ewux"];

    temp["wzuv"] -= 0.25 * S2["yvbz"] * Eta1_["xy"] * H2["bwxu"];
    temp["wzuv"] += 0.25 * S2["jvxz"] * Eta1_["xy"] * H2["ywju"];
    temp["wzuv"] += 0.25 * S2["jvzx"] * Eta1_["xy"] * H2["wyju"];
    C1["wz"] += temp["wzuv"] * Eta1_["uv"];

    C1["wz"] += 0.5 * H2["vujz"] * T2["jwyx"] * L2_["xyuv"];
    C1["wz"] += 0.5 * H2["auzx"] * S2["wvay"] * L2_["xyuv"];
    C1["wz"] -= 0.5 * H2["auxz"] * T2["wvay"] * L2_["xyuv"];
    C1["wz"] -= 0.5 * H2["auxz"] * T2["vway"] * L2_["xyvu"];

    C1["wz"] -= 0.5 * H2["bwyx"] * T2["vubz"] * L2_["xyuv"];
    C1["wz"] -= 0.5 * H2["wuix"] * S2["ivzy"] * L2_["xyuv"];
    C1["wz"] += 0.5 * H2["uwix"] * T2["ivzy"] * L2_["xyuv"];
    C1["wz"] += 0.5 * H2["uwix"] * T2["ivyz"] * L2_["xyvu"];

    C1["wz"] += 0.5 * H2["avxy"] * S2["uwaz"] * L2_["xyuv"];
    C1["wz"] -= 0.5 * H2["uviy"] * S2["iwxz"] * L2_["xyuv"];
}

void SADSRG::H_T_C2a_smallS(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                            BlockedTensor& T2, BlockedTensor& S2, BlockedTensor& C2) {
    /**
     * The following blocks should be available in memory:
     * H2: vvaa, aacc, avca, avac, vaaa, aaca, aaaa
     * T2: aavv, ccaa, caav, acav, aava, caaa, aaaa
     * S2: the same as T2
     */

    // C2["uvxy"] += H2["abxy"] * T2["uvab"];
    C2["uvxy"] += H2["efxy"] * T2["uvef"];
    C2["uvxy"] += H2["wzxy"] * T2["uvwz"];
    C2["uvxy"] += H2["ewxy"] * T2["uvew"];
    C2["uvxy"] += H2["ewyx"] * T2["vuew"];

    // C2["uvxy"] += H2["uvij"] * T2["ijxy"];
    C2["uvxy"] += H2["uvmn"] * T2["mnxy"];
    C2["uvxy"] += H2["uvwz"] * T2["wzxy"];
    C2["uvxy"] += H2["vumw"] * T2["mwyx"];
    C2["uvxy"] += H2["uvmw"] * T2["mwxy"];

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"aaaa"});
    temp["uvxy"] += H1["ax"] * T2["uvay"];
    temp["uvxy"] -= H1["ui"] * T2["ivxy"];
    temp["uvxy"] += T1["ua"] * H2["avxy"];
    temp["uvxy"] -= T1["ix"] * H2["uviy"];

    temp["uvxy"] -= 0.5 * L1_["wz"] * T2["vuaw"] * H2["azyx"];
    temp["uvxy"] -= 0.5 * Eta1_["wz"] * T2["izyx"] * H2["vuiw"];

    // temp["uvxy"] += H2["aumx"] * S2["mvay"];
    temp["uvxy"] += H2["uexm"] * S2["vmye"];
    temp["uvxy"] += H2["wumx"] * S2["mvwy"];

    temp["uvxy"] += 0.5 * L1_["wz"] * S2["zvay"] * H2["auwx"];
    temp["uvxy"] -= 0.5 * L1_["wz"] * S2["ivwy"] * H2["zuix"];

    // temp["uvxy"] -= H2["auxm"] * T2["mvay"];
    temp["uvxy"] -= H2["uemx"] * T2["vmye"];
    temp["uvxy"] -= H2["uwmx"] * T2["mvwy"];

    temp["uvxy"] -= 0.5 * L1_["wz"] * T2["zvay"] * H2["auxw"];
    temp["uvxy"] += 0.5 * L1_["wz"] * T2["ivwy"] * H2["uzix"];

    // temp["uvxy"] -= H2["avxm"] * T2["muya"];
    temp["uvxy"] -= H2["vemx"] * T2["muye"];
    temp["uvxy"] -= H2["vwmx"] * T2["muyw"];

    temp["uvxy"] -= 0.5 * L1_["wz"] * T2["uzay"] * H2["avxw"];
    temp["uvxy"] += 0.5 * L1_["wz"] * T2["iuyw"] * H2["vzix"];

    C2["uvxy"] += temp["uvxy"];
    C2["vuyx"] += temp["uvxy"];
}
} // namespace forte
