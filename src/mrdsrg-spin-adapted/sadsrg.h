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

#ifndef _sadsrg_h_
#define _sadsrg_h_

#include <cmath>
#include <memory>
#include <tuple>
#include <string>
#include <vector>
#include <map>

#include "ambit/tensor.h"
#include "ambit/blocked_tensor.h"

#include "base_classes/forte_options.h"
#include "base_classes/dynamic_correlation_solver.h"
#include "integrals/integrals.h"
#include "integrals/active_space_integrals.h"
#include "base_classes/rdms.h"
#include "base_classes/mo_space_info.h"
#include "helpers/blockedtensorfactory.h"
#include "mrdsrg-helper/dsrg_mem.h"
#include "mrdsrg-helper/dsrg_source.h"
#include "mrdsrg-helper/dsrg_time.h"
#include "mrdsrg-helper/dsrg_tensors.h"
#include "mrdsrg-helper/dsrg_transformed.h"

using namespace ambit;

namespace forte {
class SADSRG : public DynamicCorrelationSolver {
  public:
    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    SADSRG(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
           std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    virtual ~SADSRG();

    /// Compute energy
    virtual double compute_energy() = 0;

    /// Compute DSRG transformed Hamiltonian
    virtual std::shared_ptr<ActiveSpaceIntegrals> compute_Heff_actv();

    /// Set unitary matrix (in active space) from original to semicanonical
    void set_Uactv(ambit::Tensor& U);

  protected:
    /// Startup function called in constructor
    void startup();

    /// Warnings <description, changes in this run, how to get rid of it>
    std::vector<std::tuple<const char*, const char*, const char*>> warnings_;

    // ==> settings from options <==

    /// Read options
    void read_options();

    /// The flow parameter
    double s_;
    /// Source operator
    std::string source_;
    /// Source operator for the core-core-virtual-virtual block
    std::string ccvv_source_;
    /// The dsrg source operator
    std::shared_ptr<DSRG_SOURCE> dsrg_source_;
    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;

    /// Compute contributions from 3 cumulant
    bool do_cu3_;

    /// Multi-state computation if true
    bool multi_state_;
    /// Multi-state algorithm
    std::string multi_state_algorithm_;

    /// Number of amplitudes will be printed in amplitude summary
    size_t ntamp_;
    /// Threshold for amplitudes considered as intruders
    double intruder_tamp_;

    /// How to consider internal amplitudes
    std::string internal_amp_;
    /// Include which part of internal amplitudes?
    std::string internal_amp_select_;

    /// Relaxation type
    std::string relax_ref_;

    /// Timings for computing the commutators
    DSRG_TIME dsrg_time_;

    /// Active orbital rotation from semicanonicalizor (set from outside)
    ambit::BlockedTensor Uactv_;
    /// Rotate 2-body DSRG transformed integrals from semicanonical back to original
    void rotate_ints_semi_to_origin(const std::string& name, BlockedTensor& H1, BlockedTensor& H2);
    /// Rotate 3-body DSRG transformed integrals from semicanonical back to original
    void rotate_ints_semi_to_origin(const std::string& name, BlockedTensor& H1, BlockedTensor& H2,
                                    BlockedTensor& H3);

    /// Number of threads
    int n_threads_;

    // ==> system memory related <==

    /// Total memory available set by the user
    size_t mem_sys_;
    /// Memory checker and printer
    DSRG_MEM dsrg_mem_;

    /// Check initial memory
    void check_init_memory();

    // ==> some common energies for all DSRG levels <==

    /// The energy of the reference
    double Eref_;

    /// Compute reference (MK vacuum) energy from ForteIntegral and Fock_
    double compute_reference_energy_from_ints();

    /// Compute reference (MK vacuum) energy
    double compute_reference_energy(BlockedTensor H, BlockedTensor F, BlockedTensor V);

    // ==> MO space info <==

    /// Read MO space info
    void read_MOSpaceInfo();

    /// List of core MOs
    std::vector<size_t> core_mos_;
    /// List of active MOs
    std::vector<size_t> actv_mos_;
    /// List of virtual MOs
    std::vector<size_t> virt_mos_;
    /// List of the symmetry of the active MOs
    std::vector<int> actv_mos_sym_;

    /// List of active active occupied MOs (relative to active)
    std::vector<size_t> actv_occ_mos_;
    /// List of active active unoccupied MOs (relative to active)
    std::vector<size_t> actv_uocc_mos_;

    /// List of auxiliary MOs when DF/CD
    std::vector<size_t> aux_mos_;

    // ==> Ambit tensor settings <==

    /// Set Ambit tensor labels
    void set_ambit_MOSpace();

    /// Kevin's Tensor Wrapper
    std::shared_ptr<BlockedTensorFactory> BTF_;
    /// Tensor type for Ambit
    ambit::TensorType tensor_type_;

    /// Core MO label
    std::string core_label_;
    /// Active MO label
    std::string actv_label_;
    /// Virtual MO label
    std::string virt_label_;

    /// Auxillary basis label
    std::string aux_label_;

    /// Map from space label to list of MOs
    std::map<char, std::vector<size_t>> label_to_spacemo_;

    /// Compute diagonal blocks labels of a one-body operator
    std::vector<std::string> diag_one_labels();
    /// Compute diagonal blocks labels of a two-body operator
    std::vector<std::string> diag_two_labels();
    /// Compute retaining excitation blocks labels of a two-body operator
    std::vector<std::string> re_two_labels();
    /// Compute off-diagonal blocks labels of a one-body operator
    std::vector<std::string> od_one_labels();
    std::vector<std::string> od_one_labels_hp();
    std::vector<std::string> od_one_labels_ph();
    /// Compute off-diagonal blocks labels of a two-body operator
    std::vector<std::string> od_two_labels();
    std::vector<std::string> od_two_labels_hhpp();
    std::vector<std::string> od_two_labels_pphh();
    /// Compute the blocks labels used in NIVO (number of virtual < 3)
    std::vector<std::string> nivo_labels();

    // ==> fill in densities from RDMs <==

    /// Initialize density cumulants
    void init_density();
    /// Fill in density cumulants from the RDMs
    void fill_density();

    /// One-particle density matrix
    ambit::BlockedTensor L1_;
    /// One-hole density matrix
    ambit::BlockedTensor Eta1_;
    /// Two-body denisty cumulant
    ambit::BlockedTensor L2_;

    // ==> Fock matrix related <==

    /// Fock matrix
    ambit::BlockedTensor Fock_;
    /// Diagonal elements of Fock matrix
    std::vector<double> Fdiag_;

    /// Initialize Fock matrix
    void init_fock();
    /// Build Fock matrix from ForteIntegrals
    void build_fock_from_ints();
    /// Fill in diagonal elements of Fock matrix to Fdiag
    void fill_Fdiag(BlockedTensor& F, std::vector<double>& Fdiag);

    /// Check orbitals if semicanonical
    bool check_semi_orbs();
    /// Are orbitals semi-canonicalized?
    bool semi_canonical_;
    /// Checked results of each block of Fock matrix
    std::map<std::string, bool> semi_checked_results_;
    /// Unitary matrix to block diagonal Fock
    ambit::BlockedTensor U_;

    // ==> integrals <==

    /// Fill the tensor B with three-index DF or CD integrals
    void fill_three_index_ints(ambit::BlockedTensor B);

    /// Scalar of the DSRG transformed Hamiltonian
    double Hbar0_;
    /// DSRG transformed 1-body Hamiltonian (active only in DSRG-PT, but full in MRDSRG)
    ambit::BlockedTensor Hbar1_;
    /// DSRG transformed 2-body Hamiltonian (active only in DSRG-PT, but full in MRDSRG)
    ambit::BlockedTensor Hbar2_;
    /// DSRG transformed 3-body Hamiltonian (active only in DSRG-PT, but full in MRDSRG)
    ambit::BlockedTensor Hbar3_;

    /**
     * De-normal-order a 2-body DSRG transformed integrals
     * This will change H0 and H1 !!!
     */
    void deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1, BlockedTensor& H2);

    /**
     * De-normal-order a 3-body DSRG transformed integrals
     * This will change H0, H1, and H2 !!!
     */
    void deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1, BlockedTensor& H2,
                    BlockedTensor& H3);

    /**
     * De-normal-order the T1 and T2 amplitudes and return the effective T1
     * T1eff = T1 - T2["ivau"] * D1["uv"]
     *
     * This assumes no internal amplitudes !!!
     */
    ambit::BlockedTensor deGNO_Tamp(BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& D1);

    // ==> commutators <==

    /**
     * H1, C1, G1: a rank 2 tensor of all MOs in general
     * H2, C2, G2: a rank 4 tensor of all MOs in general
     * C3: a rank 6 tensor of all MOs in general
     * T1: a rank 2 tensor of hole-particle
     * T2: a rank 4 tensor of hole-hole-particle-particle
     * V: antisymmetrized 2-electron integrals
     * B: 3-index integrals from DF/CD
     */

    /// Compute zero-body term of commutator [H1, T1]
    double H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H1, T2]
    double H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T1]
    double H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T2], S2[ijab] = 2 * T[ijab] - T[ijba]
    std::vector<double> H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2,
                                 const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T2], T2 and S2 contain at least two active indices
    std::vector<double> H2_T2_C0_T2small(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2);

    /// Compute one-body term of commutator [H1, T1]
    void H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H1, T2]
    void H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T1]
    void H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T2], S2[ijab] = 2 * T[ijab] - T[ijba]
    void H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                  BlockedTensor& C1);

    /// Compute two-body term of commutator [H2, T1]
    void H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H1, T2]
    void H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2], S2[ijab] = 2 * T[ijab] - T[ijba]
    void H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                  BlockedTensor& C2);

    /// Compute two-body term of commutator [H2, T2], hole-hole contraction
    void H2_T2_C2_HH(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2], particle-particle contraction
    void H2_T2_C2_PP(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2], particle-hole contraction
    void H2_T2_C2_PH(BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                     BlockedTensor& C2);

    /// Compute zero-body term of commutator [V, T1], V is constructed from B (DF/CD)
    void V_T1_C0_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [V, T2], V is constructed from B (DF/CD)
    std::vector<double> V_T2_C0_DF(BlockedTensor& B, BlockedTensor& T1, BlockedTensor& S2,
                                   const double& alpha, double& C0);

    /// Compute one-body term of commutator [V, T1], V is constructed from B (DF/CD)
    void V_T1_C1_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [V, T2], V is constructed from B (DF/CD)
    void V_T2_C1_DF(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                    BlockedTensor& C1);

    /// Compute two-body term of commutator [V, T1], V is constructed from B (DF/CD)
    void V_T1_C2_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [V, T2], V is constructed from B (DF/CD)
    void V_T2_C2_DF(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                    BlockedTensor& C2);

    /// Compute two-body term of commutator [V, T2], hole-hole contraction
    void V_T2_C2_HH_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                        BlockedTensor& C2);
    /// Compute two-body term of commutator [V, T2], particle-particle contraction
    void V_T2_C2_PP_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                        BlockedTensor& C2);

    /// Compute two-body term of commutator [V, T2], particle-hole contraction
    void V_T2_C2_PH_DF(BlockedTensor& B, BlockedTensor& T2, BlockedTensor& S2, const double& alpha,
                        BlockedTensor& C2);
    /// Compute two-body term of commutator [V, T2], particle-hole contraction S2 related
    void V_T2_C2_PH_DF_J(BlockedTensor& B, BlockedTensor& S2, const double& alpha,
                         BlockedTensor& C2);
    /// Compute two-body term of commutator [V, T2], particle-hole contraction T2 related
    void V_T2_C2_PH_DF_K(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2);

    /// Compute the active part of commutator C1 + C2 = alpha * [H1 + H2, A1 + A2]
    void H_A_Ca(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                BlockedTensor& S2, const double& alpha, BlockedTensor& C1, BlockedTensor& C2);
    /// Compute the active part of commutator C1 + C2 = alpha * [H1 + H2, A1 + A2]
    /// G2[pqrs] = 2 * H2[pqrs] - H2[pqsr], S2[ijab] = 2 * T2[ijab] - T2[ijba]
    void H_A_Ca_small(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& G2, BlockedTensor& T1,
                      BlockedTensor& T2, BlockedTensor& S2, const double& alpha, BlockedTensor& C1,
                      BlockedTensor& C2);
    /// Compute the active part of commutator C1 = [H2, T1 + T2] that uses G2
    void H_T_C1a_smallG(BlockedTensor& G2, BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& C1);
    /// Compute the active part of commutator C1 = [H1 + H2, T1 + T2] that uses S2
    void H_T_C1a_smallS(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T2, BlockedTensor& S2,
                        BlockedTensor& C1);
    /// Compute the active part of commutator C2 = [H1 + H2, T1 + T2] that uses S2
    void H_T_C2a_smallS(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                        BlockedTensor& S2, BlockedTensor& C2);

    // ==> printing functions <==

    /// Print the summary of 2- and 3-body density cumulant
    void print_cumulant_summary();

    /// Print the contents with padding: <text> <padding with dots>
    void print_contents(const std::string& str, size_t size = 45);
    /// Print done and timing
    void print_done(double t);

    // ==> common aplitudes analysis and printing <==

    /// Prune internal amplitudes for T1
    void internal_amps_T1(BlockedTensor& T1);
    /// Prune internal amplitudes for T2
    void internal_amps_T2(BlockedTensor& T2);

    /// Check T1 and return the largest amplitudes
    std::vector<std::pair<std::vector<size_t>, double>> check_t1(BlockedTensor& T1);
    /// Check T2 and return the largest amplitudes
    std::vector<std::pair<std::vector<size_t>, double>> check_t2(BlockedTensor& T2);
    /// Analyze T1 and T2 amplitudes
    void analyze_amplitudes(std::string name, BlockedTensor& T1, BlockedTensor& T2);
    /// Print t1 amplitudes summary
    void print_t1_summary(const std::vector<std::pair<std::vector<size_t>, double>>& list,
                          const double& norm, const size_t& number_nonzero);
    /// Print t2 amplitudes summary
    void print_t2_summary(const std::vector<std::pair<std::vector<size_t>, double>>& list,
                          const double& norm, const size_t& number_nonzero);
    /// Print t1 intruder analysis
    void print_t1_intruder(const std::vector<std::pair<std::vector<size_t>, double>>& list);
    /// Print t2 intruder analysis
    void print_t2_intruder(const std::vector<std::pair<std::vector<size_t>, double>>& list);

    // ==> miscellaneous <==

    /// Diagonalize the diagonal blocks of the Fock matrix
    std::vector<double> diagonalize_Fock_diagblocks(BlockedTensor& U);

    /// Comparison function used to sort pair<vector, double>
    static bool sort_pair_second_descend(const std::pair<std::vector<size_t>, double>& left,
                                         const std::pair<std::vector<size_t>, double>& right) {
        return std::fabs(left.second) > std::fabs(right.second);
    }

    /// Separate a vector of block labels to vectors of small blocks that fit in memory
    std::vector<std::vector<std::string>>
    separate_blocks(const std::vector<std::string>& blocks, std::vector<std::string>& large_blocks,
                    const std::function<size_t(std::string)> func_Ttemp) {
        // sort the blocks according to the number of element
        std::vector<std::string> blocks_sorted(blocks);
        std::sort(blocks_sorted.begin(), blocks_sorted.end(),
                  [&](const std::string& b1, const std::string& b2) {
                      return dsrg_mem_.compute_n_elements(b1) < dsrg_mem_.compute_n_elements(b2);
                  });

        // separate blocks to small blocks that fit in memory
        // put the large block that does not fit in memory in large_blocks
        large_blocks.clear();
        std::vector<std::vector<std::string>> block_batches;
        std::vector<std::string> current_blocks;

        size_t cumulative_memory = 0;
        size_t available_memory = dsrg_mem_.available() / sizeof(double);

        for (const auto& block : blocks_sorted) {
            auto size = dsrg_mem_.compute_n_elements(block);
            auto intermediate_size = func_Ttemp(block);

            // a single block that does not fit in memory
            if (size + intermediate_size > available_memory) {
                large_blocks.push_back(block);
                continue;
            }

            // this block fit in memory
            // but adding it to current_blocks make current_blocks out of memory
            if (cumulative_memory + size + intermediate_size > available_memory) {
                block_batches.push_back(current_blocks);
                current_blocks.clear();
                cumulative_memory = 0;
            }

            // add to current_blocks
            cumulative_memory += size;
            current_blocks.push_back(block);
        }

        if (current_blocks.size()) {
            block_batches.push_back(current_blocks);
        }

        return block_batches;
    }

    /// Fill in a 3-index slice (Tsub["qrs"]) of a 4-index tensor (T["pqrs"]) for given index p
    template <class B>
    void fill_slice3_from_tensor4(BlockedTensor& T, BlockedTensor& Tsub, const std::string& block_p,
                                  size_t p, const B& blocks_qrs) {
        for (const auto& block_qrs : blocks_qrs) {
            auto& T_data = T.block(block_p + block_qrs).data();
            auto& Tsub_data = Tsub.block(block_qrs).data();

            auto chunk_size = Tsub_data.size();
            auto begin = T_data.begin() + p * chunk_size;
            std::copy(begin, begin + chunk_size, Tsub_data.begin());
        }
    }

    /// Add a 3-index slice (O["qrs"]) of index p to a 4-index tensor (C["pqrs"] and C["qpsr"])
    /// i.e., for given p, C["pqrs"] += factor * S["qrs"]; C["qpsr"] += factor * S["qrs"];
    template <class B>
    void axpy_slice3_to_tensor4_with_sym(BlockedTensor& C, BlockedTensor& S, const double factor,
                                         const std::string& block_p, size_t p,
                                         const B& blocks_qrs) {
        size_t p_size = label_to_spacemo_[block_p[0]].size();

        for (const auto& block_qrs : blocks_qrs) {
            auto q_size = label_to_spacemo_[block_qrs[0]].size();
            auto r_size = label_to_spacemo_[block_qrs[1]].size();
            auto s_size = label_to_spacemo_[block_qrs[2]].size();

            auto rs_size = r_size * s_size;
            auto qrs_size = q_size * rs_size;
            auto psr_size = p_size * rs_size;

            // C["pqrs"] += factor * S["qrs"] for given index p
            auto& Cpqrs_data = C.block(block_p + block_qrs).data();
            auto Cdata_begin = Cpqrs_data.begin() + p * qrs_size;
            std::transform(Cdata_begin, Cdata_begin + qrs_size, S.block(block_qrs).data().begin(),
                           Cdata_begin, [&factor](auto c, auto t) { return c + factor * t; });

            // C["qpsr"] += factor * S["qrs"] for given index p
            auto block_qpsr = block_qrs.substr(0, 1) + block_p;
            block_qpsr += block_qrs.substr(2, 1) + block_qrs.substr(1, 1);
            auto& Cqpsr_data = C.block(block_qpsr).data();
            S.block(block_qrs).citerate([&](const std::vector<size_t>& id, const double& value) {
                Cqpsr_data[id[0] * psr_size + p * rs_size + id[2] * r_size + id[1]] +=
                    factor * value;
            });
        }
    }
};
} // namespace forte

#endif // SADSRG_H
