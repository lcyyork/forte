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

#ifndef _trotter_so_h_
#define _trotter_so_h_

#include <cmath>
#include <unordered_set>

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"
#include "ambit/blocked_tensor.h"

#include "base_classes/forte_options.h"
#include "base_classes/dynamic_correlation_solver.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"
#include "helpers/blockedtensorfactory.h"
#include "mrdsrg-helper/dsrg_source.h"

using namespace ambit;

namespace forte {

class TROTTER_SO : public DynamicCorrelationSolver {
  public:
    // => Constructors <= //

    TROTTER_SO(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~TROTTER_SO();

    /// Compute the correlation energy
    double compute_energy();

    /// DSRG transformed Hamiltonian (not implemented)
    std::shared_ptr<ActiveSpaceIntegrals> compute_Heff_actv();

  protected:
    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();

    /// Called in the destructor
    void cleanup();

    /// Print a summary of the options
    void print_summary();

    // => Class data <= //

    std::string corr_level_;
    int trotter_level_;
    bool trotter_sym_;

    /// Print levels
    int print_;

    /// List of alpha core SOs
    std::vector<size_t> acore_sos;
    /// List of alpha active SOs
    std::vector<size_t> aactv_sos;
    /// List of alpha virtual SOs
    std::vector<size_t> avirt_sos;
    /// List of beta core SOs
    std::vector<size_t> bcore_sos;
    /// List of beta active SOs
    std::vector<size_t> bactv_sos;
    /// List of beta virtual SOs
    std::vector<size_t> bvirt_sos;

    /// List of core SOs
    std::vector<size_t> core_sos_;
    /// List of active SOs
    std::vector<size_t> actv_sos_;
    /// List of virtual SOs
    std::vector<size_t> virt_sos_;

    /// Number of spin orbitals
    size_t nso_;
    /// Number of core spin orbitals
    size_t nc_;
    /// Number of active spin orbitals
    size_t na_;
    /// Number of virtual spin orbitals
    size_t nv_;
    /// Number of hole spin orbitals
    size_t nh_;
    /// Number of particle spin orbitals
    size_t np_;

    /// The flow parameter
    double s_;

    /// Source operator
    std::string source_;
    /// The dsrg source operator
    std::shared_ptr<DSRG_SOURCE> dsrg_source_;

    std::shared_ptr<BlockedTensorFactory> BTF_;
    TensorType tensor_type_;

    /// The energy of the reference
    double Eref_;

    /// The frozen-core energy
    double Efrzc_;

    /// Convergence thresholds
    double e_conv_;
    double r_conv_;
    int maxiter_;

    // => Tensors <= //

    ambit::BlockedTensor H_;
    ambit::BlockedTensor F_;
    ambit::BlockedTensor V_;
    ambit::BlockedTensor L1_;
    ambit::BlockedTensor L2_;
    ambit::BlockedTensor L3_;
    ambit::BlockedTensor T1_;
    ambit::BlockedTensor T2_;

    /// Diagonal elements of Fock matrix
    std::vector<double> Fd_;

    /// Number of amplitudes will be printed in amplitude summary
    int ntamp_;
    /// Threshold for amplitudes considered as intruders
    double intruder_tamp_;

    /// Computes the t2 amplitudes for three different cases of spin (alpha all,
    /// beta all, and alpha beta)
    void guess_t2();
    void update_t2();
    double t2_rms_;
    double t2_norm_;
    double t2_max_;

    /// Computes the t1 amplitudes for three different cases of spin (alpha all,
    /// beta all, and alpha beta)
    void guess_t1();
    void update_t1();
    double t1_rms_;
    double t1_norm_;
    double t1_max_;

    /// Effective Hamiltonian Hbar
    double Hbar0_;
    ambit::BlockedTensor Hbar1_;
    ambit::BlockedTensor Hbar2_;

    /// Compute Hbar
    void build_ccsd_Hamiltonian(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                                BlockedTensor& T2, double& C0, BlockedTensor& C1,
                                BlockedTensor& C2);
    void compute_trotter_uccsd(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                               BlockedTensor& T2, double& C0, BlockedTensor& C1, BlockedTensor& C2);

    /// Compute C = factor * [H, Ta] where Ta contains one or more active indices
    void H_Ta_C(double factor, BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                BlockedTensor& T2, double& C0, BlockedTensor& C1, BlockedTensor& C2);
    //    /// Compute C = 0.5 * [[H, Te], Ta] + 0.5 * [[H, Ta], Te]
    //    void H_TaTe_C(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
    //                  double& C0, BlockedTensor& C1, BlockedTensor& C2);
    //    /// Compute C = 0.5 * [[H, Ta], Ta]
    //    void H_TaTa_C(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
    //                  double& C0, BlockedTensor& C1, BlockedTensor& C2);

    void transform_hamiltonian_recursive(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                                         BlockedTensor& T2, double& C0, BlockedTensor& C1,
                                         BlockedTensor& C2);
    /// Compute C = (1 / (k * (k + 1))) * [[H2, T2]_3, Ta]
    void comm2_O_Ta_C(int k, BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2, double& C0,
                      BlockedTensor& C1, BlockedTensor& C2);
    void comm2_O_Ta_C1(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& C1);
    void comm2_O_Ta_C2_ccvv(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_cavv(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_aavv(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_aavc(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_cavc(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_ccvc(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_ccva(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_cava(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_aava(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vcvv(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vavv(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vavc(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vcvc(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vcva(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vava(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vvvv(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vvvc(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vvva(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_aacc(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vaca(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_aaca(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vccc(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_cacc(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vcca(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_caca(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_cccc(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_ccca(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vvaa(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vcaa(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_ccaa(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_vaaa(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_caaa(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);
    void comm2_O_Ta_C2_aaaa(BlockedTensor& H2, BlockedTensor& T1, BlockedTensor& T2,
                            BlockedTensor& C2);

    void ccsd_hamiltonian(int level, BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                          BlockedTensor& T2, double& C0, BlockedTensor& C1, BlockedTensor& C2);
};

std::unique_ptr<TROTTER_SO> make_trotter_so(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                            std::shared_ptr<ForteOptions> options,
                                            std::shared_ptr<ForteIntegrals> ints,
                                            std::shared_ptr<MOSpaceInfo> mo_space_info);
} // namespace forte

#endif // _trotter_so_h_
