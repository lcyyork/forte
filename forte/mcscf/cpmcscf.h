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

#pragma once

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

#include "psi4/libdiis/diismanager.h"
#include "psi4/libfock/jk.h"
#include "psi4/libmints/matrix.h"
#include "psi4/lib3index/dfhelper.h"

#include "ambit/tensor.h"
#include "ambit/blocked_tensor.h"

#include "base_classes/active_space_method.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"

#include "integrals/integrals.h"
#include "base_classes/active_space_solver.h"
#include "base_classes/state_info.h"
#include "mcscf/mcscf_orb_grad.h"

namespace forte {

class CPMCSCF_SOLVER {
  public:
    CPMCSCF_SOLVER(std::shared_ptr<ForteIntegrals> ints,
                   std::shared_ptr<ActiveSpaceSolver> as_solver,
                   std::shared_ptr<ForteOptions> options,
                   std::shared_ptr<MOSpaceInfo> mo_space_info);
    // std::vector<double>& bo, std::map<StateInfo, ambit::Tensor>& bc
    /**
     * @brief
     *
     * @tparam Foo
     * @param foo
     *
     */
    void solve();

    void set_bo_cv(ambit::Tensor b_cv);
    void set_bo_ca(ambit::Tensor b_ca);
    void set_bo_av(ambit::Tensor b_av);
    void set_bo_aa(ambit::Tensor b_aa);

    void set_bc(const std::map<StateInfo, ambit::Tensor>& bc);

    std::map<std::string, ambit::Tensor> xo();
    std::map<StateInfo, ambit::Tensor> xc();

    // sigma build
    void build_sigma(ambit::BlockedTensor qo, std::map<StateInfo, ambit::Tensor>& qc,
                     ambit::BlockedTensor so, std::map<StateInfo, ambit::Tensor>& sc);

    // CI projection scheme (all or state specific)
    void project_ci(std::map<StateInfo, ambit::Tensor>& vecs, bool all = false);

    // set the active correlation energy of the target state
    void set_ecorr_actv(double ec_actv);

    bool eri_df_;

    std::shared_ptr<ForteIntegrals> ints_;
    std::shared_ptr<ActiveSpaceSolver> as_solver_;
    std::shared_ptr<ForteOptions> options_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    std::map<StateInfo, std::vector<double>> state_weights_map_;

    std::pair<StateInfo, size_t> target_root_;
    double target_e0_actv_;
    double target_ec_actv_;

    /// Map from MO space label to the absolute MO indices
    std::map<std::string, std::vector<size_t>> label_to_mos_;

    /// Relative indices within an irrep <irrep, relative indices>
    std::vector<std::pair<int, size_t>> mos_rel_;

    ambit::TensorType tensor_type_ = ambit::TensorType::CoreTensor;

    // symmetrized SA-RDMs from SA-MCSCF
    ambit::BlockedTensor D1_;
    ambit::BlockedTensor D2_;
    // the modified symmetrized SA-RDMs (multipliers)
    ambit::BlockedTensor M1_;
    ambit::BlockedTensor M2_;

    // three-index DF integrals
    ambit::BlockedTensor Q_;
    // four-index integrals
    ambit::BlockedTensor G_;

    // core Fock
    ambit::BlockedTensor Fc_;
    // total Fock
    ambit::BlockedTensor Ft_;
    // diagonal Fock elements
    std::vector<double> Fd_;
    // build Fock matrices
    void compute_fock();

    // Roothaan-Bagus supermatrix
    ambit::BlockedTensor L_;
    // SA-MCSCF orbital gradients
    ambit::BlockedTensor A_;
    // // compute orbital gradients
    // ambit::BlockedTensor compute_orb_grad(ambit::BlockedTensor D1, ambit::BlockedTensor D2);

    // diagonal orbital Hessian
    ambit::BlockedTensor Ho_;
    // diagonal CI Hessian
    std::map<StateInfo, std::vector<double>> Hc_;
    // compute diagonal Hessian
    void compute_hess_diag();
    // apply preconditioner (inverse of diagonal Hessian)
    void apply_Minv(ambit::BlockedTensor ro, std::map<StateInfo, ambit::Tensor>& rc);

    // dot product between two vectors
    double vector_dot(const ambit::BlockedTensor& vo, const std::map<StateInfo, ambit::Tensor>& vc,
                      const ambit::BlockedTensor& wo, const std::map<StateInfo, ambit::Tensor>& wc);
    // normalize the given vector
    double normalize(ambit::BlockedTensor vo, std::map<StateInfo, ambit::Tensor>& vc);
    // AXPY: y += a * x
    void axpy(double a, const ambit::BlockedTensor& xo,
              const std::map<StateInfo, ambit::Tensor>& xc, ambit::BlockedTensor yo,
              std::map<StateInfo, ambit::Tensor>& yc);

    // orbital part of b (source term)
    ambit::BlockedTensor bo_;
    // CI part of b (source term)
    std::map<StateInfo, ambit::Tensor> bc_;
    // set orbital part of b
    void set_bo_block(const std::string& block, ambit::Tensor b);

    // orbital response part of Ax
    ambit::BlockedTensor Zo_;
    // CI response part of Ax
    std::map<StateInfo, ambit::Tensor> Zc_;
    // active integrals used in CI response for SA-MCSCF orbital conditions
    ambit::BlockedTensor Z1_;
    ambit::BlockedTensor Z2_;

    // multipliers for MO coefficients
    ambit::BlockedTensor xo_;
    // multipliers for CI coefficients
    std::map<StateInfo, ambit::Tensor> xc_;

    void set_mo_space();
    void init_rdms();
    void init_ints();
    double compute_target_eref_actv();
};

// class CPMCSCF {
//   public:
//     /**
//      * @brief
//      *
//      */
//     CPMCSCF(std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<ActiveSpaceSolver> as_solver,
//             std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info);

//     std::vector<double> compute_sigma(std::vector<double>& x);

//   private:
//     std::shared_ptr<MCSCF_ORB_GRAD> orb_grad_;
//     std::shared_ptr<ActiveSpaceSolver> as_solver_;
//     std::shared_ptr<ForteOptions> options_;
//     std::shared_ptr<MOSpaceInfo> mo_space_info_;

//     void start_up();
//     void setup_mos();
//     void read_options();

//     /// The DFHelper object of Psi4
//     std::shared_ptr<psi::DFHelper> df_helper_;

//     // => MO spaces related <=

//     /// The number of irreps
//     int nirrep_;

//     /// The number of SO per irrep (AO for C matrices)
//     psi::Dimension nsopi_;
//     /// The number of MO per irrep
//     psi::Dimension nmopi_;
//     /// The number of non-frozen MO per irrep
//     psi::Dimension ncmopi_;
//     /// The number of DOCC (including frozen core) per irrep
//     psi::Dimension ndoccpi_;
//     /// The number of frozen DOCC per irrep
//     psi::Dimension nfrzcpi_;
//     /// The number of frozen UOCC per irrep
//     psi::Dimension nfrzvpi_;
//     /// The number of active per irrep
//     psi::Dimension nactvpi_;

//     /// The number of SOs
//     size_t nso_;
//     /// The number of MOs
//     size_t nmo_;
//     /// The number of non-frozen MOs
//     size_t ncmo_;
//     /// The number of active orbitals
//     size_t nactv_;
//     /// The number of frozen-core orbitals
//     size_t nfrzc_;

//     /// Ignore frozen orbitals in the input
//     bool ignore_frozen_ = true;

//     /// List of core MOs (Absolute)
//     std::vector<size_t> core_mos_;
//     /// List of active MOs (Absolute)
//     std::vector<size_t> actv_mos_;
//     /// Map from MO space label to the absolute MO indices
//     std::map<std::string, std::vector<size_t>> label_to_mos_;
//     /// Map from MO space label to the correlated MO indices
//     std::map<std::string, std::vector<size_t>> label_to_cmos_;

//     /// Relative indices within an irrep <irrep, relative indices>
//     std::vector<std::pair<int, size_t>> mos_rel_;

//     /// Relative indices within an MO space <space, relative indices>
//     std::vector<std::pair<std::string, size_t>> mos_rel_space_;
// };

// class MCSCF_ORB_GRAD {
//   public:
//     /**
//      * @brief Constructor of the AO-based CASSCF class
//      * @param options: The ForteOptions pointer
//      * @param mo_space_info: The MOSpaceInfo pointer of Forte
//      * @param ints: The ForteIntegral pointer
//      *
//      * Implementation notes:
//      *   See J. Chem. Phys. 142, 224103 (2015) and Theor. Chem. Acc. 97, 88-95 (1997)
//      */
//     MCSCF_ORB_GRAD(std::shared_ptr<ForteOptions> options, std::shared_ptr<SCFInfo> scf_info,
//                    std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ForteIntegrals>
//                    ints, bool ignore_frozen);

//     /// Evaluate the energy and orbital gradient
//     double evaluate(std::shared_ptr<psi::Vector> x, std::shared_ptr<psi::Vector> g,
//                     bool do_g = true);

//     /// Evaluate the diagonal orbital Hessian
//     void hess_diag(std::shared_ptr<psi::Vector> x, const std::shared_ptr<psi::Vector>& h0);

//     /// Set RDMs used for orbital optimization
//     void set_rdms(std::shared_ptr<RDMs> rdms);

//     /// Return active space integrals for CI
//     std::shared_ptr<ActiveSpaceIntegrals> active_space_ints();

//     /// Return the number of nonredundant orbital rotations
//     size_t nrot() const { return nrot_; }

//     /// Return the initial (not optimized) MO coefficients
//     std::shared_ptr<psi::Matrix> Ca_initial() { return C0_; }

//     /// Return the optimized MO coefficients
//     std::shared_ptr<psi::Matrix> Ca() { return C_; }

//     /// Return the generalized Fock matrix
//     std::shared_ptr<psi::Matrix> fock() { return Fock_; }

//     /// Build and return the generalized Fock matrix
//     std::shared_ptr<psi::Matrix> fock(std::shared_ptr<RDMs> rdms);

//     /// Canonicalize the final orbitals
//     void canonicalize_final(const std::shared_ptr<psi::Matrix>& U);

//     /// Compute nuclear gradient
//     void compute_nuclear_gradient();

//   private:
//     /// The Forte options
//     std::shared_ptr<ForteOptions> options_;

//     /// The SCF information
//     std::shared_ptr<SCFInfo> scf_info_;

//     /// The MOSpaceInfo object
//     std::shared_ptr<MOSpaceInfo> mo_space_info_;

//     /// The Forte integral
//     std::shared_ptr<ForteIntegrals> ints_;

//     /// Common setup for the class
//     void startup();

//     /// Read options
//     void read_options();

//     /// Prepare MO spaces
//     void setup_mos();

//     /// Number of non-redundant pairs for orbital optimization
//     void nonredundant_pairs();

//     /// Initialize/Allocate tensors and matrices
//     void init_tensors();

//     /// The JK object of Psi4
//     std::shared_ptr<psi::JK> JK_;

//     /// The DFHelper object of Psi4
//     std::shared_ptr<psi::DFHelper> df_helper_;

//     /// Algorithm for computing (pu|xy) integrals
//     enum TEIALG { JK, DF };
//     TEIALG tei_alg_ = JK;

//     // => MO spaces related <=

//     /// The number of irreps
//     int nirrep_;

//     /// The number of SO per irrep (AO for C matrices)
//     psi::Dimension nsopi_;
//     /// The number of MO per irrep
//     psi::Dimension nmopi_;
//     /// The number of non-frozen MO per irrep
//     psi::Dimension ncmopi_;
//     /// The number of DOCC (including frozen core) per irrep
//     psi::Dimension ndoccpi_;
//     /// The number of frozen DOCC per irrep
//     psi::Dimension nfrzcpi_;
//     /// The number of frozen UOCC per irrep
//     psi::Dimension nfrzvpi_;
//     /// The number of active per irrep
//     psi::Dimension nactvpi_;

//     /// The number of SOs
//     size_t nso_;
//     /// The number of MOs
//     size_t nmo_;
//     /// The number of non-frozen MOs
//     size_t ncmo_;
//     /// The number of active orbitals
//     size_t nactv_;
//     /// The number of frozen-core orbitals
//     size_t nfrzc_;

//     /// Ignore frozen orbitals in the input
//     bool ignore_frozen_ = true;

//     /// List of core MOs (Absolute)
//     std::vector<size_t> core_mos_;
//     /// List of active MOs (Absolute)
//     std::vector<size_t> actv_mos_;
//     /// Map from MO space label to the absolute MO indices
//     std::map<std::string, std::vector<size_t>> label_to_mos_;
//     /// Map from MO space label to the correlated MO indices
//     std::map<std::string, std::vector<size_t>> label_to_cmos_;

//     /// Relative indices within an irrep <irrep, relative indices>
//     std::vector<std::pair<int, size_t>> mos_rel_;

//     /// Relative indices within an MO space <space, relative indices>
//     std::vector<std::pair<std::string, size_t>> mos_rel_space_;

//     /// Number of orbital rotations considered
//     size_t nrot_;
//     /// List of rotation pairs in <irrep, index1, index2> format
//     std::vector<std::tuple<int, size_t, size_t>> rot_mos_irrep_;
//     /// List of rotation pairs in <block, index1, index2> format
//     std::vector<std::tuple<std::string, size_t, size_t>> rot_mos_block_;

//     // => Options <=

//     /// The printing level
//     int print_;
//     /// Enable debug printing or not
//     bool debug_print_;

//     /// Algorithm to compute the orbital transformation matrix from orbital rotations
//     /// 1. Cayley: U = (1 + R/2) * (1 - R/2)^-1
//     /// 2. Power: U = exp(R) ~ I + R + 1/2 * R^2 + 1/6 * R^3
//     /// 3. Pade: Psi4 implementation of U = exp(R)
//     std::string ortho_trans_algo_;

//     /// Keep internal (GASn-GASn) rotations
//     bool internal_rot_;
//     /// If the active space is from GAS
//     bool gas_ref_;

//     /// User specified zero rotations
//     /// vector of irrep, map from index i to other indices uncoupled with index i
//     std::vector<std::unordered_map<size_t, std::unordered_set<size_t>>> zero_rots_;

//     /// Integral cutoff
//     double ints_cutoff_;

//     // => Tensors and matrices <=

//     /// Initial orbital coefficients
//     std::shared_ptr<psi::Matrix> C0_;
//     /// Current orbital coefficients
//     std::shared_ptr<psi::Matrix> C_;

//     /// The inactive Fock matrix in MO basis
//     std::shared_ptr<psi::Matrix> F_closed_; // nmo x nmo
//     ambit::BlockedTensor Fc_;               // ncmo x ncmo
//     /// The generalized Fock matrix in MO basis
//     std::shared_ptr<psi::Matrix> Fock_; // nmo x nmo
//     ambit::BlockedTensor F_;            // ncmo x ncmo
//     /// Diagonal elements of the generalized Fock matrix (Pitzer ordering)
//     std::vector<double> Fd_;

//     /// Two-electron integrals in chemists' notation (pu|xy)
//     ambit::BlockedTensor V_;

//     /// Spin-summed 1-RDM
//     ambit::BlockedTensor D1_;
//     std::shared_ptr<psi::Matrix> rdm1_;
//     /// Spin-summed averaged 2-RDM in 1^+ 1 2^+ 2 ordering
//     ambit::BlockedTensor D2_;

//     /// The orbital response of MCSCF energy
//     ambit::BlockedTensor A_;
//     std::shared_ptr<psi::Matrix> Am_;

//     /// The orbital rotation matrix
//     std::shared_ptr<psi::Matrix> R_;
//     /// The orthogonal transformation matrix
//     std::shared_ptr<psi::Matrix> U_;

//     /// The orbital gradients
//     ambit::BlockedTensor g_;
//     std::shared_ptr<psi::Vector> grad_;
//     /// The orbital diagonal Hessian
//     ambit::BlockedTensor h_diag_;
//     std::shared_ptr<psi::Vector> hess_diag_;

//     /// G intermediates when forming internal diagonal Hessian
//     ambit::BlockedTensor Guu_;
//     ambit::BlockedTensor Guv_;
//     /// Intermediate (TEI) when forming internal diagonal Hessian
//     ambit::BlockedTensor jk_internal_;
//     /// Intermediate (2RDM) when forming internal diagonal Hessian
//     ambit::BlockedTensor d2_internal_;

//     // => functions used in micro iteration <=

//     /// Build integrals for gradients and Hessian
//     void build_mo_integrals();

//     /// Build two-electron integrals
//     void build_tei_jk();
//     void build_tei_df();

//     /// Fill two-electron integrals for custom integrals
//     void fill_tei_custom(ambit::BlockedTensor V);

//     /// JK build for Fock-like terms
//     void JK_build(std::shared_ptr<psi::Matrix> Cl, std::shared_ptr<psi::Matrix> Cr);

//     /// Build Fock matrix
//     void build_fock(bool rebuild_inactive = false);
//     /// Build the inactive Fock (does not depend on 1RDM), includes frozen docc
//     void build_fock_inactive();
//     /// Build the active Fock (does depend on 1RDM)
//     void build_fock_active();

//     /// Compute the energy for given sets of orbitals and density
//     void compute_reference_energy();
//     /// The energy computed using the current orbitals and CI coefficients
//     double energy_;
//     /// The closed-shell energy
//     double e_closed_;

//     /// Compute the orbital gradients
//     void compute_orbital_grad();
//     /// Compute the diagonal Hessian for orbital rotations
//     void compute_orbital_hess_diag();

//     /// Update orbitals using the given rotation matrix in vector form
//     bool update_orbitals(std::shared_ptr<psi::Vector> x);

//     /// Test if new orbitals are significantly different from the beginning orbitals
//     /// Return a tuple of <irrep, old active orbital index, new active orbital index>
//     std::vector<std::tuple<int, int, int>>
//     test_orbital_rotations(const std::shared_ptr<psi::Matrix>& U, const std::string&
//     warning_msg);

//     // => Nuclear gradient related functions <=

//     /// compute AO Lagrangian matrix and push to Psi4
//     void compute_Lagrangian();

//     /// compute AO 1-RDM and push to Psi4
//     void compute_opdm_ao();

//     /// Dump the MCSCF MO 2-RDM to file using IWL
//     void dump_tpdm_iwl();
//     /// Dump the Hartree-Fock MO 2-RDM to file using IWL
//     void dump_tpdm_iwl_hf();
//     /// Dump AO 2-RDM to file for DF-MCSCF
//     void dump_tpdm_df();
//     /// Dump the Hartree-Fock AO 2-RDM to file for DF-MCSCF
//     void dump_tpdm_df_hf(std::shared_ptr<psi::Matrix> Jm12, std::shared_ptr<psi::Matrix> d3,
//                          std::shared_ptr<psi::Matrix> d2);

//     /// Are there any frozen orbitals?
//     bool is_frozen_orbs_;

//     /// Start up for doing gradient with frozen orbitals
//     void setup_grad_frozen();

//     /// Doubly occupied MOs from Hartree-Fock
//     psi::Dimension hf_ndoccpi_;
//     /// Unoccupied MOs from Hartree-Fock
//     psi::Dimension hf_nuoccpi_;
//     /// List of occupied MOs from Hartree-Fock
//     std::vector<size_t> hf_docc_mos_;
//     /// List of unoccupied MOs from Hartree-Fock
//     std::vector<size_t> hf_uocc_mos_;

//     /// Hartree-Fock orbital energies
//     std::shared_ptr<psi::Vector> epsilon_;

//     /// Compute the frozen part of the A matrix
//     void build_Am_frozen();

//     /// Z vector for CPSCF equations
//     std::shared_ptr<psi::Matrix> Z_;
//     /// Solve Z vector equation if there are frozen orbitals
//     void solve_cpscf();

//     /**
//      * Contract Roothaan-Bagus supermatrix with Z: sum_{pq} Z_{pq} L_{pq,rs}
//      * Roothaan-Bagus supermatrix L_{pq,rs} = 4 * (pq|rs) - (pr|sq) - (ps|rq)
//      *
//      * Express contraction in AO basis:
//      * sum_{pq} sum_{PQRS} CZrow_{Pp} Z_{pq} CZcol_{Qq} L_{PQ,RS} Crow_{Rr} Ccol_{Ss}
//      */
//     std::shared_ptr<psi::Matrix> contract_RB_Z(std::shared_ptr<psi::Matrix> Z,
//                                                std::shared_ptr<psi::Matrix> C_Zrow,
//                                                std::shared_ptr<psi::Matrix> C_Zcol,
//                                                std::shared_ptr<psi::Matrix> C_row,
//                                                std::shared_ptr<psi::Matrix> C_col);

//     // => Some helper functions <=

//     /// Format the Fock matrix from SharedMatrix to BlockedTensor
//     void format_fock(std::shared_ptr<psi::Matrix> Fock, ambit::BlockedTensor F);

//     /// Format the 1RDM from BlockedTensor to SharedMatrix
//     void format_1rdm();

//     /// Fill Am_ matrix from BlockedTensor A
//     void fill_A_matrix_data(ambit::BlockedTensor A);

//     /// Reshape the orbital rotation related BlockedTensor to std::shared_ptr<psi::Vector>
//     void reshape_rot_ambit(ambit::BlockedTensor bt, const std::shared_ptr<psi::Vector>& sv);

//     /// Compute the exponential of a skew-symmetric matrix
//     std::shared_ptr<psi::Matrix> matrix_exponential(const std::shared_ptr<psi::Matrix>& A, int
//     n);

//     /// Compute Cayley transformation from skew-symmetric matrix
//     std::shared_ptr<psi::Matrix> cayley_trans(const std::shared_ptr<psi::Matrix>& A);

//     /// Grab part of the orbital coefficients
//     std::shared_ptr<psi::Matrix> C_subset(const std::string& name, std::shared_ptr<psi::Matrix>
//     C,
//                                           psi::Dimension dim_start, psi::Dimension dim_end);

//     /// Threshold for numerical zero
//     double numerical_zero_ = 1.0e-15;
// };
} // namespace forte
