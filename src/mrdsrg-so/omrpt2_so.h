#ifndef OMRPT2_SO_H
#define OMRPT2_SO_H

#include "ambit/tensor.h"
#include "ambit/blocked_tensor.h"

#include "base_classes/forte_options.h"
#include "base_classes/dynamic_correlation_solver.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"
#include "helpers/blockedtensorfactory.h"
#include "mrdsrg-helper/dsrg_source.h"

using namespace ambit;

namespace forte {

class OMRPT2_SO : public DynamicCorrelationSolver
{
public:
    OMRPT2_SO(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
              std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~OMRPT2_SO();

    /// Compute the DSRG-MRPT2 energy
    double compute_energy() override;

    /// DSRG transformed Hamiltonian (not implemented)
    std::shared_ptr<ActiveSpaceIntegrals> compute_Heff_actv() override;
protected:
    /// Called in the constructor
    void startup();
    /// Throw for not implemented features
    void not_implemented();

    /// Read options
    void read_options();
    /// Print a summary of the options
    void print_options();

    /// Read MO space info
    void read_MOSpaceInfo();

    /// List of core SOs
    std::vector<size_t> core_sos_;
    /// List of active SOs
    std::vector<size_t> actv_sos_;
    /// List of virtual SOs
    std::vector<size_t> virt_sos_;

    /// List of alpha core SOs
    std::vector<size_t> acore_sos_;
    /// List of alpha active SOs
    std::vector<size_t> aactv_sos_;
    /// List of alpha virtual SOs
    std::vector<size_t> avirt_sos_;
    /// List of beta core SOs
    std::vector<size_t> bcore_sos_;
    /// List of beta active SOs
    std::vector<size_t> bactv_sos_;
    /// List of beta virtual SOs
    std::vector<size_t> bvirt_sos_;

    /// Set Ambit tensor labels
    void set_ambit_MOSpace();
    /// Kevin's Tensor Wrapper
    std::shared_ptr<BlockedTensorFactory> BTF_;
    /// Tensor type for Ambit
    ambit::TensorType tensor_type_;

    /// Initialize density cumulants
    void init_density();
    /// One-particle density matrix
    ambit::BlockedTensor D1_;
    /// Two-body denisty matrix
    ambit::BlockedTensor D2_;

    /// Fill up integrals
    void init_ints();
    /// One-electron integral
    ambit::BlockedTensor H_;
    /// Two-electron integral
    ambit::BlockedTensor V_;
    /// Identity matrix
    ambit::BlockedTensor I_;

    /// Initialize Fock matrix
    void init_fock();
    /// Generalized Fock matrix
    ambit::BlockedTensor F_;
    /// Semicanonical orbital energies
    std::vector<double> Fd_;

    /// Check orbitals if semicanonical
    bool check_semi_orbs();
    /// Are orbitals semi-canonicalized?
    bool semi_canonical_;

    /// Max number of iterations
    int maxiter_;
    /// Energy convergence
    double e_convergence_;
    /// Density convergence
    double d_convergence_;

    /// Flow parameter
    double s_;
    /// Source operator
    std::string source_;
    /// Source operator for the core-core-virtual-virtual block
    std::string ccvv_source_;
    /// The dsrg source operator
    std::shared_ptr<DSRG_SOURCE> dsrg_source_;
    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;

    /// Number of amplitudes will be printed in amplitude summary
    size_t ntamp_;
    /// Threshold for amplitudes considered as intruders
    double intruder_tamp_;

    /// Compute T amplitudes
    void compute_amplitudes();
    /// T2 amplitudes
    ambit::BlockedTensor T2_;

    /// Compute effective first-order integrals
    void renormalize_ints();
    /// Effective 1st-order 2-body integrals
    ambit::BlockedTensor M2_;

    /// Compute reference energy
    double compute_reference_energy();
    /// Compute correlation energy
    double compute_correlation_energy();
};
}
#endif // OMRPT2_SO_H
