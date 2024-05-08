#include <memory>
#include <vector>
#include <string>

#include "psi4/libmints/dimension.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"

#include "helpers/string_algorithms.h"

#include "sparse_ci/determinant.hpp"
#include "sparse_ci/sparse_hamiltonian.h"
#include "sparse_ci/sparse_state.h"
#include "sparse_ci/sparse_exp.h"
#include "sparse_ci/sparse_operator.h"

namespace forte {

class ActiveSpaceIntegrals;
class SCFInfo;
class ForteOptions;
class MOSpaceInfo;

class SparseUCC {
  public:
    SparseUCC(std::shared_ptr<ActiveSpaceIntegrals> as_ints, std::shared_ptr<SCFInfo> scf_info,
              std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info);

    double compute_energy();

    void set_excitation_level(int ex_level) { ex_level_ = ex_level; }

  protected:
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;
    std::shared_ptr<SCFInfo> scf_info_;
    std::shared_ptr<ForteOptions> options_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    int n_threads_;
    int ex_level_;
    int maxiter_;
    double e_convergence_;
    double r_convergence_;
    double screen_thresh_ = 1.0e-15;

    double energy_;

    std::vector<size_t> rdocc_mos_;
    std::vector<size_t> ruocc_mos_;

    Determinant det0_;

    std::shared_ptr<SparseHamiltonian> hamiltonian_;
    std::shared_ptr<SparseExp> exponential_;

    SparseOperatorList cluster_operators_;
    std::vector<double> denominators_;

    // excitation operator
    struct excitation_operator {
        std::vector<size_t> a_cre;
        std::vector<size_t> b_cre;
        std::vector<size_t> a_ann;
        std::vector<size_t> b_ann;
        std::string to_string() const {
            std::vector<std::string> vec;
            for (auto a : a_cre)
                vec.push_back(std::to_string(a) + "a+");
            for (auto a : b_cre)
                vec.push_back(std::to_string(a) + "b+");
            for (auto i : b_ann)
                vec.push_back(std::to_string(i) + "b-");
            for (auto i : a_ann)
                vec.push_back(std::to_string(i) + "a-");
            return "[" + join(vec, " ") + "]";
        };
    };

    // excitation operators for each irrep
    std::vector<std::vector<excitation_operator>> excitation_operators_;

    void build_cluster_operators();

    std::vector<int> nroots_per_irrep_;

    std::vector<std::shared_ptr<psi::Matrix>> HbarIJ_per_irrep_;
    std::vector<std::shared_ptr<psi::Matrix>> S2IJ_per_irrep_;
    std::vector<std::shared_ptr<psi::Matrix>> evecs_per_irrep_;
    std::vector<std::shared_ptr<psi::Vector>> evals_per_irrep_;

    void diagonalize_eom_ee_hbar();
};
} // namespace forte