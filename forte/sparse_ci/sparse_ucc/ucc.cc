#include <sstream>
#include <functional>
#include <algorithm>
#include <numeric>

#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libdiis/diismanager.h"

#include "forte-def.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/scf_info.h"
#include "integrals/active_space_integrals.h"
#include "helpers/printing.h"
#include "helpers/string_algorithms.h"
#include "helpers/timer.h"

#include "sparse_ci/sparse_hamiltonian.h"
#include "sparse_ci/sparse_exp.h"
#include "sparse_ci/sparse_ucc/ucc.h"

using namespace psi;

namespace forte {

SparseUCC::SparseUCC(std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                     std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : as_ints_(as_ints), scf_info_(scf_info), options_(options), mo_space_info_(mo_space_info) {
    n_threads_ = omp_get_max_threads();
    std::string thread_title =
        std::to_string(n_threads_) + (n_threads_ > 1 ? " OMP threads" : " OMP thread");
    print_method_banner({"Single-Reference Unitary Coupled-Cluster Using Sparse Operators",
                         "written by Chenyang Li", thread_title});

    ex_level_ = options_->get_int("UCC_EX_LEVEL");
    maxiter_ = options_->get_int("MAXITER");
    e_convergence_ = options_->get_double("E_CONVERGENCE");
    r_convergence_ = options_->get_double("R_CONVERGENCE");

    auto frzcpi = mo_space_info_->dimension("FROZEN_DOCC");
    auto doccpi = scf_info_->doccpi() - frzcpi;
    auto nmopi = mo_space_info_->dimension("ACTIVE");
    auto nirrep = mo_space_info_->nirrep();
    for (int h = 0, shift = 0; h < nirrep; ++h) {
        for (size_t i = 0; i < doccpi[h]; ++i) {
            rdocc_mos_.push_back(i + shift);
            det0_.create_alfa_bit(i + shift);
            det0_.create_beta_bit(i + shift);
        }
        for (size_t a = doccpi[h]; a < nmopi[h]; ++a) {
            ruocc_mos_.push_back(a + shift);
        }
        shift += nmopi[h];
    }

    outfile->Printf("\n  Hartree-Fock determinants: %s", str(det0_, nmopi.sum()).c_str());
    outfile->Printf("\n  Occupied orbitals: %s", container_to_string(rdocc_mos_).c_str());
    outfile->Printf("\n  Virtual orbitals:  %s", container_to_string(ruocc_mos_).c_str());

    hamiltonian_ = std::make_shared<SparseHamiltonian>(as_ints_);
    exponential_ = std::make_shared<SparseExp>(20, 1.0e-15);

    // build excitation operators
    excitation_operators_.resize(nirrep);
    auto symmetry = mo_space_info_->symmetry("CORRELATED");

    auto combinations = [](std::vector<size_t> vec, int K) {
        auto N = vec.size();
        std::vector<std::vector<int>> out;
        std::string bitmask(K, 1); // K leading 1's
        bitmask.resize(N, 0);      // N-K trailing 0's
        do {
            std::vector<int> temp;
            temp.reserve(K);
            for (int i = 0; i < N; ++i) {
                if (bitmask[i])
                    temp.push_back(vec[i]);
            }
            out.push_back(temp);
        } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
        return out;
    };

    std::vector<std::vector<std::vector<int>>> rdocc_combinations(ex_level_);
    std::vector<std::vector<std::vector<int>>> ruocc_combinations(ex_level_);
    for (int n = 0; n < ex_level_; ++n) {
        rdocc_combinations[n] = combinations(rdocc_mos_, n + 1);
        ruocc_combinations[n] = combinations(ruocc_mos_, n + 1);
    }

    // pure alpha or beta
    for (int n = 1; n < ex_level_ + 1; ++n) {
        for (const auto& rdocc_combination : rdocc_combinations[n - 1]) {
            for (const auto& ruocc_combination : ruocc_combinations[n - 1]) {
                int irrep = 0;
                excitation_operator alpha, beta;
                for (auto a : ruocc_combination) {
                    irrep ^= symmetry[a];
                    alpha.a_cre.push_back(a);
                    beta.b_cre.push_back(a);
                }
                for (size_t i = rdocc_combination.size(); i > 0; --i) {
                    auto ni = rdocc_combination[i - 1];
                    irrep ^= symmetry[ni];
                    alpha.a_ann.push_back(ni);
                    beta.b_ann.push_back(ni);
                }
                excitation_operators_[irrep].push_back(alpha);
                excitation_operators_[irrep].push_back(beta);
            }
        }
    }

    for (int n = 1; n < ex_level_ + 1; ++n) {
        for (int ne_alpha = 1; ne_alpha < n; ++ne_alpha) {
            auto ne_beta = n - ne_alpha;
            for (const auto& ao : rdocc_combinations[ne_alpha - 1]) {
                for (const auto& bo : rdocc_combinations[ne_beta - 1]) {
                    for (const auto& av : ruocc_combinations[ne_alpha - 1]) {
                        for (const auto& bv : ruocc_combinations[ne_beta - 1]) {
                            int irrep = 0;
                            excitation_operator ex;
                            for (auto a : av) {
                                irrep ^= symmetry[a];
                                ex.a_cre.push_back(a);
                            }
                            for (auto a : bv) {
                                irrep ^= symmetry[a];
                                ex.b_cre.push_back(a);
                            }
                            for (size_t i = bo.size(); i > 0; --i) {
                                irrep ^= symmetry[bo[i - 1]];
                                ex.b_ann.push_back(bo[i - 1]);
                            }
                            for (size_t i = ao.size(); i > 0; --i) {
                                irrep ^= symmetry[ao[i - 1]];
                                ex.a_ann.push_back(ao[i - 1]);
                            }
                            excitation_operators_[irrep].push_back(ex);
                        }
                    }
                }
            }
        }
    }

    // for (int h = 0; h < nirrep; ++h) {
    //     outfile->Printf("\n Irrep %d", h);
    //     for (const auto& s : excitation_operators_[h]) {
    //         outfile->Printf("\n  %s", s.to_string().c_str());
    //     }
    // }

    outfile->Printf("\n  UCC excitation level: %d", ex_level_);
    outfile->Printf("\n  Number of cluster operators per irrep: ");
    for (int h = 0; h < nirrep; ++h) {
        outfile->Printf("%4zu", excitation_operators_[h].size());
    }
}

void SparseUCC::build_cluster_operators() {
    cluster_operators_ = SparseOperatorList();
    denominators_.clear();
    auto epsilons = scf_info_->epsilon_a();
    auto relative_mos = mo_space_info_->relative_mo("ACTIVE");
    for (const auto& ops : excitation_operators_[0]) {
        cluster_operators_.add_term_from_str(ops.to_string(), 0.0);
        double delta = 0.0;
        for (auto a : ops.a_cre) {
            auto [h, i] = relative_mos[a];
            delta += epsilons->get(h, i);
        }
        for (auto a : ops.b_cre) {
            auto [h, i] = relative_mos[a];
            delta += epsilons->get(h, i);
        }
        for (auto a : ops.a_ann) {
            auto [h, i] = relative_mos[a];
            delta -= epsilons->get(h, i);
        }
        for (auto a : ops.b_ann) {
            auto [h, i] = relative_mos[a];
            delta -= epsilons->get(h, i);
        }
        denominators_.push_back(delta);
    }
}

double SparseUCC::compute_energy() {
    build_cluster_operators();

    double e = 0.0;
    bool converged = false;
    SparseState ref;
    ref[det0_] = 1.0;

    auto diis_min_vec = options_->get_int("DIIS_MIN_VECS");
    auto diis_max_vec = options_->get_int("DIIS_MAX_VECS");

    auto diis_manager = std::make_shared<DIISManager>(diis_max_vec, "UCC DIIS",
                                                      DIISManager::RemovalPolicy::LargestError,
                                                      DIISManager::StoragePolicy::InCore);
    auto t = std::make_shared<psi::Vector>(cluster_operators_.size());
    diis_manager->set_vector_size(t);
    diis_manager->set_error_vector_size(t);
    int bad_diis_count = 0;
    double rms_old = 0.0;

    for (int iter = 0; iter < maxiter_; ++iter) {
        local_timer timer;

        // compute residuals
        local_timer t1;
        auto wfn = exponential_->apply_antiherm(cluster_operators_, ref);
        outfile->Printf("\n  build wfn: %10.4e", t1.get());
        t1.reset();
        auto Hwfn = hamiltonian_->compute_on_the_fly(wfn, screen_thresh_);
        outfile->Printf("\n  build Hwfn: %10.4e", t1.get());
        t1.reset();
        auto Hbar_wfn = exponential_->apply_antiherm(cluster_operators_, Hwfn, -1.0);
        outfile->Printf("\n  build Hbar_wfn: %10.4e", t1.get());

        t1.reset();
        auto res = get_projection(cluster_operators_, ref, Hbar_wfn);
        outfile->Printf("\n  compute residual: %10.4e", t1.get());
        t1.reset();
        std::transform(res.begin(), res.end(), denominators_.begin(), res.begin(),
                       std::divides<double>());
        auto rms = sqrt(std::inner_product(res.begin(), res.end(), res.begin(), 0.0));
        outfile->Printf("\n  compute rms: %10.4e", t1.get());

        auto e_this = Hbar_wfn[det0_];
        auto de = e_this - e;
        e = e_this;

        outfile->Printf(
            "\n  Iteration %3d  Energy: %20.15f  Delta: %10.4e  RMS: %10.4e  Time: %10.4e", iter, e,
            de, rms, timer.get());

        if (iter > 2 and fabs(de) < e_convergence_ and rms < r_convergence_) {
            converged = true;
            break;
        }

        psi::Vector dt(cluster_operators_.size());
        for (size_t i = 0, size = cluster_operators_.size(); i < size; ++i) {
            dt.set(i, -res[i]);
            t->set(i, cluster_operators_[i]);
        }
        t->add(dt);
        if (iter >= 2) {
            outfile->Printf("  ");
            if (bad_diis_count > 3) {
                psi::outfile->Printf("R/");
                diis_manager->reset_subspace();
                bad_diis_count = 0;
            }
            diis_manager->add_entry(dt, t);
            outfile->Printf("S");

            if (diis_manager->subspace_size() >= diis_min_vec) {
                dt = t->clone();
                diis_manager->extrapolate(t);
                outfile->Printf("/E");
                if (rms / rms_old > 1.0 and rms < 4.0e-3) {
                    bad_diis_count++;
                }
                dt.subtract(*t);
                outfile->Printf("\n t - told = %.15f", dt.norm());
            }
        }
        for (size_t i = 0, size = cluster_operators_.size(); i < size; ++i) {
            cluster_operators_[i] = t->get(i);
        }
        rms_old = rms;
    }

    diis_manager->reset_subspace();
    diis_manager->delete_diis_file();

    return e;
}

} // namespace forte