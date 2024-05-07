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

    auto nirrep = mo_space_info_->nirrep();

    ex_level_ = options_->get_int("UCC_EX_LEVEL");
    maxiter_ = options_->get_int("MAXITER");
    e_convergence_ = options_->get_double("E_CONVERGENCE");
    r_convergence_ = options_->get_double("R_CONVERGENCE");
    nroots_per_irrep_ = options_->get_int_list("UCC_NROOTS_PER_IRREP");
    if (nroots_per_irrep_.size() == 0)
        nroots_per_irrep_ = std::vector<int>(nirrep, 0);
    if (nroots_per_irrep_.size() != nirrep) {
        throw std::runtime_error(
            "Number of roots per irrep must be the same as the number of irreps.");
    }

    auto frzcpi = mo_space_info_->dimension("FROZEN_DOCC");
    auto doccpi = scf_info_->doccpi() - frzcpi;
    auto nmopi = mo_space_info_->dimension("ACTIVE");
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

    // pure alpha or beta excitations
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

    // mixed alpha and beta excitations
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

    outfile->Printf("\n  UCC excitation level: %d", ex_level_);
    outfile->Printf("\n  Number of cluster operators per irrep: ");
    for (int h = 0; h < nirrep; ++h) {
        outfile->Printf(" %zu", excitation_operators_[h].size());
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
    auto dt = std::make_shared<psi::Vector>(cluster_operators_.size());
    diis_manager->set_vector_size(t);
    diis_manager->set_error_vector_size(t);
    int bad_diis_count = 0;
    double rms_old = 0.0;

    for (int iter = 0; iter < maxiter_; ++iter) {
        local_timer timer;

        // compute residuals
        auto wfn = exponential_->apply_antiherm(cluster_operators_, ref);
        auto Hwfn = hamiltonian_->compute_on_the_fly(wfn, screen_thresh_);
        auto Hbar_wfn = exponential_->apply_antiherm(cluster_operators_, Hwfn, -1.0);

        auto res = get_projection(cluster_operators_, ref, Hbar_wfn);
        std::transform(res.begin(), res.end(), denominators_.begin(), res.begin(),
                       std::divides<double>());
        auto rms = sqrt(std::inner_product(res.begin(), res.end(), res.begin(), 0.0));

        auto e_this = Hbar_wfn[det0_];
        auto de = e_this - e;
        e = e_this;

        outfile->Printf(
            "\n  Iteration %3d  Energy: %20.15f  Delta: %10.3e  RMS: %10.4e  Time: %10.4e", iter, e,
            de, rms, timer.get());

        if (iter > 2 and fabs(de) < e_convergence_ and rms < r_convergence_) {
            converged = true;
            break;
        }

        // DIIS
        for (size_t i = 0, size = cluster_operators_.size(); i < size; ++i) {
            dt->set(i, -res[i]);
            t->set(i, cluster_operators_[i]);
        }
        t->add(*dt);
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
                diis_manager->extrapolate(t);
                outfile->Printf("/E");
                if (rms / rms_old > 1.0 and rms < 4.0e-3) {
                    bad_diis_count++;
                }
            }
        }
        for (size_t i = 0, size = cluster_operators_.size(); i < size; ++i) {
            cluster_operators_[i] = t->get(i);
        }
        rms_old = rms;
    }

    diis_manager->reset_subspace();
    diis_manager->delete_diis_file();

    if (not converged) {
        outfile->Printf("\n  WARNING: UCC DIIS did not converge after %d iterations", maxiter_);
    }

    if (std::reduce(nroots_per_irrep_.begin(), nroots_per_irrep_.end())) {
        if (not converged)
            throw std::runtime_error(
                "UCC ground-state energy did not converge. Cannot compute excited states.");
        energy_ = e;
        build_HbarIJ();
    }

    return e;
}

void SparseUCC::build_HbarIJ() {
    auto nirrep = mo_space_info_->nirrep();
    HbarIJ_per_irrep_.resize(nirrep);
    S2IJ_per_irrep_.resize(nirrep);
    evecs_per_irrep_.resize(nirrep);
    evals_per_irrep_.resize(nirrep);

    for (size_t h = 0; h < nirrep; ++h) {
        if (nroots_per_irrep_[h] == 0)
            continue;

        print_h2("Building EOM-UCC Hamiltonian for irrep " + std::to_string(h));

        auto dim = excitation_operators_[h].size();

        local_timer timer;
        std::vector<Determinant> ex_dets;
        ex_dets.reserve(dim);
        for (const auto& ex : excitation_operators_[h]) {
            auto det = Determinant(det0_);
            for (auto a : ex.a_ann)
                det.destroy_alfa_bit(a);
            for (auto a : ex.b_ann)
                det.destroy_beta_bit(a);
            for (auto i : ex.a_cre)
                det.create_alfa_bit(i);
            for (auto i : ex.b_cre)
                det.create_beta_bit(i);
            ex_dets.push_back(det);
        }
        outfile->Printf("\n    Time for building excited determinants: %10.4e seconds",
                        timer.get());
        timer.reset();

        auto n_threads = omp_get_max_threads();
        if (dim < n_threads)
            n_threads = dim;

        // p_I = exp(sigma) b_I^+ |ref>
        std::vector<std::unordered_map<size_t, SparseState>> pIs_t(n_threads);
        // Hp_I = H exp(sigma) b_I^+ |ref>
        std::vector<std::unordered_map<size_t, SparseState>> HpIs_t(n_threads);

#pragma omp parallel for num_threads(n_threads)
        for (size_t i = 0; i < dim; ++i) {
            auto thread = omp_get_thread_num();
            SparseState bI;
            bI[ex_dets[i]] = 1.0;
            pIs_t[thread][i] = exponential_->apply_antiherm(cluster_operators_, bI);
            HpIs_t[thread][i] = hamiltonian_->compute_on_the_fly(pIs_t[thread][i], screen_thresh_);
        }
        for (size_t i = 1; i < n_threads; ++i) {
            pIs_t[0].merge(pIs_t[i]);
            HpIs_t[0].merge(HpIs_t[i]);
        }
        outfile->Printf("\n    Time for building itermediate states:   %10.4e seconds",
                        timer.get());
        timer.reset();

        auto Hbar = std::make_shared<psi::Matrix>(dim, dim);
        auto S2 = std::make_shared<psi::Matrix>(dim, dim);

#pragma omp parallel for num_threads(n_threads)
        for (size_t i = 0; i < dim * dim; ++i) {
            auto I = i / dim;
            auto J = i % dim;
            Hbar->set(I, J, overlap(pIs_t[0][I], HpIs_t[0][J]) - (energy_ * (I == J ? 1 : 0)));
            S2->set(I, J, spin2(ex_dets[I], ex_dets[J]));
        }
        outfile->Printf("\n    Time for building Hbar_IJ and S2_IJ:    %10.4e seconds",
                        timer.get());
        timer.reset();

        HbarIJ_per_irrep_[h] = Hbar;
        S2IJ_per_irrep_[h] = S2;

        auto evecs = std::make_shared<psi::Matrix>(dim, dim);
        auto evals = std::make_shared<psi::Vector>(dim);
        Hbar->diagonalize(evecs, evals);
        outfile->Printf("\n    Time for diagonalizing Hbar_IJ matrix:  %10.4e seconds",
                        timer.get());

        evecs_per_irrep_[h] = evecs;
        evals_per_irrep_[h] = evals;

        evals->print();
    }
}

} // namespace forte