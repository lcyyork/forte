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

#include <functional>
#include <vector>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libqt/qt.h"

#include "mrdsrg-spin-integrated/dsrg_mrpt2.h"

namespace forte {

class GMRES {
  protected:
    /// the convergence criteria for residual norm
    double r_conv_;
    /// the max macro iterations allowed
    int maxiter_macro_ = 5;
    /// the max micro iterations allowed
    int maxiter_micro_ = 50;
    /// the max memory allowed in bytes (default: ~1 GB)
    size_t max_mem_ = 1e9;
    /// is the solver converged?
    bool converged_ = false;

  public:
    /**
     * @brief Constructor of the generalized minimal residual method
     * @param maxmem the max memory allowed in bytes
     * @param rconv the convergence criteria for residual norm
     *
     * Implemention notes:
     *   See Wikipedia https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
     */
    GMRES(double rconv) : r_conv_(rconv) {}

    /**
     * @brief Solve the linear system Ax = b using GMRES with restart and left preconditioning
     * @param foo Target class that should have the following methods:
     *             y = foo.compute_sigma(q) which form the sigma vector y = Aq
     * @param b the right-hand-side of the linear system
     * @param x0 the initial guess (in) / the solution vector (out)
     * @param M0 the Jacobi preconditioner (inverse of the diagonal elements of A)
     * @return the solution vector x
     */
    template <class Foo>
    void solve(Foo& foo, std::shared_ptr<psi::Vector> b, std::shared_ptr<psi::Vector> x,
               std::shared_ptr<psi::Vector> M0 = nullptr) {
        auto nirrep = b->nirrep();
        auto dimpi = b->dimpi();
        auto nelements = dimpi.sum();

        if (dimpi != x->dimpi()) {
            throw std::runtime_error("Inconsistent dimensions between b and x0");
        }

        if (M0) {
            if (dimpi != M0->dimpi()) {
                throw std::runtime_error("Inconsistent dimensions between b and M0");
            }
        }
        auto apply_M0 = [&](std::shared_ptr<psi::Vector> x) {
            if (M0) {
                for (int h = 0; h < nirrep; ++h) {
                    for (int i = 0; i < dimpi[h]; ++i) {
                        x->set(h, i, x->get(h, i) * M0->get(h, i));
                    }
                }
            }
        };

        int mmiter = std::min(max_mem_ / (8 * nelements), (size_t)maxiter_micro_) - (M0 ? 2 : 1);
        if (mmiter < 3) {
            throw std::runtime_error("Not enough memory for GMRES. Need at least " +
                                     std::to_string(40 * nelements) + " bytes of memory.");
        }

        converged_ = false;
        auto Q = std::vector<std::shared_ptr<psi::Vector>>(mmiter + 1);
        for (int iter = 0; iter < maxiter_macro_; ++iter) {
            std::vector<double> sn(mmiter), cn(mmiter), beta(mmiter + 1);
            auto H = std::make_shared<psi::Matrix>("H", mmiter + 1, mmiter);
            auto s = foo.compute_sigma(x);
            b->subtract(*s);
            apply_M0(b);
            auto bnorm = b->norm();
            b->scale(1.0 / bnorm);
            b->set_name("q0");
            Q[0] = b;
            beta[0] = bnorm;
            int k = 0;
            do {
                auto y = foo.compute_sigma(Q[k]);
                y->set_name("q" + std::to_string(k + 1));
                apply_M0(y);
                for (int j = 0; j < k + 1; ++j) {
                    auto Hjk = Q[j]->vector_dot(*y);
                    H->set(j, k, Hjk);
                    y->axpy(-Hjk, *Q[j]);
                }
                auto ynorm = y->norm();
                H->set(k + 1, k, ynorm);
                y->scale(1.0 / ynorm);
                Q[k + 1] = y;
                for (int j = 0; j < k; ++j) {
                    auto a = H->get(j, k), b = H->get(j + 1, k);
                    H->set(j + 1, k, -sn[j] * a + cn[j] * b);
                    H->set(j, k, cn[j] * a + sn[j] * b);
                }
                auto h = H->get(k, k), g = H->get(k + 1, k);
                auto rho = std::sqrt(h * h + g * g);
                sn[k] = g / rho;
                cn[k] = h / rho;
                H->set(k, k, cn[k] * h + sn[k] * g);
                H->set(k + 1, k, 0);
                beta[k + 1] = -sn[k] * beta[k];
                beta[k] = cn[k] * beta[k];
                psi::outfile->Printf("\n  macro %2d  micro %2d  error %13.6e", iter, k,
                                     beta[k + 1]);
                if (fabs(beta[++k]) < r_conv_) { // we increase k by 1 here!!!
                    converged_ = true;
                    break;
                }
            } while (k < mmiter);

            auto Hk = std::make_shared<psi::Matrix>("H", k, k);
            for (int m = 0; m < k; ++m) {
                for (int n = 0; n < k; ++n) {
                    Hk->set(m, n, H->get(m, n));
                }
            }
            auto Hk_ptr = Hk->pointer();
            psi::C_DTRSV('U', 'N', 'N', k, Hk_ptr[0], k, beta.data(), 1);

            for (int i = 0; i < k; ++i) {
                Q[i]->scale(beta[i]);
                x->add(*Q[i]);
            }
            if (converged_)
                break;
        }
    }

    /// Return true if minimization converged
    bool converged() const { return converged_; }
    /// Set max number of iterations for macro iteration
    void set_max_iter_macro(int maxiter) { maxiter_macro_ = maxiter; }
    /// Set max number of iterations for micro iteration
    void set_max_iter_micro(int maxiter) { maxiter_micro_ = maxiter; }
    /// Set max memory (in bytes) allowed
    void set_max_memory(size_t mem) { max_mem_ = mem; }
};

template void GMRES::solve(DSRG_MRPT2& func, std::shared_ptr<psi::Vector> b,
                           std::shared_ptr<psi::Vector> x,
                           std::shared_ptr<psi::Vector> M0 = nullptr);

} // namespace forte
