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
#include <numeric>
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
    void solve(Foo& foo, const std::vector<double>& b, std::vector<double>& x,
               const std::vector<double>& Minv = {}) {
        if (b.empty())
            throw std::runtime_error("Empty b vector!");
        if (x.empty())
            throw std::runtime_error("Empty x vector!");
        auto dim = b.size();
        if (x.size() != dim)
            throw std::runtime_error("Inconsistent size between b and x!");
        if (!Minv.empty() and Minv.size() != dim)
            throw std::runtime_error("Inconsistent size between b and Minv!");

        int mmiter = std::min(max_mem_ / (8 * dim), (size_t)maxiter_micro_);
        if (mmiter < 3) {
            throw std::runtime_error("Not enough memory for GMRES. Need at least " +
                                     std::to_string(24 * dim) + " bytes of memory.");
        }

        // some helper functions
        auto apply_Minv = [&](std::vector<double>& v) {
            if (!Minv.empty())
                std::transform(v.begin(), v.end(), Minv.begin(), v.begin(),
                               std::multiplies<double>());
        };
        auto vector_dot = [](const std::vector<double>& v, const std::vector<double>& y) {
            return psi::C_DDOT(v.size(), v.data(), 1, y.data(), 1);
        };
        auto norm = [&](const std::vector<double>& v) { return std::sqrt(vector_dot(v, v)); };
        auto axpy = [](double a, const std::vector<double>& x, std::vector<double>& y) {
            psi::C_DAXPY(x.size(), a, x.data(), 1, y.data(), 1);
        };
        auto scale = [](std::vector<double>& v, double scale) {
            psi::C_DSCAL(v.size(), scale, v.data(), 1);
        };

        // GMRES(m)
        converged_ = false;
        std::vector<std::vector<double>> Q(mmiter + 1);
        for (int iter = 0; iter < maxiter_macro_; ++iter) {
            std::vector<double> sn(mmiter), cs(mmiter), beta(mmiter + 1);
            std::vector<double> H(mmiter * mmiter + mmiter); // (mmiter + 1) x mmiter

            // initial residual
            auto r = foo.compute_sigma(x);
            scale(r, -1.0);
            axpy(1.0, b, r);
            apply_Minv(r); // left preconditioning
            auto rnorm = norm(r);
            scale(r, 1.0 / rnorm);
            Q[0] = r;
            beta[0] = rnorm;

            // micro iterations
            int k = 0;
            do {
                // Arnoldi
                auto y = foo.compute_sigma(Q[k]);
                apply_Minv(y); // left preconditioning
                for (int j = 0; j < k + 1; ++j) {
                    auto Hjk = vector_dot(Q[j], y);
                    H[j * mmiter + k] = Hjk;
                    axpy(-Hjk, Q[j], y);
                }
                auto ynorm = norm(y);
                H[(k + 1) * mmiter + k] = ynorm;
                scale(y, 1.0 / ynorm);
                Q[k + 1] = y;

                // Givens rotation, H -> upper triangular
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
                psi::outfile->Printf("\n  macro %2d  micro %2d  error %13.6e", iter, k,
                                     beta[k + 1]);
                if (fabs(beta[++k]) < r_conv_) { // we increase k by 1 here!!!
                    converged_ = true;
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
                scale(Q[i], beta[i]);
                axpy(1.0, Q[i], x);
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

template void GMRES::solve(DSRG_MRPT2& func, const std::vector<double>& b, std::vector<double>& x,
                           const std::vector<double>& Minv = {});

} // namespace forte
