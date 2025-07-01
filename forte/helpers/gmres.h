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
    /// the max macro iterations allowed
    int maxiter_ = 10;
    /// the max micro iterations allowed
    int maxmiter_ = 50;
    /// the max memory allowed in bytes
    size_t max_mem_;
    /// the convergence criteria for residual norm
    double r_conv_;
    /// is the solver converged?
    bool converged_;
    // /// the Jacobi preconditioner (inverse of diagonal elements of A)
    // std::shared_ptr<psi::Vector> M0_;
    // /// initial guess
    // std::shared_ptr<psi::Vector> x0_;

  public:
    /**
     * @brief Constructor of the generalized minimal residual method
     * @param maxmem the max memory allowed in bytes
     * @param rconv the convergence criteria for residual norm
     *
     * Implemention notes:
     *   See Wikipedia https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
     */
    GMRES(size_t maxiter, size_t maxmem, double rconv)
        : maxiter_(maxiter), max_mem_(maxmem), r_conv_(rconv) {}

    /**
     * @brief Solve the linear system Ax = b using GMRES with restart
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
        // TODO: do not support symmetry for now
        if (nirrep != 1) {
            throw std::runtime_error("Currently GMRES does not support symmetry!");
        }
        auto dim = b->dim();
        auto dimpi = b->dimpi();
        b->print();

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

        int mmiter = std::min(max_mem_ / dim, (size_t)maxmiter_) - (M0 ? 2 : 1);
        if (mmiter < 3) {
            throw std::runtime_error("Not enough memory for GMRES. Need at least " +
                                     std::to_string(5 * dim) + " bytes of memory.");
        }

        converged_ = false;
        auto Q = std::vector<std::shared_ptr<psi::Vector>>(mmiter + 1);
        for (int iter = 0; iter < maxiter_; ++iter) {
            std::vector<double> sn(mmiter), cn(mmiter), beta(mmiter + 1);
            auto H = std::make_shared<psi::Matrix>("H", mmiter + 1, mmiter);
            auto s = foo.compute_sigma(x);
            // apply_M0(s);
            b->subtract(*s);
            apply_M0(b);
            auto bnorm = b->norm();
            // b->print();
            // psi::outfile->Printf("\n  r norm = %20.15f", bnorm);
            b->scale(1.0 / bnorm);
            b->set_name("q0");
            Q[0] = b;
            // Q[0]->print();
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
                // auto Hsub = std::make_shared<psi::Matrix>("Hsub", k + 2, k + 1);
                // for (int i = 0; i < k + 2; ++i) {
                //     for (int j = 0; j < k + 1; ++j) {
                //         Hsub->set(i, j, H->get(i, j));
                //     }
                // }
                // Hsub->print();
                // y->print();
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
                // for (int i = 0; i < k + 2; ++i) {
                //     for (int j = 0; j < k + 1; ++j) {
                //         Hsub->set(i, j, H->get(i, j));
                //     }
                // }
                // Hsub->print();
                // psi::outfile->Printf("\ncs = %.15f, sn = %.15f", cn[k], sn[k]);
                beta[k + 1] = -sn[k] * beta[k];
                // psi::outfile->Printf("\nbeta %d = %.15f, beta %d = %.15f", k, beta[k], k + 1,
                //                      beta[k + 1]);
                beta[k] = cn[k] * beta[k];
                psi::outfile->Printf("\n  macro %2d  micro %2d  error %13.6e", iter, k,
                                     beta[k + 1]);
                if (fabs(beta[++k]) < r_conv_) { // we increase k by 1 here!!!
                    converged_ = true;
                    break;
                }
            } while (k < mmiter);

            // for (; k < mmiter; ++k) {
            //     auto y = foo.compute_sigma(Q[k]);
            //     y->set_name("q" + std::to_string(k + 1));
            //     apply_M0(y);
            //     // y->print();
            //     for (int j = 0; j < k + 1; ++j) {
            //         auto Hjk = Q[j]->vector_dot(*y);
            //         H->set(j, k, Hjk);
            //         // psi::outfile->Printf("\n  H[%d,%d] %20.15f", j, k, Hjk);
            //         y->axpy(-Hjk, *Q[j]);
            //     }
            //     // y->print();
            //     auto ynorm = y->norm();
            //     // psi::outfile->Printf("\n  y norm %20.15f", ynorm);
            //     // if (ynorm < 1.0e-15) {
            //     //     converged_ = true;
            //     //     break;
            //     // }
            //     H->set(k + 1, k, ynorm);
            //     y->scale(1.0 / ynorm);
            //     Q[k + 1] = y;
            //     auto Hsub = std::make_shared<psi::Matrix>("Hsub", k + 2, k + 1);
            //     for (int i = 0; i < k + 2; ++i) {
            //         for (int j = 0; j < k + 1; ++j) {
            //             Hsub->set(i, j, H->get(i, j));
            //         }
            //     }
            //     Hsub->print();
            //     y->print();
            //     for (int j = 0; j < k; ++j) {
            //         auto g = cn[j] * H->get(j, k) + sn[j] * H->get(j + 1, k);
            //         H->set(j + 1, k, -sn[j] * H->get(j, k) + cn[j] * H->get(j + 1, k));
            //         H->set(j, k, g);
            //     }
            //     auto h = H->get(k, k), g = H->get(k + 1, k);
            //     auto rho = std::sqrt(h * h + g * g);
            //     sn[k] = g / rho;
            //     cn[k] = h / rho;
            //     H->set(k, k, cn[k] * h + sn[k] * g);
            //     H->set(k + 1, k, 0);
            //     for (int i = 0; i < k + 2; ++i) {
            //         for (int j = 0; j < k + 1; ++j) {
            //             Hsub->set(i, j, H->get(i, j));
            //         }
            //     }
            //     Hsub->print();
            //     psi::outfile->Printf("\ncs = %.15f, sn = %.15f", cn[k], sn[k]);
            //     beta[k + 1] = -sn[k] * beta[k];
            //     psi::outfile->Printf("\nbeta %d = %.15f, beta %d = %.15f", k, beta[k], k + 1,
            //                          beta[k + 1]);
            //     beta[k] = cn[k] * beta[k];
            //     psi::outfile->Printf("\n  macro %2d  micro %2d  error %13.6e ynorm %.6e", iter,
            //     k,
            //                          beta[k + 1], ynorm);
            //     if (fabs(beta[k + 1]) < r_conv_) {
            //         converged_ = true;
            //         k += 1;
            //         break;
            //     }
            // }

            auto Hk = std::make_shared<psi::Matrix>("H", k, k);
            for (int m = 0; m < k; ++m) {
                for (int n = 0; n < k; ++n) {
                    Hk->set(m, n, H->get(m, n));
                }
            }
            // Hk->print();
            auto Hk_ptr = Hk->pointer();
            psi::C_DTRSV('U', 'N', 'N', k, Hk_ptr[0], k, beta.data(), 1);
            // auto gk = std::make_shared<psi::Vector>("g", k);
            // for (int m = 0; m < k; ++m) {
            //     gk->set(m, beta[m]);
            // }
            // gk->print();
            // psi::C_DTRSV('U', 'N', 'N', k, Hk_ptr[0], k, gk->pointer(), 1);

            // auto gy = std::make_shared<psi::Vector>("gy", k);
            // psi::C_DGEMV('N', k, k, 1.0, Hk_ptr[0], k, gk->pointer(), 1, 0.0, gy->pointer(), 1);
            // gy->print();

            for (int i = 0; i < k; ++i) {
                // Q[i]->scale(gk->get(i));
                Q[i]->scale(beta[i]);
                x->add(*Q[i]);
            }
            if (converged_)
                break;
        }
    }

    /// Return true if minimization converged
    bool converged() const { return converged_; }
};

template void GMRES::solve(DSRG_MRPT2& func, std::shared_ptr<psi::Vector> b,
                           std::shared_ptr<psi::Vector> x,
                           std::shared_ptr<psi::Vector> M0 = nullptr);

// class LBFGS {
//   public:
//     /**
//      * @brief Constructor of the Limited-BFGS class
//      * @param dim: The dimension of the problem
//      * @param param: The LBFGS_PARAM object for L-BFGS parameters
//      *
//      * Implementation notes:
//      *   See Wikipedia https://en.wikipedia.org/wiki/Limited-memory_BFGS
//      *   and <Numerical Optimization> 2nd Ed. by Jorge Nocedal and Stephen J. Wright
//      */
//     LBFGS(std::shared_ptr<LBFGS_PARAM> param);

//     /**
//      * @brief The minimization for the target function
//      * @param foo: Target class that should have the following methods:
//      *             fx = foo.evaluate(x, g, do_g=true) where gradient g is modified by the
//      function,
//      *             fx is the function return value, and g is computed when do_g is true.
//      *             If diagonal Hessian is specified, foo.hess_diag(x, h0) should be available.
//      * @param x: The initial value of x as input, the final value of x as output.
//      *
//      * @return the function value of at optimized x
//      */
//     template <class Foo> double minimize(Foo& foo, std::shared_ptr<psi::Vector> x);

//     /// Reset the L-BFGS space
//     void reset();

//     /// Return the current / final gradient vector
//     std::shared_ptr<psi::Vector> g() { return g_; }

//     /// Return the final number of iterations
//     int iter() const { return iter_; }

//     /// Return true if minimization converged
//     bool converged() const { return converged_; }

//   private:
//     /// The dimension of x
//     psi::Dimension dimpi_;

//     /// The number of irreps of x
//     int nirrep_;

//     /// The current iteration number
//     int iter_;
//     /// The shift to iteration number
//     int iter_shift_;

//     /// Parameters of L-BFGS
//     std::shared_ptr<LBFGS_PARAM> param_;

//     /// Minimization procedure converged or not
//     bool converged_;

//     /// Diagonal elements of Hessian
//     std::shared_ptr<psi::Vector> h0_;

//     /// Gradient difference vectors
//     std::vector<std::shared_ptr<psi::Vector>> y_;

//     /// Variable difference vectors
//     std::vector<std::shared_ptr<psi::Vector>> s_;

//     /// The rho vectors
//     std::vector<double> rho_;

//     /// The alpha vector
//     std::vector<double> alpha_;

//     /// The correction (moving direction) vector
//     psi::Vector p_;

//     /// The current gradient vector
//     std::shared_ptr<psi::Vector> g_;

//     /// The last gradient vector
//     std::shared_ptr<psi::Vector> g_last_;

//     /// The last solution vector
//     std::shared_ptr<psi::Vector> x_last_;

//     /// Compute gamma that can be used as inverse of diagonal Hessian
//     double compute_gamma();

//     /// Apply h0_ to some vector
//     void apply_h0(psi::Vector& q);

//     /// Generate correction (direction) vector
//     void update();

//     /// Determine step length
//     template <class Foo>
//     void next_step(Foo& foo, std::shared_ptr<psi::Vector> x, double& fx, double& step);

//     /// Determine step length using max value of direction vector
//     template <class Foo>
//     void scale_direction_vector(Foo& foo, std::shared_ptr<psi::Vector> x, double& fx, double&
//     step);

//     /// Line search using backtracking to determine step length
//     template <class Foo>
//     void line_search_backtracking(Foo& foo, std::shared_ptr<psi::Vector> x, double& fx,
//                                   double& step);

//     /// Line search using bracketing and zoom  to determine step length
//     /// See (Algorithm 3.5) of <Numerical Optimization> 2nd Ed. by Nocedal and Wright
//     template <class Foo>
//     void line_search_bracketing_zoom(Foo& foo, std::shared_ptr<psi::Vector> x, double& fx,
//                                      double& step);

//     /// Resize all vectors uisng m_
//     void resize(int m);
// };
} // namespace forte
