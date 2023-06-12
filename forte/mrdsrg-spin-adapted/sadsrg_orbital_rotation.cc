/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <algorithm>
#include <cstdio>
#include <sys/stat.h>

#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/disk_io.h"
#include "helpers/timer.h"
#include "helpers/printing.h"
#include "sadsrg.h"

using namespace psi;

namespace forte {

std::shared_ptr<psi::Matrix> SADSRG::expA1(ambit::BlockedTensor T1, bool with_symmetry) {
    // A1 = T1 - T1^+
    auto A1 = ambit::BlockedTensor::build(tensor_type_, "A1", od_one_labels());
    A1["ia"] = T1["ia"];
    A1["ai"] -= T1["ia"];

    // A1 in SharedMatrix form
    auto rel_mos = mo_space_info_->relative_mo("CORRELATED");
    auto dim_corr = mo_space_info_->dimension("CORRELATED");
    auto dim_frzc = mo_space_info_->dimension("FROZEN_DOCC");

    auto A1m = std::make_shared<psi::Matrix>("A1 Sym", dim_corr, dim_corr);
    A1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        auto [h0, i0] = rel_mos[i[0]];
        auto [h1, i1] = rel_mos[i[1]];
        if (h0 == h1) {
            A1m->set(h0, i0 - dim_frzc[h0], i1 - dim_frzc[h1], value);
        }
    });

    A1m->expm(3); // >=3 is required for high energy convergence

    auto U = std::make_shared<psi::Matrix>("exp(A1) Sym", dim_corr, dim_corr);;
    if (with_symmetry) {
        // MOM
        auto nirrep = mo_space_info_->nirrep();
        auto dim_core = mo_space_info_->dimension("RESTRICTED_DOCC");
        auto dim_actv = mo_space_info_->dimension("ACTIVE");
        auto dim_virt = mo_space_info_->dimension("RESTRICTED_UOCC");

        for (int h = 0; h < nirrep; ++h) {
            int c = 0, a = 0, v = 0;
            for (int j = 0; j < dim_corr[h]; ++j) {
                double vc = 0.0;
                for (int i = 0; i < dim_core[h]; ++i) {
                    vc += A1m->get(h, j, i);
                }

                double va = 0.0;
                for (int i = 0; i < dim_actv[h]; ++i) {
                    va += A1m->get(h, j, i + dim_core[h]);
                }

                double vv = 0.0;
                for (int i = 0; i < dim_virt[h]; ++i) {
                    vv += A1m->get(h, j, i + dim_core[h] + dim_actv[h]);
                }

                double m = std::max({std::fabs(vc), std::fabs(va), std::fabs(vv)});
                if (m == std::fabs(vc)) {
                    U->set_row(h, c, A1m->get_row(h, j));
                    c++;
                } else if (m == std::fabs(va)) {
                    U->set_row(h, a + dim_core[h], A1m->get_row(h, j));
                    a++;
                } else {
                    U->set_row(h, v + dim_core[h] + dim_actv[h], A1m->get_row(h, j));
                    v++;
                }
            }
        }
    } else {
        auto nirrep = mo_space_info_->nirrep();
        size_t ncmo = core_mos_.size() + actv_mos_.size() + virt_mos_.size();

        U = std::make_shared<psi::Matrix>("exp(A1) NoSym", ncmo, ncmo);
        for (size_t h = 0, offset = 0; h < nirrep; ++h) {
            for (size_t p = 0; p < dim_corr[h]; ++p) {
                for (size_t q = 0; q < dim_corr[h]; ++q) {
                    U->set(p + offset, q + offset, A1m->get(h, p, q));
                }
            }
            offset += dim_corr[h];
        }
    }

    return U;
}

std::shared_ptr<psi::Matrix> SADSRG::diagonalize_H1(ambit::BlockedTensor H1) {
    auto rel_mos = mo_space_info_->relative_mo("CORRELATED");
    auto dim_corr = mo_space_info_->dimension("CORRELATED");
    auto dim_frzc = mo_space_info_->dimension("FROZEN_DOCC");

    auto G = std::make_shared<psi::Matrix>("G1", dim_corr, dim_corr);
    H1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        auto [h0, i0] = rel_mos[i[0]];
        auto [h1, i1] = rel_mos[i[1]];
        if (h0 == h1) {
            G->set(h0, i0 - dim_frzc[h0], i1 - dim_frzc[h1], value);
        }
    });

    auto evals = std::make_shared<psi::Vector>("G evals", dim_corr);
    auto evecs = std::make_shared<psi::Matrix>("G evecs", dim_corr, dim_corr);
    G->diagonalize(evecs, evals);
    evecs->eivprint(evals);

    // MOM
    auto nirrep = mo_space_info_->nirrep();
    auto dim_core = mo_space_info_->dimension("RESTRICTED_DOCC");
    auto dim_actv = mo_space_info_->dimension("ACTIVE");
    auto dim_virt = mo_space_info_->dimension("RESTRICTED_UOCC");

    auto U = std::make_shared<psi::Matrix>("U", dim_corr, dim_corr);

    for (int h = 0; h < nirrep; ++h) {
        int c = 0, a = 0, v = 0;
        for (int j = 0; j < dim_corr[h]; ++j) {
            double vc = 0.0;
            for (int i = 0; i < dim_core[h]; ++i) {
                vc += evecs->get(h, i, j);
            }

            double va = 0.0;
            for (int i = 0; i < dim_actv[h]; ++i) {
                va += evecs->get(h, i + dim_core[h], j);
            }

            double vv = 0.0;
            for (int i = 0; i < dim_virt[h]; ++i) {
                vv += evecs->get(h, i + dim_core[h] + dim_actv[h], j);
            }

            outfile->Printf("\n  h = %d, j = %2d, vc = %.6f, va = %.6f, vv = %.6f", h, j, vc, va,
                            vv);

            double m = std::max({std::fabs(vc), std::fabs(va), std::fabs(vv)});
            if (m == std::fabs(vc)) {
                U->set_column(h, c, evecs->get_column(h, j));
                c++;
            } else if (m == std::fabs(va)) {
                U->set_column(h, a + dim_core[h], evecs->get_column(h, j));
                a++;
            } else {
                U->set_column(h, v + dim_core[h] + dim_actv[h], evecs->get_column(h, j));
                v++;
            }
        }
    }
    U->print();

    return U;
}

std::shared_ptr<psi::Matrix> SADSRG::R_brueckner() {
    auto nirrep = mo_space_info_->nirrep();
    auto dim_corr = mo_space_info_->dimension("CORRELATED");
    auto dim_frzc = mo_space_info_->dimension("FROZEN_DOCC");

    // build unitary rotation matrix: exp(T1 - T1^+)
    std::shared_ptr<psi::Matrix> A1m;
    if (foptions_->get_str("BRUECKNER_UPDATE") == "T1") {
        A1m = expA1(T1_, true);
    } else {
        A1m = diagonalize_H1(Hbar1_);
    }

    // include frozen orbitals for U
    auto dim_all = mo_space_info_->dimension("ALL");
    auto U = std::make_shared<psi::Matrix>("U", dim_all, dim_all);
    U->identity();

    for (size_t h = 0; h < nirrep; ++h) {
        size_t offset = dim_frzc[h];
        for (size_t p = 0; p < dim_corr[h]; ++p) {
            for (size_t q = 0; q < dim_corr[h]; ++q) {
                U->set(h, p + offset, q + offset, A1m->get(h, p, q));
            }
        }
    }

    return U;
}

void SADSRG::brueckner_orbital_rotation(ambit::BlockedTensor T1) {
    print_h2("DSRG Orbital Rotation");

    auto nirrep = mo_space_info_->nirrep();
    auto dim_corr = mo_space_info_->dimension("CORRELATED");
    auto dim_frzc = mo_space_info_->dimension("FROZEN_DOCC");

    // build unitary rotation matrix: exp(T1 - T1^+)
    auto A1m = expA1(T1, true);

    // include frozen orbitals for U
    auto dim_all = mo_space_info_->dimension("ALL");
    auto U = std::make_shared<psi::Matrix>("U", dim_all, dim_all);
    U->identity();

    for (size_t h = 0; h < nirrep; ++h) {
        size_t offset = dim_frzc[h];
        for (size_t p = 0; p < dim_corr[h]; ++p) {
            for (size_t q = 0; q < dim_corr[h]; ++q) {
                U->set(h, p + offset, q + offset, A1m->get(h, p, q));
            }
        }
    }

    // test convergence (off diagonal elements of U)
    auto Uod = U->clone();
    for (size_t h = 0; h < nirrep; ++h) {
        for (size_t p = 0; p < dim_all[h]; ++p) {
            Uod->set(h, p, p, 0.0);
        }
    }
    brueckner_absmax_ = Uod->absmax();
    outfile->Printf("\n  brueckner orb absmax = %.15f", brueckner_absmax_);
    if (brueckner_absmax_ < brueckner_conv_)
        return;

    // fix orbital phase
    ints_->fix_orbital_phases(U, true, true);

    auto Ca = ints_->Ca();
    auto Ca_new = psi::linalg::doublet(Ca, U, false, true);
    Ca_new->set_name("MO coefficients (DSRG Brueckner)");

    // update integrals
    ints_->update_orbitals(Ca_new, Ca_new, true);
}
} // namespace forte