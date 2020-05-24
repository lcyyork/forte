/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

void SADSRG::brueckner_rotation(ambit::BlockedTensor T1) {
    // build unitary rotation matrix: exp(T1 - T1^+)
    auto A1 = ambit::BlockedTensor::build(tensor_type_, "A1", {"gg"});
    A1["ia"] = T1["ia"];
    A1["ai"] -= T1["ia"];

    size_t ncmo = core_mos_.size() + actv_mos_.size() + virt_mos_.size();

    auto A1_C1 = std::make_shared<psi::Matrix>("A1 NoSym", ncmo, ncmo);
    A1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        A1_C1->set(i[0], i[1], value);
    });

    auto dim_corr = mo_space_info_->dimension("CORRELATED");
    auto A1m = std::make_shared<psi::Matrix>("A1 Sym", dim_corr, dim_corr);
    for (size_t h = 0, nirrep = mo_space_info_->nirrep(), offset = 0; h < nirrep; ++h) {
        for (size_t p = 0; p < static_cast<size_t>(dim_corr[h]); ++p) {
            for (size_t q = 0; q < static_cast<size_t>(dim_corr[h]); ++q) {
                double v = A1_C1->get(0, p + offset, q + offset);
                A1m->set(h, p, q, v);
            }
        }
        offset += dim_corr[h];
    }

    A1m->expm(3);

    // include frozen orbitals for U
    auto dim_all = mo_space_info_->dimension("ALL");
    auto dim_frzc = mo_space_info_->dimension("FROZEN_DOCC");
    auto U = std::make_shared<psi::Matrix>("U", dim_all, dim_all);
    U->identity();
    for (size_t h = 0, nirrep = mo_space_info_->nirrep(); h < nirrep; ++h) {
        size_t offset = dim_frzc[h];
        for (size_t p = 0; p < static_cast<size_t>(dim_corr[h]); ++p) {
            for (size_t q = 0; q < static_cast<size_t>(dim_corr[h]); ++q) {
                double v = A1m->get(h, p, q);
                U->set(h, p + offset, q + offset, v);
            }
        }
    }

    // fix orbital phase
    ints_->fix_orbital_phases(U, true, true);

    auto Ca = ints_->Ca();
    auto Ca_new = psi::linalg::doublet(Ca, U, false, true);
    Ca_new->set_name("MO coefficients (Brueckner)");

    // update integrals
    if (not is_brueckner_converged()) {
        ints_->update_orbitals(Ca_new, Ca_new);
    }
}
} // namespace forte
