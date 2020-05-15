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

#include "psi4/libdiis/diismanager.h"

#include "mrdsrg_so.h"

using namespace psi;

namespace forte {

void MRDSRG_SO::diis_manager_init() {
    diis_manager_ = std::make_shared<DIISManager>(diis_max_vec_, "MRDSRG_SO DIIS",
                                                  DIISManager::LargestError, DIISManager::OnDisk);

    amp_ptrs_.clear();
    res_ptrs_.clear();

    if (not do_t3_) {
        size_t t1_nel = T1.block("cv").numel();
        size_t t2_nel = T2.block("ccvv").numel();

        amp_ptrs_.push_back(T1.block("cv").data().data());
        amp_ptrs_.push_back(T2.block("ccvv").data().data());
        res_ptrs_.push_back(Hbar1.block("cv").data().data());
        res_ptrs_.push_back(Hbar2.block("ccvv").data().data());

        diis_manager_->set_error_vector_size(2, DIISEntry::Pointer, t1_nel, DIISEntry::Pointer,
                                             t2_nel);
        diis_manager_->set_vector_size(2, DIISEntry::Pointer, t1_nel, DIISEntry::Pointer, t2_nel);
    } else {
        size_t t1_nel = T1.block("cv").numel();
        size_t t2_nel = T2.block("ccvv").numel();
        size_t t3_nel = T3.block("cccvvv").numel();

        amp_ptrs_.push_back(T1.block("cv").data().data());
        amp_ptrs_.push_back(T2.block("ccvv").data().data());
        amp_ptrs_.push_back(T3.block("cccvvv").data().data());
        res_ptrs_.push_back(Hbar1.block("cv").data().data());
        res_ptrs_.push_back(Hbar2.block("ccvv").data().data());
        res_ptrs_.push_back(Hbar3.block("cccvvv").data().data());

        diis_manager_->set_error_vector_size(3, DIISEntry::Pointer, t1_nel, DIISEntry::Pointer,
                                             t2_nel, DIISEntry::Pointer, t3_nel);
        diis_manager_->set_vector_size(3, DIISEntry::Pointer, t1_nel, DIISEntry::Pointer, t2_nel,
                                       DIISEntry::Pointer, t3_nel);
    }
}

void MRDSRG_SO::diis_manager_add_entry() {
    if (not do_t3_) {
        diis_manager_->add_entry(4, res_ptrs_[0], res_ptrs_[1], amp_ptrs_[0], amp_ptrs_[1]);
    } else {
        diis_manager_->add_entry(6, res_ptrs_[0], res_ptrs_[1], res_ptrs_[2], amp_ptrs_[0],
                                 amp_ptrs_[1], amp_ptrs_[2]);
    }
}

void MRDSRG_SO::diis_manager_extrapolate() {
    if (not do_t3_) {
        diis_manager_->extrapolate(2, amp_ptrs_[0], amp_ptrs_[1]);
    } else {
        diis_manager_->add_entry(6, res_ptrs_[0], res_ptrs_[1], res_ptrs_[2], amp_ptrs_[0],
                                 amp_ptrs_[1], amp_ptrs_[2]);
    }
}

void MRDSRG_SO::diis_manager_cleanup() {
    amp_ptrs_.clear();
    res_ptrs_.clear();
}
} // namespace forte
