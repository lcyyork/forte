/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "fci_string_address.h"

#include "fci_string_lists.h"

namespace forte {

/**
 * Returns the list of alfa strings connected by a^{+}_p a^{+}_q a_q a_p
 * @param pq_sym symmetry of the pq pair
 * @param pq     relative PAIRINDEX of the pq pair
 * @param h      symmetry of the I strings in the list
 */
std::vector<StringSubstitution>& FCIStringLists::get_alfa_oo_list(int pq_sym, size_t pq, int h) {
    std::tuple<int, size_t, int> pq_pair(pq_sym, pq, h);
    return alfa_oo_list[pq_pair];
}

/**
 * Returns the list of beta strings connected by a^{+}_p a^{+}_q a_q a_p
 * @param pq_sym symmetry of the pq pair
 * @param pq     relative PAIRINDEX of the pq pair
 * @param h      symmetry of the I strings in the list
 */
std::vector<StringSubstitution>& FCIStringLists::get_beta_oo_list(int pq_sym, size_t pq, int h) {
    std::tuple<int, size_t, int> pq_pair(pq_sym, pq, h);
    return beta_oo_list[pq_pair];
}

void FCIStringLists::make_oo_list(std::shared_ptr<FCIStringAddress> addresser, OOList& list) {
    // Loop over irreps of the pair pq
    for (size_t pq_sym = 0; pq_sym < nirrep_; ++pq_sym) {
        size_t max_pq = pairpi_[pq_sym];
        for (size_t pq = 0; pq < max_pq; ++pq) {
            make_oo(addresser, list, pq_sym, pq);
        }
    }
}

void FCIStringLists::make_oo(std::shared_ptr<FCIStringAddress> addresser, OOList& list, int pq_sym,
                             size_t pq) {
    int k = addresser->nones() - 2;
    if (k >= 0) {
        int p = pair_list_[pq_sym][pq].first;
        int q = pair_list_[pq_sym][pq].second;

        int n = addresser->nbits() - 2;
        String b, I, J;
        auto b_begin = b.begin();
        auto b_end = b.begin() + n;
        for (size_t h = 0; h < nirrep_; ++h) {
            // Create the key to the map
            std::tuple<int, size_t, int> pq_pair(pq_sym, pq, h);

            // Generate the strings 1111100000
            //                      { k }{n-k}
            for (int i = 0; i < n - k; ++i)
                b[i] = false; // 0
            for (int i = n - k; i < n; ++i)
                b[i] = true; // 1
            do {
                int k = 0;
                for (int i = 0; i < q; ++i) {
                    J[i] = I[i] = b[k];
                    k++;
                }
                for (int i = q + 1; i < p; ++i) {
                    J[i] = I[i] = b[k];
                    k++;
                }
                for (int i = p + 1; i < static_cast<int>(ncmo_); ++i) {
                    J[i] = I[i] = b[k];
                    k++;
                }
                I[p] = true;
                I[q] = true;
                J[p] = true;
                J[q] = true;
                // Add the sting only of irrep(I) is h
                if (string_class_->symmetry(I) == h)
                    list[pq_pair].push_back(
                        StringSubstitution(1.0, addresser->add(I), addresser->add(J)));
            } while (std::next_permutation(b_begin, b_end));
        } // End loop over h
    }
}
} // namespace forte
