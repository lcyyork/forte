/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <map>
#include <vector>
#include <sys/stat.h>

#include "psi4/libpsi4util/process.h"
#include "helpers/disk_io.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "cc.h"

using namespace psi;

namespace forte {

void CC_SO::guess_t2() {
    local_timer timer;

    // use default file name
    std::string master_file = file_prefix_ + ".t2.master.txt";

    struct stat buf;
    if (read_amps_ and (stat(master_file.c_str(), &buf) == 0)) {
        std::string str = "Reading T2 amplitudes from files ...";
        outfile->Printf("\n    %-35s", str.c_str());
        read_disk_BT(T2_, master_file);
    } else {
        std::string str = "Computing T2 amplitudes     ...";
        outfile->Printf("\n    %-35s", str.c_str());

        T2_["ijab"] = V_["ijab"];

        T2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            value *= 1.0 / (Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]]);
        });
    }

    // norm and max
    T2max_ = 0.0, T2norm_ = T2_.norm();
    T2_.citerate(
        [&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void CC_SO::guess_t1() {
    local_timer timer;

    // use default file name
    std::string master_file = file_prefix_ + ".t1.master.txt";

    struct stat buf;
    if (read_amps_ and (stat(master_file.c_str(), &buf) == 0)) {
        std::string str = "Reading T1 amplitudes from files ...";
        outfile->Printf("\n    %-35s", str.c_str());
        read_disk_BT(T1_, master_file);
    } else {
        std::string str = "Computing T1 amplitudes     ...";
        outfile->Printf("\n    %-35s", str.c_str());

        // use simple single-reference guess
        T1_["ia"] = F_["ia"];
        T1_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            value *= 1.0 / (Fd_[i[0]] - Fd_[i[1]]);
        });
    }

    // norm and max
    T1max_ = 0.0, T1norm_ = T1_.norm();
    T1_.citerate(
        [&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        });

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void CC_SO::guess_t3() {
    local_timer timer;

    // use default file name
    std::string master_file = file_prefix_ + ".t3.master.txt";

    struct stat buf;
    if (read_amps_ and (stat(master_file.c_str(), &buf) == 0)) {
        std::string str = "Reading T3 amplitudes from files ...";
        outfile->Printf("\n    %-35s", str.c_str());
        read_disk_BT(T3_, master_file);
    } else {
        std::string str = "Computing T3 amplitudes     ...";
        outfile->Printf("\n    %-35s", str.c_str());

        ambit::BlockedTensor C3 = ambit::BlockedTensor::build(tensor_type_, "C3", {"cccvvv"});
        auto temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"cccvvv"});
        temp["g2,c0,c1,g0,g1,v0"] += -1.0 * V_["g2,v1,g0,g1"] * T2_["c0,c1,v0,v1"];
        C3["c0,c1,g2,g0,g1,v0"] += temp["g2,c0,c1,g0,g1,v0"];
        C3["c0,g2,c1,g0,g1,v0"] -= temp["g2,c0,c1,g0,g1,v0"];
        C3["g2,c0,c1,g0,g1,v0"] += temp["g2,c0,c1,g0,g1,v0"];
        C3["c0,c1,g2,g0,v0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
        C3["c0,g2,c1,g0,v0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
        C3["g2,c0,c1,g0,v0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
        C3["c0,c1,g2,v0,g0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
        C3["c0,g2,c1,v0,g0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
        C3["g2,c0,c1,v0,g0,g1"] += temp["g2,c0,c1,g0,g1,v0"];

        temp.zero();
        temp["g1,g2,c0,g0,v0,v1"] += 1.0 * V_["g1,g2,g0,c1"] * T2_["c0,c1,v0,v1"];
        C3["c0,g1,g2,g0,v0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
        C3["g1,c0,g2,g0,v0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
        C3["g1,g2,c0,g0,v0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
        C3["c0,g1,g2,v0,g0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
        C3["g1,c0,g2,v0,g0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
        C3["g1,g2,c0,v0,g0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
        C3["c0,g1,g2,v0,v1,g0"] += temp["g1,g2,c0,g0,v0,v1"];
        C3["g1,c0,g2,v0,v1,g0"] -= temp["g1,g2,c0,g0,v0,v1"];
        C3["g1,g2,c0,v0,v1,g0"] += temp["g1,g2,c0,g0,v0,v1"];

        T3_["ijkabc"] = C3["ijkabc"];
        T3_["ijkabc"] += C3["abcijk"];

        T3_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            value *= 1.0 / (Fd_[i[0]] + Fd_[i[1]] + Fd_[i[2]] - Fd_[i[3]] - Fd_[i[4]] - Fd_[i[5]]);
        });
    }

    // norm and max
    T3max_ = T3_.norm(0), T3norm_ = T3_.norm();

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void CC_SO::update_t2() {
    // compute DT2 = Hbar2 / D
    DT2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= 1.0 / (Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]]);
    });

    rms_t2_ = DT2_.norm();

    T2_["ijab"] += DT2_["ijab"];

    // norm and max
    T2max_ = T2_.norm(0), T2norm_ = T2_.norm();
}

void CC_SO::update_t1() {
    // compute DT1 = Hbar1 / D
    DT1_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= 1.0 / (Fd_[i[0]] - Fd_[i[1]]);
    });

    rms_t1_ = DT1_.norm();

    T1_["ia"] += DT1_["ia"];

    // norm and max
    T1max_ = T1_.norm(0), T1norm_ = T1_.norm();
}

void CC_SO::update_t3() {
    // compute DT3 = Hbar3 / D
    DT3_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= 1.0 / (Fd_[i[0]] + Fd_[i[1]] + Fd_[i[2]] - Fd_[i[3]] - Fd_[i[4]] - Fd_[i[5]]);
    });

    rms_t3_ = DT3_.norm();

    T3_["ijkabc"] += DT3_["ijkabc"];

    // norm and max
    T3max_ = T3_.norm(0), T3norm_ = T3_.norm();
}
} // namespace forte
