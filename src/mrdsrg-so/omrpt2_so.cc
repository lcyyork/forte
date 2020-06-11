#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsio/psio.hpp"

#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "omrpt2_so.h"

using namespace psi;

namespace forte {

OMRPT2_SO::OMRPT2_SO(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : DynamicCorrelationSolver(rdms, scf_info, options, ints, mo_space_info),
      BTF_(std::make_shared<BlockedTensorFactory>()), tensor_type_(ambit::CoreTensor) {
    print_method_banner({"Spin-Orbital Orbital-Optimized DSRG-MRPT2", "Chenyang Li"});
    startup();
}

OMRPT2_SO::~OMRPT2_SO() {}

std::shared_ptr<ActiveSpaceIntegrals> OMRPT2_SO::compute_Heff_actv() {
    throw psi::PSIEXCEPTION("Not yet implemented for spin-orbital code.");
}

void OMRPT2_SO::startup() {
    // screen not implemented features
    not_implemented();

    // read and print options
    read_options();
    print_options();

    // read orbital spaces
    read_MOSpaceInfo();

    // set Ambit MO space labels
    set_ambit_MOSpace();

    // prepare density matrix and cumulants
    init_density();

    // initialize integrals
    init_ints();

    // initialize Fock matrix
    init_fock();

    // check semicanonical orbitals
    semi_canonical_ = check_semi_orbs();
}

void OMRPT2_SO::not_implemented() {
    if (eri_df_) {
        throw PSIEXCEPTION("DF/CD not yet supported for spin-orbital code.");
    }

    std::string actv_type = foptions_->get_str("FCIMO_ACTV_TYPE");
    if (actv_type == "CIS" || actv_type == "CISD") {
        throw PSIEXCEPTION("Incomplete active space not available for spin-orbital code");
    }
}

void OMRPT2_SO::read_options() {
    maxiter_ = foptions_->get_int("MAXITER");

    s_ = foptions_->get_double("DSRG_S");
    taylor_threshold_ = foptions_->get_int("TAYLOR_THRESHOLD");

    source_ = foptions_->get_str("SOURCE");
    if (source_ != "STANDARD" && source_ != "LABS" && source_ != "DYSON") {
        outfile->Printf("\n  Warning: SOURCE option %s is not implemented.", source_.c_str());
        outfile->Printf("\n  Changed SOURCE option to STANDARD");
        source_ = "STANDARD";
    }
    if (source_ == "STANDARD") {
        dsrg_source_ = std::make_shared<STD_SOURCE>(s_, taylor_threshold_);
    } else if (source_ == "LABS") {
        dsrg_source_ = std::make_shared<LABS_SOURCE>(s_, taylor_threshold_);
    } else if (source_ == "DYSON") {
        dsrg_source_ = std::make_shared<DYSON_SOURCE>(s_, taylor_threshold_);
    }
    ccvv_source_ = foptions_->get_str("CCVV_SOURCE");
}

void OMRPT2_SO::print_options() {
    // fill in information
    std::vector<std::pair<std::string, int>> calculation_info_int{
        {"Max number of iterations", maxiter_},  {"DIIS start", diis_start_},
        {"Min DIIS vectors", diis_min_vec_},     {"Max DIIS vectors", diis_max_vec_},
        {"DIIS extrapolating freq", diis_freq_}, {"Number of amplitudes for printing", ntamp_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Flow parameter", s_},
        {"Energy convergence threshold", e_convergence_},
        {"Density convergence threshold", d_convergence_},
        {"Taylor expansion threshold", pow(10.0, -double(taylor_threshold_))},
        {"Intruder amplitudes threshold", intruder_tamp_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Integral type", ints_type_},
        {"Source operator", source_},
        {"Core-Virtual source type", ccvv_source_}};

    // print some information
    print_h2("OO-DSRG-MRPT2 Options");
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %15s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %15.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_int) {
        outfile->Printf("\n    %-40s %15d", str_dim.first.c_str(), str_dim.second);
    }
    outfile->Printf("\n");
}

void OMRPT2_SO::read_MOSpaceInfo() {
    acore_sos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    aactv_sos_ = mo_space_info_->corr_absolute_mo("ACTIVE");
    avirt_sos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");

    // put all beta behind alpha
    auto mo_shift = mo_space_info_->size("CORRELATED");
    bcore_sos_.clear();
    bactv_sos_.clear();
    bvirt_sos_.clear();
    for (size_t idx : acore_sos_)
        bcore_sos_.push_back(idx + mo_shift);
    for (size_t idx : aactv_sos_)
        bactv_sos_.push_back(idx + mo_shift);
    for (size_t idx : avirt_sos_)
        bvirt_sos_.push_back(idx + mo_shift);

    // spin orbital indices
    core_sos_ = acore_sos_;
    actv_sos_ = aactv_sos_;
    virt_sos_ = avirt_sos_;
    core_sos_.insert(core_sos_.end(), bcore_sos_.begin(), bcore_sos_.end());
    actv_sos_.insert(actv_sos_.end(), bactv_sos_.begin(), bactv_sos_.end());
    virt_sos_.insert(virt_sos_.end(), bvirt_sos_.begin(), bvirt_sos_.end());
}

void OMRPT2_SO::set_ambit_MOSpace() {
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    BTF_->add_mo_space("c", "m,n,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9", core_sos_, NoSpin);
    BTF_->add_mo_space("a", "u,v,w,x,y,z,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9", actv_sos_, NoSpin);
    BTF_->add_mo_space("v", "e,f,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9", virt_sos_, NoSpin);

    BTF_->add_composite_mo_space("h", "i,j,k,l,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9", {"c", "a"});
    BTF_->add_composite_mo_space("p", "a,b,c,d,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9", {"a", "v"});
    BTF_->add_composite_mo_space("g", "p,q,r,s,g0,g1,g2,g3,g4,g5,g6,g7,g8,g9", {"c", "a", "v"});

    I_ = BTF_->build(tensor_type_, "Identity", {"gg"}, true);
    I_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            value = 1.0;
        }
    });
}

void OMRPT2_SO::init_density() {
    auto na = actv_sos_.size();
    size_t na_mo = na / 2;
    auto na2 = na * na;
    auto na3 = na * na2;

    // 1-particle density (make a copy)
    D1_ = BTF_->build(tensor_type_, "Gamma1", {"aa"});

    (rdms_.g1a()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index = i[0] * na + i[1];
        (D1_.block("aa")).data()[index] = value;
    });

    (rdms_.g1b()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index = (i[0] + na_mo) * na + (i[1] + na_mo);
        (D1_.block("aa")).data()[index] = value;
    });

    // 2-body density cumulants (make a copy)
    D2_ = BTF_->build(tensor_type_, "Gamma2", {"aaaa"});

    (rdms_.g2aa()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index = i[0] * na3 + i[1] * na2 + i[2] * na + i[3];
        (D2_.block("aaaa")).data()[index] = value;
    });

    (rdms_.g2bb()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index =
            (i[0] + na_mo) * na3 + (i[1] + na_mo) * na2 + (i[2] + na_mo) * na + (i[3] + na_mo);
        (D2_.block("aaaa")).data()[index] = value;
    });

    (rdms_.g2ab()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t i0 = i[0];
        size_t i1 = i[1] + na_mo;
        size_t i2 = i[2];
        size_t i3 = i[3] + na_mo;

        size_t index = index = i0 * na3 + i1 * na2 + i2 * na + i3;
        (D2_.block("aaaa")).data()[index] = value;

        index = i1 * na3 + i0 * na2 + i3 * na + i2;
        (D2_.block("aaaa")).data()[index] = value;

        index = i1 * na3 + i0 * na2 + i2 * na + i3;
        (D2_.block("aaaa")).data()[index] = -value;

        index = i0 * na3 + i1 * na2 + i3 * na + i2;
        (D2_.block("aaaa")).data()[index] = -value;
    });
}

void OMRPT2_SO::init_ints() {
    size_t ncmo = mo_space_info_->size("CORRELATED");

    H_ = BTF_->build(tensor_type_, "H", {"gg"});
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] < ncmo && i[1] < ncmo) {
            value = ints_->oei_a(i[0], i[1]);
        }
        if (i[0] >= ncmo && i[1] >= ncmo) {
            value = ints_->oei_b(i[0] - ncmo, i[1] - ncmo);
        }
    });

    V_ = BTF_->build(tensor_type_, "V", {"gggg"});
    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        bool spin0 = i[0] < ncmo;
        bool spin1 = i[1] < ncmo;
        bool spin2 = i[2] < ncmo;
        bool spin3 = i[3] < ncmo;

        if (spin0 && spin1 && spin2 && spin3) {
            value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
        }

        if ((!spin0) && (!spin1) && (!spin2) && (!spin3)) {
            value = ints_->aptei_bb(i[0] - ncmo, i[1] - ncmo, i[2] - ncmo, i[3] - ncmo);
        }

        if (spin0 && (!spin1) && spin2 && (!spin3)) {
            value = ints_->aptei_ab(i[0], i[1] - ncmo, i[2], i[3] - ncmo);
        }
        if (spin1 && (!spin0) && spin3 && (!spin2)) {
            value = ints_->aptei_ab(i[1], i[0] - ncmo, i[3], i[2] - ncmo);
        }
        if (spin0 && (!spin1) && spin3 && (!spin2)) {
            value = -ints_->aptei_ab(i[0], i[1] - ncmo, i[3], i[2] - ncmo);
        }
        if (spin1 && (!spin0) && spin2 && (!spin3)) {
            value = -ints_->aptei_ab(i[1], i[0] - ncmo, i[2], i[3] - ncmo);
        }
    });
}

void OMRPT2_SO::init_fock() {
    F_ = BTF_->build(tensor_type_, "Fock", {"gg"});
    F_["pq"] = H_["pq"];
    F_["pq"] += V_["pmqn"] * I_["mn"];
    F_["pq"] += V_["pvqu"] * D1_["uv"];

    size_t ncso = 2 * mo_space_info_->size("CORRELATED");
    Fd_ = std::vector<double>(ncso);
    F_.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
            if (i[0] == i[1]) {
                Fd_[i[0]] = value;
            }
        });
}

bool OMRPT2_SO::check_semi_orbs() {
    BlockedTensor Fd = BTF_->build(tensor_type_, "Fd", {"cc", "aa", "vv"});
    Fd["pq"] = F_["pq"];

    Fd.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            value = 0.0;
        }
    });

    bool semi = true;
    std::vector<double> Fmax, Fnorm;
    double e_conv = foptions_->get_double("E_CONVERGENCE");
    e_conv = e_conv < 1.0e-12 ? 1.0e-12 : e_conv;
    double threshold_max = 10.0 * e_conv;
    for (const auto& block : {"cc", "aa", "vv"}) {
        double fmax = Fd.block(block).norm(0);
        double fnorm = Fd.block(block).norm(1);
        Fmax.push_back(fmax);
        Fnorm.push_back(fnorm);

        if (fmax > threshold_max) {
            semi = false;
        }
        if (fnorm > Fd.block(block).numel() * e_conv) {
            semi = false;
        }
    }

    std::string dash(7 + 47, '-');
    outfile->Printf("\n    Fock core, active, virtual blocks (Fij, i != j)");
    outfile->Printf("\n    %6s %15s %15s %15s", "", "core", "active", "virtual");
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    %6s %15.10f %15.10f %15.10f", "max", Fmax[0], Fmax[1], Fmax[2]);
    outfile->Printf("\n    %6s %15.10f %15.10f %15.10f", "1-norm", Fnorm[0], Fnorm[1], Fnorm[2]);
    outfile->Printf("\n    %s\n", dash.c_str());

    if (semi) {
        outfile->Printf("\n    Orbitals are semi-canonicalized.");
    } else {
        outfile->Printf("\n    Warning! Orbitals are not semi-canonicalized!");
    }

    return semi;
}

} // namespace forte
