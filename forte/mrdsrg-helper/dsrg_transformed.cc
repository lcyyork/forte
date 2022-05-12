#include "dsrg_transformed.h"
#include "helpers/timer.h"

namespace forte {

DressedQuantity::DressedQuantity() : max_body_(0), scalar_(0.0) {}

DressedQuantity::DressedQuantity(double scalar) : max_body_(0), scalar_(scalar) {}

DressedQuantity::DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b)
    : max_body_(1), scalar_(scalar), a_(a), b_(b) {}

DressedQuantity::DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b, ambit::Tensor aa,
                                 ambit::Tensor ab, ambit::Tensor bb)
    : max_body_(2), scalar_(scalar), a_(a), b_(b), aa_(aa), ab_(ab), bb_(bb) {}

DressedQuantity::DressedQuantity(double scalar, ambit::Tensor a, ambit::Tensor b, ambit::Tensor aa,
                                 ambit::Tensor ab, ambit::Tensor bb, ambit::Tensor aaa,
                                 ambit::Tensor aab, ambit::Tensor abb, ambit::Tensor bbb)
    : max_body_(3), scalar_(scalar), a_(a), b_(b), aa_(aa), ab_(ab), bb_(bb), aaa_(aaa), aab_(aab),
      abb_(abb), bbb_(bbb) {}

void DressedQuantity::rotate(const ambit::Tensor& Ua, const ambit::Tensor& Ub) {
    if (max_body_ < 1)
        return;

    timer t("Rotate DressedQuantity");

    // Transform the 1-rdms
    ambit::Tensor aT = ambit::Tensor::build(ambit::CoreTensor, "aT", a_.dims());
    ambit::Tensor bT = ambit::Tensor::build(ambit::CoreTensor, "bT", b_.dims());

    aT("pq") = Ua("ap") * a_("ab") * Ua("bq");
    bT("PQ") = Ub("AP") * b_("AB") * Ub("BQ");

    a_("pq") = aT("pq");
    b_("pq") = bT("pq");

    if (max_body_ == 1)
        return;

    // Transform the 2-rdms
    auto aaT = ambit::Tensor::build(ambit::CoreTensor, "aaT", aa_.dims());
    auto abT = ambit::Tensor::build(ambit::CoreTensor, "abT", ab_.dims());
    auto bbT = ambit::Tensor::build(ambit::CoreTensor, "bbT", bb_.dims());

    aaT("pqrs") = Ua("ap") * Ua("bq") * aa_("abcd") * Ua("cr") * Ua("ds");
    abT("pQrS") = Ua("ap") * Ub("BQ") * ab_("aBcD") * Ua("cr") * Ub("DS");
    bbT("PQRS") = Ub("AP") * Ub("BQ") * bb_("ABCD") * Ub("CR") * Ub("DS");

    aa_("pqrs") = aaT("pqrs");
    ab_("pqrs") = abT("pqrs");
    bb_("pqrs") = bbT("pqrs");

    if (max_body_ == 2)
        return;

    // Transform the 3-rdms
    auto g3T = ambit::Tensor::build(ambit::CoreTensor, "g3T", aaa_.dims());
    g3T("pqrstu") =
        Ua("ap") * Ua("bq") * Ua("cr") * aaa_("abcijk") * Ua("is") * Ua("jt") * Ua("ku");
    aaa_("pqrstu") = g3T("pqrstu");

    g3T("pqRstU") =
        Ua("ap") * Ua("bq") * Ub("CR") * aab_("abCijK") * Ua("is") * Ua("jt") * Ub("KU");
    aab_("pqrstu") = g3T("pqrstu");

    g3T("pQRsTU") =
        Ua("ap") * Ub("BQ") * Ub("CR") * abb_("aBCiJK") * Ua("is") * Ub("JT") * Ub("KU");
    abb_("pqrstu") = g3T("pqrstu");

    g3T("PQRSTU") =
        Ub("AP") * Ub("BQ") * Ub("CR") * bbb_("ABCIJK") * Ub("IS") * Ub("JT") * Ub("KU");
    bbb_("pqrstu") = g3T("pqrstu");
}

double DressedQuantity::contract_with_rdms(std::shared_ptr<RDMs> rdms) {
    double out = scalar_;
    size_t max_rdm_level = rdms->max_rdm_level();

    if (max_rdm_level >= 1 and max_body_ >= 1) {
        out += a_("uv") * rdms->g1a()("vu");
        out += b_("uv") * rdms->g1b()("vu");
    }

    if (max_rdm_level >= 2 and max_body_ >= 2) {
        out += 0.25 * aa_("uvxy") * rdms->g2aa()("xyuv");
        out += 0.25 * bb_("uvxy") * rdms->g2bb()("xyuv");
        out += ab_("uvxy") * rdms->g2ab()("xyuv");
    }

    if (max_rdm_level >= 3 and max_body_ >= 3) {
        out += (1.0 / 36.0) * aaa_("uvwxyz") * rdms->g3aaa()("xyzuvw");
        out += (1.0 / 36.0) * bbb_("uvwxyz") * rdms->g3bbb()("xyzuvw");
        out += 0.25 * aab_("uvwxyz") * rdms->g3aab()("xyzuvw");
        out += 0.25 * abb_("uvwxyz") * rdms->g3abb()("xyzuvw");
    }

    return out;
}
} // namespace forte
