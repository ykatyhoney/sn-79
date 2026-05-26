/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/process/helpers.hpp>

#include <taosim/simulation/SharedResources.hpp>

#include <cmath>
#include <cstdint>

//-------------------------------------------------------------------------

namespace taosim::process::helpers
{

//-------------------------------------------------------------------------

namespace
{

double gamma_fn(int64_t k, double H) noexcept
{
    return 0.5 * (std::pow(std::abs(k - 1), 2.0 * H)
                - 2.0 * std::pow(std::abs(k), 2.0 * H)
                + std::pow(std::abs(k + 1), 2.0 * H));
}

}  // namespace

//-------------------------------------------------------------------------

void precomputeFundamentalPriceL(Eigen::MatrixXd& L, double hurst)
{
    const Eigen::Index n = L.rows();
    if (n == 0) return;

    L(0, 0) = 1.0;

    if (n >= 2) {
        L(1, 0) = gamma_fn(1, hurst);
        L(1, 1) = std::sqrt(1.0 - std::pow(L(1, 0), 2));
    }

    for (Eigen::Index i = 2; i < n; ++i) {
        L(i, 0) = gamma_fn(i, hurst);
        for (Eigen::Index j = 1; j < i; ++j) {
            const double dot_val = L.row(i).head(j).dot(L.row(j).head(j));
            L(i, j) = (1.0 / L(j, j)) * (gamma_fn(i - j, hurst) - dot_val);
        }
        const double sumsq = L.row(i).head(i).squaredNorm();
        L(i, i) = std::sqrt(1.0 - sumsq);
    }
}

//-------------------------------------------------------------------------

void initSharedResources(
    taosim::simulation::SharedResources& shared, pugi::xml_node simuNode)
{
    auto fpNode = simuNode
        .child("Agents")
        .child("MultiBookExchangeAgent")
        .child("Books")
        .child("Processes")
        .child("FundamentalPrice");
    if (!fpNode) return;

    const auto duration = simuNode.attribute("duration").as_ullong();
    const auto updatePeriod = fpNode.attribute("updatePeriod").as_ullong(1);
    const double hurst = fpNode.attribute("Hurst").as_double(0.5);
    const auto n = duration / updatePeriod + 2;

    shared.fundamentalPriceL = Eigen::MatrixXd::Zero(n, n);
    precomputeFundamentalPriceL(shared.fundamentalPriceL, hurst);
}

//-------------------------------------------------------------------------

}  // namespace taosim::process::helpers

//-------------------------------------------------------------------------
