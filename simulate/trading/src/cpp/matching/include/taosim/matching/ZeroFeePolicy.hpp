/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/matching/FeePolicy.hpp>
#include <taosim/matching/Fees.hpp>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::matching
{

struct ZeroFeePolicy : public FeePolicy
{
    Fees calculateFees(const TradeDesc& tradeDesc) const override { return {}; };
    void updateHistory(
        Timestamp timestamp,
        BookId bookId,
        AgentId agentId,
        decimal_t volume,
        std::optional<bool> isAggressive = {}) override {}
    void resetHistory() noexcept override {};
    void resetHistory(const std::unordered_set<AgentId>& agentIds) noexcept override {};
    Fees getRates(BookId bookId, AgentId agentId) const override { return {}; };
};

}  // namespace taosim::matching

//-------------------------------------------------------------------------
