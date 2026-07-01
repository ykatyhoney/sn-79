/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/message/MultiBookMessagePayloads.hpp>
#include "Agent.hpp"
#include <common.hpp>

#include <vector>

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

class DistributedProxyAgent : public Agent
{
public:
    DistributedProxyAgent(Simulation* simulation);

    [[nodiscard]] std::span<Message::Ptr> messages() noexcept { return m_messages; }
    [[nodiscard]] auto& tradeSignal() noexcept { return m_tradeSignal; }

    void clearMessages() noexcept { m_messages.clear(); };

    virtual void receiveMessage(Message::Ptr msg) override;
    virtual void configure(const pugi::xml_node& node) override;

private:
    void handleMessageForExchangeService(Message::Ptr msg);

    std::vector<Message::Ptr> m_messages;
    bool m_exchangeServiceMode{};
    UnsyncSignal<void(EventTradePayload::Ptr)> m_tradeSignal;
};

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
