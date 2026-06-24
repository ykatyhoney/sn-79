/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/agent/DistributedProxyAgent.hpp>

#include <taosim/message/ExchangeAgentMessagePayloads.hpp>
#include "Simulation.hpp"
#include "json_util.hpp"
#include "util.hpp"

//-------------------------------------------------------------------------

namespace taosim::agent
{

//-------------------------------------------------------------------------

DistributedProxyAgent::DistributedProxyAgent(Simulation* simulation)
    : Agent{simulation, "DISTRIBUTED_PROXY_AGENT"}
{}

//-------------------------------------------------------------------------

void DistributedProxyAgent::receiveMessage(Message::Ptr msg)
{
    static const std::set<std::string> ignoredMessageTypes{
        "MULTIBOOK_STATE_PUBLISH",
        "EVENT_SIMULATION_START"
    };

    if (ignoredMessageTypes.contains(msg->type)) {
        return;
    }

    if (m_exchangeServiceMode) {
        handleMessageForExchangeService(msg);
    } else {
        m_messages.push_back(msg);
    }
}

//-------------------------------------------------------------------------

void DistributedProxyAgent::configure(const pugi::xml_node& node)
{
    Agent::configure(node);

    m_exchangeServiceMode = node.attribute("exchangeServiceMode").as_bool();
}

//-------------------------------------------------------------------------

void DistributedProxyAgent::handleMessageForExchangeService(Message::Ptr msg)
{
    if (msg->type != "EVENT_TRADE") return;

    const auto pld = std::dynamic_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
    const auto subPld = std::dynamic_pointer_cast<EventTradePayload>(pld->payload);

    if (subPld->isResting) {
        fmt::println("TRADE NOTIF {}", json::jsonSerializable2str(subPld));
        m_tradeSignal(subPld);
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
