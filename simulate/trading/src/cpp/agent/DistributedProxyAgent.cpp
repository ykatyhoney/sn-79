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
    // Terminal placement failures from the embedded simulation (insufficient
    // balance, invalid params, ...) were silently discarded here; collect them
    // so handleBatch can report them back to the validator in the response.
    if (msg->type.starts_with("ERROR_RESPONSE_DISTRIBUTED_PLACE_ORDER")) {
        const auto pld = std::static_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
        const bool isLimit = msg->type.find("LIMIT") != std::string::npos;
        if (isLimit) {
            const auto errPld =
                std::static_pointer_cast<PlaceOrderLimitErrorResponsePayload>(pld->payload);
            m_orderRejects.push_back({
                .agentId = pld->agentId,
                .bookId = errPld->requestPayload->bookId,
                .direction = errPld->requestPayload->direction,
                .reason = errPld->errorPayload->message,
                .clientOrderId = errPld->requestPayload->clientOrderId});
        } else {
            const auto errPld =
                std::static_pointer_cast<PlaceOrderMarketErrorResponsePayload>(pld->payload);
            m_orderRejects.push_back({
                .agentId = pld->agentId,
                .bookId = errPld->requestPayload->bookId,
                .direction = errPld->requestPayload->direction,
                .reason = errPld->errorPayload->message,
                .clientOrderId = errPld->requestPayload->clientOrderId});
        }
        return;
    }

    if (msg->type != "EVENT_TRADE") return;

    const auto pld = std::static_pointer_cast<DistributedAgentResponsePayload>(msg->payload);
    const auto subPld = std::static_pointer_cast<EventTradePayload>(pld->payload);

    if (subPld->isResting) {
        fmt::println("TRADE NOTIF {}", json::jsonSerializable2str(subPld));
        m_tradeSignal(subPld);
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::agent

//-------------------------------------------------------------------------
