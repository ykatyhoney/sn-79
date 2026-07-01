/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/net/net.hpp>

#include <json_util.hpp>

#include <latch>
#include <stop_token>
#include <string>

//-------------------------------------------------------------------------

namespace taosim::net
{

//-------------------------------------------------------------------------

asio::awaitable<void> session(beast::tcp_stream stream, const rapidjson::Value& responsesJson);

//-------------------------------------------------------------------------

asio::awaitable<void> listen(
    tcp::endpoint endpoint,
    const rapidjson::Value& responsesJson,
    std::latch& serverReady,
    std::stop_token stopToken);

//-------------------------------------------------------------------------

struct ServerProps
{
    std::string host;
    uint16_t port;
    rapidjson::Document responsesJson;
};

void runServer(ServerProps props, std::latch& serverReady, std::stop_token stopToken);

//-------------------------------------------------------------------------

}  // namespace taosim::net

//-------------------------------------------------------------------------
