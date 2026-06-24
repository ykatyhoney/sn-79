/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <json_util.hpp>

#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/as_tuple.hpp>
#include <boost/asio/experimental/awaitable_operators.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <fmt/format.h>
#include <rapidjson/document.h>

#include <chrono>
#include <concepts>
#include <cstdint>
#include <limits>
#include <source_location>
#include <string>
#include <string_view>
#include <thread>

//-------------------------------------------------------------------------

namespace taosim::net
{

//-------------------------------------------------------------------------

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace ip = asio::ip;
namespace this_coro = asio::this_coro;
using asio::use_awaitable;
using tcp = asio::ip::tcp;
using namespace asio::experimental::awaitable_operators;

using namespace std::literals::chrono_literals;
using std::chrono::steady_clock;

inline constexpr auto use_nothrow_awaitable = asio::as_tuple(use_awaitable);

//-------------------------------------------------------------------------

asio::awaitable<void> timeout(std::chrono::steady_clock::duration duration);
asio::awaitable<void> timeout(int64_t duration);

//-------------------------------------------------------------------------
// Generic connection parameters for sending HTTP requests. Owned by the
// driver of the send (SimulationManager, Exchange, ...) and passed by
// reference through the Context structs below. Endpoints are per-call
// (AsyncSendContext::endpoint) rather than baked in here, so callers that
// hit several endpoints against the same host only need one instance.

struct NetworkingInfo
{
    std::string host, port;
    int64_t resolveTimeout{}, connectTimeout{}, writeTimeout{}, readTimeout{};
};

//-------------------------------------------------------------------------
// Any type that exposes a string_view-taking logDebug can be used as the
// logger for asyncSendOverNetwork.

template<typename T>
concept DebugLoggable = requires(const T& t, std::string_view sv) {
    { t.logDebug(sv) };
};

//-------------------------------------------------------------------------

struct MakeHttpRequestContext
{
    const NetworkingInfo& netInfo;
    std::string_view target;
    std::string_view body;
    // Default to GET to keep most existing call sites concise. Override
    // to POST when sending a body to a server that follows RFC 7230
    // strictly (e.g. FastAPI, Flask) — those frameworks ignore GET
    // bodies and the receiver would otherwise see an empty payload.
    http::verb method{http::verb::get};
};

[[nodiscard]] http::request<http::string_body> makeHttpRequest(const MakeHttpRequestContext& ctx);

//-------------------------------------------------------------------------

template<DebugLoggable Logger>
struct AsyncSendContext
{
    const Logger& logger;
    const NetworkingInfo& netInfo;
    const rapidjson::Value& reqBody;
    std::string_view endpoint;
    rapidjson::Document& resJson;
    http::verb method{http::verb::get};
    // When set, any resolve/connect/write/read failure (timeout or error)
    // causes the coroutine to log once and return immediately, leaving
    // ctx.resJson untouched. Use for optional services where blocking
    // the caller until the service comes back is the wrong behavior.
    bool failFast{};
};

//-------------------------------------------------------------------------
// Coroutine that resolves, connects, writes, and reads a single HTTP
// exchange against ctx.netInfo.host:port at ctx.endpoint. Retries each
// stage indefinitely on timeout or error (unless ctx.failFast is set, in
// which case the first failure returns), logging the failure through
// ctx.logger.logDebug. The response body is parsed into ctx.resJson.

template<DebugLoggable Logger>
asio::awaitable<void> asyncSendOverNetwork(AsyncSendContext<Logger> ctx)
{
    const auto& netInfo = ctx.netInfo;

retry:
    auto resolver =
        use_nothrow_awaitable.as_default_on(tcp::resolver{co_await this_coro::executor});
    auto tcp_stream =
        use_nothrow_awaitable.as_default_on(beast::tcp_stream{co_await this_coro::executor});

    int attempts = 0;
    // Resolve.
    auto endpointsVariant = co_await (
        resolver.async_resolve(netInfo.host, netInfo.port) || timeout(netInfo.resolveTimeout));
    while (endpointsVariant.index() == 1) {
        ctx.logger.logDebug(fmt::format(
            "tcp::resolver timed out on {}:{}", netInfo.host, netInfo.port));
        if (ctx.failFast) co_return;
        std::this_thread::sleep_for(10s);
        endpointsVariant = co_await (
            resolver.async_resolve(netInfo.host, netInfo.port) || timeout(netInfo.resolveTimeout));
    }
    auto [e1, endpoints] = std::get<0>(endpointsVariant);
    while (e1) {
        const auto loc = std::source_location::current();
        ctx.logger.logDebug(fmt::format(
            "{}#L{}: {}:{}: {}", loc.file_name(), loc.line(), netInfo.host, netInfo.port, e1.what()));
        if (ctx.failFast) co_return;
        attempts++;
        ctx.logger.logDebug(fmt::format(
            "Unable to resolve connection to {}:{}{} - Retrying (Attempt {})",
            netInfo.host, netInfo.port, ctx.endpoint, attempts));
        std::this_thread::sleep_for(10s);
        endpointsVariant = co_await (
            resolver.async_resolve(netInfo.host, netInfo.port) || timeout(netInfo.resolveTimeout));
        auto [e11, endpoints1] = std::get<0>(endpointsVariant);
        e1 = e11;
        endpoints = endpoints1;
    }

    // Connect.
    attempts = 0;
    auto connectVariant =
        co_await (tcp_stream.async_connect(endpoints) || timeout(netInfo.connectTimeout));
    while (connectVariant.index() == 1) {
        ctx.logger.logDebug(fmt::format(
            "tcp_stream::async_connect timed out on {}:{}", netInfo.host, netInfo.port));
        if (ctx.failFast) co_return;
        std::this_thread::sleep_for(10s);
        connectVariant =
            co_await (tcp_stream.async_connect(endpoints) || timeout(netInfo.connectTimeout));
    }
    auto [e2, _2] = std::get<0>(connectVariant);
    while (e2) {
        const auto loc = std::source_location::current();
        ctx.logger.logDebug(fmt::format(
            "{}#L{}: {}:{}: {}", loc.file_name(), loc.line(), netInfo.host, netInfo.port, e2.what()));
        if (ctx.failFast) co_return;
        attempts++;
        ctx.logger.logDebug(fmt::format(
            "Unable to connect to {}:{}{} - Retrying (Attempt {})",
            netInfo.host, netInfo.port, ctx.endpoint, attempts));
        std::this_thread::sleep_for(10s);
        connectVariant =
            co_await (tcp_stream.async_connect(endpoints) || timeout(netInfo.connectTimeout));
        auto [e21, _21] = std::get<0>(connectVariant);
        e2 = e21;
        _2 = _21;
    }

    // Create the request.
    const auto req = makeHttpRequest({
        .netInfo = netInfo,
        .target = ctx.endpoint,
        .body = taosim::json::json2str(ctx.reqBody),
        .method = ctx.method
    });

    // Send the request.
    attempts = 0;
    auto writeVariant =
        co_await (http::async_write(tcp_stream, req) || timeout(netInfo.writeTimeout));
    while (writeVariant.index() == 1) {
        ctx.logger.logDebug(fmt::format(
            "http::async_write timed out on {}:{}", netInfo.host, netInfo.port));
        if (ctx.failFast) co_return;
        std::this_thread::sleep_for(5s);
        writeVariant =
            co_await (http::async_write(tcp_stream, req) || timeout(netInfo.writeTimeout));
    }
    auto [e3, _3] = std::get<0>(writeVariant);
    while (e3) {
        const auto loc = std::source_location::current();
        ctx.logger.logDebug(fmt::format(
            "{}#L{}: {}:{}: {}", loc.file_name(), loc.line(), netInfo.host, netInfo.port, e3.what()));
        if (ctx.failFast) co_return;
        attempts++;
        ctx.logger.logDebug(fmt::format(
            "Unable to send request to validator at {}:{}{} - Retrying (Attempt {})",
            netInfo.host, netInfo.port, ctx.endpoint, attempts));
        goto retry;
    }

    // Receive the response.
    attempts = 0;
    beast::flat_buffer buf;
    http::response_parser<http::string_body> parser{http::response<http::string_body>{}};
    parser.eager(true);
    parser.body_limit(std::numeric_limits<size_t>::max());
    auto readVariant =
        co_await (http::async_read(tcp_stream, buf, parser) || timeout(netInfo.readTimeout));
    if (readVariant.index() == 1) {
        ctx.logger.logDebug(fmt::format(
            "http::async_read timed out on {}:{}", netInfo.host, netInfo.port));
        if (ctx.failFast) co_return;
        goto retry;
    }
    auto [e4, _4] = std::get<0>(readVariant);
    while (e4) {
        const auto loc = std::source_location::current();
        ctx.logger.logDebug(fmt::format(
            "{}#L{}: {}:{}: {}", loc.file_name(), loc.line(), netInfo.host, netInfo.port, e4.what()));
        if (ctx.failFast) co_return;
        attempts++;
        ctx.logger.logDebug(fmt::format(
            "Unable to read response from validator at {}:{}{} : {} - re-sending request.",
            netInfo.host, netInfo.port, ctx.endpoint, e4.what(), attempts));
        goto retry;
    }

    http::response<http::string_body> res = parser.release();
    ctx.resJson.Parse(res.body().c_str());
    // ctx.logger.logDebug(fmt::format("RECEIVED RESPONSE: {}", res.body().c_str()));
}

//-------------------------------------------------------------------------

}  // namespace taosim::net

//-------------------------------------------------------------------------
