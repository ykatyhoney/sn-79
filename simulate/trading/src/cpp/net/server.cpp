/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/net/server.hpp>

#include <fmt/core.h>

#include <cstdint>
#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::net
{

//-------------------------------------------------------------------------

asio::awaitable<void> session(beast::tcp_stream stream, const rapidjson::Value& responsesJson)
{
    beast::flat_buffer buffer;
    thread_local static uint32_t counter = 0;

    try {
        for (;;) {
            http::request<http::string_body> req;
            co_await http::async_read(stream, buffer, req, use_awaitable);
            http::response<http::string_body> res{http::status::ok, req.version()};
            res.set(http::field::content_type, "application/json");
            res.body() = taosim::json::json2str(
                (counter++ == 0)
                    ? responsesJson
                    : [] {
                        rapidjson::Document json{rapidjson::kObjectType};
                        json.AddMember("responses", rapidjson::Value{}.SetArray(), json.GetAllocator());
                        return json;
                    }().Move());
            res.prepare_payload();
            co_await http::async_write(stream, res, use_awaitable);
        }
    }
    catch (const boost::system::system_error& se) {
        if (se.code() != http::error::end_of_stream) {
            throw;
        }
    }

    beast::error_code ec;
    stream.socket().shutdown(tcp::socket::shutdown_send, ec);
}

//-------------------------------------------------------------------------

asio::awaitable<void> listen(
    tcp::endpoint endpoint,
    const rapidjson::Value& responsesJson,
    std::latch& serverReady,
    std::stop_token stopToken)
{
    tcp::acceptor acceptor{co_await this_coro::executor, endpoint};
    acceptor.set_option(tcp::acceptor::reuse_address(true));
    serverReady.count_down();

    while (!stopToken.stop_requested()) {
        asio::co_spawn(
            acceptor.get_executor(),
            session(
                beast::tcp_stream{co_await acceptor.async_accept(use_awaitable)}, responsesJson),
            asio::detached);
    }
}

//-------------------------------------------------------------------------

void runServer(ServerProps props, std::latch& serverReady, std::stop_token stopToken)
{
    asio::io_context ctx;
    asio::co_spawn(
        ctx,
        listen(
            tcp::endpoint{ip::make_address(props.host), props.port},
            props.responsesJson,
            serverReady,
            stopToken),
        asio::detached);
    ctx.run();
}

//-------------------------------------------------------------------------

}  // namespace taosim::net

//-------------------------------------------------------------------------
