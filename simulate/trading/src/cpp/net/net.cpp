/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/net/net.hpp>

//-------------------------------------------------------------------------

namespace taosim::net
{

//-------------------------------------------------------------------------

asio::awaitable<void> timeout(std::chrono::steady_clock::duration duration)
{
    asio::steady_timer timer{co_await this_coro::executor};
    timer.expires_after(duration);
    co_await timer.async_wait(use_nothrow_awaitable);
}

//-------------------------------------------------------------------------

asio::awaitable<void> timeout(int64_t duration)
{
    asio::steady_timer timer{co_await this_coro::executor};
    timer.expires_after(std::chrono::seconds{duration});
    co_await timer.async_wait(use_nothrow_awaitable);
}

//-------------------------------------------------------------------------

http::request<http::string_body> makeHttpRequest(const MakeHttpRequestContext& ctx)
{
    http::request<http::string_body> req;
    req.method(ctx.method);
    req.target(std::string{ctx.target});
    req.version(11);
    req.set(http::field::host, ctx.netInfo.host);
    req.set(http::field::content_type, "application/json");
    req.body() = std::string{ctx.body};
    req.prepare_payload();
    return req;
}

//-------------------------------------------------------------------------

}  // namespace taosim::net

//-------------------------------------------------------------------------
