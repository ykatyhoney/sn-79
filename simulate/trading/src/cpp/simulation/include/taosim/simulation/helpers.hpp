/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <fmt/format.h>

#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

inline void printProgress(size_t cur, size_t tot, std::string_view label)
{
    constexpr int W = 30;
    const int filled = tot > 0 ? static_cast<int>(W * cur / tot) : 0;
    const std::string bar =
        std::string(filled > 0 ? filled - 1 : 0, '=')
        + (filled > 0 ? ">" : "")
        + std::string(W - filled, ' ');
    fmt::print(
        "\r  {:<20} [{}] {:>3}%  {}/{}",
        label, bar, tot > 0 ? 100 * cur / tot : 0, cur, tot);
    std::fflush(stdout);
    if (cur == tot) fmt::println("");
}

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
