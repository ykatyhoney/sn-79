/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <fmt/format.h>

#include <string_view>
#include <utility>

//-------------------------------------------------------------------------
// SLTPDebugger — single-flag, statically-dispatched tracer for the SL/TP
// pipeline. Mirrors Simulation::logDebug in shape so call sites read the
// same: `SLTPDebugger::log("trade fired @ {}", price)`. Output is
// gated on the `enabled` flag, set once during SimulationManager
// construction from the `sltpDebug` XML attribute. When the flag stays
// off, calls remain free at runtime.

namespace taosim::util
{

struct SLTPDebugger
{
    static inline constexpr std::string_view s_prefix{"[SLTPDebug]"};

    static inline bool enabled{};

    template<typename... Args>
    static void log(fmt::format_string<Args...> fmt, Args&&... args)
    {
        if (enabled) {
            fmt::println("{} {}", s_prefix, fmt::format(fmt, std::forward<Args>(args)...));
        }
    }

    static void log(std::string_view sv)
    {
        if (enabled) {
            fmt::println("{} {}", s_prefix, sv);
        }
    }
};

}  // namespace taosim::util

//-------------------------------------------------------------------------
