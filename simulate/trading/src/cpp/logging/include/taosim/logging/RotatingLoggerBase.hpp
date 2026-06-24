/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/simulation/TimeConfig.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <boost/signals2/signal.hpp>

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <string_view>

//-------------------------------------------------------------------------

class Simulation;

//-------------------------------------------------------------------------

namespace taosim::logging
{

//-------------------------------------------------------------------------

struct RotatingLoggerBaseDesc
{
    std::string name;
    Simulation* simulation;
    std::filesystem::path filepath;
    std::chrono::system_clock::time_point startTimePoint{};
    std::string header;
};

struct FileSinkWithInfo
{
    std::unique_ptr<spdlog::sinks::basic_file_sink_st> sink;
    bool fileExisted;
};

//-------------------------------------------------------------------------

class RotatingLoggerBase
{
public:
    explicit RotatingLoggerBase(const RotatingLoggerBaseDesc& desc) noexcept;

    [[nodiscard]] const std::filesystem::path& filepath() const noexcept { return m_filepath; }

    // The file currently being written to — equals filepath() when log
    // rotation is disabled, otherwise the windowed file for the active
    // window. Lets a consumer read back what was most recently logged.
    [[nodiscard]] const std::filesystem::path& currentFilepath() const noexcept { return m_currentFilepath; }

    // Fires with each line written to the sink (after dedup / flush).
    // No subscribers unless an optional consumer attaches.
    [[nodiscard]] auto&& loggedSignal(this auto&& self) noexcept { return self.m_loggedSignal; }

protected:
    [[nodiscard]] FileSinkWithInfo makeFileSink();
    void updateSink(std::optional<Timestamp> currentTime = {});

    std::unique_ptr<spdlog::logger> m_logger;
    Simulation* m_simulation;
    std::filesystem::path m_filepath;
    std::chrono::system_clock::time_point m_startTimePoint;
    std::filesystem::path m_currentFilepath;
    simulation::TimestampConversionFn m_timeConverter;
    Timestamp m_currentWindowBegin;
    std::string m_header;

    boost::signals2::signal<void(std::string_view)> m_loggedSignal;
};

//-------------------------------------------------------------------------

}  // namespace taosim::logging

//-------------------------------------------------------------------------