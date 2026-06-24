/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <fmt/format.h>
#include <msgpack.hpp>

#include <concepts>
#include <source_location>

//-------------------------------------------------------------------------

namespace taosim::serialization
{

//-------------------------------------------------------------------------

class HumanReadableStream
{
    msgpack::sbuffer m_underlying;

public:
    explicit HumanReadableStream(size_t initByteSize = MSGPACK_SBUFFER_INIT_SIZE)
        : m_underlying{initByteSize}
    {}

    [[nodiscard]] const char* data() const noexcept { return m_underlying.data(); }
    [[nodiscard]] size_t size() const noexcept { return m_underlying.size(); }

    void write(const char* buf, size_t len) { m_underlying.write(buf, len); }
};

class BinaryStream
{
    msgpack::sbuffer m_underlying;

public:
    explicit BinaryStream(size_t initByteSize = MSGPACK_SBUFFER_INIT_SIZE)
        : m_underlying{initByteSize}
    {}

    [[nodiscard]] const char* data() const noexcept { return m_underlying.data(); }
    [[nodiscard]] size_t size() const noexcept { return m_underlying.size(); }

    void write(const char* buf, size_t len) { m_underlying.write(buf, len); }
};

//-------------------------------------------------------------------------
// Concepts codifying the implicit contract our msgpack-using IPC paths
// rely on. A type is `MsgPackDeserializable` if a packed message can be
// decoded into it via `msgpack::object::convert` — the standard pattern
// produced by `MSGPACK_DEFINE(...)` or a custom adaptor. A type is
// `MsgPackSerializable` if `msgpack::pack` accepts the given output
// stream (defaulting to `HumanReadableStream`). `MsgPackOutputStream`
// captures any sink exposing `write` plus `data()`/`size()` read-back —
// both `HumanReadableStream` and `BinaryStream` satisfy it.

template<typename T>
concept MsgPackDeserializable = requires(const msgpack::object& o, T& v) {
    o.convert(v);
};

template<typename T>
concept MsgPackOutputStream = requires(T& s, const char* p, size_t n) {
    s.write(p, n);
    { s.data() } -> std::convertible_to<const char*>;
    { s.size() } -> std::convertible_to<size_t>;
};

template<typename T, typename Stream = HumanReadableStream>
concept MsgPackSerializable = requires(const T& v, Stream& s) {
    msgpack::pack(s, v);
};

//-------------------------------------------------------------------------

struct MsgPackError : msgpack::type_error
{
    std::string message;

    explicit MsgPackError(std::source_location sl = std::source_location::current()) noexcept
    {
        message = fmt::format("{}#L{}: {}", sl.file_name(), sl.line(), msgpack::type_error::what());
    }

    const char* what() const noexcept override { return message.c_str(); }
};

//-------------------------------------------------------------------------

}  // namespace taosim::serialization

//-------------------------------------------------------------------------