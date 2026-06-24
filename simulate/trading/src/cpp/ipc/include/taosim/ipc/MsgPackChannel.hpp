/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/ipc/PosixMessageQueue.hpp>
#include <taosim/serialization/msgpack/common.hpp>

#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <msgpack.hpp>

#include <fmt/format.h>

#include <concepts>
#include <cstring>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

//-------------------------------------------------------------------------
// Synchronous, request/response RPC over POSIX shared memory with
// msgpack-encoded payloads. Two roles (`MsgPackServer<Req, Res>` and
// `MsgPackClient<Req, Res>`) talk through three named primitives in
// each direction: a `PosixMessageQueue` carrying the next message's
// byte length, a `shared_memory_object` carrying the payload, and an
// optional `named_semaphore` for belt-and-braces synchronization
// (POSIX mq operations are already memory-sync points, so the semaphore
// is conservative: leave its name empty in `IpcChannelDesc` to omit).
//
// Wire protocol (server view of one iteration):
//   1. blocking-receive `size_t` on `reqMq`
//   2. (optional) wait on `reqSem`
//   3. open `reqShm`, read `size_t` bytes, msgpack-decode to `Request`
//   4. invoke handler, msgpack-encode result into a `HumanReadableStream`
//   5. open/truncate `resShm`, copy stream bytes in
//   6. (optional) post `resSem`
//   7. send `size_t` (response size) on `resMq`
//
// Client mirrors steps in reverse.
//
// Ownership of the named primitives: dtors only close per-process
// handles (mq_close, munmap, file descriptor release). They do NOT
// `*_unlink` the names — those are filesystem-style persistent objects
// shared across processes, and unlinking them out from under a peer
// (e.g. our simulator yanking shm that an external validator still
// has open) is the cross-process foot-gun we want to avoid. Call the
// static `cleanup(desc)` explicitly when you want the names gone;
// signal handlers are the usual driver.

namespace taosim::ipc
{

namespace bipc = boost::interprocess;

//-------------------------------------------------------------------------

struct IpcChannelDesc
{
    std::string reqMqName;
    std::string resMqName;
    std::string reqShmName;
    std::string resShmName;
    // Optional belt-and-braces synchronization atop the mq's POSIX
    // memory-sync guarantees. Set to empty to disable; both ends must
    // agree on whether to use them.
    std::string reqSemName;
    std::string resSemName;
};

//-------------------------------------------------------------------------

namespace detail
{

// Pulls the next request: blocks on the req-mq for the byte count,
// optionally drains the req-sem, then opens and maps the req-shm.
// Returns the mapped region (keep it alive while reading) and the
// payload size as written by the client.
struct ReceivedShmPayload
{
    bipc::shared_memory_object shm;
    bipc::mapped_region region;
    size_t size{};
};

[[nodiscard]] inline ReceivedShmPayload waitForRequest(
    const IpcChannelDesc& desc, PosixMessageQueue& reqMq)
{
    size_t payloadSize{};
    reqMq.blockingReceive(std::span<char>{
        reinterpret_cast<char*>(&payloadSize), sizeof(payloadSize)});

    if (!desc.reqSemName.empty()) {
        bipc::named_semaphore sem{bipc::open_or_create, desc.reqSemName.c_str(), 0};
        sem.wait();
    }

    bipc::shared_memory_object shm{
        bipc::open_only, desc.reqShmName.c_str(), bipc::read_write};
    bipc::mapped_region region{shm, bipc::read_write};
    return {std::move(shm), std::move(region), payloadSize};
}

// Writes the response: opens (or creates) the res-shm at the right
// size, copies bytes in, optionally posts the res-sem, then signals
// the size on the res-mq.
inline void publishResponse(
    const IpcChannelDesc& desc, PosixMessageQueue& resMq, std::span<const char> bytes)
{
    bipc::shared_memory_object shm{
        bipc::open_or_create, desc.resShmName.c_str(), bipc::read_write};
    shm.truncate(bytes.size());
    bipc::mapped_region region{shm, bipc::read_write};
    std::memcpy(region.get_address(), bytes.data(), bytes.size());

    if (!desc.resSemName.empty()) {
        bipc::named_semaphore sem{bipc::open_or_create, desc.resSemName.c_str(), 0};
        sem.post();
    }

    const size_t payloadSize = bytes.size();
    resMq.send(std::span<const char>{
        reinterpret_cast<const char*>(&payloadSize), sizeof(payloadSize)});
}

}  // namespace detail

//-------------------------------------------------------------------------

template<
    serialization::MsgPackDeserializable Request,
    serialization::MsgPackSerializable Response>
class MsgPackServer
{
public:
    explicit MsgPackServer(const IpcChannelDesc& desc);
    ~MsgPackServer() noexcept = default;

    MsgPackServer(const MsgPackServer&) = delete;
    MsgPackServer& operator=(const MsgPackServer&) = delete;
    MsgPackServer(MsgPackServer&&) = delete;
    MsgPackServer& operator=(MsgPackServer&&) = delete;

    // Process exactly one request → response cycle. Throws whatever the
    // handler throws; lets the caller's loop / signal logic stay in
    // charge of program lifetime. The handler constraint is checked
    // in-body via `static_assert` for a clearer diagnostic than a
    // failed-substitution error from a class-level `requires`.
    template<typename H>
    void serveOne(H&& handler);

    template<typename H>
    void serve(H&& handler);

    [[nodiscard]] const IpcChannelDesc& desc() const noexcept { return m_desc; }

    // Static so signal handlers (which can't easily reach into member
    // state) can pass a captured-on-the-side desc and call this from
    // async-signal-safe context. All four underlying removes call
    // `*_unlink(3)` variants which are async-signal-safe.
    static void cleanup(const IpcChannelDesc& desc) noexcept;

private:
    IpcChannelDesc m_desc;
    PosixMessageQueue m_reqMq;
    PosixMessageQueue m_resMq;
};

//-------------------------------------------------------------------------

template<
    serialization::MsgPackDeserializable Request,
    serialization::MsgPackSerializable Response>
MsgPackServer<Request, Response>::MsgPackServer(const IpcChannelDesc& desc)
    : m_desc{desc},
      m_reqMq{{.name = m_desc.reqMqName}},
      m_resMq{{.name = m_desc.resMqName}}
{}

template<
    serialization::MsgPackDeserializable Request,
    serialization::MsgPackSerializable Response>
template<typename H>
void MsgPackServer<Request, Response>::serveOne(H&& handler)
{
    static_assert(std::invocable<H, const Request&>);
    static_assert(std::convertible_to<std::invoke_result_t<H, const Request&>, Response>);

    auto received = detail::waitForRequest(m_desc, m_reqMq);
    msgpack::object_handle oh = msgpack::unpack(
        static_cast<const char*>(received.region.get_address()), received.size);
    Request req;
    oh.get().convert(req);

    Response res = handler(static_cast<const Request&>(req));

    serialization::HumanReadableStream stream;
    msgpack::pack(stream, res);
    detail::publishResponse(m_desc, m_resMq,
        std::span<const char>{stream.data(), stream.size()});
}

template<
    serialization::MsgPackDeserializable Request,
    serialization::MsgPackSerializable Response>
template<typename H>
void MsgPackServer<Request, Response>::serve(H&& handler)
{
    for (;;) {
        serveOne(handler);
    }
}

template<
    serialization::MsgPackDeserializable Request,
    serialization::MsgPackSerializable Response>
void MsgPackServer<Request, Response>::cleanup(const IpcChannelDesc& desc) noexcept
{
    if (!desc.reqSemName.empty()) {
        bipc::named_semaphore::remove(desc.reqSemName.c_str());
    }
    if (!desc.resSemName.empty()) {
        bipc::named_semaphore::remove(desc.resSemName.c_str());
    }
    PosixMessageQueue::remove(desc.reqMqName);
    PosixMessageQueue::remove(desc.resMqName);
    bipc::shared_memory_object::remove(desc.reqShmName.c_str());
    bipc::shared_memory_object::remove(desc.resShmName.c_str());
}

//-------------------------------------------------------------------------

// Both template parameters are unconstrained at the class level so
// `MsgPackClient<Fwd>` can be instantiated against a forward-declared
// Request type (otherwise the `MsgPackSerializable` concept would
// require Request's pack adaptor — and thus its full definition —
// at every site holding a `MsgPackClient` member, which forces
// awkward header cycles). The actual concept checks live inside
// `exchange` and `call` as `static_assert`s; Response also defaults
// to `void` for the raw-bytes-only use case.

template<typename Request, typename Response = void>
class MsgPackClient
{
public:
    explicit MsgPackClient(IpcChannelDesc desc);
    ~MsgPackClient() noexcept = default;

    MsgPackClient(const MsgPackClient&) = delete;
    MsgPackClient& operator=(const MsgPackClient&) = delete;
    MsgPackClient(MsgPackClient&&) = delete;
    MsgPackClient& operator=(MsgPackClient&&) = delete;

    // Send one request, block for the matching response. Throws on
    // msgpack decode failure (`msgpack::type_error`); mq timeouts are
    // surfaced by the underlying `PosixMessageQueue`.
    [[nodiscard]] Response call(const Request& req);

    // Raw-bytes escape hatch: pack the request, do the shm/mq dance,
    // hand the response bytes to `consumer` (a callable taking
    // `std::span<const char>`). Use this when the response shape is
    // too dynamic to express as a single `Response` type — for
    // example a map whose value is iterated manually with
    // `msgpack::object`. The bytes view is valid only inside the
    // callback; the mapping is torn down on return. The consumer
    // constraint is checked in-body via `static_assert` for the same
    // diagnostic-clarity reasons as `MsgPackServer::serveOne`.
    //
    // `Stream` defaults to `HumanReadableStream`; override with e.g.
    // `BinaryStream` when the Request's pack adaptor branches on
    // stream tag for a different on-wire encoding.
    template<
        serialization::MsgPackOutputStream Stream = serialization::HumanReadableStream,
        typename Consumer>
    void exchange(const Request& req, Consumer&& consumer);

    // Same shape but caller hands in pre-packed bytes. Useful when
    // the caller already produced a packed stream for other purposes
    // (e.g. measuring its size for a step-timing line) and doesn't
    // want this method to re-pack. The byte buffer must outlive the
    // call.
    template<typename Consumer>
    void exchange(std::span<const char> reqBytes, Consumer&& consumer);

    [[nodiscard]] const IpcChannelDesc& desc() const noexcept { return m_desc; }

private:
    IpcChannelDesc m_desc;
    PosixMessageQueue m_reqMq;
    PosixMessageQueue m_resMq;
};

//-------------------------------------------------------------------------

template<typename Request, typename Response>
MsgPackClient<Request, Response>::MsgPackClient(IpcChannelDesc desc)
    : m_desc{std::move(desc)},
      m_reqMq{{.name = m_desc.reqMqName}},
      m_resMq{{.name = m_desc.resMqName}}
{
    // Start from a clean slate: a previous run (or one interrupted mid-call)
    // can leave a stale length frame in either persistent queue, which would
    // knock the very first request/response round trip out of lock-step. We
    // drain rather than unlink — unlinking would orphan a server already
    // holding these names. The server drains on its side symmetrically.
    m_reqMq.flush();
    m_resMq.flush();
}

template<typename Request, typename Response>
template<serialization::MsgPackOutputStream Stream, typename Consumer>
void MsgPackClient<Request, Response>::exchange(const Request& req, Consumer&& consumer)
{
    static_assert(serialization::MsgPackSerializable<Request, Stream>);

    Stream stream;
    msgpack::pack(stream, req);
    exchange(
        std::span<const char>{stream.data(), stream.size()},
        std::forward<Consumer>(consumer));
}

template<typename Request, typename Response>
template<typename Consumer>
void MsgPackClient<Request, Response>::exchange(
    std::span<const char> reqBytes, Consumer&& consumer)
{
    static_assert(std::invocable<Consumer, std::span<const char>>);

    bipc::shared_memory_object reqShm{
        bipc::open_or_create, m_desc.reqShmName.c_str(), bipc::read_write};
    reqShm.truncate(reqBytes.size());
    bipc::mapped_region reqRegion{reqShm, bipc::read_write};
    std::memcpy(reqRegion.get_address(), reqBytes.data(), reqBytes.size());

    if (!m_desc.reqSemName.empty()) {
        bipc::named_semaphore sem{bipc::open_or_create, m_desc.reqSemName.c_str(), 0};
        sem.post();
    }

    const size_t reqSize = reqBytes.size();
    // mq operations are timed (see PosixMessageQueueDesc::timeout).
    // Flush before each send so a stale message from a previous run
    // doesn't block the queue; loop on timeout so a slow peer doesn't
    // drop the round-trip. Matches the existing retry shape in
    // SimulationManager::publishStateMessagePack.
    for (;;) {
        m_reqMq.flush();
        if (m_reqMq.send(std::span<const char>{
                reinterpret_cast<const char*>(&reqSize), sizeof(reqSize)})) {
            break;
        }
        fmt::println("MsgPackClient: send to /{} timed out, retrying...",
            m_desc.reqMqName);
    }

    size_t resSize{};
    for (;;) {
        if (m_resMq.receive(std::span<char>{
                reinterpret_cast<char*>(&resSize), sizeof(resSize)}) != -1) {
            break;
        }
        fmt::println("MsgPackClient: receive from /{} timed out, retrying...",
            m_desc.resMqName);
    }

    if (!m_desc.resSemName.empty()) {
        bipc::named_semaphore sem{bipc::open_or_create, m_desc.resSemName.c_str(), 0};
        sem.wait();
    }

    bipc::shared_memory_object resShm{
        bipc::open_only, m_desc.resShmName.c_str(), bipc::read_write};
    bipc::mapped_region resRegion{resShm, bipc::read_write};

    consumer(std::span<const char>{
        static_cast<const char*>(resRegion.get_address()), resSize});
}

template<typename Request, typename Response>
Response MsgPackClient<Request, Response>::call(const Request& req)
{
    static_assert(!std::is_void_v<Response>,
        "call() requires MsgPackClient to be instantiated with a non-void Response; "
        "use exchange() for the raw-bytes API instead");
    static_assert(serialization::MsgPackDeserializable<Response>,
        "Response must satisfy MsgPackDeserializable for call() to be available");

    Response res;
    exchange(req, [&](std::span<const char> bytes) {
        msgpack::object_handle oh = msgpack::unpack(bytes.data(), bytes.size());
        oh.get().convert(res);
    });
    return res;
}

//-------------------------------------------------------------------------

}  // namespace taosim::ipc

//-------------------------------------------------------------------------
