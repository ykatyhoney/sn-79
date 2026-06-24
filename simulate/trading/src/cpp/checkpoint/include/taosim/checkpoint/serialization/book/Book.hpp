/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/book/Book.hpp>
#include <taosim/book/serialization/OrderContainer.hpp>
#include <taosim/book/serialization/TickContainer.hpp>

#include <range/v3/action/reverse.hpp>

//-------------------------------------------------------------------------

namespace msgpack
{

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{

namespace adaptor
{

template<>
struct convert<taosim::book::Book>
{
    const msgpack::object& operator()(const msgpack::object& o, taosim::book::Book& v) const
    {
        if (o.type != msgpack::type::MAP) {
            throw taosim::serialization::MsgPackError{};
        }

        for (const auto& [k, val] : o.via.map) {
            auto key = k.as<std::string_view>();

            if (key == "buyQueue") {
                val.convert(v.buyQueue());
            }
            else if (key == "sellQueue") {
                val.convert(v.sellQueue());
            }
            else if (key == "orderIdCounter") {
                val.convert(v.orderIdCounter());
            }
            else if (key == "tradeIdCounter") {
                val.convert(v.tradeIdCounter());
            }
            else if (key == "orderToClientInfo") {
                val.convert(v.orderToClientInfo());
            }
        }

        return o;
    }
};

template<>
struct pack<taosim::book::Book>
{
    template<typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, const taosim::book::Book& v) const
    {
        o.pack_map(5);

        o.pack("buyQueue");
        o.pack(v.buyQueue());

        o.pack("sellQueue");
        o.pack(v.sellQueue());

        o.pack("orderIdCounter");
        o.pack(*v.orderIdCounter());

        o.pack("tradeIdCounter");
        o.pack(*v.tradeIdCounter());

        o.pack("orderToClientInfo");
        o.pack(v.orderToClientInfo());

        return o;
    }
};

}  // namespace adaptor

}  // MSGPACK_API_VERSION_NAMESPACE

}  // namespace msgpack

//-------------------------------------------------------------------------
