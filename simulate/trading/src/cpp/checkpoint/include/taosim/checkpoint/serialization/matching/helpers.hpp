/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/checkpoint/serialization/matching/DynamicFeePolicy.hpp>
#include <taosim/checkpoint/serialization/matching/TieredFeePolicy.hpp>
#include <taosim/serialization/msgpack/utils.hpp>

//-------------------------------------------------------------------------

namespace taosim::checkpoint::serialization
{

//-------------------------------------------------------------------------

void packFeePolicy(auto& o, const taosim::matching::FeePolicy& feePolicy)
{
    if (auto fp = dynamic_cast<const taosim::matching::DynamicFeePolicy*>(&feePolicy)) {
        o.pack(*fp);
    }
    else if (auto fp = dynamic_cast<const taosim::matching::TieredFeePolicy*>(&feePolicy)) {
        o.pack(*fp);
    }
    else {
        o.pack_nil();
    }
}

void unpackFeePolicy(const auto& o, taosim::matching::FeePolicy& feePolicy)
{
    const auto typeOpt = taosim::serialization::msgpackFindMap<std::string_view>(o, "type");
    if (!typeOpt) {
        throw taosim::serialization::MsgPackError{};
    }
    auto type = *typeOpt;

    if (type == "dynamic") {
        auto ptr = dynamic_cast<taosim::matching::DynamicFeePolicy*>(&feePolicy);
        o.convert(*ptr);
    }
    else if (type == "tiered") {
        auto ptr = dynamic_cast<taosim::matching::TieredFeePolicy*>(&feePolicy);
        o.convert(*ptr);
    }
}

//-------------------------------------------------------------------------

}  // namespace taosim::matching::serialization

//-------------------------------------------------------------------------