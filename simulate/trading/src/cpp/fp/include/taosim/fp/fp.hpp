#pragma once

#include <cnl/all.h>

#include <cstdint>

//-------------------------------------------------------------------------

namespace taosim::fp
{

using int128_t = cnl::wide_integer<128, int64_t>;
using uint128_t = cnl::wide_integer<128, uint64_t>;

using i64f64_t = cnl::elastic_scaled_integer<128, -64, int128_t>;
using u64f64_t = cnl::elastic_scaled_integer<128, -64, uint128_t>;

}  // namespace taosim::fp

//-------------------------------------------------------------------------