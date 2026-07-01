#include <taosim/decimal/decimal.hpp>
#include <taosim/fp/fp.hpp>

#include <gmock/gmock.h>

//-------------------------------------------------------------------------

using namespace taosim;

using namespace testing;

//-------------------------------------------------------------------------

TEST(Multiplication, WorksCorrectly)
{
    using util::double2decimal, util::decimal2double, fp::u64f64_t;

    static constexpr auto a{1.2581235782386};
    static constexpr auto b{2.5472356487651};

    static constexpr uint32_t decimalPlaces{21};

    const u64f64_t fpRes = cnl::quotient(u64f64_t{a}, u64f64_t{b});
    const auto decRes = double2decimal(a, decimalPlaces) / double2decimal(b, decimalPlaces);

    EXPECT_DOUBLE_EQ(static_cast<double>(fpRes), decimal2double(decRes));
}

//-------------------------------------------------------------------------