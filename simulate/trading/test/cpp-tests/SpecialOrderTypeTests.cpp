/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */

#include "formatting.hpp"
#include "MultiBookExchangeAgent.hpp"
#include "Order.hpp"
#include "Simulation.hpp"
#include <taosim/net/server.hpp>
#include "util.hpp"
#include <taosim/decimal/decimal.hpp>
#include <taosim/matching/ClearingManager.hpp>
#include <taosim/matching/FeePolicy.hpp>
#include <taosim/matching/FeePolicyWrapper.hpp>
#include <taosim/message/PayloadFactory.hpp>

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

//-------------------------------------------------------------------------

using namespace taosim;
using namespace taosim::accounting;
using namespace taosim::book;
using namespace taosim::matching;
using namespace taosim::literals;

using namespace testing;

namespace fs = std::filesystem;

//-------------------------------------------------------------------------

namespace
{

const auto kTestDataPath = fs::path{__FILE__}.parent_path() / "data";

//-------------------------------------------------------------------------
// Helpers
//-------------------------------------------------------------------------

void setupLimitOrder(
    MultiBookExchangeAgent* exchange,
    AgentId agentId,
    BookId bookId,
    OrderDirection direction,
    decimal_t volume,
    decimal_t price,
    decimal_t leverage)
{
    const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
        direction, volume, price, leverage, bookId, Currency::BASE,
        std::nullopt, false, TimeInForce::GTC, std::nullopt, STPFlag::CO);
    const auto orderResult = exchange->clearingManager().handleOrder(
        LimitOrderDesc{.agentId = agentId, .payload = payload});
    [[maybe_unused]] auto _ = exchange->books()[bookId]->placeLimitOrder(
        OrderClientContext{agentId},
        Timestamp{},
        orderResult.orderSize,
        payload->direction,
        payload->price,
        payload->leverage,
        payload->stpFlag);
}

OrderErrorCode submitTestOrder(
    MultiBookExchangeAgent* exchange,
    AgentId agentId,
    BookId bookId,
    OrderDirection direction,
    decimal_t volume,
    decimal_t price,
    decimal_t leverage,
    bool postOnly,
    TimeInForce timeInForce,
    STPFlag stpFlag,
    Currency currency = Currency::BASE)
{
    const auto payload = MessagePayload::create<PlaceOrderLimitPayload>(
        direction, volume, price, leverage, bookId, currency,
        std::nullopt, postOnly, timeInForce, std::nullopt, stpFlag);
    return exchange->clearingManager().handleOrder(
        LimitOrderDesc{.agentId = agentId, .payload = payload}).ec;
}

//-------------------------------------------------------------------------
// Common simulation setup
//
// Book state after setup:
//   asks: 2@303, 8@307
//   bids: 1@297, 3@291
//-------------------------------------------------------------------------

struct SimSetup
{
    static constexpr AgentId testAgent = -1;
    static constexpr AgentId bookAgent = -2;
    static constexpr BookId bookId{};

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    Book::Ptr book;

    void setUp()
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[bookId];

        exchange->accounts().registerLocal("testAgent");
        exchange->accounts().registerLocal("bookAgent");

        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::BUY, 3_dec, 291_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::BUY, 1_dec, 297_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::SELL, 2_dec, 303_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::SELL, 8_dec, 307_dec, 0_dec);
    }
};

//-------------------------------------------------------------------------
// Ghost-order simulation setup
//
// Book state after setup (ghost levels from filled orders):
//   asks: ghost@300, 2@303, 8@307
//   bids: 3@291, 1@297, ghost@298
//
// bestAsk() = 303,  sellQueue().front().price() = 300 (ghost)
// bestBid() = 297,  buyQueue().back().price()   = 298 (ghost)
//-------------------------------------------------------------------------

struct GhostSimSetup
{
    static constexpr AgentId testAgent = -1;
    static constexpr AgentId bookAgent = -2;
    static constexpr AgentId fillerAgent = -3;
    static constexpr BookId bookId{};

    taosim::util::Nodes nodes;
    std::unique_ptr<Simulation> simulation;
    MultiBookExchangeAgent* exchange{};
    Book::Ptr book;

    void setUp()
    {
        nodes = taosim::util::parseSimulationFile(kTestDataPath / "MultiAgentFees.xml");
        simulation = std::make_unique<Simulation>();
        simulation->setDebug(false);
        simulation->configure(nodes.simulation);
        exchange = simulation->exchange();
        book = exchange->books()[bookId];

        exchange->accounts().registerLocal("testAgent");
        exchange->accounts().registerLocal("bookAgent");
        exchange->accounts().registerLocal("fillerAgent");

        // Place orders that will become ghosts
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::SELL, 1_dec, 300_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::BUY, 1_dec, 298_dec, 0_dec);

        // Place normal (surviving) orders
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::BUY, 3_dec, 291_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::BUY, 1_dec, 297_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::SELL, 2_dec, 303_dec, 0_dec);
        setupLimitOrder(exchange, bookAgent, bookId, OrderDirection::SELL, 8_dec, 307_dec, 0_dec);

        // Fill the ghost-to-be orders with crossing orders
        setupLimitOrder(exchange, fillerAgent, bookId, OrderDirection::BUY, 1_dec, 300_dec, 0_dec);
        setupLimitOrder(exchange, fillerAgent, bookId, OrderDirection::SELL, 1_dec, 298_dec, 0_dec);
    }
};

//-------------------------------------------------------------------------
// IOC (Immediate or Cancel) Tests
//-------------------------------------------------------------------------

struct IOCTestParams
{
    OrderDirection direction{OrderDirection::BUY};
    decimal_t volume{};
    decimal_t price{};
    bool postOnly{false};
    OrderErrorCode expectedEc{OrderErrorCode::VALID};
};

void PrintTo(const IOCTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{direction = {}, volume = {}, price = {}, postOnly = {}, expectedEc = {}}}",
        params.direction == OrderDirection::BUY ? "BUY" : "SELL",
        params.volume,
        params.price,
        params.postOnly,
        static_cast<uint32_t>(params.expectedEc));
}

struct IOCTest : TestWithParam<IOCTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    SimSetup sim;
    IOCTestParams params;
};

TEST_P(IOCTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, SimSetup::testAgent, SimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        params.postOnly, TimeInForce::IOC, STPFlag::NONE);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    SpecialOrderTypes,
    IOCTest,
    Values(
        // BUY IOC at 303 — 2 available at 303, fills 1
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY IOC at 307 — crosses both ask levels, fills 5
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 5_dec,
            .price = 307_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY IOC at 290 — no asks at or below 290
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 290_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL IOC at 297 — 1 available at 297, fills 1
        IOCTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL IOC at 310 — no bids at or above 310
        IOCTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 310_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // IOC + postOnly — mutually exclusive, always rejected
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 303_dec,
            .postOnly = true,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        }));

//-------------------------------------------------------------------------
// IOC with ghost orders — nonZeroLevelsView must skip zero-volume levels
//-------------------------------------------------------------------------

struct GhostIOCTest : TestWithParam<IOCTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    GhostSimSetup sim;
    IOCTestParams params;
};

TEST_P(GhostIOCTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, GhostSimSetup::testAgent, GhostSimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        params.postOnly, TimeInForce::IOC, STPFlag::NONE);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    GhostOrders,
    GhostIOCTest,
    Values(
        // BUY IOC at 301 — ghost@300 skipped, no real asks at ≤301
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 301_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // BUY IOC at 303 — ghost@300 skipped, fills from 2@303
        IOCTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL IOC at 298 — ghost@298 skipped, no real bids at ≥298
        IOCTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 298_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL IOC at 297 — ghost@298 skipped, fills from 1@297
        IOCTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::VALID
        }));

//-------------------------------------------------------------------------
// FOK (Fill or Kill) Tests
//-------------------------------------------------------------------------

struct FOKTestParams
{
    OrderDirection direction{OrderDirection::BUY};
    decimal_t volume{};
    decimal_t price{};
    bool postOnly{false};
    OrderErrorCode expectedEc{OrderErrorCode::VALID};
};

void PrintTo(const FOKTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{direction = {}, volume = {}, price = {}, postOnly = {}, expectedEc = {}}}",
        params.direction == OrderDirection::BUY ? "BUY" : "SELL",
        params.volume,
        params.price,
        params.postOnly,
        static_cast<uint32_t>(params.expectedEc));
}

struct FOKTest : TestWithParam<FOKTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    SimSetup sim;
    FOKTestParams params;
};

TEST_P(FOKTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, SimSetup::testAgent, SimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        params.postOnly, TimeInForce::FOK, STPFlag::CO);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    SpecialOrderTypes,
    FOKTest,
    Values(
        // BUY FOK 2@303 — exactly 2 available at 303
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 2_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY FOK 5@303 — only 2 at 303, next level 307 > limit 303
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 5_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // BUY FOK 5@307 — 2@303 + 8@307 = 10, enough to fill 5
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 5_dec,
            .price = 307_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL FOK 4@291 — 1@297 + 3@291 = 4, exact fill
        FOKTestParams{
            .direction = OrderDirection::SELL,
            .volume = 4_dec,
            .price = 291_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL FOK 3@297 — only 1@297, next level 291 < limit 297
        FOKTestParams{
            .direction = OrderDirection::SELL,
            .volume = 3_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // FOK + postOnly — mutually exclusive, always rejected
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 2_dec,
            .price = 307_dec,
            .postOnly = true,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        }));

//-------------------------------------------------------------------------
// FOK with ghost orders — nonZeroLevelsView must skip zero-volume levels
//-------------------------------------------------------------------------

struct GhostFOKTest : TestWithParam<FOKTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    GhostSimSetup sim;
    FOKTestParams params;
};

TEST_P(GhostFOKTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, GhostSimSetup::testAgent, GhostSimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        params.postOnly, TimeInForce::FOK, STPFlag::CO);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    GhostOrders,
    GhostFOKTest,
    Values(
        // BUY FOK 2@303 — ghost@300 skipped, 2@303 fills
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 2_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY FOK 5@303 — ghost@300 skipped, only 2@303, 307>303
        FOKTestParams{
            .direction = OrderDirection::BUY,
            .volume = 5_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL FOK 1@297 — ghost@298 skipped, 1@297 fills
        FOKTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL FOK 3@297 — ghost@298 skipped, only 1@297, 291<297
        FOKTestParams{
            .direction = OrderDirection::SELL,
            .volume = 3_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        }));

//-------------------------------------------------------------------------
// Post-Only Tests
//-------------------------------------------------------------------------

struct PostOnlyTestParams
{
    OrderDirection direction{OrderDirection::BUY};
    decimal_t volume{};
    decimal_t price{};
    TimeInForce timeInForce{TimeInForce::GTC};
    STPFlag stpFlag{STPFlag::NONE};
    OrderErrorCode expectedEc{OrderErrorCode::VALID};
};

void PrintTo(const PostOnlyTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{direction = {}, volume = {}, price = {}, timeInForce = {}, stpFlag = {}, expectedEc = {}}}",
        params.direction == OrderDirection::BUY ? "BUY" : "SELL",
        params.volume,
        params.price,
        static_cast<uint32_t>(params.timeInForce),
        static_cast<uint32_t>(params.stpFlag),
        static_cast<uint32_t>(params.expectedEc));
}

struct PostOnlyTest : TestWithParam<PostOnlyTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    SimSetup sim;
    PostOnlyTestParams params;
};

TEST_P(PostOnlyTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, SimSetup::testAgent, SimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        true, params.timeInForce, params.stpFlag);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    SpecialOrderTypes,
    PostOnlyTest,
    Values(
        // BUY post-only at 302 — below best ask 303, does not cross
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 302_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY post-only at 303 — equals best ask, would cross
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 303_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // BUY post-only at 310 — above best ask, would cross
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 310_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL post-only at 298 — above best bid 297, does not cross
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 298_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL post-only at 297 — equals best bid, would cross
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 297_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL post-only at 290 — below best bid, would cross
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 290_dec,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // post-only + IOC — mutually exclusive, rejected at timeInForce check
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 302_dec,
            .timeInForce = TimeInForce::IOC,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // post-only + FOK — mutually exclusive, rejected at timeInForce check
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 302_dec,
            .timeInForce = TimeInForce::FOK,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        }));

//-------------------------------------------------------------------------
// Post-Only with ghost orders
//
// Ghost levels: ghost@300 on sell side, ghost@298 on buy side.
// checkPostOnly with CO uses nonZeroLevelsView and skips ghosts correctly.
// checkPostOnly NONE/CN else-branch uses sellQueue().front() / buyQueue().back()
// directly which may include ghost levels.
//-------------------------------------------------------------------------

struct GhostPostOnlyTest : TestWithParam<PostOnlyTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    GhostSimSetup sim;
    PostOnlyTestParams params;
};

TEST_P(GhostPostOnlyTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, GhostSimSetup::testAgent, GhostSimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        true, params.timeInForce, params.stpFlag);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    GhostOrders,
    GhostPostOnlyTest,
    Values(
        // BUY post-only at 301 (CO) — ghost@300 skipped via nonZeroLevelsView,
        // no real asks cross 301 (best real ask is 303)
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 301_dec,
            .stpFlag = STPFlag::CO,
            .expectedEc = OrderErrorCode::VALID
        },
        // BUY post-only at 303 (CO) — ghost@300 skipped, real ask at 303 crosses
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 303_dec,
            .stpFlag = STPFlag::CO,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // SELL post-only at 298 (CO) — ghost@298 skipped via nonZeroLevelsView,
        // no real bids cross 298 (best real bid is 297)
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 298_dec,
            .stpFlag = STPFlag::CO,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL post-only at 297 (CO) — ghost@298 skipped, real bid at 297 crosses
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 297_dec,
            .stpFlag = STPFlag::CO,
            .expectedEc = OrderErrorCode::CONTRACT_VIOLATION
        },
        // BUY post-only at 301 (NONE) — ghost@300 present at sellQueue().front(),
        // real best ask is 303 so 301 should not cross
        PostOnlyTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 301_dec,
            .stpFlag = STPFlag::NONE,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL post-only at 298 (NONE) — ghost@298 present at buyQueue().back(),
        // real best bid is 297 so 298 should not cross
        PostOnlyTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 298_dec,
            .stpFlag = STPFlag::NONE,
            .expectedEc = OrderErrorCode::VALID
        }));

//-------------------------------------------------------------------------
// Minimum Order Size Tests
//
// Config: volumeDecimals=4 → minOrderSize = 10^(-4) = 0.0001
//-------------------------------------------------------------------------

struct MinOrderSizeTestParams
{
    OrderDirection direction{OrderDirection::BUY};
    decimal_t volume{};
    decimal_t price{};
    Currency currency{Currency::BASE};
    OrderErrorCode expectedEc{OrderErrorCode::VALID};
};

void PrintTo(const MinOrderSizeTestParams& params, std::ostream* os)
{
    *os << fmt::format(
        "{{direction = {}, volume = {}, price = {}, currency = {}, expectedEc = {}}}",
        params.direction == OrderDirection::BUY ? "BUY" : "SELL",
        params.volume,
        params.price,
        params.currency == Currency::BASE ? "BASE" : "QUOTE",
        static_cast<uint32_t>(params.expectedEc));
}

struct MinOrderSizeTest : TestWithParam<MinOrderSizeTestParams>
{
    void SetUp() override
    {
        sim.setUp();
        params = GetParam();
    }

    SimSetup sim;
    MinOrderSizeTestParams params;
};

TEST_P(MinOrderSizeTest, WorksCorrectly)
{
    const auto ec = submitTestOrder(
        sim.exchange, SimSetup::testAgent, SimSetup::bookId,
        params.direction, params.volume, params.price, 0_dec,
        false, TimeInForce::GTC, STPFlag::CO, params.currency);
    EXPECT_EQ(ec, params.expectedEc);
}

INSTANTIATE_TEST_SUITE_P(
    SpecialOrderTypes,
    MinOrderSizeTest,
    Values(
        // BASE vol=1 — well above minOrderSize 0.0001
        MinOrderSizeTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 302_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // BASE vol=0.0001 — exactly at minOrderSize boundary
        MinOrderSizeTestParams{
            .direction = OrderDirection::BUY,
            .volume = DEC(0.0001),
            .price = 302_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // SELL BASE vol=1 — above minOrderSize
        MinOrderSizeTestParams{
            .direction = OrderDirection::SELL,
            .volume = 1_dec,
            .price = 298_dec,
            .expectedEc = OrderErrorCode::VALID
        },
        // QUOTE vol=0.01 at price 302 — base equivalent rounds to 0.0000 < 0.0001
        MinOrderSizeTestParams{
            .direction = OrderDirection::BUY,
            .volume = DEC(0.01),
            .price = 302_dec,
            .currency = Currency::QUOTE,
            .expectedEc = OrderErrorCode::MINIMUM_ORDER_SIZE_VIOLATION
        },
        // QUOTE vol=1 at price 302 — base equivalent ≈ 0.0033, above minOrderSize
        MinOrderSizeTestParams{
            .direction = OrderDirection::BUY,
            .volume = 1_dec,
            .price = 302_dec,
            .currency = Currency::QUOTE,
            .expectedEc = OrderErrorCode::VALID
        }));

//-------------------------------------------------------------------------

}  // namespace

//-------------------------------------------------------------------------
