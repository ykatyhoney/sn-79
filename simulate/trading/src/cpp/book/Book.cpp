/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/book/Book.hpp>

#include <taosim/simulation/SimulationException.hpp>
#include "Simulation.hpp"

#include <range/v3/action/remove_if.hpp>

//-------------------------------------------------------------------------

namespace taosim::book
{

//-------------------------------------------------------------------------

Book::Book(
    Simulation* simulation,
    BookId id,
    size_t maxDepth,
    size_t detailedDepth,
    std::shared_ptr<OrderID> orderIdCounter,
    std::shared_ptr<TradeID> tradeIdCounter)
    : m_simulation{simulation},
      m_id{id},
      m_orderIdCounter{orderIdCounter ? orderIdCounter : std::make_shared<OrderID>()},
      m_tradeIdCounter{tradeIdCounter ? tradeIdCounter : std::make_shared<TradeID>()}
{
    if (maxDepth == 0) {
        throw std::invalid_argument("Book maximum depth must be non-zero");
    }
    m_maxDepth = maxDepth;
    m_detailedDepth = std::min(detailedDepth, maxDepth);

    simulation->logDebug(
        orderIdCounter
            ? fmt::format("Using shared orderIdCounter ({}) for book {}", *m_orderIdCounter, simulation->bookIdCanon(id))
            : fmt::format("Creating unique orderIdCounter ({}) for book {}", *m_orderIdCounter, simulation->bookIdCanon(id)));
    simulation->logDebug(
        tradeIdCounter
            ? fmt::format("Using shared tradeIdCounter ({}) for book {}", *m_tradeIdCounter, simulation->bookIdCanon(id))
            : fmt::format("Creating unique tradeIdCounter ({}) for book {}", *m_tradeIdCounter, simulation->bookIdCanon(id)));

    setupL2Signal();
}

//-------------------------------------------------------------------------

taosim::decimal_t Book::midPrice() const noexcept
{
    const auto bid = bestBid();
    const auto ask = bestAsk();

    if (bid == 0_dec || ask == 0_dec) [[unlikely]] {
        return {};
    }
    return DEC(0.5) * (bid + ask);
}

//-------------------------------------------------------------------------

taosim::decimal_t Book::bestBid() const noexcept
{
    return bestBuyLevel()
        .transform([](auto&& level) { return level.get().price(); })
        .value_or(0_dec);
}

//-------------------------------------------------------------------------

taosim::decimal_t Book::bestAsk() const noexcept
{
    return bestSellLevel()
        .transform([](auto&& level) { return level.get().price(); })
        .value_or(0_dec);
}

//-------------------------------------------------------------------------

void Book::refreshTopOfBook() const noexcept
{
    auto& buyQueue = const_cast<OrderContainer&>(m_buyQueue);
    auto& sellQueue = const_cast<OrderContainer&>(m_sellQueue);

    auto buyView = buyQueue | ranges::views::reverse;
    auto buyIt = ranges::find_if(buyView, [](auto&& level) { return level.hasActiveOrders(); });
    m_topOfBook.bestBuyLevel = (buyIt != ranges::end(buyView))
        ? std::make_optional(std::ref(*buyIt)) : std::nullopt;

    auto sellIt = ranges::find_if(sellQueue, [](auto&& level) { return level.hasActiveOrders(); });
    m_topOfBook.bestSellLevel = (sellIt != ranges::end(sellQueue))
        ? std::make_optional(std::ref(*sellIt)) : std::nullopt;

    m_topOfBook.isDirty = false;
}

//-------------------------------------------------------------------------

Book::OptLevelRef Book::bestBuyLevel() const noexcept
{
    if (m_topOfBook.isDirty) {
        refreshTopOfBook();
    }
    return m_topOfBook.bestBuyLevel;
}

//-------------------------------------------------------------------------

Book::OptLevelRef Book::bestSellLevel() const noexcept
{
    if (m_topOfBook.isDirty) {
        refreshTopOfBook();
    }
    return m_topOfBook.bestSellLevel;
}

//-------------------------------------------------------------------------

void Book::placeOrder(const MarketOrder::Ptr& order)
{
    using namespace taosim::util;

    const auto clientCtx = m_order2clientCtx.at(order->id());
    const auto slip = order->maxSlippage();
    const auto priceDecimals =
        m_simulation->exchange()->config().parameters().priceIncrementDecimals;

    // The processing step is conditional on the opposite side having any
    // active orders (the match loop dereferences bestBuyLevel()/bestSellLevel()
    // and would otherwise throw on the empty `std::optional`). We must still
    // emit `marketOrderProcessed` at the end even when nothing traded, so
    // marketOrderProcessedCallback can release the order's untouched
    // reservation and drop it from the agent's activeOrders — otherwise the
    // order permanently lingers there with a live reservation.
    if (order->direction() == OrderDirection::BUY) {
        if (m_sellQueue.hasActiveOrders()) {
            const auto maxPrice = (slip > 0_dec)
                ? round(bestAsk() * dec1p(slip), priceDecimals)
                : std::numeric_limits<taosim::decimal_t>::max();
            processAgainstTheSellQueue(order, maxPrice);
        }
    } else {
        if (m_buyQueue.hasActiveOrders()) {
            const auto minPrice = (slip > 0_dec)
                ? round(bestBid() * dec1m(slip), priceDecimals)
                : std::numeric_limits<taosim::decimal_t>::min();
            processAgainstTheBuyQueue(order, minPrice);
        }
    }

    m_signals.marketOrderProcessed(
        order, OrderContext{clientCtx.agentId, m_id, clientCtx.clientOrderId});
}

//-------------------------------------------------------------------------

void Book::placeOrder(const LimitOrder::Ptr& order)
{
    const auto clientCtx = m_order2clientCtx.at(order->id());

    if (order->direction() == OrderDirection::BUY) {
        placeLimitBuy(order);
    } else {
        placeLimitSell(order);
    }

    m_signals.limitOrderProcessed(
        order, OrderContext{clientCtx.agentId, m_id, clientCtx.clientOrderId});
}

//-------------------------------------------------------------------------

void Book::placeLimitBuy(const LimitOrder::Ptr& order)
{
    if (!m_sellQueue.hasActiveOrders() || order->price() < bestAsk()) {
        auto firstLessThan = std::find_if(
            m_buyQueue.rbegin(),
            m_buyQueue.rend(),
            [&order](const auto& level) { return level.price() <= order->price(); });

        if (firstLessThan != m_buyQueue.rend() && firstLessThan->price() == order->price()) {
            registerLimitOrder(order);
            firstLessThan->push_back(order);
        }
        else {
            taosim::book::TickContainer tov{&m_buyQueue, order->price()};
            registerLimitOrder(order);
            tov.push_back(order);
            m_buyQueue.insert(firstLessThan.base(), std::move(tov));
        }
        m_topOfBook.invalidate();
    }
    else {
        const auto volBefore = order->volume();
        processAgainstTheSellQueue(order, order->price());

        if (order->volume() > 0_dec) {
            if (order->volume() == volBefore) {
                // No progress was made — place as passive to avoid infinite recursion
                auto firstLessThan = std::find_if(
                    m_buyQueue.rbegin(),
                    m_buyQueue.rend(),
                    [&order](const auto& level) { return level.price() <= order->price(); });

                if (firstLessThan != m_buyQueue.rend() && firstLessThan->price() == order->price()) {
                    registerLimitOrder(order);
                    firstLessThan->push_back(order);
                }
                else {
                    taosim::book::TickContainer tov{&m_buyQueue, order->price()};
                    registerLimitOrder(order);
                    tov.push_back(order);
                    m_buyQueue.insert(firstLessThan.base(), std::move(tov));
                }
                m_topOfBook.invalidate();
            } else {
                placeOrder(order);
            }
        } else {
            unregisterLimitOrder(order);
        }
    }
}

//-------------------------------------------------------------------------

void Book::placeLimitSell(const LimitOrder::Ptr& order)
{
    if (!m_buyQueue.hasActiveOrders() || order->price() > bestBid()) {
        auto firstGreaterThan = std::find_if(
            m_sellQueue.begin(),
            m_sellQueue.end(),
            [&order](const auto& level) { return level.price() >= order->price(); });

        if (firstGreaterThan != m_sellQueue.end() && firstGreaterThan->price() == order->price()) {
            registerLimitOrder(order);
            firstGreaterThan->push_back(order);
        }
        else {
            taosim::book::TickContainer tov{&m_sellQueue, order->price()};
            registerLimitOrder(order);
            tov.push_back(order);
            m_sellQueue.insert(firstGreaterThan, std::move(tov));
        }
        m_topOfBook.invalidate();
    }
    else {
        const auto volBefore = order->volume();
        processAgainstTheBuyQueue(order, order->price());

        if (order->volume() > 0_dec) {
            if (order->volume() == volBefore) {
                // No progress was made — place as passive to avoid infinite recursion
                auto firstGreaterThan = std::find_if(
                    m_sellQueue.begin(),
                    m_sellQueue.end(),
                    [&order](const auto& level) { return level.price() >= order->price(); });

                if (firstGreaterThan != m_sellQueue.end() && firstGreaterThan->price() == order->price()) {
                    registerLimitOrder(order);
                    firstGreaterThan->push_back(order);
                }
                else {
                    taosim::book::TickContainer tov{&m_sellQueue, order->price()};
                    registerLimitOrder(order);
                    tov.push_back(order);
                    m_sellQueue.insert(firstGreaterThan, std::move(tov));
                }
                m_topOfBook.invalidate();
            } else {
                placeOrder(order);
            }
        } else {
            unregisterLimitOrder(order);
        }
    }
}

//-------------------------------------------------------------------------

bool Book::cancelOrder(OrderID orderId, std::optional<taosim::decimal_t> volumeToCancel)
{
    auto it = m_orderIdMap.find(orderId);
    if (it == m_orderIdMap.end()) { return false; }

    const auto maxDecimals = std::max({
        m_simulation->exchange()->config().parameters().volumeIncrementDecimals,
        m_simulation->exchange()->config().parameters().priceIncrementDecimals,
        m_simulation->exchange()->config().parameters().quoteIncrementDecimals,
        m_simulation->exchange()->config().parameters().baseIncrementDecimals
    });

    auto order = it->second;

    const taosim::decimal_t orderVolume = order->volume();
    if (orderVolume == 0_dec) { return false; }

    taosim::decimal_t volumeToCancelActual =
        std::min(volumeToCancel.value_or(orderVolume), orderVolume);

    volumeToCancelActual = taosim::util::round(volumeToCancelActual, maxDecimals);

    if (m_simulation->debug()) {
        const auto ctx = m_order2clientCtx[order->id()];
        const auto& balances = m_simulation->exchange()->accounts()[ctx.agentId][m_id];
        m_simulation->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", 
            m_simulation->currentTimestamp(), ctx.agentId, m_id, *balances.quote, balances.base);
    }
    m_signals.cancelOrderDetails(order, volumeToCancelActual, m_id);

    auto& orderSideLevels = order->direction() == OrderDirection::BUY ? m_buyQueue : m_sellQueue;
    auto levelIt = std::lower_bound(orderSideLevels.begin(), orderSideLevels.end(), order->price());

    if (volumeToCancelActual == orderVolume) {
        std::erase_if(
            *levelIt, [orderId](const auto orderOnLevel) { return orderOnLevel->id() == orderId; });
        levelIt->updateVolume(-volumeToCancelActual);
        if (levelIt->empty()) {
            orderSideLevels.erase(levelIt);
        }
        unregisterLimitOrder(order);
    }
    else {
        order->removeVolume(volumeToCancelActual);
        levelIt->updateVolume(-volumeToCancelActual);
    }

    m_signals.cancel(order->id(), volumeToCancelActual);
    m_topOfBook.invalidate();

    return true;
}

//-------------------------------------------------------------------------

bool Book::restoreRestingOrderVolume(OrderID orderId, taosim::decimal_t deltaVolume)
{
    if (deltaVolume <= 0_dec) { return false; }

    auto it = m_orderIdMap.find(orderId);
    if (it == m_orderIdMap.end()) { return false; }

    const auto& order = it->second;
    auto& orderSideLevels = order->direction() == OrderDirection::BUY ? m_buyQueue : m_sellQueue;
    auto levelIt = std::lower_bound(orderSideLevels.begin(), orderSideLevels.end(), order->price());
    if (levelIt == orderSideLevels.end() || levelIt->price() != order->price()) { return false; }

    order->setVolume(order->volume() + deltaVolume);
    levelIt->updateVolume(deltaVolume);
    m_topOfBook.invalidate();

    return true;
}

//-------------------------------------------------------------------------

std::optional<LimitOrder::Ptr> Book::getOrder(OrderID orderId) const
{
    auto it = m_orderIdMap.find(orderId);
    return it != m_orderIdMap.end() ? std::make_optional(it->second) : std::nullopt;
}

//-------------------------------------------------------------------------

void Book::registerLimitOrder(const LimitOrder::Ptr& order)
{
    m_orderIdMap[order->id()] = order;
    if (m_simulation->debug()) {
        const auto ctx = m_order2clientCtx[order->id()];
        auto& balances = m_simulation->exchange()->accounts().at(ctx.agentId).at(m_id);
        fmt::println("{} | AGENT #{} BOOK {} : REGISTERED {} ORDER #{} FOR {}@{}| RESERVED {} QUOTE + {} BASE | BALANCES : QUOTE {}  BASE {}", m_simulation->currentTimestamp(), 
            ctx.agentId, m_simulation->bookIdCanon(m_id), order->direction() == OrderDirection::BUY ? "BUY" : "SELL",
            order->id(), order->leverage() > 0_dec ? fmt::format("{}x{}",1_dec + order->leverage(),order->volume()) : fmt::format("{}",order->volume()), order->price(),
            balances.quote->getReservation(order->id()).value_or(0_dec), balances.base.getReservation(order->id()).value_or(0_dec),
            *balances.quote, balances.base);
    } 
}

//-------------------------------------------------------------------------

void Book::unregisterLimitOrder(const LimitOrder::Ptr& order)
{
    m_signals.unregister(order, m_id);
    m_orderIdMap.erase(order->id());
    m_order2clientCtx.erase(order->id());
}

//-------------------------------------------------------------------------

taosim::decimal_t Book::processAgainstTheBuyQueue(const Order::Ptr& order, taosim::decimal_t minPrice)
{
    using namespace taosim::util;

    taosim::decimal_t processedQuote = {};
    const auto volumeDecimals = m_simulation->exchange()->config().parameters().volumeIncrementDecimals;
    const auto priceDecimals = m_simulation->exchange()->config().parameters().priceIncrementDecimals;
    const auto maxDecimals = std::max({
        volumeDecimals,
        priceDecimals,
        m_simulation->exchange()->config().parameters().quoteIncrementDecimals,
        m_simulation->exchange()->config().parameters().baseIncrementDecimals
    });
    const auto agentId = m_order2clientCtx[order->id()].agentId;

    auto bestBuyDeque = &bestBuyLevel().value().get();

    order->setVolume(taosim::util::round(order->volume(), volumeDecimals));
    order->setLeverage(taosim::util::round(order->leverage(), volumeDecimals));

    while (order->volume() > 0_dec && bestBuyDeque->price() >= minPrice) {
        LimitOrder::Ptr iop = bestBuyDeque->getFirstActiveOrder();
        if (!iop) {
            // All orders on this level are ghosts, advance to next level
            auto optLevel = bestBuyLevel();
            if (!optLevel || &optLevel->get() == bestBuyDeque) {
                break;
            }
            bestBuyDeque = &optLevel->get();
            continue;
        }
        const auto iopAgentId = m_order2clientCtx[iop->id()].agentId;
        if (agentId == iopAgentId && order->stpFlag() != STPFlag::NONE){
            bestBuyDeque = preventSelfTrade(bestBuyDeque, iop, order, agentId);
            if (bestBuyDeque == nullptr)
                break;
            continue;
        }

        const taosim::decimal_t usedVolume = std::min(iop->totalVolume(), order->totalVolume());

        OrderClientContext aggCtx, restCtx;
        if (m_simulation->debug()) {
            aggCtx = m_order2clientCtx[order->id()];
            const auto& aggBalances = m_simulation->exchange()->accounts()[aggCtx.agentId][m_id];
            restCtx = m_order2clientCtx[iop->id()];
            const auto& restingBalances = m_simulation->exchange()->accounts()[restCtx.agentId][m_id];
            m_simulation->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", m_simulation->currentTimestamp(), restCtx.agentId, m_id, *restingBalances.quote, restingBalances.base);
        }

        if (usedVolume > 0_dec) {
            processedQuote += usedVolume * bestBuyDeque->price();
            logTrade(
                m_simulation->currentTimestamp(),
                OrderDirection::SELL,
                order->id(),
                iop->id(),
                usedVolume,
                bestBuyDeque->price());
        }

        order->removeLeveragedVolume(usedVolume);
        iop->removeLeveragedVolume(usedVolume);

        order->setVolume(taosim::util::round(order->volume(), maxDecimals));
        iop->setVolume(taosim::util::round(iop->volume(), maxDecimals));

        auto& accounts = m_simulation->exchange()->accounts();
        const auto& aggBalances = accounts[agentId][m_id];
        if ((order->volume() > 0_dec && taosim::util::round(order->volume(), volumeDecimals) == 0_dec) ||
            aggBalances.getReservationInBase(order->id(), 1_dec) == 0_dec){
            order->setVolume(0_dec);
        }

        const auto& restingBalances = accounts[iopAgentId][m_id];
        if ((iop->volume() > 0_dec && taosim::util::round(iop->volume(), volumeDecimals) == 0_dec) ||
            restingBalances.getReservationInBase(iop->id(), 1_dec) == 0_dec){
            bestBuyDeque->updateVolume(-taosim::util::round(iop->totalVolume(), maxDecimals));
            iop->setVolume(0_dec);
        }

        bestBuyDeque->updateVolume(-taosim::util::round(usedVolume, maxDecimals));
        m_topOfBook.invalidate();

        // Ghost order: leave fully filled resting orders in place for reconciliation
        if (taosim::util::round(iop->totalVolume(), volumeDecimals) == 0_dec) {
            m_simulation->logDebug("BOOK {} : ORDER #{} FULLY FILLED (GHOST)", m_id, iop->id());
        }

        if (m_simulation->debug()) {
            const auto& restingBalances = m_simulation->exchange()->accounts()[restCtx.agentId][m_id];
            m_simulation->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", m_simulation->currentTimestamp(), aggCtx.agentId, m_id, *aggBalances.quote, aggBalances.base);
            m_simulation->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", m_simulation->currentTimestamp(), restCtx.agentId, m_id, *restingBalances.quote, restingBalances.base);
        }

        if (!bestBuyDeque->hasActiveOrders()) {
            auto optLevel = bestBuyLevel();
            if (!optLevel) {
                break;
            }
            bestBuyDeque = &optLevel->get();
        }
    }

    return processedQuote;
}

//-------------------------------------------------------------------------

taosim::decimal_t Book::processAgainstTheSellQueue(const Order::Ptr& order, taosim::decimal_t maxPrice)
{
    using namespace taosim::util;

    taosim::decimal_t processedQuote = {};
    const auto volumeDecimals = m_simulation->exchange()->config().parameters().volumeIncrementDecimals;
    const auto priceDecimals = m_simulation->exchange()->config().parameters().priceIncrementDecimals;
    const auto maxDecimals = std::max({
        volumeDecimals,
        priceDecimals,
        m_simulation->exchange()->config().parameters().quoteIncrementDecimals,
        m_simulation->exchange()->config().parameters().baseIncrementDecimals
    });
    const auto agentId = m_order2clientCtx[order->id()].agentId;

    auto bestSellDeque = &bestSellLevel().value().get();

    order->setVolume(taosim::util::round(order->volume(), volumeDecimals));
    order->setLeverage(taosim::util::round(order->leverage(), volumeDecimals));

    while (order->volume() > 0_dec && bestSellDeque->price() <= maxPrice) {
        LimitOrder::Ptr iop = bestSellDeque->getFirstActiveOrder();
        if (!iop) {
            // All orders on this level are ghosts, advance to next level
            auto optLevel = bestSellLevel();
            if (!optLevel || &optLevel->get() == bestSellDeque) {
                break;
            }
            bestSellDeque = &optLevel->get();
            continue;
        }
        const auto iopAgentId = m_order2clientCtx[iop->id()].agentId;
        if (agentId == iopAgentId && order->stpFlag() != STPFlag::NONE){
            bestSellDeque = preventSelfTrade(bestSellDeque, iop, order, agentId);
            if (bestSellDeque == nullptr)
                break;
            continue;
        }

        const taosim::decimal_t usedVolume = std::min(iop->totalVolume(), order->totalVolume());

        OrderClientContext aggCtx, restCtx;
        if (m_simulation->debug()) {
            aggCtx = m_order2clientCtx[order->id()];
            const auto& aggBalances = m_simulation->exchange()->accounts()[aggCtx.agentId][m_id];
            restCtx = m_order2clientCtx[iop->id()];
            const auto& restingBalances = m_simulation->exchange()->accounts()[restCtx.agentId][m_id];
            m_simulation->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}",
                m_simulation->currentTimestamp(), restCtx.agentId, m_id, *restingBalances.quote, restingBalances.base);
        }

        if (usedVolume > 0_dec) {
            processedQuote += usedVolume * bestSellDeque->price();
            logTrade(
                m_simulation->currentTimestamp(),
                OrderDirection::BUY,
                order->id(),
                iop->id(),
                usedVolume,
                bestSellDeque->price());
        }

        order->removeLeveragedVolume(usedVolume);
        iop->removeLeveragedVolume(usedVolume);

        order->setVolume(taosim::util::round(order->volume(), maxDecimals));
        iop->setVolume(taosim::util::round(iop->volume(), maxDecimals));

        auto& accounts = m_simulation->exchange()->accounts();
        const auto& aggBalances = accounts[agentId][m_id];
        if ((order->volume() > 0_dec && taosim::util::round(order->volume(), volumeDecimals) == 0_dec) ||
            aggBalances.getReservationInBase(order->id(), 1_dec) == 0_dec){
            order->setVolume(0_dec);
        }

        const auto& restingBalances = accounts[iopAgentId][m_id];
        if ((iop->volume() > 0_dec && taosim::util::round(iop->volume(), volumeDecimals) == 0_dec) ||
            restingBalances.getReservationInBase(iop->id(), 1_dec) == 0_dec){
            bestSellDeque->updateVolume(-taosim::util::round(iop->totalVolume(), maxDecimals));
            iop->setVolume(0_dec);
        }

        bestSellDeque->updateVolume(-taosim::util::round(usedVolume, maxDecimals));
        m_topOfBook.invalidate();

        // Ghost order: leave fully filled resting orders in place for reconciliation
        if (taosim::util::round(iop->totalVolume(), volumeDecimals) == 0_dec) {
            m_simulation->logDebug("BOOK {} : ORDER #{} FULLY FILLED (GHOST)", m_id, iop->id());
        }

        if (m_simulation->debug()) {
            m_simulation->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", m_simulation->currentTimestamp(), aggCtx.agentId, m_id, *aggBalances.quote, aggBalances.base);
            m_simulation->logDebug("{} | AGENT #{} BOOK {} : QUOTE : {}  BASE : {}", m_simulation->currentTimestamp(), restCtx.agentId, m_id, *restingBalances.quote, restingBalances.base);
        }

        if (!bestSellDeque->hasActiveOrders()) {
            auto optLevel = bestSellLevel();
            if (!optLevel) {
                break;
            }
            bestSellDeque = &optLevel->get();
        }
    }

    return processedQuote;
}

//-------------------------------------------------------------------------

taosim::book::TickContainer* Book::preventSelfTrade(
    taosim::book::TickContainer* queue, const LimitOrder::Ptr& iop, const Order::Ptr& order, AgentId agentId)
{
    auto stpFlag = order->stpFlag();
    auto now = m_simulation->currentTimestamp();

    auto cancelAndLog = [&](OrderID orderId, std::optional<taosim::decimal_t> volume = {}) {
        if (cancelOrder(orderId, volume)) {
            taosim::event::Cancellation cancellation{orderId, volume};
            m_simulation->exchange()->signals().at(m_id)->cancelLog(CancellationWithLogContext(
                cancellation,
                std::make_shared<CancellationLogContext>(
                    agentId,
                    m_id,
                    now)));
            m_simulation->logDebug("{} | AGENT #{} BOOK {} : SELF TRADE PREVENTION CANCELED {}ORDER {}", 
                now, agentId, m_id, 
                volume.has_value() ? fmt::format("{} volume of ", volume.value()) : "",
                orderId);
            return true;
        } else {
            m_simulation->logDebug("{} | AGENT #{} BOOK {} : SELF TRADE PREVENTION OF ORDER {} FAILED", now, agentId, m_id, orderId);
            return false;
        }
    };

    if (stpFlag == STPFlag::CN || stpFlag == STPFlag::CB) {
        order->removeVolume(order->volume());
        m_simulation->logDebug("{} | AGENT #{} BOOK {} : SELF TRADE PREVENTION CANCELED ORDER {}", now, agentId, m_id, order->id());
        if (stpFlag == STPFlag::CN) {
            return nullptr;
        }
    }

    // After cancelOrder, the queue pointer may be dangling (deque erase invalidates),
    // so always re-derive from bestBuyLevel/bestSellLevel.
    auto freshQueue = [this](bool isBuy) -> TickContainer* {
        auto optLevel = isBuy ? bestBuyLevel() : bestSellLevel();
        return optLevel ? &optLevel->get() : nullptr;
    };

    if (stpFlag == STPFlag::CO || stpFlag == STPFlag::CB) {
        const bool isBuy = iop->direction() == OrderDirection::BUY;
        cancelAndLog(iop->id());
        queue = freshQueue(isBuy);
        if (stpFlag == STPFlag::CB) {
            return nullptr;
        }
        return queue;
    }

    if (stpFlag == STPFlag::DC) {
        if (iop->totalVolume() == order->totalVolume()){
            order->removeVolume(order->volume());
            m_simulation->logDebug("{} | AGENT #{} BOOK {} : SELF TRADE PREVENTION CANCELED ORDER {}", now, agentId, m_id, order->id());
            cancelAndLog(iop->id());
            return nullptr;
        } else if (iop->totalVolume() < order->totalVolume()){
            auto volumeToCancel = taosim::util::round(iop->totalVolume() / taosim::util::dec1p(order->leverage()),
                m_simulation->exchange()->config().parameters().volumeIncrementDecimals);
            const bool isBuy = iop->direction() == OrderDirection::BUY;
            if (cancelAndLog(iop->id())){
                queue = freshQueue(isBuy);
                if (!queue) return nullptr;
                order->removeVolume(volumeToCancel);
                return queue;
            }
        } else {
            auto volumeToCancel = taosim::util::round(order->totalVolume() / taosim::util::dec1p(iop->leverage()),
                m_simulation->exchange()->config().parameters().volumeIncrementDecimals);
            order->removeVolume(order->volume());
            m_simulation->logDebug("{} | AGENT #{} BOOK {} : SELF TRADE PREVENTION CANCELED ORDER {}", now, agentId, m_id, order->id());
            cancelAndLog(iop->id(), volumeToCancel);
            return nullptr;
        }
    }

    return queue;
}

//-------------------------------------------------------------------------

void Book::clearFilledOrders() noexcept
{
    auto impl = [this](auto&& side) {
        for (auto it = side.begin(); it != side.end(); ) {
            // Remove ghost orders (zero volume) from this level
            for (auto orderIt = it->begin(); orderIt != it->end(); ) {
                if ((*orderIt)->totalVolume() == 0_dec) {
                    unregisterLimitOrder(*orderIt);
                    orderIt = it->erase(orderIt);
                } else {
                    ++orderIt;
                }
            }
            // Remove empty levels
            if (it->empty()) {
                it = side.erase(it);
            } else {
                ++it;
            }
        }
    };

    impl(m_buyQueue);
    impl(m_sellQueue);
    m_topOfBook.invalidate();
}

//-------------------------------------------------------------------------

void Book::printCSV(uint32_t depth) const
{
    std::cout << "ask";
    dumpCSVLOB(m_sellQueue.cbegin(), m_sellQueue.cend(), depth);
    std::cout << "\n";

    std::cout << "bid";
    dumpCSVLOB(m_buyQueue.crbegin(), m_buyQueue.crend(), depth);
    std::cout << "\n";
}

//-------------------------------------------------------------------------

void Book::jsonSerialize(rapidjson::Document& json, const std::string& key) const
{
    auto serialize = [this](rapidjson::Document& json) {
        json.SetObject();
        auto& allocator = json.GetAllocator();

        auto serializeLevelBroad = [](rapidjson::Document& json, const taosim::book::TickContainer& level) {
            json.SetObject();
            auto& allocator = json.GetAllocator();
            json.AddMember(
                "price", rapidjson::Value{taosim::util::decimal2double(level.price())}, allocator);
            json.AddMember(
                "volume", rapidjson::Value{taosim::util::decimal2double(level.volume())}, allocator);
        };

        rapidjson::Value bidsJson{rapidjson::kArrayType};
        for (const auto& level : m_buyQueue | views::reverse | views::take(m_detailedDepth)) {
            rapidjson::Document levelJson{&allocator};
            level.jsonSerialize(levelJson);
            bidsJson.PushBack(levelJson, allocator);
        }
        for (const auto& level : m_buyQueue | views::reverse | views::drop(m_detailedDepth)) {
            rapidjson::Document levelJson{&allocator};
            serializeLevelBroad(levelJson, level);
            bidsJson.PushBack(levelJson, allocator);
        }
        json.AddMember(
            "bid", bidsJson.Size() > 0 ? bidsJson : rapidjson::Value{}.SetNull(), allocator);

        rapidjson::Value asksJson{rapidjson::kArrayType};
        for (const auto& level : m_sellQueue | views::take(m_detailedDepth)) {
            rapidjson::Document levelJson{&allocator};
            level.jsonSerialize(levelJson);
            asksJson.PushBack(levelJson, allocator);
        }
        for (const auto& level : m_sellQueue | views::drop(m_detailedDepth)) {
            rapidjson::Document levelJson{&allocator};
            serializeLevelBroad(levelJson, level);
            asksJson.PushBack(levelJson, allocator);
        }
        json.AddMember(
            "ask", asksJson.Size() > 0 ? asksJson : rapidjson::Value{}.SetNull(), allocator);
    };
    taosim::json::serializeHelper(json, key, serialize);
}

//-------------------------------------------------------------------------

void Book::setupL2Signal()
{
    m_signals.limitOrderProcessed.connect([this](auto&&... args) {
        if (!m_initMode) {
            emitL2Signal(std::forward<decltype(args)>(args)...);
        }
    });
    m_signals.marketOrderProcessed.connect([this](auto&&... args) {
        emitL2Signal(std::forward<decltype(args)>(args)...);
    });
    m_signals.trade.connect([this](auto&&... args) {
        emitL2Signal(std::forward<decltype(args)>(args)...);
    });
    m_signals.cancel.connect([this](auto&&... args) {
        emitL2Signal(std::forward<decltype(args)>(args)...);
    });
}

//-------------------------------------------------------------------------

}  // namespace taosim::book

//-------------------------------------------------------------------------
