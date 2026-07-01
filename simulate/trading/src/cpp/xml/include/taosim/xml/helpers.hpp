/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <pugixml.hpp>

#include <string_view>

//-------------------------------------------------------------------------

namespace taosim::xml
{

//-------------------------------------------------------------------------

template<typename T>
requires requires(pugi::xml_attribute attr, T val) {
    { attr.set_value(val) } -> std::same_as<bool>;
}
void setAttribute(pugi::xml_node node, std::string_view name, const T& value)
{
    if (auto attr = node.attribute(name.data())) {
        attr.set_value(value);
    } else {
        node.append_attribute(name.data()) = value;
    }
}

//-------------------------------------------------------------------------

[[nodiscard]] inline pugi::xml_node findChildByName(pugi::xml_node node, std::string_view needle)
{
    return node.find_child([needle](pugi::xml_node child) { return needle == child.name(); });
}

//-------------------------------------------------------------------------

inline size_t removeChildren(pugi::xml_node node, std::function<bool(pugi::xml_node)> criterion)
{
    size_t removeCounter{};
    auto child = node.first_child();
    while (child) {
        const auto nextChild = child.next_sibling();
        if (criterion(child)) {
            node.remove_child(child);
            ++removeCounter;
        }
        child = nextChild;
    }
    return removeCounter;
}

//-------------------------------------------------------------------------

}  // namespace taosim::xml

//-------------------------------------------------------------------------