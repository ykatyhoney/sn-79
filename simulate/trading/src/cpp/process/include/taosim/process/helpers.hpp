/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <Eigen/Dense>
#include <pugixml.hpp>

//-------------------------------------------------------------------------

namespace taosim::simulation
{
struct SharedResources;
}

namespace taosim::process::helpers
{

//-------------------------------------------------------------------------

void precomputeFundamentalPriceL(Eigen::MatrixXd& L, double hurst);

void initSharedResources(
    taosim::simulation::SharedResources& shared, pugi::xml_node simuNode);

//-------------------------------------------------------------------------

}  // namespace taosim::process::helpers

//-------------------------------------------------------------------------
