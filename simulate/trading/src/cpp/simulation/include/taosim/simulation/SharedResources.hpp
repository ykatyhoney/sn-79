/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <Eigen/Dense>

//-------------------------------------------------------------------------

namespace taosim::simulation
{

//-------------------------------------------------------------------------

struct SharedResources
{
    Eigen::MatrixXd fundamentalPriceL;
};

//-------------------------------------------------------------------------

}  // namespace taosim::simulation

//-------------------------------------------------------------------------
