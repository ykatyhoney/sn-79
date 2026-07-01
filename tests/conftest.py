# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Pytest fixtures for the public-surface unit tests.

These tests exercise pure logic (blacklist policy, protocol integrity, reward
maths) without standing up a validator/miner or the C++ engine, mirroring the
stub-the-heavy-bits pattern used in GenTRX/tests.
"""
