# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
CPU core allocation: distributes logical cores across validator sub-processes
(validator, query, reward, reporting, IPC, and optional co-located servers —
SL/TP service and/or GenTRX gradient server).
"""
import multiprocessing

def get_core_allocation(sltp_cores_count: int = 0, grad_server_cores: int = 0):
    """
    Allocate CPU cores across validator components using percentage-based allocation.

    Reserves the last `sltp_cores_count` logical cores for the SL/TP service,
    then the next-to-last `grad_server_cores` for the GenTRX gradient server, and
    distributes the remainder across the validator, query, reward, reporting, and IPC
    sub-processes.

    Args:
        sltp_cores_count: Cores to reserve for the SL/TP service (last N). 0 = no reservation.
        grad_server_cores: Cores to reserve for the GenTRX gradient server. 0 = no reservation.

    Returns:
        dict: Mapping of component name to list of core indices. Keys include 'validator',
            'query', 'reward', 'reporting', 'ipc', and optionally 'sltp' and 'gradient_server'.

    Raises:
        Exception: If fewer than 8 cores remain after all reservations.
    """
    total_cores = multiprocessing.cpu_count()
    available_cores = total_cores - sltp_cores_count - grad_server_cores

    sltp_cores = (
        list(range(total_cores - sltp_cores_count, total_cores))
        if sltp_cores_count > 0 else []
    )
    grad_cores = (
        list(range(available_cores, available_cores + grad_server_cores))
        if grad_server_cores > 0 else []
    )

    if available_cores < 8:
        raise Exception(
            f"Validator requires a minimum of 8 cores to run! "
            f"(total={total_cores}, sltp_cores_count={sltp_cores_count}, "
            f"grad_server_cores={grad_server_cores}, available={available_cores})"
        )

    if available_cores == 8:
        result = {
            'validator': [0, 1],
            'query': [2, 3],
            'reward': [4, 5],
            'reporting': [6],
            'ipc': [7],
        }
        if sltp_cores:
            result['sltp'] = sltp_cores
        if grad_cores:
            result['gradient_server'] = grad_cores
        return result

    validator_pct = 0.20
    query_pct = 0.20
    reward_pct = 0.25
    reporting_pct = 0.1
    ipc_pct = 0.15

    validator_count = max(2, int(available_cores * validator_pct))
    query_count = max(2, int(available_cores * query_pct))
    reward_count = max(2, int(available_cores * reward_pct))
    reporting_count = max(1, int(available_cores * reporting_pct))
    ipc_count = max(2, int(available_cores * ipc_pct))

    allocated = validator_count + query_count + reward_count + reporting_count + ipc_count
    if allocated > available_cores:
        scale = available_cores / allocated
        validator_count = max(2, int(validator_count * scale))
        query_count = max(2, int(query_count * scale))
        reward_count = max(2, int(reward_count * scale))
        reporting_count = max(1, int(reporting_count * scale))
        ipc_count = max(2, int(ipc_count * scale))

    offset = 0

    validator_cores = list(range(offset, offset + validator_count))
    offset += validator_count

    query_cores = list(range(offset, offset + query_count))
    offset += query_count

    reward_cores = list(range(offset, offset + reward_count))
    offset += reward_count

    reporting_cores = list(range(offset, min(available_cores, offset + reporting_count)))
    offset = min(available_cores, offset + reporting_count)

    ipc_cores = list(range(offset, min(available_cores, offset + ipc_count)))

    result = {
        'validator': validator_cores,
        'query': query_cores,
        'reward': reward_cores,
        'reporting': reporting_cores,
        'ipc': ipc_cores,
    }
    if sltp_cores:
        result['sltp'] = sltp_cores
    if grad_cores:
        result['gradient_server'] = grad_cores
    return result
