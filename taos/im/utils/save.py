# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Validator and simulation state persistence: atomic msgpack serialisation to
disk, designed for use in a ProcessPoolExecutor worker.
"""
import os
from typing import Dict
import msgpack


def _fsync_dir(path):
    """fsync the directory entry so a preceding os.replace is durable.

    Without this, a power loss right after the rename can leave the directory
    pointing at the OLD inode on remount (the temp's bytes are already fsynced,
    but the rename itself isn't). Best-effort: filesystems without O_DIRECTORY
    support raise OSError and we silently fall through — at worst we lose the
    same durability guarantee that's missing without the fix.
    """
    try:
        dfd = os.open(path or ".", os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def _pack_stream_fsync(path, obj):
    """Stream-pack `obj` straight into `path` and fsync. Returns bytes written.

    Byte-identical to msgpack.packb(obj, use_bin_type=True): a msgpack map is its
    header followed by the packed key/value pairs in iteration order. Streaming
    avoids materialising the full multi-hundred-MB intermediate buffer and
    overlaps serialization with the file write. fsync BEFORE the atomic
    os.replace guarantees the temp file's bytes are durable before it takes the
    state file's place — without it a power loss right after the rename could
    leave a truncated state with the previous good file already gone. Pair this
    with _fsync_dir on the parent after os.replace to make the rename itself
    durable.
    """
    packer = msgpack.Packer(use_bin_type=True)
    total = 0
    with open(path, 'wb', buffering=1024 * 1024) as f:
        if isinstance(obj, dict):
            chunk = packer.pack_map_header(len(obj))
            f.write(chunk)
            total += len(chunk)
            for k, v in obj.items():
                chunk = packer.pack(k)
                f.write(chunk)
                total += len(chunk)
                chunk = packer.pack(v)
                f.write(chunk)
                total += len(chunk)
        else:
            chunk = packer.pack(obj)
            f.write(chunk)
            total += len(chunk)
        f.flush()
        os.fsync(f.fileno())
    return total


def save_state_worker(simulation_state_data: Dict, validator_state_data: Dict,
                     simulation_state_file: str, validator_state_file: str) -> Dict:
    """
    Worker function for saving validator and simulation state to disk - picklable for ProcessPoolExecutor.
    
    Args:
        simulation_state_data (Dict): Dictionary containing simulation state to save
        validator_state_data (Dict): Dictionary containing validator state to save
        simulation_state_file (str): Path to save the simulation state file
        validator_state_file (str): Path to save the validator state file
        
    Returns:
        Dict: Result with success status and timing information
    """
    import os
    import time
    import traceback
    
    result = {
        'success': False,
        'error': None,
        'simulation_save_time': 0,
        'validator_save_time': 0,
        'total_time': 0
    }
    
    total_start = time.time()
    
    try:
        sim_start = time.time()
        sim_tmp = simulation_state_file + ".tmp"
        _pack_stream_fsync(sim_tmp, simulation_state_data)
        os.replace(sim_tmp, simulation_state_file)
        _fsync_dir(os.path.dirname(simulation_state_file))
        result['simulation_save_time'] = time.time() - sim_start

        val_start = time.time()
        val_tmp = validator_state_file + ".tmp"
        _pack_stream_fsync(val_tmp, validator_state_data)
        os.replace(val_tmp, validator_state_file)
        _fsync_dir(os.path.dirname(validator_state_file))
        result['validator_save_time'] = time.time() - val_start
        result['total_time'] = time.time() - total_start
        result['success'] = True
        
    except Exception as ex:
        result['error'] = str(ex)
        result['traceback'] = traceback.format_exc()
        for tmp in [simulation_state_file + ".tmp", validator_state_file + ".tmp"]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return result