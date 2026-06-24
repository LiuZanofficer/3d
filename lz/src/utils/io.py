from __future__ import annotations

from typing import List

import numpy as np


def pack_list_of_np_arrays(arrays: List[np.ndarray]) -> dict:
    """Pack list of numpy arrays into a dict for np.savez."""
    return {str(i): arr for i, arr in enumerate(arrays)}


def unpack_list_of_np_arrays(npz_file) -> List[np.ndarray]:
    """Unpack list of numpy arrays from a np.load result."""
    def unpack_single_level(packed, lengths):
        packed_arr = np.asarray(packed)
        lengths_arr = np.asarray(lengths).astype(np.int64).reshape(-1)
        out = []
        ptr = 0
        for n in lengths_arr.tolist():
            n = int(n)
            out.append(np.asarray(packed_arr[ptr : ptr + n]))
            ptr += n
        if ptr != int(packed_arr.shape[0]):
            raise ValueError("Packed lengths do not match packed array size.")
        return out

    def unpack_two_level(packed, outer_lengths, inner_lengths):
        inner = unpack_single_level(packed, inner_lengths)
        outer_lengths_arr = np.asarray(outer_lengths).astype(np.int64).reshape(-1)
        out = []
        ptr = 0
        for n in outer_lengths_arr.tolist():
            n = int(n)
            out.append(inner[ptr : ptr + n])
            ptr += n
        if ptr != len(inner):
            raise ValueError("Outer lengths do not match number of inner arrays.")
        return out

    data = npz_file
    if hasattr(npz_file, "files"):
        files = list(npz_file.files)
        if "packed" in files and "lengths" in files:
            return unpack_single_level(npz_file["packed"], npz_file["lengths"])
        if "packed" in files and "outer_lengths" in files and "inner_lengths" in files:
            return unpack_two_level(
                npz_file["packed"],
                npz_file["outer_lengths"],
                npz_file["inner_lengths"],
            )
        keys = sorted(npz_file.files, key=lambda x: int(x))
        return [npz_file[k] for k in keys]
    if isinstance(data, dict):
        if "packed" in data and "lengths" in data:
            return unpack_single_level(data["packed"], data["lengths"])
        if "packed" in data and "outer_lengths" in data and "inner_lengths" in data:
            return unpack_two_level(
                data["packed"],
                data["outer_lengths"],
                data["inner_lengths"],
            )
        keys = sorted(data.keys(), key=lambda x: int(x))
        return [data[k] for k in keys]
    raise ValueError("Unsupported npz container type.")
