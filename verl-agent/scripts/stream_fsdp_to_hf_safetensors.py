#!/usr/bin/env python3
"""Merge FSDP model shards into HF sharded safetensors with low peak memory."""

import argparse
import gc
import json
import os
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor


FSDP_MODEL_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?B?)?\s*", value, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid size: {value}")
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    multipliers = {
        "B": 1,
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
    }
    return int(number * multipliers[unit])


def output_tensor_nbytes(tensor) -> int:
    return int(tensor.numel()) * torch.tensor([], dtype=torch.bfloat16).element_size()


def find_world_size(local_dir: Path) -> int:
    world_sizes = set()
    ranks_by_world_size: dict[int, set[int]] = {}
    for path in local_dir.glob("model_world_size_*_rank_*.pt"):
        match = FSDP_MODEL_RE.match(path.name)
        if not match:
            continue
        world_size = int(match.group(1))
        rank = int(match.group(2))
        world_sizes.add(world_size)
        ranks_by_world_size.setdefault(world_size, set()).add(rank)

    if len(world_sizes) != 1:
        raise RuntimeError(f"expected exactly one world size under {local_dir}, found {sorted(world_sizes)}")

    world_size = next(iter(world_sizes))
    expected = set(range(world_size))
    found = ranks_by_world_size[world_size]
    if found != expected:
        raise RuntimeError(f"missing rank files for world_size={world_size}: expected {sorted(expected)}, found {sorted(found)}")
    return world_size


def hf_weights_complete(target_dir: Path) -> bool:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = target_dir / name
        if path.is_file() and path.stat().st_size > 0:
            return True

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = target_dir / index_name
        if not index_path.is_file() or index_path.stat().st_size == 0:
            continue
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        except Exception:
            continue
        shard_names = set(weight_map.values())
        if shard_names and all((target_dir / shard).is_file() and (target_dir / shard).stat().st_size > 0 for shard in shard_names):
            return True
    return False


def remove_stale_hf_weights(target_dir: Path) -> None:
    stale_patterns = (
        "model.safetensors",
        "model.safetensors.index.json",
        "model-*-of-*.safetensors",
        ".model-*-of-*.safetensors.tmp",
        ".model.safetensors.index.json.tmp",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-*-of-*.bin",
    )
    for pattern in stale_patterns:
        for path in target_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def load_rank_state_dicts(local_dir: Path, world_size: int) -> list[dict]:
    state_dicts = []
    for rank in range(world_size):
        path = local_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        print(f"Loading rank {rank}: {path}", flush=True)
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
        state_dicts.append(state_dict)
    return state_dicts


def build_key_groups(rank_zero_state_dict: dict, max_shard_size: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0

    for key in sorted(rank_zero_state_dict):
        size = output_tensor_nbytes(rank_zero_state_dict[key])
        if current and current_size + size > max_shard_size:
            groups.append(current)
            current = []
            current_size = 0
        current.append(key)
        current_size += size

    if current:
        groups.append(current)
    return groups


def local_tensor_and_placements(value):
    if isinstance(value, DTensor):
        return value._local_tensor, tuple(value.placements), tuple(value.shape)
    return value, None, tuple(value.shape)


def merge_one_key(key: str, state_dicts: list[dict]) -> torch.Tensor:
    pieces = []
    expected_placements = None
    expected_shape = None

    for rank, state_dict in enumerate(state_dicts):
        value = state_dict[key]
        local_tensor, placements, global_shape = local_tensor_and_placements(value)
        if expected_shape is None:
            expected_shape = global_shape
            expected_placements = placements
        elif expected_shape != global_shape or expected_placements != placements:
            raise RuntimeError(f"inconsistent metadata for {key} at rank {rank}")
        pieces.append(local_tensor.to(dtype=torch.bfloat16, device="cpu").contiguous())

    if expected_placements is None:
        merged = torch.cat(pieces, dim=0).contiguous()
    elif len(expected_placements) == 1 and expected_placements[0].is_replicate():
        merged = pieces[0].contiguous()
    elif len(expected_placements) == 1 and expected_placements[0].is_shard():
        merged = torch.cat(pieces, dim=expected_placements[0].dim).contiguous()
    else:
        raise NotImplementedError(f"unsupported placement for {key}: {expected_placements}")

    if tuple(merged.shape) != expected_shape:
        raise RuntimeError(f"shape mismatch for {key}: merged {tuple(merged.shape)} vs expected {expected_shape}")
    return merged


def save_group(group: list[str], state_dicts: list[dict], target_dir: Path, shard_name: str) -> int:
    tensors = {}
    total_size = 0
    for key in group:
        tensor = merge_one_key(key, state_dicts)
        tensors[key] = tensor
        total_size += tensor.numel() * tensor.element_size()

    tmp_path = target_dir / f".{shard_name}.tmp"
    final_path = target_dir / shard_name
    print(f"Saving {shard_name}: {len(tensors)} tensors, {total_size / 1024**3:.2f} GiB", flush=True)
    save_file(tensors, str(tmp_path), metadata={"format": "pt"})
    os.replace(tmp_path, final_path)

    del tensors
    gc.collect()
    return total_size


def write_index(target_dir: Path, weight_map: dict[str, str], total_size: int) -> None:
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    tmp_path = target_dir / ".model.safetensors.index.json.tmp"
    final_path = target_dir / "model.safetensors.index.json"
    tmp_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, final_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_dir", required=True, type=Path)
    parser.add_argument("--target_dir", type=Path, default=None)
    parser.add_argument("--max-shard-size", type=parse_size, default=parse_size("2GB"))
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    local_dir = args.local_dir
    target_dir = args.target_dir or local_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    if hf_weights_complete(target_dir) and not args.overwrite_existing:
        print(f"HF weights already complete under {target_dir}; nothing to do", flush=True)
        return
    if args.overwrite_existing:
        remove_stale_hf_weights(target_dir)

    world_size = find_world_size(local_dir)
    state_dicts = load_rank_state_dicts(local_dir, world_size)

    key_sets = [set(state_dict) for state_dict in state_dicts]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise RuntimeError("rank state_dict keys differ")

    key_groups = build_key_groups(state_dicts[0], args.max_shard_size)
    shard_count = len(key_groups)
    print(f"Writing {shard_count} safetensors shards to {target_dir}", flush=True)

    weight_map = {}
    total_size = 0
    for index, group in enumerate(key_groups, start=1):
        shard_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        total_size += save_group(group, state_dicts, target_dir, shard_name)
        for key in group:
            weight_map[key] = shard_name

    write_index(target_dir, weight_map, total_size)
    print(f"Done: wrote {len(weight_map)} tensors, {total_size / 1024**3:.2f} GiB total", flush=True)


if __name__ == "__main__":
    main()
