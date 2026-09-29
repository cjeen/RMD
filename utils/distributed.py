from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

from utils.device import get_default_device, get_distributed_backend, set_current_device


def canonicalize_wrapped_module_state_dict(state_dict):
    """Remove wrapper-only path components from checkpoint parameter names."""
    wrapper_prefixes = (
        "_fsdp_wrapped_module.",
        "_checkpoint_wrapped_module.",
        "_orig_mod.",
    )
    canonical_state_dict = {}
    for name, value in state_dict.items():
        canonical_name = name
        for prefix in wrapper_prefixes:
            canonical_name = canonical_name.replace(prefix, "")
        if canonical_name in canonical_state_dict:
            raise ValueError(
                f"Checkpoint keys collide after removing wrapper prefixes: {canonical_name}"
            )
        canonical_state_dict[canonical_name] = value
    return canonical_state_dict


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    device = get_default_device()
    # On NPU, moving a full FP32 module to the device before FSDP constructs
    # and offloads its shards defeats CPU offload and can exceed the per-device
    # memory limit during initialization. Keep the module on CPU for that
    # opt-in path and let FSDP materialize/offload its wrapped units. Preserve
    # the existing eager-move behavior for CUDA and all non-offloaded modules.
    defer_npu_move_for_cpu_offload = device.type == "npu" and cpu_offload
    if device.type != "cpu" and not defer_npu_move_for_cpu_offload:
        module = module.to(device)

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str | None = None):
    if backend is None:
        backend = get_distributed_backend()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    set_current_device(local_rank)
    if rank == 0:
        print(
            f"[distributed] backend={backend} local_rank={local_rank} device={get_default_device()}",
            flush=True,
        )
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))
    if backend == "hccl":
        warmup = torch.zeros((), device=get_default_device())
        dist.all_reduce(warmup, op=dist.ReduceOp.SUM)
        del warmup
