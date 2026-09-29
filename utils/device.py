import os

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def has_npu() -> bool:
    if not hasattr(torch, "npu"):
        return False
    try:
        if torch.npu.is_available():
            return True
    except Exception:
        pass
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("ASCEND_VISIBLE_DEVICES"):
        return True
    try:
        return torch.npu.device_count() > 0
    except Exception:
        return False


def has_cuda() -> bool:
    return torch.cuda.is_available()


def get_accelerator_type() -> str:
    if has_npu():
        return "npu"
    if has_cuda():
        return "cuda"
    return "cpu"


def get_current_device_index(device_type: str | None = None) -> int:
    device_type = device_type or get_accelerator_type()
    if device_type == "npu":
        try:
            index = torch.npu.current_device()
        except Exception:
            index = -1
        if index is None or index < 0:
            index = int(os.environ.get("LOCAL_RANK", 0))
        return index
    if device_type == "cuda":
        return torch.cuda.current_device()
    return 0


def get_default_device(index: int | None = None) -> torch.device:
    device_type = get_accelerator_type()
    if device_type == "cpu":
        return torch.device("cpu")
    if index is None:
        index = get_current_device_index(device_type)
    return torch.device(f"{device_type}:{index}")


def set_current_device(index: int, device_type: str | None = None) -> None:
    device_type = device_type or get_accelerator_type()
    if device_type == "npu":
        torch.npu.set_device(f"npu:{index}")
    elif device_type == "cuda":
        torch.cuda.set_device(index)


def get_distributed_backend() -> str:
    device_type = get_accelerator_type()
    if device_type == "npu":
        return "hccl"
    if device_type == "cuda":
        return "nccl"
    return "gloo"


def empty_cache() -> None:
    if has_npu() and hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()
    elif has_cuda():
        torch.cuda.empty_cache()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if has_npu() and hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(seed)
    if has_cuda():
        torch.cuda.manual_seed_all(seed)


def get_module_device(module: torch.nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return get_default_device()
