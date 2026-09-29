import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def single_process_context(use_cuda: bool) -> DistributedContext:
    device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
    return DistributedContext(rank=0, world_size=1, device=device)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def init_distributed(rank: int, world_size: int, port: int) -> DistributedContext:
    """Joins the process group of a single machine. Each process drives the GPU with index == rank.

    Without CUDA, all processes run on the CPU, which is only useful for testing.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # NCCL is the fast GPU backend but only exists on Linux. Gloo is the fallback on Windows.
    backend = "nccl" if dist.is_nccl_available() and device.type == "cuda" else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    return DistributedContext(rank=rank, world_size=world_size, device=device)


def print_on_main(context: DistributedContext, message: str) -> None:
    if context.is_main_process:
        print(message)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.is_distributed:
        dist.destroy_process_group()


def per_process_batch_size(global_batch_size: int, world_size: int) -> int:
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Batch size {global_batch_size} must be divisible by the number of GPUs ({world_size})."
        )
    return global_batch_size // world_size


def sum_across_processes(values: list[float], context: DistributedContext) -> list[float]:
    if not context.is_distributed:
        return values

    totals = torch.tensor(values, dtype=torch.float64, device=context.device)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals.tolist()


def mean_across_processes(value: float, context: DistributedContext) -> float:
    total = sum_across_processes([value], context)[0]
    return total / context.world_size
