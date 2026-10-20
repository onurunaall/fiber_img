import socket
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Rank, number of processes and device of the current training process."""
    rank: int
    world_size: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        """True when training runs in more than one process."""
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """True for the process with rank 0, which prints and saves checkpoints."""
        return self.rank == 0


def single_process_context(use_cuda: bool) -> DistributedContext:
    """Returns the context of a single process on GPU 0, or on the CPU without use_cuda."""
    device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
    return DistributedContext(rank=0, world_size=1, device=device)


def find_free_port() -> int:
    """Returns a TCP port on 127.0.0.1 that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def use_libuv() -> bool:
    """libuv is the default TCPStore backend, but the Windows builds of torch have none."""
    return sys.platform != "win32"


def open_meeting_point() -> dist.TCPStore:
    """Opens the place where the training processes meet, on a free port of 127.0.0.1.

    The launching program keeps it open while the processes run, so no other program can take the
    port between choosing it and using it. Its port is passed on to init_distributed.
    """
    return dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, use_libuv=use_libuv())


def init_distributed(rank: int, world_size: int, port: int) -> DistributedContext:
    """Joins the process group of a single machine. Each process drives the GPU with index == rank.

    Without CUDA, all processes run on the CPU, which is only useful for testing.
    port: the port of open_meeting_point() in the launching program.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # NCCL is the fast GPU backend but only exists on Linux. Gloo is the fallback on Windows.
    backend = "nccl" if dist.is_nccl_available() and device.type == "cuda" else "gloo"
    store = dist.TCPStore("127.0.0.1", port, is_master=False, use_libuv=use_libuv())
    dist.init_process_group(backend=backend, store=store, rank=rank, world_size=world_size)
    return DistributedContext(rank=rank, world_size=world_size, device=device)


def print_on_main(context: DistributedContext, message: str) -> None:
    """Prints message in the main process only."""
    if context.is_main_process:
        print(message)


def cleanup_distributed(context: DistributedContext) -> None:
    """Leaves the process group, if training is distributed."""
    if context.is_distributed:
        dist.destroy_process_group()


def sum_across_processes(values: list[float], context: DistributedContext) -> list[float]:
    """Returns the element-wise sum of values over all processes."""
    if not context.is_distributed:
        return values

    totals = torch.tensor(values, dtype=torch.float64, device=context.device)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals.tolist()


def mean_across_processes(value: float, context: DistributedContext) -> float:
    """Returns the mean of value over all processes."""
    total = sum_across_processes([value], context)[0]
    return total / context.world_size
