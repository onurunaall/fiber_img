import argparse
import copy
import datetime
import os

import torch
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from distributed_utils import (
    DistributedContext,
    cleanup_distributed,
    find_free_port,
    init_distributed,
    mean_across_processes,
    per_process_batch_size,
    print_on_main,
    single_process_context,
    sum_across_processes,
)
from imageDatastore import imageDatastore
from model_factory import COMPILE_MODES, available_architectures, build_model, compile_model
from utils import Timer, adjust_learning_rate, load_checkpoint, save_checkpoint, setup_seed, time_text
from validation import ValidationTotals, accumulate_validation_totals, compute_validation_result


DEFAULT_DATA_PATH = r"C:\Users\onue687i\Documents\AirDent\TW Source Material\720"

type LossStats = dict[str, list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="net")

    # Training
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="global batch size, split across all GPUs")
    parser.add_argument("--val_batch_size", default=4, type=int, help="global batch size, split across all GPUs")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_DropFactor", default=0.5, type=float)
    parser.add_argument("--lr_DropPeriod", default=10, type=int)
    parser.add_argument("--save_everyEpoch", default=1, type=int)
    parser.add_argument("--save_path", default="./folder/", type=str)
    parser.add_argument("--valid_everyEpoch", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int, help="data loading workers per GPU")
    parser.add_argument("--num_GPUs", default=1, type=int, help="0: CPU, 1: one GPU, >1: DistributedDataParallel")

    # Model
    parser.add_argument("--arch", default="unet", choices=available_architectures())
    parser.add_argument("--compile", action="store_true", help="optimize the model with torch.compile")
    parser.add_argument("--compile_mode", default="default", choices=COMPILE_MODES)

    # Dataset
    parser.add_argument("--dir_ZTrain", default=os.path.join(DEFAULT_DATA_PATH, "HR_Train"), type=str)
    parser.add_argument("--dir_XTrain", default=os.path.join(DEFAULT_DATA_PATH, "sim_MCF_Train"), type=str)
    parser.add_argument("--dir_ZValid", default=os.path.join(DEFAULT_DATA_PATH, "HR_Valid"), type=str)
    parser.add_argument("--dir_XValid", default=os.path.join(DEFAULT_DATA_PATH, "sim_MCF_Valid"), type=str)
    parser.add_argument("--n_colors", default=1, type=int, help="number of color channels to use")

    # Pre-train
    parser.add_argument("--pretrain", action="store_true", help="load pre-trained model")
    parser.add_argument("--dir_pretrain", default="", type=str, help="path1 to checkpoint")
    parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number")

    parser.add_argument("--save_model_arch", action="store_true", help="save model architecture")
    parser.add_argument("--rgb_range", default=1, type=int, help="maxium value of RGB")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--ext", default=".png", type=str)
    return parser.parse_args()


def create_train_loader(
    args: argparse.Namespace,
    context: DistributedContext,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = imageDatastore(
        args.dir_ZTrain,
        args.dir_XTrain,
        n_colors=args.n_colors,
        rgb_range=args.rgb_range,
        ext=args.ext,
    )

    sampler: DistributedSampler | None = None
    if context.is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )

    loader = DataLoader(
        dataset,
        batch_size=per_process_batch_size(args.batch_size, context.world_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    return loader, sampler


def create_valid_loader(args: argparse.Namespace, context: DistributedContext) -> DataLoader:
    dataset = imageDatastore(
        args.dir_ZValid,
        args.dir_XValid,
        n_colors=args.n_colors,
        rgb_range=args.rgb_range,
        ext=args.ext,
    )

    # Every process validates every world_size-th image. Unlike DistributedSampler,
    # this never duplicates images to even out the split, so the metrics stay exact.
    shard = Subset(dataset, range(context.rank, len(dataset), context.world_size))

    return DataLoader(
        shard,
        batch_size=per_process_batch_size(args.val_batch_size, context.world_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def prepare_models(
    network: nn.Module,
    args: argparse.Namespace,
    context: DistributedContext,
) -> tuple[nn.Module, nn.Module]:
    """Returns (training model, validation model). Both share the parameters of `network`.

    Validation uses the network without the DDP wrapper: GPUs can get a different number of
    validation images, and a DDP forward pass may wait for the other GPUs, which would hang.
    """
    training_model = network
    validation_model = network

    if context.is_distributed:
        device_ids = [context.device.index] if context.device.type == "cuda" else None
        training_model = DistributedDataParallel(network, device_ids=device_ids)

    if args.compile:
        training_model = compile_model(training_model, args.compile_mode)
        validation_model = compile_model(validation_model, args.compile_mode)

    return training_model, validation_model


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    context: DistributedContext,
) -> float:
    model.train()
    adjust_learning_rate(optimizer, epoch - 1, args.lr, args.lr_DropFactor, args.lr_DropPeriod)

    learning_rate = optimizer.param_groups[0]["lr"]
    print_on_main(context, f"\n------- Epoch {epoch}/{args.num_epochs} \t\t lr = {learning_rate}")

    progress = tqdm(loader, desc="In training", leave=True, ncols=80, disable=not context.is_main_process)
    total_loss = 0.0

    for inputs, targets in progress:
        inputs = inputs.to(context.device)
        targets = targets.float().to(context.device)

        predictions = model(inputs)
        loss = criterion(predictions, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        progress.set_postfix(loss=loss_value)
        total_loss += loss_value

    epoch_loss = mean_across_processes(total_loss / len(loader), context)
    print_on_main(context, f"\nTrain loss: {epoch_loss:.4f}")
    return epoch_loss


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    context: DistributedContext,
) -> float:
    model.eval()

    local_totals = accumulate_validation_totals(model, loader, criterion, context.device)
    totals = ValidationTotals.from_list(sum_across_processes(local_totals.as_list(), context))
    result = compute_validation_result(totals)

    print_on_main(
        context,
        f"Valid loss: {result.loss:.4f} \t\t PSNR: {result.psnr:.4f} \t\t SSIM: {result.ssim:.4f}",
    )
    return result.loss


def run_training(args: argparse.Namespace, context: DistributedContext) -> LossStats:
    setup_seed(args.seed + context.rank)

    train_loader, train_sampler = create_train_loader(args, context)
    valid_loader = create_valid_loader(args, context)

    network = build_model(args.arch, in_channels=args.n_colors, out_channels=args.n_colors)
    model_arch = copy.deepcopy(network)
    network = network.to(context.device)

    criterion = nn.L1Loss().to(context.device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    loss_stats: LossStats = {"train": [], "valid": [], "valid_expt": []}

    if args.pretrain:
        # weights_only=False: the file also stores 'args'. Only load checkpoint files you trust.
        checkpoint = torch.load(args.dir_pretrain, map_location=context.device, weights_only=False)
        load_checkpoint(checkpoint, network)
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_stats = dict(checkpoint["loss_stats"])

    loss_stats.setdefault("valid_epoch", list(range(1, len(loss_stats["valid"]) + 1)))
    best_valid_loss = min(loss_stats["valid"]) if loss_stats["valid"] else float("inf")

    training_model, validation_model = prepare_models(network, args, context)

    print_on_main(context, "===> Training")
    code_start = datetime.datetime.now()
    timer = Timer()

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        epoch_start = timer.t()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        loss_train = train_one_epoch(epoch, training_model, train_loader, criterion, optimizer, args, context)
        loss_stats["train"].append(loss_train)

        do_valid = epoch % args.valid_everyEpoch == 0 or epoch == args.num_epochs
        if do_valid:
            loss_valid = validate(validation_model, valid_loader, criterion, context)
            loss_stats["valid"].append(loss_valid)
            loss_stats["valid_epoch"].append(epoch)

            # Save the best model now, only when it gets better
            if loss_valid < best_valid_loss:
                best_valid_loss = loss_valid
                if context.is_main_process:
                    save_checkpoint(network, model_arch, optimizer, epoch, loss_stats, args, text="_best")

        do_save = epoch % args.save_everyEpoch == 0 or epoch == args.num_epochs
        if do_save and context.is_main_process:
            save_checkpoint(network, model_arch, optimizer, epoch, loss_stats, args)

        epoch_end = timer.t()
        progress = (epoch - args.start_epoch + 1) / (args.num_epochs - args.start_epoch + 1)
        epoch_time = time_text(epoch_end - epoch_start)
        elapsed_time = time_text(epoch_end)
        expected_total_time = time_text(epoch_end / progress)
        print_on_main(context, f"Time cost: {epoch_time}, {elapsed_time}/{expected_total_time}")

    code_end = datetime.datetime.now()
    total_time = str(code_end - code_start).split(".", 2)[0]
    print_on_main(context, f"Total cost times: {total_time}")

    return loss_stats


def plot_training_progress(loss_stats: LossStats) -> None:
    plt.figure("Training progress", figsize=(10, 5))
    plt.title("Training progress")
    plt.plot(range(1, len(loss_stats["train"]) + 1), loss_stats["train"], label="Train")
    plt.plot(loss_stats["valid_epoch"], loss_stats["valid"], label="Valid")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.show()


def distributed_worker(rank: int, args: argparse.Namespace, port: int) -> None:
    context = init_distributed(rank, args.num_GPUs, port)
    try:
        loss_stats = run_training(args, context)
    finally:
        cleanup_distributed(context)

    if context.is_main_process:
        plot_training_progress(loss_stats)
    torch.cuda.empty_cache()


def launch_distributed_training(args: argparse.Namespace) -> None:
    available_gpus = torch.cuda.device_count()
    if args.num_GPUs > available_gpus:
        raise ValueError(f"--num_GPUs is {args.num_GPUs}, but only {available_gpus} GPU(s) are available.")

    # One process per GPU. Process number i (its "rank") trains on GPU i.
    port = find_free_port()
    mp.spawn(distributed_worker, args=(args, port), nprocs=args.num_GPUs)


def main() -> None:
    args = parse_args()

    # Create the checkpoint folder if it does not exist yet
    os.makedirs(args.save_path, exist_ok=True)

    if args.num_GPUs > 1:
        launch_distributed_training(args)
        return

    context = single_process_context(use_cuda=args.num_GPUs > 0)
    loss_stats = run_training(args, context)
    plot_training_progress(loss_stats)
    torch.cuda.empty_cache()


# Needed on Windows: every DataLoader worker and every GPU process re-runs this file from the top.
# The guard makes them skip main(), so only the launching program starts the training.
if __name__ == "__main__":
    main()
