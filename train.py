import argparse
import datetime
import math
import os
import re
import sys
import traceback
from typing import Any

import torch
import torch.multiprocessing as mp
from matplotlib.figure import Figure
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from config_file import parse_args_with_config
from distributed_utils import (
    DistributedContext,
    cleanup_distributed,
    init_distributed,
    mean_across_processes,
    open_meeting_point,
    print_on_main,
    single_process_context,
    sum_across_processes,
)
from imageDatastore import imageDatastore
from model_factory import (
    COMPILE_MODES,
    EDSR_DEFAULT_FEATS,
    EDSR_DEFAULT_RESBLOCKS,
    ModelDescription,
    available_architectures,
    build_described_model,
    compile_model,
    model_description,
    saved_model_description,
)
from utils import (
    Timer,
    adjust_learning_rate,
    load_checkpoint,
    save_checkpoint,
    setup_seed,
    time_text,
)
from validation import ValidationResult, ValidationTotals, accumulate_validation_totals, compute_validation_result


DATA_FOLDERS = ("dir_ZTrain", "dir_XTrain", "dir_ZValid", "dir_XValid")
# The progress bar shows the loss of every PROGRESS_EVERY-th batch. Reading a loss (.item()) makes the
# CPU wait until the GPU has finished, so it is not done for every batch.
PROGRESS_EVERY = 20

type LossStats = dict[str, list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="net")

    # Training
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="global batch size, split across all GPUs")
    parser.add_argument("--val_batch_size", default=4, type=int, help="global batch size, split across all GPUs")
    parser.add_argument("--lr", default=1e-4, type=float, help="initial learning rate")
    parser.add_argument("--lr_DropFactor", default=0.5, type=float,
                        help="the learning rate is multiplied by this every lr_DropPeriod epochs")
    parser.add_argument("--lr_DropPeriod", default=10, type=int)
    parser.add_argument("--patch_size", default=0, type=int,
                        help="train on random square crops of this size, 0: full images")
    parser.add_argument("--augment", default="none", choices=["none", "flips", "flips+rot90"],
                        help="random flips / 90-degree rotations of the training images")
    parser.add_argument("--save_everyEpoch", default=10, type=int, help="save net_epoch_N.pth.tar every this many epochs")
    parser.add_argument("--save_path", default="./folder/", type=str,
                        help="every new run saves its files in a new subfolder <date>_<time>_<arch> of this folder; "
                             "--pretrain without --start_epoch continues in the folder of its checkpoint instead")
    parser.add_argument("--valid_everyEpoch", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int, help="data loading workers per GPU")
    parser.add_argument("--num_GPUs", default=1, type=int, help="0: CPU, 1: one GPU, >1: DistributedDataParallel")

    # Model
    parser.add_argument("--arch", default="unet", choices=available_architectures())
    parser.add_argument("--compile", action="store_true", help="optimize the model with torch.compile")
    parser.add_argument("--compile_mode", default="default", choices=COMPILE_MODES)
    parser.add_argument("--amp", action="store_true",
                        help="train in bfloat16 mixed precision: about half the GPU memory and faster. Needs an "
                             "Ampere or newer GPU (e.g. RTX A6000, A100). Validation stays in float32.")
    parser.add_argument("--edsr_n_resblocks", default=EDSR_DEFAULT_RESBLOCKS, type=int,
                        help="unet_edsr and unet_v2_edsr: number of EDSR residual blocks")
    parser.add_argument("--edsr_n_feats", default=EDSR_DEFAULT_FEATS, type=int,
                        help="unet_edsr and unet_v2_edsr: number of EDSR feature channels")
    parser.add_argument("--edsr_res_scale", default=1.0, type=float,
                        help="unet_edsr and unet_v2_edsr: each EDSR block's output is multiplied by this before "
                             "its skip is added (the EDSR paper uses 0.1 for 32 blocks x 256 features)")

    # Dataset. No defaults: set the folders in the config file or on the command line.
    parser.add_argument("--dir_ZTrain", default=None, type=str, help="folder of the training targets (HR)")
    parser.add_argument("--dir_XTrain", default=None, type=str, help="folder of the training inputs (sim_MCF)")
    parser.add_argument("--dir_ZValid", default=None, type=str, help="folder of the validation targets (HR)")
    parser.add_argument("--dir_XValid", default=None, type=str, help="folder of the validation inputs (sim_MCF)")
    parser.add_argument("--n_colors", default=1, type=int, help="number of color channels to use")
    parser.add_argument("--reference_image", default=None, type=str,
                        help="white image of the fiber (e.g. calib/sim_cfb_white.png): the model gets it as an extra "
                             "input channel next to every input image. Changes the model, so it needs its own training.")

    # Pre-train
    parser.add_argument("--pretrain", action="store_true", help="load pre-trained model")
    parser.add_argument("--dir_pretrain", default="", type=str, help="path1 to checkpoint")
    parser.add_argument("--start_epoch", "--start-epoch", default=None, type=int,
                        help="first epoch to train (default: the epoch after the --pretrain checkpoint, otherwise 1). "
                             "Only with --pretrain; to train on new data use --finetune_from. "
                             "The old spelling --start-epoch also works.")
    parser.add_argument("--finetune_from", default="", type=str,
                        help="transfer learning: start a new run from the weights of this checkpoint. Only the weights "
                             "are loaded: epoch 1, new optimizer, new loss history, new run folder.")

    parser.add_argument("--save_model_arch", action="store_true",
                        help="also save the whole trained model object in every checkpoint")
    parser.add_argument("--rgb_range", default=1, type=int, help="maxium value of RGB")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--ext", default=".png", type=str)
    args = parse_args_with_config(parser)

    missing = [name for name in DATA_FOLDERS if getattr(args, name) is None]
    if missing:
        parser.error(f"no data folder given for {', '.join(missing)}: set it in the config file (--config) "
                     f"or on the command line, e.g. --{missing[0]} FOLDER")

    # Checked here: zero would crash later with "division by zero", in the middle of training.
    for name in ("num_epochs", "valid_everyEpoch", "save_everyEpoch"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1, got {getattr(args, name)}")

    # Checked here: a resumed run saves its files in the folder of this checkpoint.
    if args.pretrain and not os.path.isfile(args.dir_pretrain):
        parser.error(f"--dir_pretrain '{args.dir_pretrain}' is not a checkpoint file")
    # Checked here: with a forgotten --pretrain, training would silently start from random weights.
    if args.dir_pretrain and not args.pretrain:
        parser.error("--dir_pretrain is given without --pretrain, so the checkpoint would not be loaded. Add --pretrain "
                     "to continue that run, or use --finetune_from FILE to start a new run from its weights.")
    # The learning rate is computed from the epoch number, so a new run must start at epoch 1.
    if args.start_epoch is not None and not args.pretrain:
        parser.error("--start_epoch only works together with --pretrain. To train on new data from the weights of "
                     "a checkpoint, use --finetune_from FILE (it starts at epoch 1).")
    if args.finetune_from:
        if args.pretrain:
            parser.error("--finetune_from starts a new run and --pretrain continues an old one: use only one of them")
        if not os.path.isfile(args.finetune_from):
            parser.error(f"--finetune_from '{args.finetune_from}' is not a checkpoint file")
    if args.reference_image is not None and not os.path.isfile(args.reference_image):
        parser.error(f"--reference_image '{args.reference_image}' is not a file")

    # Each GPU gets an equal share of every batch. Checked here, before the GPU processes start.
    for name in ("batch_size", "val_batch_size"):
        size = getattr(args, name)
        if args.num_GPUs > 1 and size % args.num_GPUs != 0:
            parser.error(f"--{name} {size} must be divisible by --num_GPUs {args.num_GPUs}")
    return args


def create_train_loader(
    args: argparse.Namespace,
    context: DistributedContext,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = imageDatastore(
        args.dir_ZTrain,
        args.dir_XTrain,
        patches=args.patch_size > 0,
        patch_size=args.patch_size,
        n_colors=args.n_colors,
        rgb_range=args.rgb_range,
        hflip=args.augment != "none",
        vflip=args.augment != "none",
        rotate=args.augment == "flips+rot90",
        ext=args.ext,
        reference_image=args.reference_image,
    )

    if args.patch_size > 0:
        image_height, image_width = dataset._load_file(0)[0].shape[:2]
        if args.patch_size > min(image_height, image_width):
            raise ValueError(
                f"--patch_size {args.patch_size} is larger than the training images ({image_height} x {image_width}). "
                "Use a smaller --patch_size, or --patch_size 0 to train on full images."
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

    # parse_args checked that the batch size splits evenly across the GPUs.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size // context.world_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=context.device.type == "cuda",
        drop_last=True,
        persistent_workers=False,
    )
    if len(loader) == 0:
        raise ValueError(
            f"{args.dir_XTrain} has {len(dataset)} training images, too few for one batch of "
            f"--batch_size {args.batch_size}. Use a smaller --batch_size."
        )
    return loader, sampler


def create_valid_loader(args: argparse.Namespace, context: DistributedContext) -> DataLoader:
    dataset = imageDatastore(
        args.dir_ZValid,
        args.dir_XValid,
        n_colors=args.n_colors,
        rgb_range=args.rgb_range,
        ext=args.ext,
        reference_image=args.reference_image,
    )

    # Every process validates every world_size-th image. Unlike DistributedSampler,
    # this never duplicates images to even out the split, so the metrics stay exact.
    shard = Subset(dataset, range(context.rank, len(dataset), context.world_size))

    return DataLoader(
        shard,
        batch_size=args.val_batch_size // context.world_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=context.device.type == "cuda",
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
    """Returns the training L1 loss averaged over the epoch."""
    model.train()
    adjust_learning_rate(optimizer, epoch - 1, args.lr, args.lr_DropFactor, args.lr_DropPeriod)

    learning_rate = optimizer.param_groups[0]["lr"]
    print_on_main(context, f"\n------- Epoch {epoch}/{args.num_epochs} \t\t lr = {learning_rate}")

    progress = tqdm(loader, desc="In training", leave=True, ncols=80, disable=not context.is_main_process)
    # The sum stays on the GPU, so the CPU never has to wait for the GPU to add it up.
    total_loss = torch.zeros((), device=context.device)

    for batch_index, (inputs, targets) in enumerate(progress):
        # non_blocking: the copy runs while the CPU already queues the next GPU work (the loader pins the memory).
        inputs = inputs.to(context.device, non_blocking=True)
        targets = targets.float().to(context.device, non_blocking=True)

        # --amp: the forward pass runs in bfloat16 where it is safe (e.g. convolutions); the weights,
        # their gradients and the optimizer stay float32. bfloat16 needs no gradient scaling.
        with torch.autocast(context.device.type, dtype=torch.bfloat16, enabled=args.amp):
            predictions = model(inputs)
            loss = criterion(predictions, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()
        if batch_index % PROGRESS_EVERY == 0:
            progress.set_postfix(loss=loss.item())

    epoch_loss = mean_across_processes(total_loss.item() / len(loader), context)
    print_on_main(context, f"\nTrain loss: {epoch_loss:.4f}")
    return epoch_loss


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    context: DistributedContext,
) -> ValidationResult:
    model.eval()

    local_totals = accumulate_validation_totals(model, loader, criterion, context.device)
    totals = ValidationTotals.from_list(sum_across_processes(local_totals.as_list(), context))
    result = compute_validation_result(totals)

    print_on_main(context, result.text())
    return result


def check_amp_support(args: argparse.Namespace, context: DistributedContext) -> None:
    """Stops with a clear message if --amp is on but the GPU cannot compute in bfloat16."""
    if args.amp and context.device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError(
            f"--amp needs a GPU with bfloat16 support (Ampere or newer), "
            f"but {torch.cuda.get_device_name(context.device)} has none. Train without --amp."
        )


def read_checkpoint(path: str, description: ModelDescription, context: DistributedContext) -> dict[str, Any]:
    """Reads a checkpoint file. Stops if it holds another model than the settings of this run build."""
    # weights_only=False: old checkpoints store 'args' as an argparse.Namespace, and --save_model_arch stores the whole
    # model object. Only load checkpoint files you trust.
    checkpoint: dict[str, Any] = torch.load(path, map_location=context.device, weights_only=False)
    saved = saved_model_description(checkpoint)
    if saved != description:
        raise ValueError(
            f"{path} holds the model {saved}, but the settings of this run build {description}. "
            "Use the same --arch, --n_colors, --edsr_* options and --reference_image (given or not) as the checkpoint."
        )
    return checkpoint


def load_pretrained(
    args: argparse.Namespace,
    description: ModelDescription,
    network: nn.Module,
    optimizer: optim.Optimizer,
    context: DistributedContext,
) -> tuple[int, LossStats]:
    """Loads the --pretrain checkpoint into network and optimizer. Returns (first epoch to train, its loss history).

    Stops if the checkpoint holds another model than the settings of this run build.
    """
    checkpoint = read_checkpoint(args.dir_pretrain, description, context)
    load_checkpoint(checkpoint, network)
    optimizer.load_state_dict(checkpoint["optimizer"])

    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    else:
        start_epoch = checkpoint["epoch"] + 1
        if start_epoch > args.num_epochs:
            raise ValueError(
                f"{args.dir_pretrain} ends at epoch {checkpoint['epoch']}, so there is nothing left to train "
                f"up to --num_epochs {args.num_epochs}. Raise --num_epochs to train longer."
            )
    print_on_main(context, f"===> Starting at epoch {start_epoch} from {args.dir_pretrain}")
    return start_epoch, dict(checkpoint["loss_stats"])


def complete_loss_stats(loss_stats: LossStats) -> LossStats:
    """Adds the lists that a new run, or a checkpoint from before they were saved, does not have yet."""
    loss_stats.setdefault("valid_epoch", list(range(1, len(loss_stats["valid"]) + 1)))
    # Checkpoints from before PSNR and SSIM were saved get nan for their earlier validations.
    for key in ("valid_psnr", "valid_ssim"):
        loss_stats.setdefault(key, [math.nan] * len(loss_stats["valid"]))
    return loss_stats


def check_nothing_overwritten(run_folder: str, start_epoch: int) -> None:
    """Stops before training overwrites checkpoints of later epochs, e.g. when resuming from an older checkpoint.

    A resumed run continues in the folder of its checkpoint (a new run's folder is empty).
    """
    later_checkpoints = sorted(
        name for name in os.listdir(run_folder)
        if (match := re.fullmatch(r"net_epoch_(\d+)\.pth\.tar", name)) and int(match.group(1)) >= start_epoch
    )
    if later_checkpoints:
        raise ValueError(
            f"{run_folder} already has {', '.join(later_checkpoints)}, which training from epoch {start_epoch} "
            "would overwrite. Resume from the last checkpoint, or add --start_epoch to train into a new folder."
        )


def best_valid_loss_so_far(run_folder: str, loss_stats: LossStats) -> float:
    """The lowest validation L1 of the loss history and of run_folder/net_best.pth.tar.

    net_best.pth.tar can be newer than the resumed checkpoint (e.g. after a crash). Only a better model may replace it.
    """
    best_valid_loss = min(loss_stats["valid"]) if loss_stats["valid"] else float("inf")
    best_path = os.path.join(run_folder, "net_best.pth.tar")
    if os.path.isfile(best_path):
        # mmap: only the loss history is read from the disk, not the weights and the optimizer state.
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
        best_valid_loss = min([best_valid_loss, *best_checkpoint["loss_stats"]["valid"]])
        del best_checkpoint
    return best_valid_loss


def run_training(args: argparse.Namespace, context: DistributedContext, run_folder: str) -> None:
    """Trains in run_folder: a new, empty folder, or the folder of the --pretrain checkpoint when resuming."""
    check_amp_support(args, context)
    setup_seed(args.seed + context.rank)
    # All images of a run have the same size, so cuDNN can time its algorithms once and keep the fastest.
    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = create_train_loader(args, context)
    valid_loader = create_valid_loader(args, context)

    description = model_description(args)
    network = build_described_model(description).to(context.device)

    criterion = nn.L1Loss().to(context.device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    start_epoch = 1 if args.start_epoch is None else args.start_epoch
    loss_stats: LossStats = {"train": [], "valid": [], "valid_expt": []}
    if args.pretrain:
        start_epoch, loss_stats = load_pretrained(args, description, network, optimizer, context)
    elif args.finetune_from:
        # Transfer learning: only the weights are taken over. The optimizer, the loss history and the epoch count
        # start new, so net_best.pth.tar is the best model on the new data.
        load_checkpoint(read_checkpoint(args.finetune_from, description, context), network)
        print_on_main(context, f"===> Fine-tuning from the weights of {args.finetune_from} "
                               "(new optimizer, new loss history, epoch 1)")
    loss_stats = complete_loss_stats(loss_stats)
    check_nothing_overwritten(run_folder, start_epoch)
    best_valid_loss = best_valid_loss_so_far(run_folder, loss_stats)

    training_model, validation_model = prepare_models(network, args, context)

    print_on_main(
        context,
        f"===> Training {args.arch}: lr {args.lr}, batch size {args.batch_size}, "
        f"patch size {args.patch_size or 'full image'}, augment {args.augment}, "
        f"bfloat16 mixed precision {'on' if args.amp else 'off'}",
    )
    timer = Timer()

    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start = timer.t()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        loss_train = train_one_epoch(epoch, training_model, train_loader, criterion, optimizer, args, context)
        loss_stats["train"].append(loss_train)

        do_valid = epoch % args.valid_everyEpoch == 0 or epoch == args.num_epochs
        if do_valid:
            result = validate(validation_model, valid_loader, criterion, context)
            loss_stats["valid"].append(result.loss)
            loss_stats["valid_psnr"].append(float(result.psnr))
            loss_stats["valid_ssim"].append(float(result.ssim))
            loss_stats["valid_epoch"].append(epoch)

            # Save the best model now, only when it gets better. It replaces the previous best.
            if result.loss < best_valid_loss:
                best_valid_loss = result.loss
                if context.is_main_process:
                    save_checkpoint(network, optimizer, epoch, loss_stats, args, description, run_folder,
                                    "net_best.pth.tar")
            # Drawn after every validation, so a run that stops early still has its plot.
            if context.is_main_process:
                save_training_plot(loss_stats, run_folder)

        do_save = epoch % args.save_everyEpoch == 0 or epoch == args.num_epochs
        if do_save and context.is_main_process:
            save_checkpoint(network, optimizer, epoch, loss_stats, args, description, run_folder,
                            f"net_epoch_{epoch}.pth.tar")

        epoch_end = timer.t()
        progress = (epoch - start_epoch + 1) / (args.num_epochs - start_epoch + 1)
        epoch_time = time_text(epoch_end - epoch_start)
        elapsed_time = time_text(epoch_end)
        expected_total_time = time_text(epoch_end / progress)
        print_on_main(context, f"Time cost: {epoch_time}, {elapsed_time}/{expected_total_time}")

    print_on_main(context, f"Total cost times: {time_text(timer.t())}")


def save_training_plot(loss_stats: LossStats, folder: str) -> None:
    """Saves the training and validation loss curves as training_progress.png in folder."""
    # A Figure without pyplot needs no screen and never waits for a window to be closed.
    figure = Figure(figsize=(10, 5))
    axes = figure.add_subplot()
    axes.set_title("Training progress")
    axes.plot(range(1, len(loss_stats["train"]) + 1), loss_stats["train"], label="Train")
    axes.plot(loss_stats["valid_epoch"], loss_stats["valid"], label="Valid")
    axes.set_xlabel("Epochs")
    axes.set_ylabel("Loss")
    axes.set_yscale("log")
    axes.legend()
    figure.savefig(os.path.join(folder, "training_progress.png"))


def make_run_folder(save_path: str, arch: str) -> str:
    """Makes a new, empty folder save_path/<date>_<time>_<arch> for the files of this run."""
    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{arch}"
    path = os.path.join(save_path, run_name)
    os.makedirs(path)  # fails if the folder exists, so a run never writes into another run's folder
    return path


class Tee:
    """Writes everything printed to the screen and to a log file."""

    def __init__(self, path: str) -> None:
        self.screen = sys.stdout
        self.file = open(path, "a", encoding="utf-8")

    def write(self, text: str) -> int:
        self.screen.write(text)
        self.file.write(text)
        self.file.flush()  # so the log is complete even if the run is killed
        return len(text)

    def flush(self) -> None:
        self.screen.flush()
        self.file.flush()

    def __getattr__(self, name: str) -> object:
        return getattr(self.screen, name)


def log_to_file(folder: str) -> Tee:
    """From now on, everything this process prints also goes to folder/train.log (a resumed run adds to it)."""
    log = Tee(os.path.join(folder, "train.log"))
    sys.stdout = log
    return log


def distributed_worker(rank: int, args: argparse.Namespace, port: int, run_folder: str) -> None:
    if rank == 0:
        log_to_file(run_folder)
    context = init_distributed(rank, args.num_GPUs, port)
    try:
        run_training(args, context, run_folder)
    finally:
        cleanup_distributed(context)


def launch_distributed_training(args: argparse.Namespace, run_folder: str) -> None:
    available_gpus = torch.cuda.device_count()
    if torch.cuda.is_available() and args.num_GPUs > available_gpus:
        raise ValueError(f"--num_GPUs is {args.num_GPUs}, but only {available_gpus} GPU(s) are available.")
    if not torch.cuda.is_available():
        print(f"===> No GPU found: running {args.num_GPUs} processes on the CPU (only useful for testing)")

    # One process per GPU. Process number i (its "rank") trains on GPU i. The processes meet at a port
    # that this process keeps open while they run, so no other program can take it in between.
    meeting_point = open_meeting_point()
    mp.spawn(distributed_worker, args=(args, meeting_point.port, run_folder), nprocs=args.num_GPUs)


def main() -> None:
    args = parse_args()

    # Decided here, before any GPU process starts, so that all of them save into the same folder.
    # args.save_path stays the parent folder of the settings; run_folder is the folder of this run.
    if args.pretrain and args.start_epoch is None:
        # Resuming continues the same run, in the folder of its checkpoint.
        run_folder = os.path.dirname(os.path.abspath(args.dir_pretrain))
    else:
        # A new run gets a new folder, so it never overwrites the files of an earlier run.
        run_folder = make_run_folder(args.save_path, args.arch)
    log = log_to_file(run_folder)
    print(f"===> Saving the files of this run in {run_folder}")

    try:
        if args.num_GPUs > 1:
            launch_distributed_training(args, run_folder)
        else:
            run_training(args, single_process_context(use_cuda=args.num_GPUs > 0), run_folder)
    except BaseException:
        # The error is shown on the screen as usual; this also keeps it in train.log.
        traceback.print_exc(file=log.file)
        raise


# Needed on Windows: every DataLoader worker and every GPU process re-runs this file from the top.
# The guard makes them skip main(), so only the launching program starts the training.
if __name__ == "__main__":
    main()
