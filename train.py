import argparse
import copy
import datetime
import os

import torch
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from config_file import parse_args_with_config
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
from losses import SGARNetLoss, load_vgg19_weights
from model_factory import (
    COMPILE_MODES,
    EDSR_DEFAULT_FEATS,
    EDSR_DEFAULT_RESBLOCKS,
    architecture_options,
    available_architectures,
    build_model,
    compile_model,
)
from utils import (
    Timer,
    adjust_learning_rate,
    load_checkpoint,
    save_checkpoint,
    set_cosine_learning_rate,
    setup_seed,
    time_text,
)
from validation import ValidationTotals, accumulate_validation_totals, compute_validation_result


DEFAULT_DATA_PATH = r"C:\Users\onue687i\Documents\AirDent\TW Source Material\720"

type LossStats = dict[str, list[float]]

# Training recipes:
#   unet:    (also used for --arch unet_edsr) L1 loss, Adam, step decay (lr_DropFactor every lr_DropPeriod epochs), full images, no augmentation.
#   sgarnet: options/train/MCFArtifactFree.yml of https://github.com/THUHoloLab/SGARNet: PSNR loss
#            + 0.01 * VGG19 perceptual loss, AdamW, cosine decay per iteration, gradient clipping,
#            random crops with flips and rot90.
# --recipe auto picks the recipe named like --arch. Options left out on the command line get the
# recipe's default below.
RECIPE_DEFAULTS: dict[str, dict[str, object]] = {
    "unet": {"lr": 1e-4, "batch_size": 4, "patch_size": 0, "augment": "none"},
    "sgarnet": {"lr": 1e-3, "batch_size": 8, "patch_size": 368, "augment": "flips+rot90"},
}
SGARNET_WEIGHT_DECAY = 1e-3
SGARNET_BETAS = (0.9, 0.9)
SGARNET_ETA_MIN = 1e-7
SGARNET_GRAD_CLIP_NORM = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="net")

    # Training
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--recipe", default="auto", choices=["auto", *RECIPE_DEFAULTS],
                        help="training recipe; auto: sgarnet for --arch sgarnet, otherwise unet")
    parser.add_argument("--batch_size", default=None, type=int,
                        help="global batch size, split across all GPUs (default: unet 4, sgarnet 8)")
    parser.add_argument("--val_batch_size", default=4, type=int, help="global batch size, split across all GPUs")
    parser.add_argument("--lr", default=None, type=float, help="initial learning rate (default: unet 1e-4, sgarnet 1e-3)")
    parser.add_argument("--lr_DropFactor", default=0.5, type=float, help="unet recipe only")
    parser.add_argument("--lr_DropPeriod", default=10, type=int, help="unet recipe only")
    parser.add_argument("--patch_size", default=None, type=int,
                        help="train on random square crops of this size, 0: full images (default: unet 0, sgarnet 368)")
    parser.add_argument("--augment", default=None, choices=["none", "flips", "flips+rot90"],
                        help="random flips / 90-degree rotations of the training images (default: unet none, sgarnet flips+rot90)")
    parser.add_argument("--save_everyEpoch", default=1, type=int)
    parser.add_argument("--save_path", default="./folder/", type=str)
    parser.add_argument("--valid_everyEpoch", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int, help="data loading workers per GPU")
    parser.add_argument("--num_GPUs", default=1, type=int, help="0: CPU, 1: one GPU, >1: DistributedDataParallel")

    # Model
    parser.add_argument("--arch", default="unet", choices=available_architectures())
    parser.add_argument("--compile", action="store_true", help="optimize the model with torch.compile")
    parser.add_argument("--compile_mode", default="default", choices=COMPILE_MODES)
    parser.add_argument("--sgarnet_lattice_period", default=None, type=float,
                        help="SGARNet: core lattice period in input pixels, from estimate_lattice_period.py")
    parser.add_argument("--sgarnet_lattice_angle", default=None, type=float,
                        help="SGARNet: core lattice angle in degrees, from estimate_lattice_period.py")
    parser.add_argument("--edsr_n_resblocks", default=EDSR_DEFAULT_RESBLOCKS, type=int,
                        help="UNET_EDSR: number of EDSR residual blocks")
    parser.add_argument("--edsr_n_feats", default=EDSR_DEFAULT_FEATS, type=int,
                        help="UNET_EDSR: number of EDSR feature channels")

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
    return resolve_recipe(parse_args_with_config(parser))


def resolve_recipe(args: argparse.Namespace) -> argparse.Namespace:
    if args.recipe == "auto":
        args.recipe = "sgarnet" if args.arch == "sgarnet" else "unet"

    for name, value in RECIPE_DEFAULTS[args.recipe].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
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
) -> tuple[float, float]:
    """Returns (L1 loss, training loss) averaged over the epoch. With the unet recipe both are the L1 loss."""
    model.train()
    first_iteration = (epoch - 1) * len(loader)
    total_iterations = args.num_epochs * len(loader)
    if args.recipe == "sgarnet":
        set_cosine_learning_rate(optimizer, first_iteration, total_iterations, args.lr, SGARNET_ETA_MIN)
    else:
        adjust_learning_rate(optimizer, epoch - 1, args.lr, args.lr_DropFactor, args.lr_DropPeriod)

    learning_rate = optimizer.param_groups[0]["lr"]
    print_on_main(context, f"\n------- Epoch {epoch}/{args.num_epochs} \t\t lr = {learning_rate}")

    progress = tqdm(loader, desc="In training", leave=True, ncols=80, disable=not context.is_main_process)
    total_loss = 0.0
    total_l1 = 0.0

    for batch_index, (inputs, targets) in enumerate(progress):
        if args.recipe == "sgarnet":
            # Like the original, the cosine learning rate changes every iteration.
            set_cosine_learning_rate(optimizer, first_iteration + batch_index, total_iterations, args.lr, SGARNET_ETA_MIN)

        inputs = inputs.to(context.device)
        targets = targets.float().to(context.device)

        predictions = model(inputs)
        loss = criterion(predictions, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.recipe == "sgarnet":
            nn.utils.clip_grad_norm_(model.parameters(), SGARNET_GRAD_CLIP_NORM)
        optimizer.step()

        loss_value = loss.item()
        if args.recipe == "unet":
            l1_value = loss_value
            progress.set_postfix(loss=loss_value)
        else:
            l1_value = F.l1_loss(predictions.detach(), targets).item()
            progress.set_postfix(loss=loss_value, l1=l1_value)
        total_loss += loss_value
        total_l1 += l1_value

    epoch_loss = mean_across_processes(total_loss / len(loader), context)
    if args.recipe == "unet":
        print_on_main(context, f"\nTrain loss: {epoch_loss:.4f}")
        return epoch_loss, epoch_loss

    epoch_l1 = mean_across_processes(total_l1 / len(loader), context)
    print_on_main(context, f"\nTrain loss: {epoch_loss:.4f} \t\t Train L1: {epoch_l1:.4f}")
    return epoch_l1, epoch_loss


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


def create_optimizer(network: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    if args.recipe == "sgarnet":
        return optim.AdamW(network.parameters(), lr=args.lr, weight_decay=SGARNET_WEIGHT_DECAY, betas=SGARNET_BETAS)
    return optim.Adam(network.parameters(), lr=args.lr)


def run_training(args: argparse.Namespace, context: DistributedContext) -> LossStats:
    setup_seed(args.seed + context.rank)

    train_loader, train_sampler = create_train_loader(args, context)
    valid_loader = create_valid_loader(args, context)

    network = build_model(
        args.arch, in_channels=args.n_colors, out_channels=args.n_colors, **architecture_options(args.arch, args)
    )
    model_arch = copy.deepcopy(network)
    network = network.to(context.device)

    # Validation (and the choice of the _best checkpoint) always uses the L1 loss, so all recipes are compared alike.
    criterion = nn.L1Loss().to(context.device)
    training_criterion = SGARNetLoss().to(context.device) if args.recipe == "sgarnet" else criterion
    optimizer = create_optimizer(network, args)
    loss_stats: LossStats = {"train": [], "valid": [], "valid_expt": []}

    if args.pretrain:
        # weights_only=False: the file also stores 'args'. Only load checkpoint files you trust.
        checkpoint = torch.load(args.dir_pretrain, map_location=context.device, weights_only=False)
        load_checkpoint(checkpoint, network)
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_stats = dict(checkpoint["loss_stats"])

    loss_stats.setdefault("valid_epoch", list(range(1, len(loss_stats["valid"]) + 1)))
    if args.recipe != "unet":
        loss_stats.setdefault("train_objective", [])
    best_valid_loss = min(loss_stats["valid"]) if loss_stats["valid"] else float("inf")

    training_model, validation_model = prepare_models(network, args, context)

    print_on_main(
        context,
        f"===> Training {args.arch} with the {args.recipe} recipe: lr {args.lr}, batch size {args.batch_size}, "
        f"patch size {args.patch_size or 'full image'}, augment {args.augment}",
    )
    code_start = datetime.datetime.now()
    timer = Timer()

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        epoch_start = timer.t()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        loss_train, objective_train = train_one_epoch(
            epoch, training_model, train_loader, training_criterion, optimizer, args, context
        )
        loss_stats["train"].append(loss_train)
        if args.recipe != "unet":
            loss_stats["train_objective"].append(objective_train)

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

    # Stop now, before any GPU process starts, if a setting of the architecture is missing.
    architecture_options(args.arch, args)

    # Create the checkpoint folder if it does not exist yet
    os.makedirs(args.save_path, exist_ok=True)

    if args.recipe == "sgarnet":
        # Download the VGG19 weights once here, so the GPU processes do not all download them at the same time.
        load_vgg19_weights()

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
