"""Trains on tiny random images on the CPU, to catch breakages of train.py in a few minutes.

Run:  python smoke_test.py

Two quick checks first: the validation PSNR / SSIM equal skimage's, and a small unet can memorize one image.
Then it covers unet, unet_v2, a small unet_edsr and unet_v2_edsr, --amp, the config file (also its type
checks), resuming with --pretrain (and that it stops when the model settings differ from the checkpoint's),
--start_epoch, --save_model_arch, the setting checks, the files a run writes (checkpoints, train.log, plot), the model
description in the checkpoints (also for old checkpoints without one), the run folders (a new run never touches an
earlier run's files), evaluate.py with --config and without --arch, transfer learning with --finetune_from (and the
checks against a forgotten --pretrain), the fiber's reference image as a 2nd input channel (--reference_image), and,
on a computer without a GPU, a run with 2 processes (DistributedDataParallel).
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def make_images(folder: Path, count: int, seed: int, darkest: int = 0) -> None:
    """Writes count random 64 x 64 grayscale PNGs named img_0001.png, img_0002.png, ... into folder.

    Their pixel values are random between darkest and 255.
    """
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for number in range(1, count + 1):
        cv2.imwrite(str(folder / f"img_{number:04d}.png"), rng.integers(darkest, 256, (64, 64), dtype=np.uint8))


def run_script(script: str, *options: str) -> subprocess.CompletedProcess[str]:
    """Runs a script of this folder with options and returns the finished process with its output."""
    print(f"$ python {script}", " ".join(options), flush=True)
    return subprocess.run([sys.executable, script, *options], capture_output=True, text=True)


def train(*options: str) -> subprocess.CompletedProcess[str]:
    """Runs train.py with options and returns the finished process with its output."""
    return run_script("train.py", *options)


def train_ok(*options: str) -> None:
    """Runs train.py with options and stops the test with its output if it fails."""
    run = train(*options)
    if run.returncode != 0:
        sys.exit(f"train.py failed:\n{run.stdout}\n{run.stderr}")


def train_fails(expected_message: str, *options: str) -> None:
    """Runs train.py with options and checks that it stops with an error that contains expected_message."""
    run = train(*options)
    assert run.returncode != 0 and expected_message in run.stderr, run.stderr


def load_checkpoint(path: Path) -> dict[str, object]:
    """Loads a checkpoint written by train.py."""
    assert path.is_file(), f"missing {path}"
    checkpoint: dict[str, object] = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint


def run_folders(parent: Path) -> list[Path]:
    """Returns the run folders train.py made in parent, oldest first (their names start with date and time)."""
    return sorted(path for path in parent.iterdir() if path.is_dir())


def check_metrics() -> None:
    """The batched PSNR / SSIM of validation.py give the same numbers as skimage (utils.compute_psnr / compute_ssim)."""
    from utils import compute_psnr, compute_ssim, tensor2np
    from validation import psnr_ssim_8bit

    generator = torch.Generator().manual_seed(0)
    targets = torch.rand(3, 1, 48, 64, generator=generator)
    # Values below 0 and above 1 too, like a real network output.
    predictions = targets + 0.2 * torch.randn(targets.shape, generator=generator)
    psnr, ssim = psnr_ssim_8bit(predictions, targets)
    for index in range(3):
        before = predictions[index].clone()
        prediction_image, target_image = tensor2np(predictions[index]), tensor2np(targets[index])
        assert torch.equal(predictions[index], before), "tensor2np changed the tensor it was given"
        assert abs(psnr[index].item() - compute_psnr(target_image, prediction_image)) < 1e-9, "PSNR differs from skimage"
        assert abs(ssim[index].item() - compute_ssim(prediction_image, target_image)) < 1e-9, "SSIM differs from skimage"
    print("Validation PSNR / SSIM equal skimage's.", flush=True)


def check_memorize() -> None:
    """A small unet must learn to clean up one image: the most basic sign that the model and its gradients work."""
    from model_factory import build_model

    torch.manual_seed(0)
    y, x = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing="ij")
    target = (0.5 + 0.4 * torch.sin(12 * x) * torch.cos(9 * y)).view(1, 1, 64, 64)
    inputs = (F.avg_pool2d(target, 5, 1, 2) + 0.05 * torch.randn_like(target)).clamp(0, 1)  # blurred and noisy
    model = build_model("unet", 1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with torch.no_grad():
        first_loss = F.l1_loss(model(inputs), target).item()
    for _ in range(300):
        loss = F.l1_loss(model(inputs), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Measured with 3 seeds: 23 to 33 times lower after 300 steps, and 2 to 7 times below the input's own error.
    input_loss = F.l1_loss(inputs, target).item()
    assert loss.item() < first_loss / 10 and loss.item() < input_loss, (first_loss, loss.item(), input_loss)
    print(f"A unet memorizes one image: L1 {first_loss:.4f} -> {loss.item():.4f}.", flush=True)


def check_reference_follows_input(root: Path, reference: Path) -> None:
    """With a reference image, random crops and flips move it exactly like the input image."""
    from imageDatastore import imageDatastore

    # The only input image and its target are copies of the reference image, so both channels of X and Z must
    # stay equal, whatever crop and flips are drawn.
    image = cv2.imread(str(reference), cv2.IMREAD_UNCHANGED)
    for name in ("reference_X", "reference_Z"):
        (root / name).mkdir()
        cv2.imwrite(str(root / name / "img_0001.png"), image)
    dataset = imageDatastore(str(root / "reference_Z"), str(root / "reference_X"), patches=True, patch_size=32,
                             hflip=True, vflip=True, rotate=True, reference_image=str(reference))
    for _ in range(20):
        x, z = dataset[0]
        assert x.shape == (2, 32, 32) and z.shape == (1, 32, 32), (x.shape, z.shape)
        assert torch.equal(x[0], x[1]) and torch.equal(x[0], z[0]), "the reference image was not cropped and flipped like the input"
    print("The reference image is cropped and flipped like the input image.", flush=True)


def main() -> None:
    """Trains, resumes and checks the results; exits with an error message on the first failure."""
    check_metrics()
    check_memorize()

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for name, count, seed in [("HR_Train", 8, 1), ("sim_MCF_Train", 8, 2), ("HR_Valid", 4, 3), ("sim_MCF_Valid", 4, 4)]:
            make_images(root / name, count, seed)

        config = root / "smoke.toml"
        config.write_text(
            f'dir_ZTrain = "{(root / "HR_Train").as_posix()}"\n'
            f'dir_XTrain = "{(root / "sim_MCF_Train").as_posix()}"\n'
            f'dir_ZValid = "{(root / "HR_Valid").as_posix()}"\n'
            f'dir_XValid = "{(root / "sim_MCF_Valid").as_posix()}"\n'
            "num_GPUs = 0\nnum_workers = 0\nbatch_size = 4\nval_batch_size = 2\nsave_everyEpoch = 1\n"
            "\n[unet_edsr]\nedsr_n_resblocks = 2\nedsr_n_feats = 16\n"
            "\n[unet_v2_edsr]\nedsr_n_resblocks = 2\nedsr_n_feats = 16\n"
        )

        runs = root / "runs"
        train_ok("--config", str(config), "--arch", "unet", "--num_epochs", "2", "--save_path", str(runs))
        [unet] = run_folders(runs)
        assert unet.name.endswith("_unet"), unet.name
        stats = load_checkpoint(unet / "net_epoch_2.pth.tar")["loss_stats"]
        assert isinstance(stats, dict)
        assert len(stats["train"]) == 2 and stats["valid_epoch"] == [1, 2], stats
        assert len(stats["valid_psnr"]) == len(stats["valid_ssim"]) == 2, "PSNR and SSIM are not saved"
        assert (unet / "net_best.pth.tar").is_file(), "no net_best.pth.tar"
        assert not list(unet.glob("net_epoch_*_best.pth.tar")), "old-style _best checkpoints were written"
        assert (unet / "training_progress.png").is_file(), "no training_progress.png"
        log = (unet / "train.log").read_text(encoding="utf-8")
        assert "Saving the files of this run" in log and log.count("Valid L1") == 2, log

        # evaluate.py takes the data folders from the same config and skips the options only train.py has.
        run = run_script("evaluate.py", "--checkpoint", str(unet / "net_best.pth.tar"), "--config", str(config),
                         "--warmup_iterations", "1", "--timed_iterations", "1")
        assert run.returncode == 0 and "Valid L1" in run.stdout, run.stdout + run.stderr

        # Resuming continues after the checkpoint's epoch without --start_epoch, in the checkpoint's folder.
        resume = ("--config", str(config), "--arch", "unet", "--num_epochs", "3", "--save_path", str(runs),
                  "--pretrain", "--dir_pretrain", str(unet / "net_epoch_2.pth.tar"))
        train_ok(*resume)
        assert run_folders(runs) == [unet], "resuming made a new run folder"
        checkpoint = load_checkpoint(unet / "net_epoch_3.pth.tar")
        stats = checkpoint["loss_stats"]
        assert isinstance(stats, dict)
        assert checkpoint["epoch"] == 3 and len(stats["train"]) == 3 and stats["valid_epoch"] == [1, 2, 3], stats
        log = (unet / "train.log").read_text(encoding="utf-8")
        assert "Starting at epoch 3" in log and log.count("Valid L1") == 3, "the resumed run did not add to train.log"

        # Resuming from net_epoch_2 again would overwrite net_epoch_3, so it stops.
        run = train(*resume)
        assert run.returncode != 0 and "net_epoch_3.pth.tar" in run.stderr, run.stderr
        assert "net_epoch_3.pth.tar" in (unet / "train.log").read_text(encoding="utf-8"), "the error is not in train.log"

        # The old spelling --start-epoch still works. With it, training goes into a new folder.
        train_ok(*resume[:-4], str(root / "start_epoch"), *resume[-3:], "--start-epoch", "3")  # another --save_path
        [start_epoch_run] = run_folders(root / "start_epoch")
        assert (start_epoch_run / "net_epoch_3.pth.tar").is_file(), "--start-epoch did not train epoch 3"

        # A new run gets its own folder and leaves the files of earlier runs alone.
        unet_files = {path.name: path.read_bytes() for path in unet.iterdir()}
        # Also covers --amp (bfloat16 works on the CPU too) and --edsr_res_scale.
        train_ok("--config", str(config), "--arch", "unet_edsr", "--num_epochs", "1", "--save_path", str(runs),
                 "--amp", "--edsr_res_scale", "0.1")
        [_, edsr] = run_folders(runs)
        assert edsr.name.endswith("_unet_edsr"), edsr.name
        assert {path.name: path.read_bytes() for path in unet.iterdir()} == unet_files, "a new run changed an earlier run's files"
        # The checkpoint says which model it holds, and stores the settings as plain text and numbers.
        edsr_checkpoint = load_checkpoint(edsr / "net_epoch_1.pth.tar")
        assert edsr_checkpoint["model"] == {"arch": "unet_edsr", "n_colors": 1, "in_channels": 1,
                                            "options": {"n_resblocks": 2, "n_feats": 16, "res_scale": 0.1}}, edsr_checkpoint
        edsr_args = edsr_checkpoint["args"]
        assert isinstance(edsr_args, dict) and edsr_args["amp"] is True, edsr_args
        # save_path stays the parent folder of the settings; the run has its own folder inside it.
        assert edsr_args["save_path"] == str(runs), edsr_args["save_path"]
        # Plain data only (no PosixPath, no argparse.Namespace): a Linux checkpoint opens on Windows.
        torch.load(edsr / "net_epoch_1.pth.tar", map_location="cpu", weights_only=True)

        # Resuming with other model settings than the checkpoint's stops with a clear message.
        resume_edsr = ("--config", str(config), "--num_epochs", "2", "--pretrain", "--dir_pretrain",
                       str(edsr / "net_epoch_1.pth.tar"))
        train_fails("holds the model", *resume_edsr, "--arch", "unet_edsr", "--edsr_res_scale", "1.0")
        train_fails("holds the model", *resume_edsr, "--arch", "unet_v2_edsr", "--edsr_res_scale", "0.1")

        # unet_v2 trains, and ends with 8 channels at full size (ConvOut: 8 -> 1 channels).
        # --save_model_arch saves the trained model object, with the same weights as the checkpoint.
        train_ok("--config", str(config), "--arch", "unet_v2", "--num_epochs", "1", "--save_path", str(runs),
                 "--save_model_arch")
        [unet_v2] = [path for path in run_folders(runs) if path.name.endswith("_unet_v2")]
        checkpoint = load_checkpoint(unet_v2 / "net_epoch_1.pth.tar")
        state = checkpoint["state_dict"]
        assert state["ConvOut.weight"].shape == (1, 8, 3, 3), state["ConvOut.weight"].shape  # type: ignore[index]
        saved_model_state = checkpoint["model_arch"].state_dict()  # type: ignore[attr-defined]
        assert all(torch.equal(saved_model_state[key], value) for key, value in state.items()), (  # type: ignore[attr-defined]
            "--save_model_arch did not save the trained weights")

        # unet_v2_edsr trains and gets the EDSR size of its [unet_v2_edsr] table.
        train_ok("--config", str(config), "--arch", "unet_v2_edsr", "--num_epochs", "1", "--save_path", str(runs))
        [unet_v2_edsr] = [path for path in run_folders(runs) if path.name.endswith("_unet_v2_edsr")]
        assert load_checkpoint(unet_v2_edsr / "net_epoch_1.pth.tar")["model"]["options"]["n_feats"] == 16  # type: ignore[index]

        # evaluate.py needs no --arch: the checkpoint says which model it holds.
        run = run_script("evaluate.py", "--checkpoint", str(unet_v2_edsr / "net_epoch_1.pth.tar"), "--config", str(config),
                         "--warmup_iterations", "1", "--timed_iterations", "1")
        assert run.returncode == 0 and "Valid L1" in run.stdout, run.stdout + run.stderr

        # A checkpoint from before the model description (no "model", settings as an argparse.Namespace) still loads.
        from model_factory import load_trained_model
        old_style = load_checkpoint(unet_v2_edsr / "net_epoch_1.pth.tar")
        del old_style["model"]
        old_style["args"] = argparse.Namespace(**old_style["args"])  # type: ignore[arg-type]
        torch.save(old_style, root / "old_style.pth.tar")
        model_state = load_trained_model(root / "old_style.pth.tar")[0].state_dict()
        assert all(torch.equal(model_state[key], value) for key, value in old_style["state_dict"].items()), (  # type: ignore[union-attr]
            "an old-style checkpoint did not load")

        # Transfer learning (--finetune_from): a new run on other data starts from the weights of a checkpoint only.
        # The new targets are bright, so the new validation loss is worse than the best of the first run.
        for name, count, seed, darkest in [("HR_Train_new", 8, 5, 200), ("sim_MCF_Train_new", 8, 6, 0),
                                           ("HR_Valid_new", 4, 7, 200), ("sim_MCF_Valid_new", 4, 8, 0)]:
            make_images(root / name, count, seed, darkest)
        new_data = ("--dir_ZTrain", str(root / "HR_Train_new"), "--dir_XTrain", str(root / "sim_MCF_Train_new"),
                    "--dir_ZValid", str(root / "HR_Valid_new"), "--dir_XValid", str(root / "sim_MCF_Valid_new"))
        source = unet / "net_epoch_3.pth.tar"
        # --lr 0: the weights cannot change, so the saved weights show which weights were loaded.
        train_ok("--config", str(config), *new_data, "--arch", "unet", "--num_epochs", "2", "--lr", "0",
                 "--save_path", str(root / "finetune"), "--finetune_from", str(source))
        [finetune] = run_folders(root / "finetune")
        source_checkpoint = load_checkpoint(source)
        checkpoint = load_checkpoint(finetune / "net_epoch_2.pth.tar")
        source_state, state = source_checkpoint["state_dict"], checkpoint["state_dict"]
        assert isinstance(source_state, dict) and isinstance(state, dict)
        assert all(torch.equal(state[key], value) for key, value in source_state.items()), (
            "--finetune_from did not load the weights of the checkpoint")
        source_stats, stats = source_checkpoint["loss_stats"], checkpoint["loss_stats"]
        assert isinstance(source_stats, dict) and isinstance(stats, dict)
        # Only the history of the new run, counted from epoch 1.
        assert len(stats["train"]) == 2 and stats["valid_epoch"] == [1, 2], stats
        assert min(stats["valid"]) > min(source_stats["valid"]), "the new data is not harder, so the next check shows nothing"
        assert (finetune / "net_best.pth.tar").is_file(), "fine-tuning on harder data saved no net_best.pth.tar"
        # The optimizer started new: 2 epochs x 2 batches = 4 steps (the checkpoint's optimizer had done 6).
        steps = next(iter(checkpoint["optimizer"]["state"].values()))["step"]  # type: ignore[index]
        assert int(steps) == 4, steps
        assert "Fine-tuning from the weights" in (finetune / "train.log").read_text(encoding="utf-8")

        # A forgotten --pretrain, a --start_epoch for a new run, and --finetune_from together with --pretrain all stop.
        train_fails("--dir_pretrain is given without --pretrain", "--config", str(config),
                    "--dir_pretrain", str(source), "--save_path", str(root / "x"))
        train_fails("--start_epoch only works together with --pretrain", "--config", str(config),
                    "--start_epoch", "2", "--save_path", str(root / "x"))
        train_fails("use only one of them", "--config", str(config), "--pretrain", "--dir_pretrain", str(source),
                    "--finetune_from", str(source), "--save_path", str(root / "x"))

        # The fiber's reference image: the model gets it as a 2nd input channel, and the checkpoint says so.
        reference = root / "white.png"
        cv2.imwrite(str(reference), np.random.default_rng(9).integers(0, 256, (64, 64), dtype=np.uint8))
        check_reference_follows_input(root, reference)
        train_ok("--config", str(config), "--arch", "unet", "--num_epochs", "1", "--save_path", str(root / "reference"),
                 "--reference_image", str(reference), "--patch_size", "32", "--augment", "flips+rot90")
        [reference_run] = run_folders(root / "reference")
        reference_checkpoint = load_checkpoint(reference_run / "net_epoch_1.pth.tar")
        assert reference_checkpoint["model"] == {"arch": "unet", "n_colors": 1, "in_channels": 2, "options": {}}, (
            reference_checkpoint["model"])
        first_layer = reference_checkpoint["state_dict"]["ConvIn.0.weight"]  # type: ignore[index]
        assert first_layer.shape == (8, 2, 3, 3), first_layer.shape
        # evaluate.py needs the reference image for this checkpoint, and refuses it for one trained without.
        evaluate = ("evaluate.py", "--config", str(config), "--warmup_iterations", "1", "--timed_iterations", "1")
        run = run_script(*evaluate, "--checkpoint", str(reference_run / "net_epoch_1.pth.tar"),
                         "--reference_image", str(reference))
        assert run.returncode == 0 and "Valid L1" in run.stdout, run.stdout + run.stderr
        run = run_script(*evaluate, "--checkpoint", str(reference_run / "net_epoch_1.pth.tar"))
        assert run.returncode != 0 and "trained with a reference image" in run.stderr, run.stderr
        run = run_script(*evaluate, "--checkpoint", str(unet / "net_best.pth.tar"), "--reference_image", str(reference))
        assert run.returncode != 0 and "trained without a reference image" in run.stderr, run.stderr
        # A checkpoint trained without the reference image cannot start a run with it: the first layer differs.
        train_fails("holds the model", "--config", str(config), "--arch", "unet", "--reference_image", str(reference),
                    "--finetune_from", str(unet / "net_best.pth.tar"), "--save_path", str(root / "x"))

        # A batch size that does not split across the GPUs stops before any GPU process starts.
        run = train("--config", str(config), "--num_GPUs", "2", "--val_batch_size", "1", "--save_path", str(root / "x"))
        assert run.returncode == 2 and "--val_batch_size 1" in run.stderr, run.stderr

        # Wrong settings stop with a clear message instead of a confusing crash later.
        wrong_type = root / "wrong_type.toml"
        wrong_type.write_text('compile = "false"\n' + config.read_text())  # text, not false: would switch compile on
        train_fails("must be true or false", "--config", str(wrong_type), "--save_path", str(root / "x"))
        train_fails("--valid_everyEpoch must be at least 1", "--config", str(config), "--valid_everyEpoch", "0",
                    "--save_path", str(root / "x"))
        train_fails("no data folder given for dir_ZTrain", "--save_path", str(root / "x"))
        train_fails("too few for one batch", "--config", str(config), "--batch_size", "16", "--save_path", str(root / "x"))

        # 2 processes (DistributedDataParallel). Only on a computer without a GPU, where they run on the CPU:
        # on a GPU machine (e.g. Kai) this would take 2 GPUs that other people may be using.
        if torch.cuda.is_available():
            print("Skipped the 2-process run: this computer has a GPU.")
        else:
            train_ok("--config", str(config), "--num_GPUs", "2", "--num_epochs", "1", "--save_path", str(root / "ddp"))
            [ddp] = run_folders(root / "ddp")
            assert (ddp / "net_epoch_1.pth.tar").is_file(), "the 2-process run saved no checkpoint"
            assert "Valid L1" in (ddp / "train.log").read_text(encoding="utf-8"), "the 2-process run did not validate"

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
