"""Trains on tiny random images on the CPU, to catch breakages of train.py in a few minutes.

Run:  python smoke_test.py

Two quick checks first: the validation PSNR / SSIM equal skimage's, and a small unet can memorize one image.
Then it covers unet, unet_v2, a small unet_edsr and unet_v2_edsr, --amp, the config file (also its type
checks), resuming with --pretrain, --start_epoch, --save_model_arch, the setting checks, the files a run writes
(checkpoints, train.log, plot), the run folders (a new run never touches an earlier run's files),
evaluate.py with --config, and, on a computer without a GPU, a run with 2 processes (DistributedDataParallel).
sgarnet is left out: its loss downloads the VGG19 weights (about 550 MB).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def make_images(folder: Path, count: int, seed: int) -> None:
    """Writes count random 64 x 64 grayscale PNGs named img_0001.png, img_0002.png, ... into folder."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for number in range(1, count + 1):
        cv2.imwrite(str(folder / f"img_{number:04d}.png"), rng.integers(0, 256, (64, 64), dtype=np.uint8))


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
        prediction_image, target_image = tensor2np(predictions[index]), tensor2np(targets[index])
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
        edsr_args = load_checkpoint(edsr / "net_epoch_1.pth.tar")["args"]
        assert edsr_args.edsr_n_feats == 16 and edsr_args.edsr_res_scale == 0.1 and edsr_args.amp, edsr_args  # type: ignore[attr-defined]

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
        assert load_checkpoint(unet_v2_edsr / "net_epoch_1.pth.tar")["args"].edsr_n_feats == 16  # type: ignore[attr-defined]

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
