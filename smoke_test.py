"""Trains on tiny random images on the CPU, to catch breakages of train.py in about a minute.

Run:  python smoke_test.py

Covers unet and a small unet_edsr, the config file, resuming with --pretrain, the batch size check
and the files a run writes. sgarnet is left out: its loss downloads the VGG19 weights (about 550 MB).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


def make_images(folder: Path, count: int, seed: int) -> None:
    """Writes count random 64 x 64 grayscale PNGs named img_0001.png, img_0002.png, ... into folder."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for number in range(1, count + 1):
        cv2.imwrite(str(folder / f"img_{number:04d}.png"), rng.integers(0, 256, (64, 64), dtype=np.uint8))


def train(*options: str) -> subprocess.CompletedProcess[str]:
    """Runs train.py with options and returns the finished process with its output."""
    print("$ python train.py", " ".join(options), flush=True)
    return subprocess.run([sys.executable, "train.py", *options], capture_output=True, text=True)


def train_ok(*options: str) -> None:
    """Runs train.py with options and stops the test with its output if it fails."""
    run = train(*options)
    if run.returncode != 0:
        sys.exit(f"train.py failed:\n{run.stdout}\n{run.stderr}")


def load_checkpoint(path: Path) -> dict[str, object]:
    """Loads a checkpoint written by train.py."""
    assert path.is_file(), f"missing {path}"
    checkpoint: dict[str, object] = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint


def main() -> None:
    """Trains, resumes and checks the results; exits with an error message on the first failure."""
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
        )

        unet = root / "unet"
        train_ok("--config", str(config), "--arch", "unet", "--num_epochs", "2", "--save_path", str(unet))
        stats = load_checkpoint(unet / "net_epoch_2.pth.tar")["loss_stats"]
        assert isinstance(stats, dict)
        assert len(stats["train"]) == 2 and stats["valid_epoch"] == [1, 2], stats
        assert len(stats["valid_psnr"]) == len(stats["valid_ssim"]) == 2, "PSNR and SSIM are not saved"
        assert (unet / "net_best.pth.tar").is_file(), "no net_best.pth.tar"
        assert not list(unet.glob("net_epoch_*_best.pth.tar")), "old-style _best checkpoints were written"
        assert (unet / "training_progress.png").is_file(), "no training_progress.png"

        # Resuming continues after the checkpoint's epoch without --start-epoch.
        train_ok("--config", str(config), "--arch", "unet", "--num_epochs", "3", "--save_path", str(unet),
                 "--pretrain", "--dir_pretrain", str(unet / "net_epoch_2.pth.tar"))
        checkpoint = load_checkpoint(unet / "net_epoch_3.pth.tar")
        stats = checkpoint["loss_stats"]
        assert isinstance(stats, dict)
        assert checkpoint["epoch"] == 3 and len(stats["train"]) == 3 and stats["valid_epoch"] == [1, 2, 3], stats

        edsr = root / "unet_edsr"
        train_ok("--config", str(config), "--arch", "unet_edsr", "--num_epochs", "1", "--save_path", str(edsr))
        assert load_checkpoint(edsr / "net_epoch_1.pth.tar")["args"].edsr_n_feats == 16  # type: ignore[attr-defined]

        # A batch size that does not split across the GPUs stops before any GPU process starts.
        run = train("--config", str(config), "--num_GPUs", "2", "--val_batch_size", "1", "--save_path", str(root / "x"))
        assert run.returncode == 2 and "--val_batch_size 1" in run.stderr, run.stderr

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
