#!/usr/bin/env bash
# Quick look at the trained models: scores, prediction images and U-Net vs SGARNet comparisons.
#
# Usage:
#     bash run_evaluation.sh             # every model that has a *_best checkpoint
#     bash run_evaluation.sh unet        # only the UNet (or: sgarnet, both)
#
# Output (replaced on every run; it can always be made again from the checkpoints):
#     /workspace/results/summary.txt                  scores of the baseline, UNet and SGARNet
#     /workspace/results/unet/                        scores.txt, pred/ (predictions), compare/ (input | UNet | target)
#     /workspace/results/sgarnet/                     the same for SGARNet
#     /workspace/results/compare_unet_vs_sgarnet/     input | U-Net | SGARNet | target
#     /workspace/results.zip                          all of the above, without checkpoints
#
# It uses the newest *_best.pth.tar of each model, and also works while a training is still running
# (it then shares the GPU). For the final export with checkpoints, loss curves, ONNX and TensorRT,
# use after_training.sh.

set -eo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-$WORKSPACE/venv}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
CKPT_DIR="${CKPT_DIR:-$WORKSPACE/checkpoints}"
SGARNET_CKPT_DIR="${SGARNET_CKPT_DIR:-$WORKSPACE/checkpoints_sgarnet}"
RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE/results}"
NUM_WORKERS="${NUM_WORKERS:-4}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$VENV/bin/python"
DIR_Z_VALID="$DATA_DIR/HR_Valid"
DIR_X_VALID="$DATA_DIR/sim_MCF_Valid"

fail() { echo; echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "================ $* ================"; }

MODE="${1:-auto}"
case "$MODE" in
    auto|unet|sgarnet|both) ;;
    *) echo "Usage: bash $0 [unet|sgarnet|both]" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------- checks
[ -x "$PY" ] || fail "No Python found at $PY. Run setup_pod.sh first."
[ -d "$DIR_Z_VALID" ] || fail "Missing folder $DIR_Z_VALID"
[ -d "$DIR_X_VALID" ] || fail "Missing folder $DIR_X_VALID"

newest_best() { ls -v "$1"/*_best.pth.tar 2> /dev/null | tail -n 1 || true; }
UNET_BEST=$(newest_best "$CKPT_DIR")
SG_BEST=$(newest_best "$SGARNET_CKPT_DIR")

EVAL_UNET=0
EVAL_SGARNET=0
case "$MODE" in
    auto)
        [ -n "$UNET_BEST" ] && EVAL_UNET=1
        [ -n "$SG_BEST" ] && EVAL_SGARNET=1
        [ "$EVAL_UNET" -eq 1 ] || [ "$EVAL_SGARNET" -eq 1 ] || fail "No *_best.pth.tar in $CKPT_DIR or $SGARNET_CKPT_DIR" ;;
    unet) EVAL_UNET=1 ;;
    sgarnet) EVAL_SGARNET=1 ;;
    both) EVAL_UNET=1; EVAL_SGARNET=1 ;;
esac
[ "$EVAL_UNET" -eq 0 ] || [ -n "$UNET_BEST" ] || fail "No *_best.pth.tar in $CKPT_DIR"
[ "$EVAL_SGARNET" -eq 0 ] || [ -n "$SG_BEST" ] || fail "No *_best.pth.tar in $SGARNET_CKPT_DIR"

rm -rf "$RESULTS_DIR" "$RESULTS_DIR.zip"
mkdir -p "$RESULTS_DIR"
exec > >(tee "$RESULTS_DIR/run_evaluation_output.txt") 2>&1
cd "$REPO_DIR"

[ "$EVAL_UNET" -eq 1 ] && echo "UNet checkpoint:    $UNET_BEST"
[ "$EVAL_SGARNET" -eq 1 ] && echo "SGARNet checkpoint: $SG_BEST"
if pgrep -f "python.*train\.py" > /dev/null; then
    echo "NOTE: a training is still running. This evaluation shares the GPU and uses the best checkpoint saved so far."
fi

# ---------------------------------------------------------------- evaluation
evaluate_model() {   # evaluate_model ARCH CHECKPOINT
    local arch="$1" checkpoint="$2"
    mkdir -p "$RESULTS_DIR/$arch"
    echo "Checkpoint: $checkpoint" > "$RESULTS_DIR/$arch/scores.txt"
    "$PY" evaluate.py --checkpoint "$checkpoint" --arch "$arch" --dir_ZValid "$DIR_Z_VALID" --dir_XValid "$DIR_X_VALID" \
        --num_workers "$NUM_WORKERS" 2>&1 | tee -a "$RESULTS_DIR/$arch/scores.txt"
    "$PY" save_test_preds.py "$checkpoint" "$DIR_Z_VALID" "$DIR_X_VALID" "$RESULTS_DIR/$arch" "$arch"
}

step "Baseline (raw input vs target, no network)"
"$PY" baseline.py "$DIR_Z_VALID" "$DIR_X_VALID" 2>&1 | tee "$RESULTS_DIR/baseline.txt"

if [ "$EVAL_UNET" -eq 1 ]; then
    step "UNet: scores and prediction images"
    evaluate_model unet "$UNET_BEST"
fi

if [ "$EVAL_SGARNET" -eq 1 ]; then
    step "SGARNet: scores and prediction images"
    evaluate_model sgarnet "$SG_BEST"
fi

if [ "$EVAL_UNET" -eq 1 ] && [ "$EVAL_SGARNET" -eq 1 ]; then
    step "U-Net vs SGARNet comparison images"
    "$PY" compare_predictions.py "$DIR_Z_VALID" "$DIR_X_VALID" "$RESULTS_DIR/unet/pred" "$RESULTS_DIR/sgarnet/pred" \
        "$RESULTS_DIR/compare_unet_vs_sgarnet"
fi

step "Summary"
"$PY" - "$RESULTS_DIR" <<'EOF' | tee "$RESULTS_DIR/summary.txt"
import os
import re
import sys

results_dir = sys.argv[1]

def read(file_name, pattern):
    path = os.path.join(results_dir, file_name)
    if not os.path.exists(path):
        return None
    text = open(path).read()
    match = re.search(pattern, text)
    checkpoint = re.search(r"Checkpoint: (.*)", text)
    return (float(match.group(1)), float(match.group(2)), checkpoint.group(1) if checkpoint else "")

baseline = read("baseline.txt", r"PSNR ([\d.]+)\s+SSIM ([\d.]+)")
scores = {name: read(f"{name}/scores.txt", r"PSNR: ([\d.]+)\s+SSIM: ([\d.]+)") for name in ("unet", "sgarnet")}

print(f"Baseline (no network): PSNR {baseline[0]:.4f}  SSIM {baseline[1]:.4f}")
for name, label in (("unet", "UNet"), ("sgarnet", "SGARNet")):
    if scores[name]:
        psnr, ssim, checkpoint = scores[name]
        print(f"{label + ':':<22} PSNR {psnr:.4f}  SSIM {ssim:.4f}   ({os.path.basename(checkpoint)})")
if scores["unet"] and scores["sgarnet"]:
    print(f"SGARNet minus UNet:    PSNR {scores['sgarnet'][0] - scores['unet'][0]:+.4f} dB  "
          f"SSIM {scores['sgarnet'][1] - scores['unet'][1]:+.4f}")
EOF

"$PY" -c "import shutil, os; shutil.make_archive('$RESULTS_DIR', 'zip', os.path.dirname('$RESULTS_DIR'), os.path.basename('$RESULTS_DIR'))"
echo
echo "Created $RESULTS_DIR.zip ($(du -h "$RESULTS_DIR.zip" | cut -f1))"
