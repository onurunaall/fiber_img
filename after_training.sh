#!/usr/bin/env bash
# Collects everything worth keeping after train.py has finished, into one zip file.
#
# Usage (on the RunPod pod, after training has ended):
#     bash /workspace/tu_dresden/after_training.sh
#     bash /workspace/tu_dresden/after_training.sh --no-optional   # skip torch.compile and TensorRT
#
# Safe steps (always run, the script stops at the first error):
#     baseline + UNet scores, prediction images, loss curve, checkpoints, ONNX export + ONNX check
#     -> /workspace/export.zip
# Optional steps (run unless --no-optional; a failure is reported but does not stop the script):
#     compiled evaluation, 2-epoch compiled training test, TensorRT build + evaluation
#     -> /workspace/export.zip is rebuilt with their results
#
# All paths can be changed with environment variables, e.g. DATA_DIR=/workspace/mydata bash after_training.sh

set -eo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-$WORKSPACE/venv}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
CKPT_DIR="${CKPT_DIR:-$WORKSPACE/checkpoints}"
TRAIN_LOG="${TRAIN_LOG:-$WORKSPACE/train_log.txt}"
OUT_DIR="${OUT_DIR:-$WORKSPACE/export}"
TRT_DIR="${TRT_DIR:-$WORKSPACE/trt}"
COMPILE_TEST_DIR="${COMPILE_TEST_DIR:-$WORKSPACE/checkpoints_compile_test}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

RUN_OPTIONAL=1
for arg in "$@"; do
    case "$arg" in
        --no-optional) RUN_OPTIONAL=0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$VENV/bin/python"
DIR_Z_VALID="$DATA_DIR/HR_Valid"
DIR_X_VALID="$DATA_DIR/sim_MCF_Valid"

fail() { echo; echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "================ $* ================"; }

# ---------------------------------------------------------------- checks
step "Checking the setup"
[ -x "$PY" ] || fail "No Python found at $PY. Create the environment first (Part B)."
command -v uv > /dev/null || fail "uv is not installed. Run: pip install uv"
[ -d "$DIR_Z_VALID" ] || fail "Missing folder $DIR_Z_VALID"
[ -d "$DIR_X_VALID" ] || fail "Missing folder $DIR_X_VALID"
[ -d "$CKPT_DIR" ] || fail "Missing folder $CKPT_DIR. Has training run?"
if pgrep -f "python.*train\.py" > /dev/null; then
    fail "train.py is still running. Wait until it has finished, then run this script again."
fi

BEST=$(ls -v "$CKPT_DIR"/*_best.pth.tar 2> /dev/null | tail -n 1 || true)
LAST=$(ls -v "$CKPT_DIR"/net_epoch_*.pth.tar 2> /dev/null | grep -v _best | tail -n 1 || true)
[ -n "$BEST" ] || fail "No *_best.pth.tar file in $CKPT_DIR"
[ -n "$LAST" ] || fail "No net_epoch_N.pth.tar file in $CKPT_DIR"

read -r H W < <("$PY" -c "
import cv2, glob, sys
files = sorted(glob.glob('$DIR_X_VALID/*.png'))
if not files:
    sys.exit('no .png files in $DIR_X_VALID')
img = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
print(img.shape[0], img.shape[1])
")
[ -n "$H" ] && [ -n "$W" ] || fail "Could not read the image size from $DIR_X_VALID"

HAS_CUDA=$("$PY" -c "import torch; print(int(torch.cuda.is_available()))")

echo "Best checkpoint: $BEST"
echo "Last checkpoint: $LAST"
echo "Image size:      ${H} x ${W}"
echo "CUDA available:  $HAS_CUDA"

# Never overwrite an earlier export: move it aside.
if [ -e "$OUT_DIR" ]; then
    OLD="${OUT_DIR}_previous_$(date +%Y%m%d_%H%M%S)"
    mv "$OUT_DIR" "$OLD"
    echo "Moved the existing $OUT_DIR to $OLD"
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/after_training_output.txt") 2>&1

cd "$REPO_DIR"

make_zip() {
    rm -f "$OUT_DIR.zip"
    "$PY" -c "import shutil, os; shutil.make_archive('$OUT_DIR', 'zip', os.path.dirname('$OUT_DIR'), os.path.basename('$OUT_DIR'))"
    echo "Created $OUT_DIR.zip ($(du -h "$OUT_DIR.zip" | cut -f1))"
}

# ---------------------------------------------------------------- safe steps
step "1/8 Copying the training log and checkpoints"
if [ -f "$TRAIN_LOG" ]; then
    cp "$TRAIN_LOG" "$OUT_DIR/"
else
    echo "WARNING: $TRAIN_LOG not found, skipping it."
fi
cp "$BEST" "$LAST" "$OUT_DIR/"

step "2/8 Baseline (raw input vs target, no UNet)"
"$PY" baseline.py "$DIR_Z_VALID" "$DIR_X_VALID" 2>&1 | tee "$OUT_DIR/baseline.txt"

step "3/8 UNet scores and speed (PyTorch)"
"$PY" evaluate.py --checkpoint "$BEST" --dir_ZValid "$DIR_Z_VALID" --dir_XValid "$DIR_X_VALID" \
    --num_workers "$NUM_WORKERS" 2>&1 | tee "$OUT_DIR/eval_pytorch.txt"

step "4/8 Prediction images"
"$PY" save_test_preds.py "$BEST" "$DIR_Z_VALID" "$DIR_X_VALID" "$OUT_DIR/predictions"

step "5/8 Loss curve"
"$PY" - "$LAST" "$OUT_DIR" <<'EOF'
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

checkpoint_path, out_dir = sys.argv[1:3]
stats = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["loss_stats"]
train_epochs = list(range(1, len(stats["train"]) + 1))

with open(f"{out_dir}/loss_values.txt", "w") as f:
    f.write("epoch  train_loss\n")
    for epoch, loss in zip(train_epochs, stats["train"]):
        f.write(f"{epoch}  {loss:.6f}\n")
    f.write("\nepoch  valid_loss\n")
    for epoch, loss in zip(stats["valid_epoch"], stats["valid"]):
        f.write(f"{epoch}  {loss:.6f}\n")

plt.figure(figsize=(10, 5))
plt.plot(train_epochs, stats["train"], label="Train")
plt.plot(stats["valid_epoch"], stats["valid"], label="Valid")
plt.xlabel("Epoch")
plt.ylabel("L1 loss")
plt.yscale("log")
plt.legend()
plt.title("Training progress")
plt.savefig(f"{out_dir}/loss_curve.png", dpi=150, bbox_inches="tight")
print(f"Saved {out_dir}/loss_curve.png and {out_dir}/loss_values.txt")
EOF

step "6/8 Software and GPU versions"
uv pip freeze --python "$PY" > "$OUT_DIR/installed_packages.txt"
if command -v nvidia-smi > /dev/null; then
    nvidia-smi > "$OUT_DIR/nvidia_smi.txt" || true
fi

step "7/8 ONNX export (fp32 and fp16) and check against PyTorch"
uv pip install --python "$PY" onnx onnxscript onnxruntime
for P in fp32 fp16; do
    "$PY" onnx_export.py --checkpoint "$BEST" --output "$OUT_DIR/unet_$P.onnx" \
        --height "$H" --width "$W" --precision "$P"
done
"$PY" - "$BEST" "$OUT_DIR" "$DIR_Z_VALID" "$DIR_X_VALID" <<'EOF' 2>&1 | tee "$OUT_DIR/onnx_check.txt"
import sys
import numpy as np
import onnxruntime
import torch
from imageDatastore import imageDatastore
from model_factory import load_trained_model

checkpoint, out_dir, dir_z, dir_x = sys.argv[1:5]
model = load_trained_model(checkpoint, "unet", n_colors=1)
dataset = imageDatastore(dir_z, dir_x)
count = min(20, len(dataset))
for precision in ("fp32", "fp16"):
    onnx_file = f"{out_dir}/unet_{precision}.onnx"
    session = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    largest_difference = 0.0
    for index in range(count):
        inputs = dataset[index][0].unsqueeze(0)
        with torch.no_grad():
            torch_output = model(inputs).numpy()
        onnx_output = session.run(None, {"input": inputs.numpy()})[0]
        largest_difference = max(largest_difference, float(np.abs(torch_output - onnx_output).max()))
    verdict = "OK" if largest_difference < 0.01 else "TOO LARGE, something is wrong"
    print(f"{onnx_file}: {count} images, largest difference ONNX vs PyTorch: {largest_difference:.6f} ({verdict})")
EOF

step "8/8 Summary"
"$PY" - "$OUT_DIR" <<'EOF' | tee "$OUT_DIR/summary.txt"
import re
import sys

out_dir = sys.argv[1]

def read_scores(file_name, pattern):
    text = open(f"{out_dir}/{file_name}").read()
    match = re.search(pattern, text)
    return (float(match.group(1)), float(match.group(2))) if match else None

baseline = read_scores("baseline.txt", r"PSNR ([\d.]+)\s+SSIM ([\d.]+)")
unet = read_scores("eval_pytorch.txt", r"PSNR: ([\d.]+)\s+SSIM: ([\d.]+)")
print(f"Baseline (no UNet): PSNR {baseline[0]:.4f}  SSIM {baseline[1]:.4f}")
print(f"UNet:               PSNR {unet[0]:.4f}  SSIM {unet[1]:.4f}")
print(f"UNet better PSNR than baseline: {'YES' if unet[0] > baseline[0] else 'NO'}")
print(f"UNet better SSIM than baseline: {'YES' if unet[1] > baseline[1] else 'NO'}")
print("Also look at predictions/compare/*.png: input | prediction | target")
EOF

step "Creating the zip with the safe results"
make_zip

if [ "$RUN_OPTIONAL" -eq 0 ]; then
    echo
    echo "Done. Download $OUT_DIR.zip (optional steps were skipped)."
    exit 0
fi

# ---------------------------------------------------------------- optional steps
RESULTS=()
run_optional() {
    local name="$1"
    shift
    step "OPTIONAL: $name"
    "$@"
    case $? in
        0) RESULTS+=("OK       $name") ;;
        2) RESULTS+=("SKIPPED  $name (no CUDA GPU)") ;;
        *) RESULTS+=("FAILED   $name (see the output above)") ;;
    esac
}

compiled_evaluation() {
    "$PY" evaluate.py --checkpoint "$BEST" --compile --dir_ZValid "$DIR_Z_VALID" --dir_XValid "$DIR_X_VALID" \
        --num_workers "$NUM_WORKERS" 2>&1 | tee "$OUT_DIR/eval_pytorch_compiled.txt"
}

compiled_training_test() {
    if [ "$HAS_CUDA" != "1" ]; then echo "No CUDA GPU, skipping."; return 2; fi
    rm -rf "$COMPILE_TEST_DIR"
    MPLBACKEND=Agg "$PY" train.py --compile --num_GPUs "$NUM_GPUS" --num_epochs 2 --batch_size 4 \
        --num_workers "$NUM_WORKERS" --save_path "$COMPILE_TEST_DIR" \
        --dir_ZTrain "$DATA_DIR/HR_Train" --dir_XTrain "$DATA_DIR/sim_MCF_Train" \
        --dir_ZValid "$DIR_Z_VALID" --dir_XValid "$DIR_X_VALID" 2>&1 | tee "$OUT_DIR/train_compile_test.txt"
}

tensorrt_steps() {
    if [ "$HAS_CUDA" != "1" ]; then echo "No CUDA GPU, skipping."; return 2; fi

    # Keep torch and its CUDA libraries exactly as they are: if TensorRT needs other
    # versions, the install fails instead of silently changing torch.
    local constraints="$OUT_DIR/.torch_constraints.txt"
    grep -iE '^(torch|torchvision|triton|nvidia-|cuda-)' "$OUT_DIR/installed_packages.txt" > "$constraints" || true
    uv pip install --python "$PY" tensorrt --constraint "$constraints" || return 1
    rm -f "$constraints"
    "$PY" -c "import tensorrt; print('TensorRT', tensorrt.__version__)" || return 1

    mkdir -p "$TRT_DIR"
    local P
    for P in fp32 fp16; do
        "$PY" tensorrt_engine.py --onnx "$OUT_DIR/unet_$P.onnx" --engine "$TRT_DIR/unet_$P.engine" \
            --max_batch_size 4 || return 1
        "$PY" evaluate.py --engine "$TRT_DIR/unet_$P.engine" --dir_ZValid "$DIR_Z_VALID" --dir_XValid "$DIR_X_VALID" \
            --num_workers "$NUM_WORKERS" 2>&1 | tee "$OUT_DIR/eval_tensorrt_$P.txt" || return 1
    done
}

set +e
run_optional "torch.compile evaluation" compiled_evaluation
run_optional "torch.compile training test (2 epochs)" compiled_training_test
run_optional "TensorRT fp32 + fp16" tensorrt_steps
set -e

step "Optional steps"
printf '%s\n' "${RESULTS[@]}" | tee "$OUT_DIR/optional_steps.txt"

step "Creating the final zip"
make_zip

echo
echo "Done. Download $OUT_DIR.zip, check it opens on your PC, then terminate the pod."
