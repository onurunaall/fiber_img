#!/usr/bin/env bash
# One-time setup of a fresh RunPod pod: tools, Python environment, data and sanity checks.
#
# Usage (after cloning the repo into /workspace and uploading /workspace/data.zip):
#     bash /workspace/tu_dresden/setup_pod.sh
#
# Steps (each one is skipped when it is already done, so the script can be run again):
#     zip/unzip -> uv -> venv (Python 3.12) -> Python packages -> GPU check -> UNet and SGARNet test runs
#     -> unzip /workspace/data.zip -> data check -> baseline score
# Then train with:  bash run_training.sh both
#
# All paths can be changed with environment variables, e.g. DATA_ZIP=/workspace/mydata.zip bash setup_pod.sh
# FORCE_UNZIP=1 unzips data.zip again even if the data folder already exists.

set -eo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-$WORKSPACE/venv}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
DATA_ZIP="${DATA_ZIP:-$WORKSPACE/data.zip}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
FORCE_UNZIP="${FORCE_UNZIP:-0}"
PACKAGES=(torch torchvision numpy opencv-python-headless scikit-image albumentations lpips piq pytorch-msssim matplotlib tqdm)
DATA_FOLDERS=(HR_Train sim_MCF_Train HR_Valid sim_MCF_Valid)

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$VENV/bin/python"

fail() { echo; echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "================ $* ================"; }

mkdir -p "$WORKSPACE"
exec > >(tee -a "$WORKSPACE/setup_log.txt") 2>&1
cd "$REPO_DIR"

step "1/8 Code"
echo "Repository: $REPO_DIR"
git -C "$REPO_DIR" log -1 --format="Branch $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD), commit %h: %s" || true
[ -f "$REPO_DIR/model_SGARNet.py" ] || fail "model_SGARNet.py is missing: this checkout has no SGARNet. Clone the branch that has it, e.g.
    git clone -b cl/zealous-clarke-kwd62x https://github.com/onurunaall/fiber_img.git $REPO_DIR"

step "2/8 System tools (zip, unzip) and uv"
if command -v zip > /dev/null && command -v unzip > /dev/null; then
    echo "zip and unzip are installed."
else
    apt-get update
    apt-get install -y zip unzip
fi
command -v uv > /dev/null || pip install uv
uv --version

step "3/8 Python environment"
if [ -x "$PY" ]; then
    echo "Using the existing environment $VENV"
else
    uv venv --python "$PYTHON_VERSION" "$VENV"
fi
uv pip install --python "$PY" "${PACKAGES[@]}"
"$PY" --version

step "4/8 GPU"
HAS_CUDA=$("$PY" -c "import torch; print(int(torch.cuda.is_available()))")
"$PY" -c "
import torch
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
for index in range(torch.cuda.device_count()):
    print(f'GPU {index}:', torch.cuda.get_device_name(index))
"
[ "$HAS_CUDA" = "1" ] || echo "WARNING: no CUDA GPU found. Training would run on the CPU (very slow)."

step "5/8 Test runs of both models (random input)"
"$PY" -c "
import torch
from model_factory import build_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(2, 1, 256, 256, device=device)
# The lattice values here only make the model buildable; training measures the real ones.
for name, options in (('unet', {}), ('sgarnet', {'lattice_period_px': 5.0, 'lattice_angle_deg': 0.0})):
    model = build_model(name, 1, 1, **options).to(device).eval()
    with torch.no_grad():
        y = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'{name}: output {tuple(y.shape)}, {params:.2f}M parameters, device {device}')
"

step "6/8 Data"
have_all_folders() {
    local folder
    for folder in "${DATA_FOLDERS[@]}"; do
        [ -d "$DATA_DIR/$folder" ] || return 1
    done
}
if have_all_folders && [ "$FORCE_UNZIP" != "1" ]; then
    echo "$DATA_DIR already has ${DATA_FOLDERS[*]}, not unzipping (FORCE_UNZIP=1 unzips again)."
else
    [ -f "$DATA_ZIP" ] || fail "Missing $DATA_ZIP. Upload it first."
    ls -lh "$DATA_ZIP"
    unzip -o -q "$DATA_ZIP" -d "$WORKSPACE"
    have_all_folders || fail "After unzipping, $DATA_DIR does not have all of: ${DATA_FOLDERS[*]}"
fi
ls "$DATA_DIR"

step "7/8 Data check"
"$PY" - "$DATA_DIR" <<'EOF'
import glob
import sys

import cv2

data_dir = sys.argv[1]
counts = {}
for folder in ["HR_Train", "sim_MCF_Train", "HR_Valid", "sim_MCF_Valid"]:
    files = sorted(glob.glob(f"{data_dir}/{folder}/*.png"))
    counts[folder] = len(files)
    print(folder, "| number of images:", len(files))
    if files:
        img = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
        print("    first file:", files[0], "| shape:", img.shape, "| type:", img.dtype,
              "| divisible by 16:", img.shape[0] % 16 == 0 and img.shape[1] % 16 == 0)
        if min(img.shape[:2]) < 368:
            print("    NOTE: smaller than SGARNet's 368 px training crops; train SGARNet with "
                  "SGARNET_ARGS=\"--patch_size N\" (N <= image size, multiple of 16)")

problems = [f"{a} has {counts[a]} images but {b} has {counts[b]}"
            for a, b in (("HR_Train", "sim_MCF_Train"), ("HR_Valid", "sim_MCF_Valid")) if counts[a] != counts[b]]
problems += [f"{folder} has no .png images" for folder, count in counts.items() if count == 0]
if problems:
    sys.exit("Data problem: " + "; ".join(problems))
print("Data looks complete.")
EOF

step "8/8 Baseline (raw input vs target, no network)"
"$PY" baseline.py "$DATA_DIR/HR_Valid" "$DATA_DIR/sim_MCF_Valid"

echo
echo "Setup finished (log: $WORKSPACE/setup_log.txt)."
echo "Next:  bash $REPO_DIR/run_training.sh both"
echo "To use the environment in this terminal:  source $VENV/bin/activate"
