#!/usr/bin/env bash
# Trains the UNet, SGARNet or both. By default it runs in the background, so closing the terminal
# does not stop the training.
#
# Usage:
#     bash run_training.sh unet                 # UNet only
#     bash run_training.sh sgarnet              # SGARNet only
#     bash run_training.sh both                 # UNet, then SGARNet
#     bash run_training.sh status               # running or not, current epoch, last scores, GPU use
#     bash run_training.sh stop                 # stop the background training
#     bash run_training.sh both --foreground    # stay attached instead (Ctrl+C stops the training)
#
# Output:
#     UNet:    /workspace/checkpoints/          log /workspace/train_log.txt
#     SGARNet: /workspace/checkpoints_sgarnet/  log /workspace/train_log_sgarnet.txt
#     This script's own log: /workspace/run_training.log
# An earlier checkpoint folder or log is moved aside (name + _previous_<date>), never overwritten.
#
# SGARNet needs the lattice period and angle of the fiber cores. Unless LATTICE_PERIOD and LATTICE_ANGLE
# are given, they are measured with estimate_lattice_period.py before any training starts; its output
# and spectrum plot are saved in the SGARNet checkpoint folder (check that the circles sit on the peaks).
#
# Settings (environment variables, defaults in brackets):
#     UNET_EPOCHS [50], SGARNET_EPOCHS [50], NUM_GPUS [1], NUM_WORKERS [4], SAVE_EVERY [10],
#     LATTICE_PERIOD, LATTICE_ANGLE [measured], UNET_ARGS, SGARNET_ARGS [extra train.py options]
#     e.g. SGARNET_EPOCHS=100 SGARNET_ARGS="--augment flips" bash run_training.sh sgarnet
# Batch size, learning rate, crops and augmentation follow each model's recipe in train.py
# (UNet: batch 4, full images; SGARNet: batch 8, 368 px crops, flips + rot90).
#
# When it has finished:  bash after_training.sh (full export)  or  bash run_evaluation.sh (quick look)

set -eo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-$WORKSPACE/venv}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
CKPT_DIR="${CKPT_DIR:-$WORKSPACE/checkpoints}"
TRAIN_LOG="${TRAIN_LOG:-$WORKSPACE/train_log.txt}"
SGARNET_CKPT_DIR="${SGARNET_CKPT_DIR:-$WORKSPACE/checkpoints_sgarnet}"
SGARNET_TRAIN_LOG="${SGARNET_TRAIN_LOG:-$WORKSPACE/train_log_sgarnet.txt}"
RUN_LOG="${RUN_LOG:-$WORKSPACE/run_training.log}"
LOCK_FILE="${LOCK_FILE:-$WORKSPACE/.run_training.lock}"
UNET_EPOCHS="${UNET_EPOCHS:-50}"
SGARNET_EPOCHS="${SGARNET_EPOCHS:-50}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_EVERY="${SAVE_EVERY:-10}"
LATTICE_PERIOD="${LATTICE_PERIOD:-}"
LATTICE_ANGLE="${LATTICE_ANGLE:-}"
UNET_ARGS="${UNET_ARGS:-}"
SGARNET_ARGS="${SGARNET_ARGS:-}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$REPO_DIR/$(basename "${BASH_SOURCE[0]}")"
PY="$VENV/bin/python"

fail() { echo; echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "================ $* ================"; }

usage() {
    echo "Usage: bash $SCRIPT unet|sgarnet|both|status|stop [--foreground]" >&2
    exit 1
}

MODE=""
FOREGROUND=0
CHILD=0
for arg in "$@"; do
    case "$arg" in
        unet|sgarnet|both|status|stop) MODE="$arg" ;;
        --foreground) FOREGROUND=1 ;;
        --child) CHILD=1 ;;   # internal: the background copy of this script
        *) usage ;;
    esac
done
[ -n "$MODE" ] || usage

TRAIN_UNET=0
TRAIN_SGARNET=0
case "$MODE" in
    unet) TRAIN_UNET=1 ;;
    sgarnet) TRAIN_SGARNET=1 ;;
    both) TRAIN_UNET=1; TRAIN_SGARNET=1 ;;
esac

# ---------------------------------------------------------------- status
last_line() {   # last_line FILE PATTERN: last matching line of a log that tqdm filled with \r
    tr '\r' '\n' < "$1" | grep -a -E -e "$2" | tail -n 1 || true
}

show_status() {
    echo "Running training processes:"
    pgrep -af "run_training\.sh .*--child" || true
    pgrep -af "python.*train\.py" || echo "    none"
    local name log
    for name in UNet SGARNet; do
        log="$TRAIN_LOG"; [ "$name" = SGARNet ] && log="$SGARNET_TRAIN_LOG"
        echo
        if [ ! -f "$log" ]; then
            echo "$name: no log yet ($log)"
            continue
        fi
        echo "$name ($log):"
        local pattern line
        for pattern in '-- Epoch [0-9]+/' '^Train loss' '^Valid loss' 'Error|Traceback|Total cost'; do
            line="$(last_line "$log" "$pattern")"
            if [ -n "$line" ]; then echo "    $line"; fi
        done
    done
    if [ -f "$RUN_LOG" ]; then
        echo
        echo "Last lines of $RUN_LOG:"
        tr '\r' '\n' < "$RUN_LOG" | grep -a -v -e '^\s*$' -e 'it/s\]' | tail -n 4 | sed 's/^/    /' || true
    fi
    if command -v nvidia-smi > /dev/null; then
        echo
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    fi
}

stop_training() {
    local stopped=0
    # The background copy of this script first, so that it does not start the next model.
    if pkill -f "run_training\.sh .*--child"; then stopped=1; fi
    if pkill -f "python.*train\.py"; then stopped=1; fi
    if [ "$stopped" -eq 1 ]; then echo "Stopped the training."; else echo "No training was running."; fi
}

if [ "$MODE" = status ]; then
    show_status
    exit 0
fi
if [ "$MODE" = stop ]; then
    stop_training
    exit 0
fi

cd "$REPO_DIR"
DATA_ARGS=(--dir_ZTrain "$DATA_DIR/HR_Train" --dir_XTrain "$DATA_DIR/sim_MCF_Train"
           --dir_ZValid "$DATA_DIR/HR_Valid" --dir_XValid "$DATA_DIR/sim_MCF_Valid")

# ---------------------------------------------------------------- preparation (before going to the background)
move_aside() {   # move_aside PATH: renames an existing file or non-empty folder to NAME_previous_<date>[.ext]
    local path="$1" new
    if [ -f "$path" ]; then
        new="${path%.*}_previous_$(date +%Y%m%d_%H%M%S).${path##*.}"
    elif [ -d "$path" ] && [ -n "$(ls -A "$path")" ]; then
        new="${path}_previous_$(date +%Y%m%d_%H%M%S)"
    else
        return 0
    fi
    mv "$path" "$new"
    echo "Moved the existing $path to $new"
}

estimate_lattice() {
    local out="$SGARNET_CKPT_DIR/lattice_estimate.txt"
    "$PY" estimate_lattice_period.py --dir_X "$DATA_DIR/sim_MCF_Train" --plot "$SGARNET_CKPT_DIR/lattice_spectrum.png" \
        2>&1 | tee "$out" || fail "The lattice estimate failed (see above). SGARNet was not started.
Give the values yourself with LATTICE_PERIOD=... LATTICE_ANGLE=... if you know them."
    LATTICE_PERIOD=$(sed -n 's/.*--sgarnet_lattice_period \([0-9.]*\).*/\1/p' "$out" | tail -n 1)
    LATTICE_ANGLE=$(sed -n 's/.*--sgarnet_lattice_angle \([0-9.]*\).*/\1/p' "$out" | tail -n 1)
    [ -n "$LATTICE_PERIOD" ] && [ -n "$LATTICE_ANGLE" ] || fail "Could not read the lattice values from $out"
}

if [ "$CHILD" -eq 0 ]; then
    step "Checking the setup"
    [ -x "$PY" ] || fail "No Python found at $PY. Run setup_pod.sh first."
    # The lock stays held by the background copy of this script and by train.py, until the training ends.
    exec 9> "$LOCK_FILE"
    flock -n 9 || fail "run_training.sh is already running. See: bash $SCRIPT status"
    for folder in HR_Train sim_MCF_Train HR_Valid sim_MCF_Valid; do
        [ -d "$DATA_DIR/$folder" ] || fail "Missing folder $DATA_DIR/$folder. Run setup_pod.sh first."
    done
    if pgrep -f "python.*train\.py" > /dev/null; then
        fail "A training is already running:
$(pgrep -af "python.*train\.py")
Wait for it to finish, or stop it with: bash $SCRIPT stop"
    fi
    if { [ -n "$LATTICE_PERIOD" ] && [ -z "$LATTICE_ANGLE" ]; } || { [ -z "$LATTICE_PERIOD" ] && [ -n "$LATTICE_ANGLE" ]; }; then
        fail "Give both LATTICE_PERIOD and LATTICE_ANGLE, or neither (then they are measured)."
    fi

    if [ "$TRAIN_SGARNET" -eq 1 ] && [[ "$SGARNET_ARGS" != *--patch_size* ]]; then
        IMAGE_SIZE=$("$PY" -c "
import cv2, glob
files = sorted(glob.glob('$DATA_DIR/sim_MCF_Train/*.png'))
print(min(cv2.imread(files[0], cv2.IMREAD_UNCHANGED).shape[:2]) if files else 0)
")
        [ "$IMAGE_SIZE" -ge 368 ] || fail "The training images ($IMAGE_SIZE px) are smaller than SGARNet's 368 px crops.
Use e.g. SGARNET_ARGS=\"--patch_size 256\" (at most the image size, a multiple of 16)."
    fi

    step "Preparing the output folders"
    if [ "$TRAIN_UNET" -eq 1 ]; then
        move_aside "$CKPT_DIR"
        move_aside "$TRAIN_LOG"
    fi
    if [ "$TRAIN_SGARNET" -eq 1 ]; then
        move_aside "$SGARNET_CKPT_DIR"
        move_aside "$SGARNET_TRAIN_LOG"
        mkdir -p "$SGARNET_CKPT_DIR"
        if [ -n "$LATTICE_PERIOD" ]; then
            echo "Lattice from LATTICE_PERIOD / LATTICE_ANGLE: period $LATTICE_PERIOD px, angle $LATTICE_ANGLE deg"
        else
            step "Measuring the core lattice for SGARNet"
            estimate_lattice
        fi
    fi
    move_aside "$RUN_LOG"

    if [ "$FOREGROUND" -eq 1 ]; then
        exec > >(tee -a "$RUN_LOG") 2>&1
    else
        export LATTICE_PERIOD LATTICE_ANGLE
        nohup setsid bash "$SCRIPT" "$MODE" --child > "$RUN_LOG" 2>&1 < /dev/null &
        echo
        echo "Training ($MODE) started in the background, process $!."
        echo "Check it with:   bash $SCRIPT status"
        echo "Follow it with:  tail -f $RUN_LOG"
        echo "Stop it with:    bash $SCRIPT stop"
        exit 0
    fi
fi

# ---------------------------------------------------------------- training
train_model() {   # train_model NAME EPOCHS CHECKPOINT_DIR LOG EXTRA_ARGS...
    local name="$1" epochs="$2" ckpt_dir="$3" log="$4"
    shift 4
    step "Training $name: $epochs epochs -> $ckpt_dir (log: $log)"
    echo "Started $(date)"
    MPLBACKEND=Agg "$PY" -u train.py --arch "$name" --num_GPUs "$NUM_GPUS" --num_epochs "$epochs" \
        --num_workers "$NUM_WORKERS" --save_everyEpoch "$SAVE_EVERY" --save_path "$ckpt_dir" \
        "${DATA_ARGS[@]}" "$@" 2>&1 | tee "$log"
}

RESULTS=()
set +e
if [ "$TRAIN_UNET" -eq 1 ]; then
    # shellcheck disable=SC2086  # UNET_ARGS is a list of options
    train_model unet "$UNET_EPOCHS" "$CKPT_DIR" "$TRAIN_LOG" $UNET_ARGS
    status=$?
    RESULTS+=("UNet:    $([ $status -eq 0 ] && echo OK || echo "FAILED (exit code $status), see $TRAIN_LOG")")
fi
if [ "$TRAIN_SGARNET" -eq 1 ]; then
    # shellcheck disable=SC2086  # SGARNET_ARGS is a list of options
    train_model sgarnet "$SGARNET_EPOCHS" "$SGARNET_CKPT_DIR" "$SGARNET_TRAIN_LOG" \
        --sgarnet_lattice_period "$LATTICE_PERIOD" --sgarnet_lattice_angle "$LATTICE_ANGLE" $SGARNET_ARGS
    status=$?
    RESULTS+=("SGARNet: $([ $status -eq 0 ] && echo OK || echo "FAILED (exit code $status), see $SGARNET_TRAIN_LOG")")
fi
set -e

step "Finished $(date)"
printf '%s\n' "${RESULTS[@]}"
echo "Next: bash $REPO_DIR/after_training.sh (full export) or bash $REPO_DIR/run_evaluation.sh (quick look)"
for result in "${RESULTS[@]}"; do
    [[ "$result" == *FAILED* ]] && exit 1
done
exit 0
