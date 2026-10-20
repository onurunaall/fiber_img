# Project review: tracker

Review 1: 2026-10-19 (session 1). Review 2: 2026-10-19 (session 2). **Review 3: 2026-10-20 (session 3)**,
fresh full re-read of every file at commit a9c66a6, extra focus on "too complex / wasteful code that can be simpler".
**Session 3, part 2:** every issue double-checked by running code. Issues with a mistake were dropped (see below);
M1, M3 and most L issues were fixed (not committed yet).
Branch `training-improvements`. SGARNet is left out on purpose (not in use right now).

How to read this: each issue has an ID. **Status** shows if it has been explained to you yet, or fixed.
Ask "explain M2" (or any ID) whenever you want. "was M1" = the ID from review 2.

---

## Dropped after the double-check: my analysis had a mistake. Look at these again later.

None of these were fixed. The underlying problem may still be real; the description was wrong somewhere.

| ID | What the issue said | The mistake |
|----|---------------------|-------------|
| H1 | A checkpoint does not describe its own model; the code rebuilds it from what you type. | The explanation said the output change was "as big as the output itself". It compared the *largest* change (0.036) with the *average* output (0.047). The average change is 0.013, about ¼ of the output. Also "save_test_preds silently uses unet" was misleading: a wrong arch there crashes loudly. The main point (wrong settings load with no error) was re-verified and holds. |
| H2 | `--pretrain` carries over optimizer, history and best score when fine-tuning. | "no net_best gets saved (the new data never beats the old score)": "never" is wrong. It *can* beat the old score; it just might not. |
| M2 | `utils.py` junk drawer, heavy imports. | "heavy imports load in every data worker and in every script": 4 scripts don't import utils (tensorrt_engine, smoke_test, visualize_model, estimate_lattice_period), and data workers re-import only on Windows. The "~350 lines" was never counted exactly. |
| L1 | Data workers restart every epoch. | "re-imports ~23 s each time": measured again, 4 parallel imports of train.py take 15.6 s. |
| L2 | Data loader traps + leftover machinery. | "each image is copied 3 times": for gray images `ascontiguousarray` does not copy, so it is 2 copies (one needed). "`targets.float()` converts again": it is a no-op (returns the same tensor). Both checked. |
| L6 | `setup_seed` / `PYTHONHASHSEED`. | "set too late to matter": wrong reason. Data workers on Windows are new processes and *do* get it. (It still has no effect: nothing here depends on string hashing.) |
| L8 | Dead code in the model file. | "unused Upsampler": EDSR builds it when scale ≥ 2 (EDSR's own default is 2); it only never happens in this project. "inplace ignored" is only true for UNET_EDSR, not UNET. |
| L11 | `--compile` compiles two wrappers. | Implied avoidable waste, but the training graph (with gradients) and the validation graph (without) must be compiled separately anyway. And with 100 validation images and batch 4 or 1 a smaller last batch never happens. |

---

## Already fixed

| ID | What | Where / commit | Check |
|----|------|----------------|-------|
| H1 (review 2) | Runs overwrote each other's files → every run gets its own folder. | b5e140d | smoke test |
| H2 (review 2) | UNet squeezes to 1 channel → `--arch unet_v2` / `unet_v2_edsr`. | 0a20e06 | needs a real training run to compare |
| part of M2/M4 (review 2) | `--amp`, `--edsr_res_scale`. | a9c66a6 | smoke test |
| M1 | Defaults in 3 places. Now: no personal data path (data folders must be given; clear error), argparse defaults = the used values (`save_everyEpoch` 10, `num_workers` 4), recipe numbers in `--help` come from `RECIPE_DEFAULTS` (copy in config.toml removed), `evaluate.py --config` reads the data folders (skips train-only keys). | session 3, not committed | smoke test (evaluate with config, missing folder error) |
| M3 | `cudnn.benchmark` on; loss summed on the GPU, `.item()` only every 20 steps and once per epoch; `non_blocking` copies (train + validation). | session 3 | smoke test (speed not measured: no GPU on this PC) |
| L3 | `assert` → real errors in imageDatastore and `UNET.forward`. | session 3 | models bit-identical |
| L4 | `unwrap_module` / `load_checkpoint` also recognize DDP (code kept). | session 3 | — |
| L5 | No more `deepcopy` of the model; `--save_model_arch` saves the *trained* model. | session 3 | smoke test |
| L7 | `num_epochs`, `valid_everyEpoch`, `save_everyEpoch` must be ≥ 1; too few images for one batch → clear error. | session 3 | smoke test |
| L10 | Printout says "Valid L1 (raw output)" and "PSNR / SSIM (8-bit output)", with a comment why; data range fixed at 255. | session 3 | — |
| L12 | Smoke test adds: metric test (= skimage), memorize-one-image test, 2-process CPU run (only without GPU), train.log, config type error, setting errors, `--start-epoch`, `--save_model_arch`, evaluate with config. Not added: Linux → Windows test (part of H1). | session 3 | runs |
| L13 | `--start_epoch` (old `--start-epoch` still works). `save_test_preds.py`, `baseline.py`, `compare_predictions.py` use argparse with the same positional arguments; the first two get `--n_colors`; compare says "grayscale only". MATLAB-style names kept (your choice). | session 3 | smoke test (alias) |
| L14 | README written (UTF-8): setup, data, config, train/resume, run files, evaluate, predictions, ONNX, TensorRT, test. | session 3 | — |
| L15 | The launching program opens the meeting port itself and keeps it open (`open_meeting_point`). Same launch command. | session 3 | 2-process run works |
| L16 | `.venv` rebuilt with `uv sync`: Python 3.13.15, torch 2.14.1+cu126, OpenCV 4.14. | session 3 | smoke test before/after |
| L17 | `train.log` in every run folder (resumed runs add to it; crash error included); plot redrawn after every validation. | session 3 | smoke test |
| L18 | Config values must have the right type (true/false, whole number, number, text). | session 3 | smoke test |
| L19 | PSNR/SSIM computed in torch for the whole batch on the GPU (float64). Same numbers as skimage: on the 100 real validation images the totals differ by < 1e-12. | session 3 | metric test + real data |
| L20 | 2 builders instead of 4; one timer; duplicate batch check removed; `net_best` read with `mmap`; `empty_cache` removed; `pin_memory` only on CUDA; `.gitignore` duplicates removed; `visualize_model.py` docstring says what it shows. | session 3 | models bit-identical |

New finding (session 3): on the Windows torch builds, multi-process training could not start at all
("PyTorch was built without libuv support"). The L15 change uses `use_libuv=False` on Windows, so it works now.
It prints a few harmless "client socket has failed to connect" warnings on Windows before it connects.

---

## Still open (not touched in session 3)

| ID | Issue (plain words) | Where | Status |
|----|---------------------|-------|--------|
| M4 | (was M2) `config.toml` still gives unet_edsr batch size 4 → "out of memory" right away on one 48 GB GPU (~132 GiB needed in fp32). `--amp` and `--edsr_res_scale` exist but the config doesn't use them. | [config.toml:49-57](config.toml:49) | Explained (session 2). You said: leave for now. |
| M5 | `run_training` is one long function doing ~8 jobs; resume logic is spread over `parse_args`, `main`, `run_training`. | [train.py](train.py) | Not yet. You said: leave for now. |
| L9 | Down-block shortcut keeps only every 2nd pixel → possible aliasing. An idea to test with a training run, not a proven bug. | [model_UNET_EDSR.py:131](model_UNET_EDSR.py:131) | Skipped (your choice). |
| H1, H2, M2, L1, L2, L6, L8, L11 | See "Dropped" above. | | |

---

## Facts checked for real (not guesses)

Session 3:
- unet_edsr with `res_scale 0.1` loads into one with `res_scale 1.0` with no error. Output difference on a test
  input: largest 0.036, average 0.013; average output size 0.047. unet_edsr ↔ unet_v2_edsr also loads with no error.
- `config.toml` values were not type-checked: `compile = "false"` became True; `batch_size = 2.5` was accepted. (Fixed.)
- PSNR + SSIM on the CPU (skimage): 113 ms per 720 x 720 image, measured twice.
- 4 parallel imports of train.py: 15.6 s (one import: 14 s).
- `DistributedDataParallel` is not a subclass of `nn.DataParallel`. `cudnn.benchmark` is False by default.
- Model outputs of unet, unet_v2, unet_edsr, unet_v2_edsr are bit-identical before and after the session-3 changes.
- New metrics vs skimage on the 100 real validation images: totals differ by at most 7e-13.
- unet memorizes one 64 x 64 image: L1 23–33x lower after 300 steps (3 seeds).

From review 2 (still true):
- All images are 720 x 720, 8-bit, grayscale. 5000 train / 100 valid / 400 test pairs. No file appears twice across splits.
- A checkpoint whose `args` holds a Linux path (`PosixPath`) fails to load on Windows.
- UNET_EDSR 32 x 256 at 720 x 720: 39.8 TFLOP and ~33 GiB per image (fp32), ~19 GiB with `--amp`.
- Kai: 8 x RTX A6000 48 GB, shared. Earlier unet_edsr run: 33.7 min/epoch, valid PSNR 27.38 / SSIM 0.760.

---

## Things that are done well (don't "fix" these)

- Checkpoints are written to a temp file first, then renamed: an interrupted save never leaves a broken file.
- Every run has its own folder; resuming refuses to overwrite later epochs or a better `net_best`.
- Validation splits images across GPUs without duplicates, so the numbers are exact.
- Validation runs without the DDP wrapper, which avoids a known hang.
- The learning-rate schedule is computed from the epoch number, so resuming needs no extra state.
- The config reader rejects unknown keys, unknown tables and (now) wrong types with clear messages.
- The smoke test runs the real train.py end to end.
- The model registry (`MODEL_BUILDERS`) is one simple place to add an architecture.
- The data loader checks that input file i and target file i carry the same number.

---

## Words explained so far (mini glossary)

- **Channel**: one "layer of notes" the network keeps about each pixel. A gray image has 1 channel.
- **Skip connection**: a shortcut that carries early, detailed information directly to a later part of the network.
- **ReLU**: a rule that turns every negative number into 0.
- **Checkpoint**: a saved file of the model's learned numbers (weights), plus extra info.
- **Optimizer state**: the optimizer's "memory" of past steps.
- **Fine-tuning**: starting from an already-trained model and training it further on new data.
- **Validation**: after each epoch, the model is tested on 100 images it never trains on.
- **net_best.pth.tar**: the checkpoint with the best validation score so far.
- **Run folder**: one folder that holds all files of one training run and nothing else.
- **Resolution / size**: how many pixels a layer works on (720 -> 360 -> 180 -> 90 -> 45 and back).
- **Encoder / decoder**: the shrinking half / the growing half of the UNet.
- **Learned numbers (parameters / weights)**: the numbers training adjusts.
- **Residual output**: the network predicts only the *change* to the input.
- **FLOP / TFLOP**: one small math step / 1 trillion of them.
- **Out of memory (OOM)**: the GPU has no room left; the run stops with an error.
- **Residual scaling**: each EDSR block's change is multiplied by a small number (e.g. 0.1) before it is added.
- **Mixed precision (AMP)**: most math with shorter numbers (bfloat16); less memory, faster.
- **state_dict**: the dictionary inside a checkpoint that holds only the learned numbers, by layer name.
- **Architecture settings**: the settings that decide what the model *is* (arch, n_colors, EDSR size, res_scale).
- **Pickle**: Python's way to save any Python object to a file. Loading one can run code.
- **Sync / waiting for the GPU**: (session 3) the GPU works on a queue of jobs. Reading a number back (`.item()`) makes the CPU stop and wait until the queue is empty.
- **Port**: (session 3) a numbered "door" on a computer through which programs talk to each other.
