# Project review: tracker

Review 1: 2026-10-19 (session 1). Review 2: 2026-10-19 (session 2). Review 3: 2026-10-20 (session 3); its fixes
are committed in 69e8e64. **Review 4: 2026-10-21 (session 4)**, full re-read at commit 69e8e64, focus: "how can the
whole codebase be simpler and more readable" + design patterns. Every claim checked by running code.
Branch `training-improvements`. SGARNet is left out on purpose (not in use right now).

How to read this: each issue has an ID. **Status** shows if it has been explained to you yet, or fixed.
Ask "explain M2" (or any ID) whenever you want. "was M1" = the ID from review 2.
Review 4 kept the old IDs for re-checked issues and continued the numbering for new ones (M6..., L21...).

---

## Review 4 (session 4): issues. Session 5 (2026-10-22): H1, M5, L21, L22, L26 fixed; see the Status column.

Line numbers in the "Where" column refer to commit 69e8e64 (before the session-5 changes).

All recommended fixes were tried in a scratch copy (outside the project) and tested there: the adapted smoke
test passes, and the rewritten model file and data loader give bit-identical results.

| ID | Danger | Issue (plain words) | Where | Status |
|----|--------|---------------------|-------|--------|
| H1 | High | (re-checked) A checkpoint does not say which model it holds. A wrong `--arch` (unet_edsr ↔ unet_v2_edsr) or a different `--edsr_res_scale` loads with no error and computes something else. A Linux checkpoint cannot be opened on Windows (PosixPath). | utils.py:70, model_factory.py:123, train.py:385 | Fixed (session 5): `checkpoint["model"]`, plain-text `args`; no `--arch` in evaluate / onnx_export / save_test_preds; `--pretrain` stops on a mismatch. Rule (your choice: args stay a dictionary): never load a new checkpoint with old code (Kai clone before `git pull`, old `fiber_img` clone): old code silently uses the default EDSR settings, e.g. res_scale 1.0 instead of 0.1 (checked) |
| M2 | Medium | (re-checked) 293 of utils.py's 563 lines are functions nothing calls. lpips, piq, pytorch_msssim (utils.py) and albumentations (lib/common.py) are loaded but only used by that dead code. albumentations alone: 8.4–9.8 s per start of train.py on this PC. | utils.py, lib/common.py:6 | Explained (session 4). Kept on purpose: you want to reuse this code later |
| M5 | Medium | `run_training` does 11 jobs in 119 lines; the resume logic is spread over 3 functions. | train.py:372 | Fixed (session 5): split into check_amp_support, load_pretrained, complete_loss_stats, check_nothing_overwritten, best_valid_loss_so_far |
| M6 | Medium | (replaces H2) `--pretrain` does two jobs: resume, and (with `--start_epoch`) fine-tune. Fine-tuning keeps the old optimizer memory and loss history, and net_best is only written if it beats the *old* run's best score. | train.py:136, 397, 574 | Explained (session 4). Postponed: --pretrain is not used now |
| M7 | Medium | `--rgb_range` only works at 1: with 255, PSNR / SSIM are silently wrong. | train.py:144, lib/common.py:48, validation.py:51 | Explained again (session 5). Not fixed |
| M8 | Medium | (replaces L2) The data loader carries leftovers from the EDSR code (scale, repeat, patches flag, .npy, `--ext`); its `augment` parameter is silently ignored. | imageDatastore.py, lib/common.py | Explained again (session 5). Not fixed |
| M9 | Medium | The `inplace` switch of the model is broken: on, the output changes and training crashes. Nothing turns it on today. | model_UNET_EDSR.py:101, 145, 184, 313 | Explained (session 4). Postponed (may be useful later) |
| L8 | Low | (re-checked) Model file clutter: unused classes, a never-built Upsampler, a dead `outScale` path, ignored parameters, lines that do nothing, commented-out code. | model_UNET_EDSR.py | Explained (session 4). Later: separate analysis of unet / unet_v2 |
| L21 | Low | `tensor2np` changes the tensor it is given (CPU float tensors). Harmless for today's callers. | utils.py:140 | Fixed (session 5): `clamp_` → `clamp` |
| L22 | Low | `save_path` means the parent folder in the config, but the run folder after main() overwrites it. | train.py:574 | Fixed (session 5): `run_folder` is passed explicitly; `args.save_path` stays the parent folder |
| L23 | Low | The loss history is a loose dictionary with a dead key (`valid_expt`) and old-format patches. | train.py:394, 412 | Explained again (session 5). Not fixed (the patch lines only moved into complete_loss_stats) |
| L24 | Low | Small copies of the same code in several scripts (validation loader, data options, 8-bit rounding). | train.py, evaluate.py, baseline.py | Explained (session 4) |
| L25 | Low | smoke_test.py only works when started in the project folder. | smoke_test.py:34 | Explained (session 4) |
| L26 | Low | No setting for which GPU; on Kai you need `CUDA_VISIBLE_DEVICES`, and the README does not say so. | distributed_utils.py:29, 62 | Not fixed: the README with the Kai guide was written in session 5, then deleted (your choice). config.toml has the Kai paths. |
| L27 | Low | The TensorRT engine is tuned for the largest batch, not batch 1. Idea to test on Kai, not measured. | tensorrt_engine.py:44 | Explained again (session 5). Not fixed |
| L28 | Low | visualize_model.py: 453 lines to draw one picture that is already saved and is out of date. Your call. | visualize_model.py | Explained (session 4) |

---

## Dropped after the double-check: my analysis had a mistake. Look at these again later.

None of these were fixed. The underlying problem may still be real; the description was wrong somewhere.
H1, H2, M2, L2 and L8 were re-checked in review 4 with correct claims (see the table above: H2 → M6, L2 → M8).

| ID | What the issue said | The mistake |
|----|---------------------|-------------|
| L1 | Data workers restart every epoch. | "re-imports ~23 s each time": measured again, 4 parallel imports of train.py take 15.6 s. |
| L6 | `setup_seed` / `PYTHONHASHSEED`. | "set too late to matter": wrong reason. Data workers on Windows are new processes and *do* get it. (It still has no effect: nothing here depends on string hashing.) |
| L11 | `--compile` compiles two wrappers. | Implied avoidable waste, but the training graph (with gradients) and the validation graph (without) must be compiled separately anyway. And with 100 validation images and batch 4 or 1 a smaller last batch never happens. |

---

## Already fixed

| ID | What | Where / commit | Check |
|----|------|----------------|-------|
| H1 (review 2) | Runs overwrote each other's files → every run gets its own folder. | b5e140d | smoke test |
| H2 (review 2) | UNet squeezes to 1 channel → `--arch unet_v2` / `unet_v2_edsr`. | 0a20e06 | needs a real training run to compare |
| part of M2/M4 (review 2) | `--amp`, `--edsr_res_scale`. | a9c66a6 | smoke test |
| M1 | Defaults in 3 places. Now: no personal data path (data folders must be given; clear error), argparse defaults = the used values (`save_everyEpoch` 10, `num_workers` 4), recipe numbers in `--help` come from `RECIPE_DEFAULTS` (copy in config.toml removed), `evaluate.py --config` reads the data folders (skips train-only keys). | 69e8e64 | smoke test (evaluate with config, missing folder error) |
| M3 | `cudnn.benchmark` on; loss summed on the GPU, `.item()` only every 20 steps and once per epoch; `non_blocking` copies (train + validation). | session 3 | smoke test (speed not measured: no GPU on this PC) |
| L3 | `assert` → real errors in imageDatastore and `UNET.forward`. | session 3 | models bit-identical |
| L4 | `unwrap_module` / `load_checkpoint` also recognize DDP (code kept). | session 3 | — |
| L5 | No more `deepcopy` of the model; `--save_model_arch` saves the *trained* model. | session 3 | smoke test |
| L7 | `num_epochs`, `valid_everyEpoch`, `save_everyEpoch` must be ≥ 1; too few images for one batch → clear error. | session 3 | smoke test |
| L10 | Printout says "Valid L1 (raw output)" and "PSNR / SSIM (8-bit output)", with a comment why; data range fixed at 255. | session 3 | — |
| L12 | Smoke test adds: metric test (= skimage), memorize-one-image test, 2-process CPU run (only without GPU), train.log, config type error, setting errors, `--start-epoch`, `--save_model_arch`, evaluate with config. Not added: Linux → Windows test (part of H1). | session 3 | runs |
| L13 | `--start_epoch` (old `--start-epoch` still works). `save_test_preds.py`, `baseline.py`, `compare_predictions.py` use argparse with the same positional arguments; the first two get `--n_colors`; compare says "grayscale only". MATLAB-style names kept (your choice). | session 3 | smoke test (alias) |
| L14 | README written (UTF-8): setup, data, config, train/resume, run files, evaluate, predictions, ONNX, TensorRT, test. README.md was deleted in session 5 (your choice). | session 3 | — |
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

## Still open from earlier reviews

| ID | Issue (plain words) | Where | Status |
|----|---------------------|-------|--------|
| M4 | (was M2) `config.toml` still gives unet_edsr batch size 4 → "out of memory" right away on one 48 GB GPU (~132 GiB needed in fp32). `--amp` and `--edsr_res_scale` exist but the config doesn't use them. | [config.toml:49-57](config.toml:49) | Fixed (session 5): [unet_edsr] and [unet_v2_edsr] now have batch_size 1, amp = true, edsr_res_scale 0.1 (the settings of your 2026-10-19 Kai run). |
| L9 | Down-block shortcut keeps only every 2nd pixel → possible aliasing. An idea to test with a training run, not a proven bug. | [model_UNET_EDSR.py:131](model_UNET_EDSR.py:131) | Skipped (your choice). |
| L1, L6, L11 | See "Dropped" above. | | |
| M5 | Now in the review-4 table. | | |

---

## Facts checked for real (not guesses)

Session 5 (2026-10-22, after the H1 / M5 / L21 / L22 fixes, this PC, CPU):
- Old code (69e8e64) and new code train exactly the same: identical weights, optimizer state, loss history and epoch
  for unet (2 epochs), a --pretrain resume, --pretrain --start_epoch, net_best, and unet_v2_edsr (2 x 16, res_scale 0.1).
- Old-format checkpoints (no "model", Namespace args) work in the new code: evaluate.py without --arch, and --pretrain.
- New checkpoints open with torch.load(weights_only=True): only plain data inside, so no PosixPath.
- --pretrain with another --edsr_res_scale, or --arch unet_v2_edsr on a unet_edsr checkpoint, stops with "holds the model".
- save_test_preds.py without arch works; evaluate.py and onnx_export.py have no --arch. smoke_test.py passes (509 s).
- Old code (69e8e64) loading a new checkpoint (32 x 256, res_scale 0.1): no error, but res_scale 1.0 (it reads the
  dictionary 'args' with getattr, which falls back to the defaults).
- Not tested on Linux / Kai (you chose not to run anything there): the 2-process CPU part of the smoke test.
- Kai (read-only look): clone fiber_img_v2 is at a9c66a6 (needs git pull), no local changes; the old venv has torch 2.14.1, onnx 1.23.2,
  TensorRT 11.4; tmux, nohup and uv (~/.local/bin/uv) exist; /scratch is 80 % full; GPU 0 was in use by someone else.
  Your running unet_v2_edsr run (started 2026-10-19 without --config): memory cap 60 % (set_per_process_memory_fraction),
  --num_workers 2, --batch_size 1, --amp, --edsr_res_scale 0.1. Its checkpoints hold no Linux path.

Session 4 (this PC, Windows, no GPU; commit 69e8e64):
- The smoke test passes at 69e8e64: 546 s. With albumentations replaced by an empty stand-in: 394 s (one run each).
- Importing train.py: 32.5–33.7 s; without albumentations 23.0–25.0 s (3 paired runs). Import breakdown (2 runs):
  torch ~8 s, albumentations ~10 s, torchvision (via losses.py) ~8 s, matplotlib ~2 s, rest of utils ~2.6 s.
- unet_edsr weights load into unet_v2_edsr with "All keys matched successfully"; small model, random weights:
  mean |difference| 0.032, mean |output| 0.056. res_scale 0.1 weights load into res_scale 1.0 the same way.
- `pickle` of `PosixPath` on Windows: "cannot instantiate 'PosixPath' on your system". `torch.load(weights_only=True)`
  refuses a checkpoint holding an `argparse.Namespace`, and loads one holding a plain dict of text / numbers.
- `--rgb_range 255`: a near-perfect prediction gets PSNR 23.66 / SSIM 0.7033 instead of 39.91 / 0.9994.
- `imageDatastore(..., augment=True)` gives `dataset.augment == False`. UpBlock kernel_size 3 and 7: same layers.
- UNET with `inplace=True`, same weights: mean |difference| 0.197 vs mean |output| 0.317; backward crashes
  ("modified by an inplace operation").
- `tensor2np` on a CPU float tensor [2, -1, 0.5] leaves it as [1, 0, 0.5].
- A checkpoint holding a pickled model object cannot be opened at all after the class is renamed (AttributeError).
- Fine-tune with `--pretrain --start_epoch 2` on a harder dataset (new folder): no net_best.pth.tar; the history
  starts with the old run's valid L1 (0.284, then 0.638, 0.630); Adam's step count carried over (6 = 2 old + 4 new).
- smoke_test.py started from the parent folder: "can't open file ...\train.py".
- Proposed model file (scratch copy): same layer names, same parameter order, same start weights (same seed), same
  output and gradients as now, for unet, unet_v2, unet_edsr, unet_v2_edsr. Note: the parameter ORDER matters, the
  optimizer state in a checkpoint is stored by position (a first draft got this wrong; caught by the test).
- Proposed data loader: bit-identical tensors for full images, crops, crops + flips, crops + flips + rot90 (20 seeds x 5 pairs).

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
- (review 4) validation.py, distributed_utils.py, config_file.py, tensorrt_engine.py are short, clear and well commented.
- (review 4) Design patterns used well: registry + factory (model_factory), strategy (validation takes any
  "predict" function: model, compiled model, TensorRT engine, identity), adapter (TensorRTModel), wrappers
  (HalfPrecisionModel, Tee, DDP, torch.compile), frozen dataclasses instead of global variables (DistributedContext).

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
- **Dead code**: (session 4) code that is never run. It still has to be read, installed and kept working.
- **Library / dependency**: (session 4) code written by others that this project installs and loads (e.g. albumentations).
- **Import**: (session 4) loading a file or library into the running program. Slow libraries make every start slow.
- **Design pattern**: (session 4) a well-known, named way to arrange code for a common problem (like a named recipe).
- **Factory / registry**: (session 4) one function that builds objects for you / a dictionary "name → builder".
- **Strategy**: (session 4) the part that changes is passed in as an argument (e.g. which "predict" function to use).
- **Adapter**: (session 4) a plug converter: makes one thing look like another (TensorRT engine → PyTorch model).
- **Wrapper / decorator**: (session 4) an object around another one that keeps its interface and adds one job.
- **Side effect**: (session 4) a function changes something outside itself, e.g. the tensor you gave it.
- **In place**: (session 4) overwrite the numbers instead of making a new copy. Saves memory; dangerous if the old numbers are still needed.
- **Resume vs fine-tune**: (session 4) resume = continue the same run; fine-tune = start a new run from trained weights.
- **Bit-identical**: (session 4) exactly the same numbers, down to the last bit.
- **Model description**: (session 4) the "instruction booklet" of a checkpoint: arch, n_colors, EDSR sizes.
