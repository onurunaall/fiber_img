# Project review: tracker

Review 1: 2026-10-19 (session 1). Review 2: 2026-10-19 (session 2). Review 3: 2026-10-20 (session 3); its fixes
are committed in 69e8e64. **Review 4: 2026-10-21 (session 4)**, full re-read at commit 69e8e64, focus: "how can the
whole codebase be simpler and more readable" + design patterns. Every claim checked by running code.
Branch `training-improvements`. SGARNet is left out on purpose (not in use right now). It was removed from the code on 2026-11-08 (session 7).

How to read this: each issue has an ID. **Status** shows if it has been explained to you yet, or fixed.
Ask "explain M2" (or any ID) whenever you want. "was M1" = the ID from review 2.
Review 4 kept the old IDs for re-checked issues and continued the numbering for new ones (M6..., L21...).

---

## Review 6 (session 8, 2026-11-10): bugs that matter for transfer learning. Session 9 (2026-11-13): T1, T2, T3, T7 applied; T4 dropped.

Full re-read of the working tree (SGARNet already removed). Focus: can this code do what the two Wang papers do
(pre-train on simulated data, then transfer learning on ~50 real image pairs)? Every "checked" line was run today
on this PC (CPU, temp folder). T = transfer learning.

| ID | Danger | Issue (plain words) | Where | Checked | Status |
|----|--------|---------------------|-------|---------|--------|
| T1 | High | (= M6, no longer "not used") Fine-tuning with `--pretrain --start_epoch 1` on new data: `net_best.pth.tar` is only written if the new data's validation L1 beats the *old* run's best; the loss history and plot mix both runs; Adam's memory is carried over. | train.py:403-409, 344, 332 | Old valid L1 0.31 / 0.30, new 0.52 / 0.51 / 0.51: no net_best; valid_epoch [1, 2, 1, 2, 3]; Adam step 10 instead of 6 | Fixed (session 9): new option `--finetune_from FILE` = new run from the weights only (epoch 1, new optimizer, new history, new folder). `--pretrain` is unchanged and is for continuing a run |
| T2 | Medium | `--dir_pretrain FILE` without `--pretrain` is silently ignored: training starts from random weights, no warning. | train.py:105-106, 405 | exit code 0, no "Starting at epoch", history length 1 | Fixed (session 9): train.py stops with a message that names both ways out (`--pretrain` or `--finetune_from`) |
| T3 | Medium | Trap, not a bug: the learning rate is computed from the epoch number. `--start_epoch 51` starts fine-tuning at lr 3.1e-6 instead of 1e-4. Fine-tuning must use `--start_epoch 1`. | train.py:253, utils.py:123 | lr at epoch 1: 1e-4; at epoch 51: 3.125e-6 | Fixed (session 9): `--finetune_from` always starts at epoch 1; `--start_epoch` without `--pretrain` stops with a message |
| T4 | High (real data) | A 16-bit image is divided by 255 like an 8-bit one: values go up to 257 instead of 1. No error. | lib/common.py:48 | 16-bit PNG 0..65535 → tensor 0..257 | **Dropped in the session-9 double-check** (the bug is real; my planned fix was wrong, see the "Dropped" table). Not fixed |
| T5 | High (real data) | A gray picture saved as a 3-channel file comes out as 16..235 instead of 0..255, and red / blue are swapped in the gray conversion. A 1-channel file of the same picture gives 0..255, so input and target can get different scales. | lib/common.py:34-35, imageDatastore.py:17-26 | same picture: 1-channel 0..1, 3-channel 0.0627..0.9216; pure red → 41.0 (correct 81.5), pure blue → 81.5 (correct 41.0) | Explained (session 8). Not fixed |
| T6 | Low | Only `.png` and `.npy` files are seen. A folder of `.tif` files gives an empty dataset with no error at that point; train.py then stops with a confusing message that does not name the cause. | imageDatastore.py:12, 32 | dataset length 0, no error; train.py: "num_samples should be a positive integer value, but got num_samples=0" | Explained (session 8). Not fixed |
| T7 | Design gap | The papers give the network the white reference image of the fiber as a 2nd input channel. This code cannot: `n_colors` sets input *and* output channels, and nothing reads `data/calib`. | model_factory.py:105 | n_colors 2 → output has 2 channels; no code mentions `calib` | Added (session 9): option `--reference_image FILE` in train.py, evaluate.py, save_test_preds.py. Off by default. A model trained with it needs its own pre-training (first layer 8 x 2 x 3 x 3 instead of 8 x 1 x 3 x 3). Whether it improves the results is not measured yet |

Session 9 (2026-11-13): double-check of the plan, then T1, T2, T3, T7 applied (not committed). T4 dropped.
All fixes were first built in a scratch copy and compared with the code before the change (this PC, CPU):
- No new option used → old and new code train exactly the same (weights, optimizer state, loss history, epoch):
  unet 2 epochs; resume with `--pretrain`; net_best; `--pretrain --start_epoch 3` into a new folder;
  unet_v2_edsr (2 x 16) with crops + flips + rot90 + amp.
- A checkpoint written by the old code works in the new code: evaluate.py gives the same numbers, train.py resumes it.
  The 2026-11-03 Kai checkpoint (unet_v2_edsr 16 x 128, res_scale 1.0) loads and gives the same output as before.
- `--finetune_from` on harder data: source valid L1 0.285 / 0.281 / 0.279; fine-tune 0.507 / 0.490 / 0.468;
  net_best written (epoch 3); valid_epoch [1, 2, 3]; Adam steps 6; first lr 1e-4. Old way on the same data: no net_best.
- The 5 new stops (forgotten `--pretrain`, `--start_epoch` alone, `--finetune_from` with `--pretrain`, missing
  fine-tune file, missing reference file) all end with exit code 2 before any run folder is made.
- `--reference_image`: checkpoint says `in_channels` 2; fine-tuning from it without the image stops ("holds the model");
  save_test_preds.py and evaluate.py need the image and refuse it for a checkpoint trained without; a reference of
  another size or with 3 channels stops with a clear message; crops and flips move the reference exactly like the input.
- Not tested here: onnx_export.py (the onnx package is not installed on this PC; only the input shape it builds was
  checked: 2 channels for a reference checkpoint) and the TensorRT path (no GPU). evaluate.py `--engine` has no
  checkpoint to check the reference against.
- Changed for users: onnx_export.py has no `--n_colors` any more (the checkpoint knows its input channels);
  `load_trained_model` returns (model, description).
- Wording of the plan that was too loose: "adapt the one existing assertion" in smoke_test.py: two lines had to be
  adapted (the description assertion and the `load_trained_model` call).

Facts checked in session 8:
- smoke_test.py passes on the working tree (after the SGARNet removal, including the 2-process CPU run).
- `data/calib/sim_cfb_white.png`: 5941 cores found, core pitch 9.2 px; peak brightness of the cores within 300 px of
  the center: std / mean 12 % (part of this can come from reading the peak at whole pixels).
- Wang, SPIE 2025, Table 1 (glioblastoma test set), PSNR / SSIM: raw fiber image 20.44 / 0.547; pre-trained only
  18.89 / 0.763; transfer learning on 1 / 5 / 10 / 25 / 50 pairs: 22.46 / 0.886, 26.31 / 0.920, 27.26 / 0.932,
  28.47 / 0.936, 28.83 / 0.941. Recipe: 160 x 160 random crops, batch 4, lr 1e-4, L1; pre-training 100 epochs,
  lr x 0.1 every 20; transfer learning 300 epochs, lr x 0.1 every 60.
- TW's slides (2024-12 slide 8, 2025-11 slide 26): the real fiber / sharp image pairs are registered with a *learned*
  affine matrix ("accurate image registration is difficult"). train_tw.py has leftovers of it (`pespective`,
  `optimizer_align`); the module `models/spatialTransformer.py` was not shared.

---

## Review 5 (session 6, 2026-10-23): readability. Nothing changed in the project yet.

Goal: code that is clear to read, keeping classes and design patterns. Left out: SGARNet, the unused parts of
utils.py (you keep them), UNET / DownBlock / UpBlock (your separate analysis). Every "after" version was built in a
scratch copy of f900261 and tested: training identical (model, weights, optimizer, history, epoch; 5 cases), config
reader identical in 15 cases (including all error messages), smoke test passes, and checkpoints work in both
directions between f900261 and the new code. Line numbers refer to f900261.

| ID | Gain | What is needlessly complex (plain words) | Where | Simpler form | Status |
|----|------|------------------------------------------|-------|--------------|--------|
| R1 | Big | No single owner of the checkpoint file: written in utils.py, read in model_factory.py and in 2 places of train.py, each with its own string keys. | utils.py:57, model_factory.py:124, train.py:384, 447 | `Checkpoint` class in a new checkpoint.py (save / load / trained_model); old formats handled in one place | Explained (session 6) |
| R2 | Big | What a model *is* is a raw dict with 3 text keys, built and used by 4 functions. | model_factory.py:115-151 | `ModelSpec` frozen dataclass: `from_settings(args)`, `build()`, `to_dict()` | Explained (session 6) |
| R3 | Big | (= L23) The loss history is a dict whose 7 key names are typed as text 18 times in train.py, plus a patch function. | train.py:417, 481, 507-517, 544 | `TrainingHistory` dataclass: `add_validation()`, `best_valid_loss()`, `to_dict()` / `from_dict()` | Explained (session 6) |
| R4 | Medium | train.py (653 lines) mixes 3 jobs: settings, run files (folder, log, plot), training. | train.py | train_options.py (settings), run_files.py (folder, log, plot), train.py (training, 431 lines) | Explained (session 6) |
| R5 | Medium | parse_args_with_config does every step in one 49-line block and deletes from dicts while looping over them. | config_file.py:27 | 4 named steps: read_config, checked_section, the 3-pass parse, check_choices (identical results in 15 cases) | Explained (session 6) |
| R6 | Medium | smoke_test.py: one 143-line main() where each check depends on files of earlier checks; a command built by slicing a tuple (`*resume[:-4] ... *resume[-3:]`). | smoke_test.py:113, 171 | Named check functions (outline only, not built); the slicing written out (built) | Explained (session 6) |
| R7 | Medium | (= M8) The data loader carries EDSR leftovers over 2 files. | imageDatastore.py, lib/common.py | The ~80-line version tested in session 4 | Explained (session 4) |
| R8 | Small | `print_on_main(context, ...)`, `sum_across_processes(..., context)`, `mean_across_processes(..., context)` are free functions about the context. | distributed_utils.py:74-99 | Methods: `context.print_main()`, `context.sum()`, `context.mean()` | Explained (session 6) |
| R9 | Small | Validation result in 3 steps on one line: `from_list(sum_across_processes(as_list()))` then `compute_validation_result()`. | train.py:361-363, validation.py:124 | `ValidationTotals.mean()` | Explained (session 6) |
| R10 | Small | main() calls `architecture_options(...)` and throws the result away (it only checks SGARNet settings). | train.py:622 | The check moves into parse_args as a clear `parser.error` | Gone: the call was removed with SGARNet (session 7) |
| R11 | Small | train.py reads the image size through the dataset's private `_load_file`. | train.py:202 | `dataset.image_size()` | Explained (session 6) |
| R12 | Small | `Timer().t()` (cryptic names) and the rule "every N epochs or the last one" written twice. | train.py:496-541 | `time.perf_counter()` and `is_due(epoch, every, last_epoch)` | Explained (session 6) |
| R13 | Small | (= part of L8) EDSR / UNET_EDSR: ignored parameters, lines that do nothing, commented-out code. UNET itself left out. | model_UNET_EDSR.py:256-354 | See L8 | Explained (session 4) |

Consequences of R1-R12 (for your decision): utils.save_checkpoint, load_checkpoint, unwrap_module,
fix_state_dict_for_loading and Timer would no longer be called (utils.py stays as you decided). The checkpoint file keeps
its keys; "valid_expt" is no longer written and "train_objective" is always written (empty for unet). The --pretrain
mismatch message shows `ModelSpec(...)` instead of a dict; missing SGARNet settings stop with a short argparse error
(exit code 2) instead of a traceback; "=> Loading checkpoint" is no longer printed. The total gets ~100 lines longer
(1835 -> 1933): the gain is clarity, not length.

Not recommended now: a typed settings object instead of `args` (16 functions take all ~40 settings, but every option
would be written twice); a Strategy object for the recipe (removes the 14 `args.recipe` checks, 5 of them in
train_one_epoch; SGARNet only); renaming X / Z (your choice to keep the MATLAB names).

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
| M6 | Medium | (replaces H2) `--pretrain` does two jobs: resume, and (with `--start_epoch`) fine-tune. Fine-tuning keeps the old optimizer memory and loss history, and net_best is only written if it beats the *old* run's best score. | train.py:136, 397, 574 | Explained (session 4). Session 9: fine-tuning now has its own option `--finetune_from` (see T1). `--pretrain --start_epoch N` still behaves as described here; use it only to continue the same data in a new folder |
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
| T4 | 16-bit images are divided by 255 (values up to 257). The bug itself is confirmed. | The fix in the session-8 plan (divide by the largest value of the image's number type) was wrong in scope: it also changes `.npy` data stored as int16 / int32 / int64. Values 0..255 that give 1.0 today would give 0.0078 / 1.2e-7 / 2.8e-17 (measured). A narrower fix (only uint16 → 65535, everything else as today) is not applied yet. |
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
- **Transfer learning**: (session 8) train a little more on a small amount of new data, starting from a model that already learned on other data. Same idea as fine-tuning.
- **Image pair**: (session 8) one fiber image plus one sharp image of the exact same spot. The sharp one is the "ground truth".
- **Reference image**: (session 8) a photo of the fiber looking at an evenly lit surface. It shows where each core is and how bright each core is.
- **Input channel**: (session 9) one layer of the picture stack the network gets. With the reference image the stack has 2 layers: fiber image + reference.
- **Registration**: (session 8) shifting / turning two images until they overlap pixel by pixel.
