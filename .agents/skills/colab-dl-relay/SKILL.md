---
name: colab-dl-relay
description: >-
  Pre-configure and launch cross-instance serial relay training for deep learning
  models on Google Colab via colab-cli and Google Drive (rclone), with Cloud
  First-Step Smoke verification and automatic fail-fast colab stop protection.
  Use this skill when the user wants to adapt a PyTorch, HuggingFace Trainer,
  PyTorch Lightning, YOLO, or custom deep learning training script for multi-round
  Colab relay training, sync datasets and training results with Google Drive, or
  launch unattended serial relay training monitored by Colab Pilot.
---

# Colab Deep Learning Serial Relay Skill (`colab-dl-relay`)

This skill standardizes any local deep learning training script for **cross-instance serial relay training** on Google Colab, performs a **Colab Cloud First-Step Smoke Test** on the real cloud GPU (with automatic `colab stop` fail-fast protection if CUDA OOM or Drive sync fails), and orchestrates multi-round execution via `colab-cli` + `rclone` while reporting live progress to **Colab Pilot**.

## Core Design Principles

1. **Pre-configure Before Cloud Upload**: Inject checkpoint saving/loading, Google Drive dataset/results sync, and telemetry reporting on the local machine *before* provisioning a Colab VM.
2. **Two-Tier Verification (Local Dry-Run + Cloud First-Step Real Smoke)**:
   - **Local (< 5s)**: Check Python syntax and optionally verify tensor shapes & checkpoint dictionary keys via [`smoke_test_dl.py`](./scripts/smoke_test_dl.py).
   - **Colab Cloud First-Step Smoke (`CLOUD_SMOKE_PASSED`)**: Because local OS/GPU differs from Colab Linux + CUDA, the real smoke test runs on **Epoch 1 on the Colab GPU**—immediately testing real dataset loading, real batch VRAM usage, and forcing an immediate `rclone` write of `checkpoint.pt` + `results/` to Google Drive.
   - **Fail-Fast Auto-Stop**: If Epoch 1 on Colab fails (e.g. `CUDA out of memory`, missing package, or Drive write failure), [`dl_relay_runner.py`](./scripts/dl_relay_runner.py) immediately runs `colab stop -s <session>` to terminate the VM and prevent idle Compute Unit (CU) billing.
3. **Make-After-Break Serial Handover (Zero Double Billing)**: At 22.5 hours, the instance saves its checkpoint to Google Drive, exits with `HANDOVER_REQUESTED`, terminates the old VM (`colab stop`), and immediately provisions the next VM to resume from `last_epoch + 1`.

---

## Fast-Path Execution & Windows PowerShell Rules

1. **Omit Empty String Flags in PowerShell**:
   - In Windows PowerShell 5.1, passing `--gdrive-dataset-dir ""` strips the empty quotes and causes `argparse` to fail with `expected one argument`. Omit `--gdrive-dataset-dir` when no remote dataset directory is needed.
2. **Avoid Ad-Hoc `colab exec` One-Liners with Nested Quotes**:
   - Do not construct complex `python -c "..."` strings with nested `\"` or shell pipes `||` in PowerShell. Always invoke `smoke_test_dl.py` and `dl_relay_runner.py` directly via `python -u`.
3. **Windowless Background Processes on Windows**:
   - When spawning background daemon processes on Windows, use `creationflags = 0x08000000` (`CREATE_NO_WINDOW`) only; never combine with `DETACHED_PROCESS (0x00000008)` which causes child console binaries (`colab.exe`, `rclone.exe`) to pop up visible flashing console windows.

---

## Step-by-Step Workflow

### Step 1: Choose the Adaptation Pattern & Configure the Script

Copy or import [`colab_relay_adapter.py`](./scripts/colab_relay_adapter.py) into the target workspace and apply one of the three patterns:

#### Pattern A: Custom Training Loop (PyTorch / TensorFlow / JAX)
Inject `ColabRelayContext` into the training script (`train.py`):

```python
from colab_relay_adapter import ColabRelayContext

relay = ColabRelayContext(
    model_name="resnet50-cifar",
    framework="pytorch",
    total_epochs=int(os.environ.get("TOTAL_EPOCHS", "100")),
    gdrive_dir=os.environ.get("GDRIVE_DIR", "gdrive:colab_relay/resnet50"),
    gdrive_dataset_dir=os.environ.get("GDRIVE_DATASET_DIR", ""),
    local_dataset_dir="./dataset",
    results_dir="./results",
)

# 1. Pull dataset & latest checkpoint from Google Drive
relay.prepare_cloud_workspace()

# 2. Resume model(s), optimizer(s), scheduler(s), and AMP GradScaler
start_epoch, extra = relay.load_torch_checkpoint(
    models=model,
    optimizers=optimizer,
    schedulers=scheduler,
    scaler=scaler,
)

# 3. Training loop (Epoch 1 on Colab automatically triggers Cloud First-Step Smoke + Drive write)
for epoch in range(start_epoch, relay.total_epochs + 1):
    loss, metrics = train_one_epoch(model, dataloader, optimizer)
    if relay.step_end(
        epoch=epoch,
        loss=loss,
        models=model,
        optimizers=optimizer,
        schedulers=scheduler,
        scaler=scaler,
        metrics=metrics,
    ):
        sys.exit(0)

# 4. All epochs completed -> upload final model, results/, and training_done.flag
relay.finish_training(final_epoch=relay.total_epochs, final_loss=loss, metrics=metrics)
```

#### Pattern B: High-Level Frameworks (HuggingFace `Trainer` / Lightning / YOLO)
Attach `make_hf_trainer_callback(relay)` without modifying framework internals.

#### Pattern C: Black-Box Third-Party CLI Project
Use `colab_relay_adapter.py --wrap "<train_cmd>"` to wrap any unmodified repository.

---

### Step 2: (Optional) Local Tensor & Checkpoint Dry-Run

If you want to verify local tensor shapes and checkpoint save/load keys before uploading, run:

```powershell
python -u "C:\Users\86177\.gemini\config\skills\colab-dl-relay\scripts\smoke_test_dl.py" `
  --script train.py `
  --gdrive-dir gdrive:colab_relay
```

---

### Step 3: Upload to Colab & Launch Serial Relay (with Cloud First-Step Smoke)

Launch [`dl_relay_runner.py`](./scripts/dl_relay_runner.py) to execute the training on Google Colab via `colab-cli`:

```powershell
python -u "C:\Users\86177\.gemini\config\skills\colab-dl-relay\scripts\dl_relay_runner.py" `
  --session <workspace-session-name> `
  --workspace . `
  --script train.py `
  --gpu T4 `
  --high-mem `
  --gdrive-dir gdrive:colab_relay `
  --total-epochs 100 `
  --max-session-hours 22.5 `
  --backup-interval-minutes 30 `
  --reset-done-flag `
  --detach
```

- On Round #1 Epoch 1, the cloud script automatically performs the **Colab Cloud First-Step Smoke Test** (`CLOUD_SMOKE_PASSED`) and pushes the first checkpoint to Google Drive.
- If the cloud smoke check fails, `dl_relay_runner.py` immediately executes `colab stop -s <session>` to stop billing and reports the error.
